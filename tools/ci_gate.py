"""
tools/ci_gate.py

Local/CI quality gate:
- compile check
- tool entrypoint health check
- universal model build + validation threshold gate
- multi-agent communication regression gate
- broad-exception regression gate
- artifact hygiene routine
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Dict, List, Set


ROOT = Path(__file__).resolve().parent.parent


def _run(
    cmd: List[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        capture_output=bool(capture_output),
        text=bool(capture_output),
    )
    if proc.returncode != 0:
        detail = ""
        if capture_output:
            text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if text:
                detail = f"\n--- command output ---\n{text[-4000:]}"
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}{detail}")
    if capture_output:
        return proc.stdout or ""
    return ""


def _python_files() -> List[Path]:
    out: List[Path] = []
    skip_exact = {
        "__pycache__",
        "node_modules",
        ".git",
    }
    for p in ROOT.rglob("*.py"):
        parts = {part.lower() for part in p.parts}
        if parts.intersection(skip_exact):
            continue
        # Ignore common virtualenv-style directories (.venv, .venv312, venv, venv311, etc.).
        if any(part.startswith(".venv") or part.startswith("venv") for part in parts):
            continue
        out.append(p)
    return out


def _tool_files() -> List[Path]:
    tools_dir = ROOT / "tools"
    return sorted([p for p in tools_dir.glob("*.py") if p.is_file()])


def _list_runs(runs_root: Path) -> List[Path]:
    runs = runs_root
    if not runs.is_dir():
        return []
    out = []
    for p in runs.iterdir():
        if p.is_dir() and (p / "events.jsonl").is_file():
            out.append(p)
    return sorted(out)


def _repo_diff_paths() -> Set[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("workspace hygiene requires a git repository")
    out: Set[str] = set()
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip("\n")
        if not line:
            continue
        entry = line[3:] if len(line) >= 4 else ""
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        entry = entry.strip().strip('"').replace("\\", "/")
        if entry:
            out.add(entry)
    return out


def _rel_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def workspace_hygiene_gate(*, baseline: Set[str], allow_prefixes: List[str]) -> None:
    current = _repo_diff_paths()
    new_paths = sorted(p for p in (current - baseline))

    normalized: List[str] = []
    for p in allow_prefixes:
        v = str(p).strip().replace("\\", "/").strip("/")
        if v:
            normalized.append(v)

    violations: List[str] = []
    for p in new_paths:
        pp = p.strip("/").replace("\\", "/")
        allowed = any(pp == pref or pp.startswith(pref + "/") for pref in normalized)
        if not allowed:
            violations.append(p)

    if violations:
        sample = ", ".join(violations[:8])
        raise RuntimeError(
            f"workspace hygiene failed: {len(violations)} new changed paths outside allowlist; sample={sample}"
        )


def compile_gate() -> None:
    for p in _python_files():
        try:
            # Syntax-check without writing __pycache__ artifacts (avoids file-lock issues).
            with tokenize.open(str(p)) as f:
                src = f.read()
            compile(src, str(p), "exec")
        except Exception as exc:
            raise RuntimeError(f"compile gate failed for {p}") from exc


def help_gate() -> None:
    for p in _tool_files():
        # Keep CI logs concise: capture help text and only surface it on failure.
        _run([sys.executable, str(p), "--help"], capture_output=True)


def ensure_bootstrap_run(*, runs_root: Path) -> None:
    if _list_runs(runs_root):
        return
    runs_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "tools/train_agent.py",
            "--algo",
            "random",
            "--verse",
            "line_world",
            "--episodes",
            "2",
            "--max_steps",
            "5",
            "--runs_root",
            str(runs_root),
        ]
    )


def model_gate(min_coverage: float, min_action_accuracy: float, *, runs_root: Path, model_out_dir: Path) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    model_out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "tools/universal_model.py",
            "build",
            "--runs_root",
            str(runs_root),
            "--out_dir",
            str(model_out_dir),
            "--snapshot_memory",
            "--max_events",
            "200",
        ]
    )
    _run(
        [
            sys.executable,
            "tools/universal_model.py",
            "validate_set",
            "--model_dir",
            str(model_out_dir),
            "--runs_root",
            str(runs_root),
            "--split",
            "all",
            "--min_coverage",
            str(min_coverage),
            "--min_action_accuracy",
            str(min_action_accuracy),
        ]
    )


def promotion_gate_smoke(manifest_path: str, verse: str, suite: str) -> None:
    cfg = json.dumps({"manifest_path": manifest_path, "verse_name": verse})
    _run(
        [
            sys.executable,
            "tools/eval_harness.py",
            "--baseline_algo",
            "gateway",
            "--baseline_config_json",
            cfg,
            "--candidate_algo",
            "gateway",
            "--candidate_config_json",
            cfg,
            "--suite",
            suite,
            "--verse",
            verse,
            "--bootstrap_samples",
            "200",
            "--fail_on_gate",
        ]
    )


def production_readiness_gate(
    *,
    manifest_path: str,
    bench_json: str,
    min_verses: int,
    min_episodes: int,
    min_success_rate: float,
    max_bench_age_hours: float,
    max_safety_violation_rate: float,
    require_run_dirs: bool,
    require_benchmark: bool,
) -> None:
    cmd = [
        sys.executable,
        "tools/production_readiness_gate.py",
        "--manifest_path",
        manifest_path,
        "--bench_json",
        bench_json,
        "--min_verses",
        str(int(min_verses)),
        "--min_episodes",
        str(int(min_episodes)),
        "--min_success_rate",
        str(float(min_success_rate)),
        "--max_bench_age_hours",
        str(float(max_bench_age_hours)),
        "--max_safety_violation_rate",
        str(float(max_safety_violation_rate)),
    ]
    if require_run_dirs:
        cmd.append("--require_run_dirs")
    if require_benchmark:
        cmd.append("--require_benchmark")
    _run(cmd)


def memory_soak_gate(
    *,
    duration_sec: int,
    sample_interval_sec: int,
    max_slope_mb_per_hour: float,
    output_json: str,
) -> None:
    cmd = [
        sys.executable,
        "tools/memory_soak_gate.py",
        "--duration_sec",
        str(max(2, int(duration_sec))),
        "--sample_interval_sec",
        str(max(1, int(sample_interval_sec))),
        "--max_slope_mb_per_hour",
        str(float(max_slope_mb_per_hour)),
        "--output_json",
        str(output_json),
        "--fail_on_slope",
    ]
    _run(cmd)


def _load_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _count_broad_exception_blocks(files: List[Path]) -> int:
    re_broad = re.compile(r"^\s*except\s+Exception\b")
    re_bare = re.compile(r"^\s*except\s*:\s*$")
    count = 0
    for p in files:
        try:
            with tokenize.open(str(p)) as f:
                for line in f:
                    if re_broad.match(line) or re_bare.match(line):
                        count += 1
        except Exception:
            # compile_gate surfaces syntax issues separately; skip unreadable files here.
            continue
    return int(count)


def exception_debt_gate(
    *,
    baseline_json: str,
    max_broad: int,
    fail_on_regression: bool,
) -> Dict[str, int]:
    files = _python_files()
    broad = _count_broad_exception_blocks(files)
    baseline = _load_json_object((ROOT / str(baseline_json)).resolve())
    baseline_max = baseline.get("broad_exception_max")
    if max_broad >= 0:
        allowed = int(max_broad)
    elif isinstance(baseline_max, int):
        allowed = int(baseline_max)
    else:
        allowed = int(broad)

    if bool(fail_on_regression) and int(broad) > int(allowed):
        raise RuntimeError(
            f"exception debt regression: broad_exception_blocks={broad} exceeds allowed={allowed}"
        )
    return {"broad_exception_blocks": int(broad), "allowed": int(allowed)}


def artifact_hygiene_gate(
    *,
    root: str,
    patterns: List[str],
    keep: int,
    min_age_days: float,
    delete: bool,
) -> None:
    selected_patterns = [str(p).strip() for p in patterns if str(p).strip()]
    if not selected_patterns:
        selected_patterns = ["runs*", "central_memory*"]
    cmd: List[str] = [
        sys.executable,
        "tools/cleanup_artifacts.py",
        "--root",
        str(root),
        "--keep",
        str(max(0, int(keep))),
        "--min_age_days",
        str(max(0.0, float(min_age_days))),
    ]
    for p in selected_patterns:
        cmd.extend(["--pattern", str(p)])
    if bool(delete):
        cmd.append("--delete")
    _run(cmd)


def _cli_flag_present(cli_args: List[str], name: str) -> bool:
    raw_name = str(name).strip()
    flag_dash = "--" + raw_name.replace("_", "-")
    flag_underscore = "--" + raw_name
    neg_flag_dash = "--no-" + raw_name.replace("_", "-")
    neg_flag_underscore = "--no-" + raw_name
    for raw in cli_args:
        token = str(raw or "").strip()
        if not token:
            continue
        if (
            token == flag_dash
            or token.startswith(flag_dash + "=")
            or token == flag_underscore
            or token.startswith(flag_underscore + "=")
        ):
            return True
        if (
            token == neg_flag_dash
            or token.startswith(neg_flag_dash + "=")
            or token == neg_flag_underscore
            or token.startswith(neg_flag_underscore + "=")
        ):
            return True
    return False


def _set_profile_value(
    args: argparse.Namespace,
    *,
    cli_args: List[str],
    name: str,
    value: bool,
) -> None:
    if _cli_flag_present(cli_args, name):
        return
    setattr(args, name, value)


def _apply_profile_defaults(args: argparse.Namespace, cli_args: List[str] | None = None) -> None:
    profile = str(getattr(args, "profile", "custom") or "custom").strip().lower()
    provided = list(cli_args or [])
    if profile == "fast":
        _set_profile_value(args, cli_args=provided, name="skip_compile", value=False)
        _set_profile_value(args, cli_args=provided, name="skip_help", value=True)
        _set_profile_value(args, cli_args=provided, name="skip_model", value=True)
        _set_profile_value(args, cli_args=provided, name="bootstrap_run", value=False)
        _set_profile_value(args, cli_args=provided, name="run_comm_gate", value=True)
        _set_profile_value(args, cli_args=provided, name="run_exception_debt_gate", value=True)
        _set_profile_value(args, cli_args=provided, name="run_production_readiness", value=False)
        _set_profile_value(args, cli_args=provided, name="run_memory_soak", value=False)
        _set_profile_value(args, cli_args=provided, name="run_artifact_hygiene", value=False)
        _set_profile_value(args, cli_args=provided, name="enforce_workspace_hygiene", value=False)
        return
    if profile == "full":
        _set_profile_value(args, cli_args=provided, name="skip_compile", value=False)
        _set_profile_value(args, cli_args=provided, name="skip_help", value=False)
        _set_profile_value(args, cli_args=provided, name="skip_model", value=False)
        _set_profile_value(args, cli_args=provided, name="bootstrap_run", value=True)
        _set_profile_value(args, cli_args=provided, name="run_comm_gate", value=True)
        _set_profile_value(args, cli_args=provided, name="run_exception_debt_gate", value=True)
        _set_profile_value(args, cli_args=provided, name="run_artifact_hygiene", value=True)
        return


def communication_gate(*, tests: List[str] | None = None) -> None:
    selected = list(tests or [])
    if not selected:
        selected = [
            "tests/test_shared_memory_pool.py",
            "tests/test_marl_social_frontier.py",
            "tests/test_marl_aware_communication.py",
        ]
    cmd = [sys.executable, "-m", "pytest", "-q", *selected]
    # Keep logs focused; pytest output is shown on failure.
    _run(cmd, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        type=str,
        default="custom",
        choices=["custom", "fast", "full"],
        help="Preset gate profile. custom keeps explicit flag behavior.",
    )
    ap.add_argument("--release_gate", action="store_true", help="Run release-profile quality gate.")
    ap.add_argument("--skip_compile", action="store_true")
    ap.add_argument("--skip_help", action="store_true")
    ap.add_argument("--skip_model", action="store_true")
    ap.add_argument("--bootstrap_run", action="store_true")
    ap.add_argument("--min_coverage", type=float, default=0.05)
    ap.add_argument("--min_action_accuracy", type=float, default=0.05)
    ap.add_argument("--run_promotion_gate", action="store_true")
    ap.add_argument("--promotion_manifest", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument("--promotion_verse", type=str, default="line_world")
    ap.add_argument("--promotion_suite", type=str, default="quick", choices=["target", "quick", "full"])
    ap.add_argument("--run_production_readiness", action="store_true")
    ap.add_argument("--prod_manifest", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument("--prod_bench_json", type=str, default=os.path.join("models", "benchmarks_smoke", "latest.json"))
    ap.add_argument("--prod_min_verses", type=int, default=1)
    ap.add_argument("--prod_min_episodes", type=int, default=50)
    ap.add_argument("--prod_min_success_rate", type=float, default=0.6)
    ap.add_argument("--prod_max_bench_age_hours", type=float, default=72.0)
    ap.add_argument("--prod_max_safety_violation_rate", type=float, default=0.2)
    ap.add_argument("--prod_require_run_dirs", action="store_true")
    ap.add_argument("--prod_require_benchmark", action="store_true")
    ap.add_argument("--run_memory_soak", action="store_true")
    ap.add_argument("--run_comm_gate", action="store_true")
    ap.add_argument("--run_exception_debt_gate", action="store_true")
    ap.add_argument("--run_artifact_hygiene", action="store_true")
    ap.add_argument(
        "--comm_test",
        action="append",
        default=None,
        help="Optional pytest target for communication gate (repeatable).",
    )
    ap.add_argument(
        "--exception_baseline_json",
        type=str,
        default=os.path.join("tools", "quality_baseline.json"),
    )
    ap.add_argument(
        "--exception_max_broad",
        type=int,
        default=-1,
        help="Allowed broad exception blocks. -1 uses baseline json.",
    )
    ap.add_argument(
        "--exception_fail_on_regression",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--artifact_root", type=str, default=".")
    ap.add_argument("--artifact_pattern", action="append", default=None)
    ap.add_argument("--artifact_keep", type=int, default=6)
    ap.add_argument("--artifact_min_age_days", type=float, default=7.0)
    ap.add_argument("--artifact_delete", action="store_true")
    ap.add_argument("--memory_soak_duration_sec", type=int, default=60)
    ap.add_argument("--memory_soak_sample_interval_sec", type=int, default=8)
    ap.add_argument("--memory_soak_max_slope_mb_per_hour", type=float, default=48.0)
    ap.add_argument(
        "--memory_soak_output_json",
        type=str,
        default=os.path.join("models", "memory_health", "latest.json"),
    )
    ap.add_argument("--ci_output_root", type=str, default=".ci_artifacts")
    ap.add_argument("--clean_ci_output_root", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--enforce_workspace_hygiene", action="store_true")
    ap.add_argument(
        "--hygiene_require_git",
        action="store_true",
        help="Fail when workspace hygiene is requested but git metadata is unavailable.",
    )
    ap.add_argument("--hygiene_allow_prefix", action="append", default=None)
    raw_argv = sys.argv[1:]
    args = ap.parse_args(raw_argv)

    _apply_profile_defaults(args, cli_args=raw_argv)

    if args.release_gate:
        args.skip_compile = False
        args.skip_help = False
        args.skip_model = False
        args.bootstrap_run = True
        args.run_production_readiness = True
        args.prod_require_benchmark = True
        args.enforce_workspace_hygiene = True
        args.run_memory_soak = True
        args.run_comm_gate = True
        args.run_exception_debt_gate = True
        args.run_artifact_hygiene = True

    ci_output_root = (ROOT / str(args.ci_output_root)).resolve()
    runs_root = ci_output_root / "runs"
    model_out_dir = ci_output_root / "models" / "universal_model"
    allow_prefixes = list(args.hygiene_allow_prefix or [])
    allow_prefixes.append(_rel_to_root(ci_output_root))
    baseline_paths: Set[str] = set()
    hygiene_active = False

    if bool(args.enforce_workspace_hygiene):
        try:
            baseline_paths = _repo_diff_paths()
            hygiene_active = True
        except RuntimeError as exc:
            if (not bool(args.hygiene_require_git)) and ("requires a git repository" in str(exc)):
                print("workspace hygiene gate: skipped (not a git repository)")
                hygiene_active = False
            else:
                raise

    if bool(args.clean_ci_output_root) and ci_output_root.is_dir():
        shutil.rmtree(ci_output_root, ignore_errors=True)

    if not args.skip_compile:
        compile_gate()
        print("compile gate: ok")
    if not args.skip_help:
        help_gate()
        print("help gate: ok")
    if not args.skip_model:
        if args.bootstrap_run:
            ensure_bootstrap_run(runs_root=runs_root)
        model_gate(
            args.min_coverage,
            args.min_action_accuracy,
            runs_root=runs_root,
            model_out_dir=model_out_dir,
        )
        print("model gate: ok")
    if args.run_promotion_gate:
        promotion_gate_smoke(args.promotion_manifest, args.promotion_verse, args.promotion_suite)
        print("promotion gate: ok")
    if args.run_production_readiness:
        production_readiness_gate(
            manifest_path=args.prod_manifest,
            bench_json=args.prod_bench_json,
            min_verses=args.prod_min_verses,
            min_episodes=args.prod_min_episodes,
            min_success_rate=args.prod_min_success_rate,
            max_bench_age_hours=args.prod_max_bench_age_hours,
            max_safety_violation_rate=args.prod_max_safety_violation_rate,
            require_run_dirs=bool(args.prod_require_run_dirs),
            require_benchmark=bool(args.prod_require_benchmark),
        )
        print("production readiness gate: ok")
    if args.run_memory_soak:
        memory_soak_gate(
            duration_sec=int(args.memory_soak_duration_sec),
            sample_interval_sec=int(args.memory_soak_sample_interval_sec),
            max_slope_mb_per_hour=float(args.memory_soak_max_slope_mb_per_hour),
            output_json=str(args.memory_soak_output_json),
        )
        print("memory soak gate: ok")
    if args.run_comm_gate:
        communication_gate(tests=list(args.comm_test or []))
        print("communication gate: ok")
    if args.run_exception_debt_gate:
        debt = exception_debt_gate(
            baseline_json=str(args.exception_baseline_json),
            max_broad=int(args.exception_max_broad),
            fail_on_regression=bool(args.exception_fail_on_regression),
        )
        print(
            "exception debt gate: ok "
            f"(broad_exception_blocks={int(debt['broad_exception_blocks'])}, allowed={int(debt['allowed'])})"
        )
    if args.run_artifact_hygiene:
        artifact_hygiene_gate(
            root=str(args.artifact_root),
            patterns=list(args.artifact_pattern or []),
            keep=max(0, int(args.artifact_keep)),
            min_age_days=max(0.0, float(args.artifact_min_age_days)),
            delete=bool(args.artifact_delete),
        )
        print("artifact hygiene gate: ok")

    if bool(args.clean_ci_output_root) and ci_output_root.is_dir():
        shutil.rmtree(ci_output_root, ignore_errors=True)
    if bool(args.enforce_workspace_hygiene) and bool(hygiene_active):
        workspace_hygiene_gate(baseline=baseline_paths, allow_prefixes=allow_prefixes)
        print("workspace hygiene gate: ok")

    print("ci gate: all checks passed")


if __name__ == "__main__":
    main()
