"""
tools/run_recall_lift_preset.py

Run a reproducible recall-lift experiment profile (train + causal analysis).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from tools.measure_recall_lift import analyze  # type: ignore


_RUN_RE = re.compile(r"\brun_id\s*:\s*(run_[0-9a-f]{32})\b", re.IGNORECASE)
_RUN_DIR_RE = re.compile(r"runs[\\/](run_[0-9a-f]{32})\b", re.IGNORECASE)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    return obj


def _fmt_kv(key: str, value: Any) -> str:
    if isinstance(value, bool):
        v = "true" if value else "false"
    elif isinstance(value, (int, float)):
        v = str(value)
    elif isinstance(value, (list, dict)):
        v = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    else:
        v = str(value)
    return f"{key}={v}"


def _build_train_cmd(profile: Dict[str, Any], *, episodes: int | None, seed: int | None) -> List[str]:
    train = dict(profile.get("train") or {})
    verse = str(train.get("verse", "")).strip()
    algo = str(train.get("algo", "")).strip()
    if not verse or not algo:
        raise ValueError("Profile train section requires verse and algo")
    cmd: List[str] = [sys.executable, "tools/train_agent.py", "--algo", algo, "--verse", verse]
    cmd.extend(["--episodes", str(int(episodes if episodes is not None else train.get("episodes", 200)))])
    cmd.extend(["--max_steps", str(int(train.get("max_steps", 50)))])
    cmd.extend(["--seed", str(int(seed if seed is not None else train.get("seed", 123)))])
    if bool(train.get("train_flag", True)):
        cmd.append("--train")

    for k, v in dict(train.get("vparam") or {}).items():
        cmd.extend(["--vparam", _fmt_kv(str(k), v)])
    for k, v in dict(train.get("aconfig") or {}).items():
        cmd.extend(["--aconfig", _fmt_kv(str(k), v)])
    return cmd


def _extract_run_id(stdout: str, stderr: str) -> str:
    blob = f"{stdout}\n{stderr}"
    m = _RUN_RE.search(blob)
    if m:
        return str(m.group(1))
    m = _RUN_DIR_RE.search(blob)
    if m:
        return str(m.group(1))
    raise RuntimeError("Could not parse run_id from train_agent output")


def _make_out_path(profile_name: str, run_id: str) -> str:
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(os.path.join("models", "validation"), exist_ok=True)
    return os.path.join("models", "validation", f"recall_lift_{profile_name}_{run_id}_{ts}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a recall-lift experiment preset (train + analyze)")
    ap.add_argument("--preset_file", type=str, default=os.path.join("experiment", "recall_lift_presets.json"))
    ap.add_argument("--profile", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=None, help="Override train episodes")
    ap.add_argument("--seed", type=int, default=None, help="Override train seed")
    ap.add_argument("--bootstrap", type=int, default=None, help="Override analysis bootstrap resamples")
    ap.add_argument("--horizon", type=int, default=None, help="Override analysis hazard horizon")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--save_json", type=str, default=None, help="Output path for combined result JSON")
    args = ap.parse_args()

    preset = _load_json(str(args.preset_file))
    profiles = preset.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"Invalid preset file: missing profiles in {args.preset_file}")
    if str(args.profile) not in profiles:
        raise KeyError(f"Unknown profile '{args.profile}'. Available: {sorted(profiles.keys())}")

    profile = dict(profiles[str(args.profile)] or {})
    train_cmd = _build_train_cmd(profile, episodes=args.episodes, seed=args.seed)
    if bool(args.dry_run):
        print(" ".join(train_cmd))
        return

    proc = subprocess.run(train_cmd, capture_output=True, text=True, cwd=os.getcwd())
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    if int(proc.returncode) != 0:
        raise SystemExit(int(proc.returncode))

    run_id = _extract_run_id(proc.stdout or "", proc.stderr or "")
    run_dir = os.path.join("runs", run_id)

    analysis_cfg = dict(profile.get("analysis") or {})
    result = analyze(
        run_dir=run_dir,
        horizon=int(args.horizon if args.horizon is not None else analysis_cfg.get("horizon", 20)),
        n_boot=max(100, int(args.bootstrap if args.bootstrap is not None else analysis_cfg.get("bootstrap", 2000))),
        seed=int(analysis_cfg.get("seed", 7)),
        require_gate_passed=bool(analysis_cfg.get("require_gate_passed", False)),
    )
    payload = {
        "preset_file": str(args.preset_file),
        "profile": str(args.profile),
        "description": str(profile.get("description", "")),
        "train_command": train_cmd,
        "run_id": run_id,
        "run_dir": run_dir,
        "analysis": result,
    }
    out_path = str(args.save_json or _make_out_path(str(args.profile), run_id))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps({"run_id": run_id, "run_dir": run_dir, "result_json": out_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
