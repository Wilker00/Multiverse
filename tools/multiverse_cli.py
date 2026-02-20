#!/usr/bin/env python3
"""
tools/multiverse_cli.py

Unified convenience CLI for Multiverse:
- list universes (verses)
- run single/distributed training
- browse run artifacts quickly
- launch the terminal hub
- status snapshot and training profiles
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from verses.registry import list_verses, register_builtin


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    modified_ts: float
    total_size_bytes: int


TRAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "verse": "line_world",
        "algo": "random",
        "episodes": 20,
        "max_steps": 40,
        "eval": False,
        "make_index": False,
    },
    "balanced": {
        "verse": "line_world",
        "algo": "q",
        "episodes": 80,
        "max_steps": 60,
        "eval": True,
        "make_index": True,
    },
    "research": {
        "verse": "warehouse_world",
        "algo": "q",
        "episodes": 200,
        "max_steps": 100,
        "eval": True,
        "make_index": True,
    },
}


DISTRIBUTED_PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "mode": "sharded",
        "verse": "line_world",
        "algo": "q",
        "episodes": 100,
        "max_steps": 50,
        "workers": 2,
    },
    "balanced": {
        "mode": "sharded",
        "verse": "warehouse_world",
        "algo": "q",
        "episodes": 240,
        "max_steps": 100,
        "workers": 4,
    },
    "research": {
        "mode": "pbt",
        "verse": "warehouse_world",
        "algo": "q",
        "episodes": 400,
        "max_steps": 120,
        "workers": 6,
    },
}


def _normalize_remainder(extra: Iterable[str] | None) -> List[str]:
    parts = [str(x) for x in (extra or [])]
    if parts and parts[0] == "--":
        return parts[1:]
    return parts


def _flag_present(raw_argv: Iterable[str], names: Iterable[str]) -> bool:
    candidates = [str(x).strip() for x in names if str(x).strip()]
    for token in [str(x) for x in raw_argv]:
        for name in candidates:
            if token == name or token.startswith(name + "="):
                return True
            if name.startswith("--"):
                neg = "--no-" + name[2:]
                if token == neg or token.startswith(neg + "="):
                    return True
    return False


def _maybe_set_attr(args: argparse.Namespace, raw_argv: Iterable[str], flag_names: Iterable[str], attr: str, value: Any) -> None:
    if _flag_present(raw_argv, flag_names):
        return
    setattr(args, attr, value)


def apply_train_profile(args: argparse.Namespace, raw_argv: Iterable[str]) -> None:
    profile = str(getattr(args, "profile", "custom") or "custom").strip().lower()
    if profile == "custom":
        return
    if profile not in TRAIN_PROFILES:
        raise ValueError(f"Unknown train profile '{profile}'.")
    cfg = TRAIN_PROFILES[profile]
    _maybe_set_attr(args, raw_argv, ["--universe", "--verse"], "verse", str(cfg["verse"]))
    _maybe_set_attr(args, raw_argv, ["--algo"], "algo", str(cfg["algo"]))
    _maybe_set_attr(args, raw_argv, ["--episodes"], "episodes", int(cfg["episodes"]))
    _maybe_set_attr(args, raw_argv, ["--max-steps", "--max_steps"], "max_steps", int(cfg["max_steps"]))
    _maybe_set_attr(args, raw_argv, ["--eval"], "eval", bool(cfg["eval"]))
    _maybe_set_attr(args, raw_argv, ["--make-index", "--make_index"], "make_index", bool(cfg["make_index"]))


def apply_distributed_profile(args: argparse.Namespace, raw_argv: Iterable[str]) -> None:
    profile = str(getattr(args, "profile", "custom") or "custom").strip().lower()
    if profile == "custom":
        return
    if profile not in DISTRIBUTED_PROFILES:
        raise ValueError(f"Unknown distributed profile '{profile}'.")
    cfg = DISTRIBUTED_PROFILES[profile]
    _maybe_set_attr(args, raw_argv, ["--mode"], "mode", str(cfg["mode"]))
    _maybe_set_attr(args, raw_argv, ["--universe", "--verse"], "verse", str(cfg["verse"]))
    _maybe_set_attr(args, raw_argv, ["--algo"], "algo", str(cfg["algo"]))
    _maybe_set_attr(args, raw_argv, ["--episodes"], "episodes", int(cfg["episodes"]))
    _maybe_set_attr(args, raw_argv, ["--max-steps", "--max_steps"], "max_steps", int(cfg["max_steps"]))
    _maybe_set_attr(args, raw_argv, ["--workers"], "workers", int(cfg["workers"]))


def _human_bytes(n: int) -> str:
    size = float(max(0, int(n)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(size)}{units[idx]}"
    return f"{size:.1f}{units[idx]}"


def _iso_local(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _render_cmd(cmd: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(x) for x in cmd])
    return shlex.join([str(x) for x in cmd])


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                total += int(p.stat().st_size)
            except OSError:
                continue
    return total


def discover_runs(runs_root: str) -> List[RunRecord]:
    root = Path(runs_root)
    if not root.is_dir():
        return []

    out: List[RunRecord] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        events = item / "events.jsonl"
        if not events.is_file():
            continue
        try:
            modified = float(max(item.stat().st_mtime, events.stat().st_mtime))
        except OSError:
            modified = 0.0
        out.append(
            RunRecord(
                run_id=item.name,
                run_dir=item,
                modified_ts=modified,
                total_size_bytes=_dir_size_bytes(item),
            )
        )
    out.sort(key=lambda r: r.modified_ts, reverse=True)
    return out


def resolve_run_dir(runs_root: str, run_id: str | None) -> Path:
    root = Path(runs_root)
    if run_id:
        candidate = root / str(run_id)
        if not candidate.is_dir():
            raise FileNotFoundError(f"Run not found: {candidate}")
        return candidate
    runs = discover_runs(runs_root)
    if not runs:
        raise FileNotFoundError(f"No runs with events.jsonl found under: {root}")
    return runs[0].run_dir


def run_process(cmd: List[str], *, capture_output: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=bool(capture_output),
        text=bool(capture_output),
    )
    if int(proc.returncode) != 0:
        detail = ""
        if capture_output:
            text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if text:
                detail = f"\n--- command output ---\n{text[-4000:]}"
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}{detail}")
    if capture_output:
        return str(proc.stdout or "")
    return ""


def _line_count(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


def _run_snapshot(runs_root: str) -> Dict[str, Any]:
    rows = discover_runs(runs_root)
    latest = rows[0] if rows else None
    return {
        "runs_root": str(runs_root),
        "run_count": int(len(rows)),
        "latest": {
            "run_id": latest.run_id,
            "path": str(latest.run_dir),
            "modified": _iso_local(latest.modified_ts),
            "size_bytes": latest.total_size_bytes,
            "size_human": _human_bytes(latest.total_size_bytes),
        }
        if latest
        else None,
    }


def build_train_agent_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_agent.py"),
        "--algo",
        str(args.algo),
        "--verse",
        str(args.verse),
        "--episodes",
        str(int(args.episodes)),
        "--max_steps",
        str(int(args.max_steps)),
        "--seed",
        str(int(args.seed)),
        "--runs_root",
        str(args.runs_root),
    ]
    if args.policy_id:
        cmd.extend(["--policy_id", str(args.policy_id)])
    if bool(args.train):
        cmd.append("--train")
    if bool(args.eval):
        cmd.append("--eval")
    if bool(args.make_index):
        cmd.append("--make_index")
    cmd.extend(_normalize_remainder(args.extra))
    return cmd


def build_train_distributed_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_distributed.py"),
        "--mode",
        str(args.mode),
        "--algo",
        str(args.algo),
        "--verse",
        str(args.verse),
        "--episodes",
        str(int(args.episodes)),
        "--max_steps",
        str(int(args.max_steps)),
        "--workers",
        str(int(args.workers)),
        "--seed",
        str(int(args.seed)),
        "--run_root",
        str(args.runs_root),
    ]
    if args.policy_id:
        cmd.extend(["--policy_id", str(args.policy_id)])
    if bool(args.train):
        cmd.append("--train")
    cmd.extend(_normalize_remainder(args.extra))
    return cmd


def cmd_universe_list(args: argparse.Namespace) -> int:
    register_builtin()
    names = sorted(list_verses().keys())
    if args.contains:
        q = str(args.contains).strip().lower()
        names = [n for n in names if q in str(n).lower()]
    if bool(args.json):
        print(json.dumps({"count": len(names), "universes": names}, indent=2))
        return 0
    for name in names:
        print(name)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cmd = build_train_agent_cmd(args)
    if bool(args.dry_run):
        print(_render_cmd(cmd))
        return 0
    capture = bool(getattr(args, "_capture_subprocess_output", False))
    out = run_process(cmd, capture_output=capture)
    if capture and out:
        print(out.rstrip("\n"))
    return 0


def cmd_distributed(args: argparse.Namespace) -> int:
    cmd = build_train_distributed_cmd(args)
    if bool(args.dry_run):
        print(_render_cmd(cmd))
        return 0
    capture = bool(getattr(args, "_capture_subprocess_output", False))
    out = run_process(cmd, capture_output=capture)
    if capture and out:
        print(out.rstrip("\n"))
    return 0


def cmd_runs_list(args: argparse.Namespace) -> int:
    rows = discover_runs(args.runs_root)
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    if not rows:
        print(f"No runs found under {args.runs_root}")
        return 0
    if bool(args.json):
        payload = [
            {
                "run_id": r.run_id,
                "path": str(r.run_dir),
                "modified": _iso_local(r.modified_ts),
                "size_bytes": r.total_size_bytes,
                "size_human": _human_bytes(r.total_size_bytes),
            }
            for r in rows
        ]
        print(json.dumps({"runs": payload}, indent=2))
        return 0
    print("RUN ID                           MODIFIED             SIZE")
    for row in rows:
        rid = row.run_id[:32].ljust(32)
        print(f"{rid} {_iso_local(row.modified_ts)}  {_human_bytes(row.total_size_bytes)}")
    return 0


def cmd_runs_latest(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.runs_root, run_id=None)
    if bool(args.json):
        st = run_dir.stat()
        payload = {
            "run_id": run_dir.name,
            "path": str(run_dir),
            "modified": _iso_local(st.st_mtime),
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.path_only:
        print(str(run_dir))
    else:
        print(f"Latest run: {run_dir.name}")
        print(str(run_dir))
    return 0


def cmd_runs_files(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.runs_root, run_id=args.run_id)
    iterator = run_dir.rglob("*") if bool(args.recursive) else run_dir.glob("*")
    files = [p for p in iterator if p.is_file()]
    files.sort(key=lambda p: str(p.relative_to(run_dir)).lower())
    if args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        print(f"No files found in {run_dir}")
        return 0
    if bool(args.json):
        payload = []
        for p in files:
            rel = p.relative_to(run_dir).as_posix()
            try:
                size = int(p.stat().st_size)
            except OSError:
                size = -1
            payload.append({"file": rel, "size_bytes": size, "size_human": _human_bytes(max(size, 0)) if size >= 0 else "?"})
        print(json.dumps({"run_id": run_dir.name, "files": payload}, indent=2))
        return 0
    for p in files:
        rel = p.relative_to(run_dir).as_posix()
        try:
            size = _human_bytes(int(p.stat().st_size))
        except OSError:
            size = "?"
        print(f"{rel}\t{size}")
    return 0


def cmd_runs_tail(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.runs_root, run_id=args.run_id)
    file_path = run_dir / str(args.file)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    tail = deque(maxlen=max(1, int(args.lines)))
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tail.append(line.rstrip("\n"))
    for line in tail:
        print(line)
    return 0


def cmd_runs_inspect(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.runs_root, run_id=args.run_id)
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    events_path = run_dir / "events.jsonl"
    episode_path = run_dir / "episodes.jsonl"
    events_count = None
    if bool(args.count_events) and events_path.is_file():
        events_count = _line_count(events_path)
    payload = {
        "run_id": run_dir.name,
        "path": str(run_dir),
        "modified": _iso_local(run_dir.stat().st_mtime),
        "size_bytes": _dir_size_bytes(run_dir),
        "size_human": _human_bytes(_dir_size_bytes(run_dir)),
        "file_count": len(files),
        "has_events_jsonl": bool(events_path.is_file()),
        "has_episodes_jsonl": bool(episode_path.is_file()),
        "events_line_count": events_count,
    }
    if bool(args.json):
        print(json.dumps(payload, indent=2))
        return 0
    print(f"Run ID          : {payload['run_id']}")
    print(f"Path            : {payload['path']}")
    print(f"Modified        : {payload['modified']}")
    print(f"Size            : {payload['size_human']} ({payload['size_bytes']} bytes)")
    print(f"File count      : {payload['file_count']}")
    print(f"events.jsonl    : {'yes' if payload['has_events_jsonl'] else 'no'}")
    print(f"episodes.jsonl  : {'yes' if payload['has_episodes_jsonl'] else 'no'}")
    if events_count is not None:
        print(f"events lines    : {events_count}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    register_builtin()
    verses = sorted(list_verses().keys())
    snap = _run_snapshot(args.runs_root)
    payload = {
        "universes": {"count": len(verses), "sample": verses[:10]},
        "runs": snap,
    }
    if bool(args.json):
        print(json.dumps(payload, indent=2))
        return 0

    print("Multiverse Status")
    print("-----------------")
    print(f"Universes      : {len(verses)}")
    print(f"Runs root      : {snap['runs_root']}")
    print(f"Run count      : {snap['run_count']}")
    latest = snap.get("latest")
    if latest:
        print(f"Latest run     : {latest['run_id']}")
        print(f"Latest path    : {latest['path']}")
        print(f"Latest updated : {latest['modified']}")
        print(f"Latest size    : {latest['size_human']}")
    else:
        print("Latest run     : none")
    print("")
    print("Quick Start")
    print("-----------")
    print("multiverse universe list")
    print("multiverse train --profile quick")
    print("multiverse runs latest")
    return 0


def cmd_hub(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "universe_hub.py"),
        "--runs_root",
        str(args.runs_root),
        "--refresh_sec",
        str(float(args.refresh_sec)),
    ]
    if bool(args.once):
        cmd.append("--once")
    cmd.extend(_normalize_remainder(args.extra))
    if bool(args.dry_run):
        print(_render_cmd(cmd))
        return 0
    capture = bool(getattr(args, "_capture_subprocess_output", False))
    out = run_process(cmd, capture_output=capture)
    if capture and out:
        print(out.rstrip("\n"))
    return 0


def _split_soft_lines(text: str, width: int) -> List[str]:
    w = max(8, int(width))
    out: List[str] = []
    for raw in str(text).splitlines() or [""]:
        line = raw
        if not line:
            out.append("")
            continue
        while len(line) > w:
            out.append(line[:w])
            line = line[w:]
        out.append(line)
    return out


def _windows_ctrl_esc_pressed() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes  # type: ignore

        user32 = ctypes.windll.user32
        vk_control = 0x11
        vk_escape = 0x1B
        ctrl_down = bool(int(user32.GetAsyncKeyState(vk_control)) & 0x8000)
        esc_down = bool(int(user32.GetAsyncKeyState(vk_escape)) & 0x8000)
        return bool(ctrl_down and esc_down)
    except Exception:
        return False


def _enable_windows_vt() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes  # type: ignore

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


class InteractiveShell:
    def __init__(self, *, runs_root: str = "runs") -> None:
        self.runs_root = str(runs_root)
        self.logs: deque[str] = deque(maxlen=1200)
        self.input_buf = ""
        self.history: List[str] = []
        self.history_pos = 0
        self.last_command = ""
        self.last_rc = 0
        self.running = True
        self._cursor_visible = True
        self._busy = False
        self.log_scroll = 0
        self.suggestion_idx = 0
        self.suggestion_page = 0
        self.theme = "dark"
        self.intensity = 1
        self._ac_seed = ""
        self._ac_matches: List[str] = []
        self._ac_idx = 0
        self.layout = "compact"
        self.max_ui_width = 136
        self.max_ui_height = 30

    def _log(self, text: str) -> None:
        for line in _split_soft_lines(str(text), width=400):
            self.logs.append(line)
        self.log_scroll = 0

    def _suggestion_pages(self) -> List[Dict[str, Any]]:
        base = [
            "status",
            "u ls --contains line",
            "t --profile quick",
            "t --profile research --dry-run",
            "r ls --limit 10",
            "r i --count-events",
            "r tail --file events.jsonl --lines 20",
            "h --once",
            ":theme dark",
            ":theme matrix",
            ":intensity 2",
            "clear",
        ]
        train = [
            "t --profile quick",
            "t --profile balanced",
            "t --profile research",
            "t --universe warehouse_world --algo q --episodes 120 --eval",
            "dist --profile balanced --dry-run",
            "dist --mode pbt --profile research --dry-run",
        ]
        runs = [
            "r latest",
            "r ls --limit 15",
            "r i --count-events",
            "r f --recursive --limit 80",
            "r tail --file events.jsonl --lines 40",
            "r tail --file episodes.jsonl --lines 20",
        ]
        system = [
            "status",
            "u ls",
            "help",
            ":layout compact",
            ":layout full",
            ":theme dark",
            ":theme glass",
            ":theme matrix",
            ":theme contrast",
            ":intensity 0",
            ":intensity 1",
            ":intensity 2",
            ":intensity 3",
            "clear",
        ]
        lc = str(self.last_command).strip().lower()
        if lc.startswith("u ") or lc.startswith("universe "):
            train.insert(0, "u ls --contains world")
        if lc.startswith("t ") or lc.startswith("train "):
            runs.insert(0, "r latest")
            runs.insert(1, "r i --count-events")
        if lc.startswith("r ") or lc.startswith("runs "):
            runs.insert(0, "r i --count-events")
        return [
            {"title": "NEXT ACTIONS", "items": base},
            {"title": "TRAIN FLOW", "items": train},
            {"title": "RUNS FLOW", "items": runs},
            {"title": "SHELL", "items": system},
        ]

    def _active_suggestions(self) -> List[str]:
        pages = self._suggestion_pages()
        if not pages:
            return []
        idx = max(0, min(self.suggestion_page, len(pages) - 1))
        self.suggestion_page = idx
        items = pages[idx].get("items", [])
        return [str(x) for x in items]

    def _theme_codes(self) -> Dict[str, str]:
        themes = {
            "dark": {
                "header": "\x1b[1;38;5;117m",
                "border": "\x1b[38;5;240m",
                "panel": "\x1b[38;5;251m",
                "hint": "\x1b[38;5;244m",
                "accent": "\x1b[38;5;153m",
                "select": "\x1b[48;5;24;38;5;231m",
            },
            "glass": {
                "header": "\x1b[1;36m",
                "border": "\x1b[38;5;110m",
                "panel": "\x1b[38;5;147m",
                "hint": "\x1b[38;5;248m",
                "accent": "\x1b[38;5;81m",
                "select": "\x1b[48;5;24;38;5;231m",
            },
            "matrix": {
                "header": "\x1b[1;32m",
                "border": "\x1b[38;5;34m",
                "panel": "\x1b[38;5;77m",
                "hint": "\x1b[38;5;71m",
                "accent": "\x1b[38;5;46m",
                "select": "\x1b[48;5;22;38;5;231m",
            },
            "contrast": {
                "header": "\x1b[1;97m",
                "border": "\x1b[38;5;250m",
                "panel": "\x1b[38;5;255m",
                "hint": "\x1b[38;5;245m",
                "accent": "\x1b[38;5;226m",
                "select": "\x1b[48;5;236;38;5;226m",
            },
        }
        t = themes.get(str(self.theme).lower(), themes["dark"])
        return t

    def _dim_code(self) -> str:
        level = int(max(0, min(3, self.intensity)))
        # Transparent-like effect by dimming panel text intensity.
        if level <= 0:
            return ""
        if level == 1:
            return "\x1b[2m"
        if level == 2:
            return "\x1b[2;37m"
        return "\x1b[2;90m"

    def _right_panel_lines(self, width: int, height: int) -> List[str]:
        pages = self._suggestion_pages()
        if not pages:
            return [""] * height
        page = pages[self.suggestion_page % len(pages)]
        cmd_lines = [str(x) for x in page.get("items", [])]
        if cmd_lines:
            self.suggestion_idx = max(0, min(self.suggestion_idx, len(cmd_lines) - 1))
        else:
            self.suggestion_idx = 0
        lines: List[str] = []
        lines.append(f"{page.get('title', 'NEXT')}  [{self.suggestion_page + 1}/{len(pages)}]")
        lines.append("")
        for idx, cmd in enumerate(cmd_lines, start=1):
            marker = ">" if (idx - 1) == self.suggestion_idx else " "
            lines.extend(_split_soft_lines(f"{marker} {idx}. {cmd}", max(8, width - 2)))
        lines.append("")
        lines.append("KEYS")
        lines.append("Up/Down: select item")
        lines.append("Left/Right: page")
        lines.append("Enter: run selected")
        lines.append("Tab: autocomplete")
        lines.append("PgUp/PgDn: scroll log")
        lines.append(":layout compact|full")
        lines.append("Ctrl+Esc: exit shell")
        return (lines + [""] * height)[:height]

    def _left_panel_lines(self, width: int, height: int) -> List[str]:
        snap = _run_snapshot(self.runs_root)
        latest = snap.get("latest") or {}
        head = [
            "MULTIVERSE LIVE SHELL",
            f"runs_root={self.runs_root} | runs={snap.get('run_count', 0)} | last_rc={self.last_rc}",
            f"latest={latest.get('run_id', 'none')}",
            f"layout={self.layout} theme={self.theme} intensity={self.intensity} scroll={self.log_scroll}",
        ]
        header_lines: List[str] = []
        for raw in head:
            header_lines.extend(_split_soft_lines(raw, max(8, width)))
        header_lines.append("")

        log_h = max(1, int(height) - len(header_lines))
        all_logs: List[str] = []
        for raw in list(self.logs):
            all_logs.extend(_split_soft_lines(raw, max(8, width)))
        if not all_logs:
            all_logs = ["(no output yet)"]

        if self.log_scroll <= 0:
            view = all_logs[-log_h:]
        else:
            end = max(0, len(all_logs) - self.log_scroll)
            start = max(0, end - log_h)
            view = all_logs[start:end]
        if len(view) < log_h:
            view = ([""] * (log_h - len(view))) + view

        out = (header_lines + view)[:height]
        if len(out) < height:
            out.extend([""] * (height - len(out)))
        return out

    def _autocomplete_candidates(self) -> List[str]:
        base = [
            "status",
            "st",
            "u ls",
            "universe list",
            "t --profile quick",
            "t --profile balanced",
            "t --profile research",
            "dist --profile balanced",
            "r ls",
            "r latest",
            "r i --count-events",
            "r f --recursive",
            "r tail --file events.jsonl --lines 40",
            "h --once",
            "help",
            "clear",
            ":theme dark",
            ":layout compact",
            ":layout full",
            ":theme glass",
            ":theme matrix",
            ":theme contrast",
            ":intensity 0",
            ":intensity 1",
            ":intensity 2",
            ":intensity 3",
        ]
        for it in self._active_suggestions():
            if it not in base:
                base.append(it)
        return base

    def _autocomplete(self) -> None:
        current = str(self.input_buf).strip()
        if not current:
            s = self._active_suggestions()
            if s:
                self.input_buf = s[self.suggestion_idx]
            return
        if current != self._ac_seed:
            self._ac_seed = current
            self._ac_idx = 0
            self._ac_matches = [c for c in self._autocomplete_candidates() if c.startswith(current)]
            if not self._ac_matches:
                # fallback: substring match
                self._ac_matches = [c for c in self._autocomplete_candidates() if current in c]
        if not self._ac_matches:
            return
        self.input_buf = self._ac_matches[self._ac_idx % len(self._ac_matches)]
        self._ac_idx += 1

    def _reset_autocomplete(self) -> None:
        self._ac_seed = ""
        self._ac_matches = []
        self._ac_idx = 0

    def _render(self) -> None:
        tw, th = shutil.get_terminal_size((120, 32))
        full_w = max(72, tw - 2)
        full_h = max(16, th - 1)
        if str(self.layout).lower() == "compact":
            ui_w = max(72, min(full_w, int(self.max_ui_width)))
            ui_h = max(16, min(full_h, int(self.max_ui_height)))
        else:
            ui_w = full_w
            ui_h = full_h

        left_pad = max(0, (tw - ui_w) // 2)
        top_pad = max(0, (th - ui_h) // 2)

        right_w = max(34, min(46, ui_w // 3))
        left_w = max(28, ui_w - right_w - 5)
        content_h = max(8, ui_h - 5)

        left_lines = self._left_panel_lines(left_w, content_h)
        right_lines = self._right_panel_lines(right_w, content_h)

        reset = "\x1b[0m"
        style = self._theme_codes()
        header = style["header"]
        border = style["border"]
        panel = style["panel"]
        hint = style["hint"]
        accent = style["accent"]
        select = style["select"]
        dim = self._dim_code()

        out: List[str] = []
        out.append("\x1b[2J\x1b[H")
        if top_pad > 0:
            out.extend([""] * top_pad)
        prefix = " " * left_pad
        status = "RUNNING" if self._busy else "READY"
        out.append(prefix + f"{header}Multiverse Shell [{status}]{reset}".ljust(ui_w))
        out.append(prefix + f"{border}+" + "-" * (left_w + 2) + "+" + "-" * (right_w + 2) + f"+{reset}")
        selected_mark = f"{select}> {reset}"
        normal_mark = f"{panel}{dim}  {reset}"
        for i in range(content_h):
            l = left_lines[i][:left_w].ljust(left_w)
            r_raw = right_lines[i][:right_w]
            if r_raw.startswith("> "):
                rr = r_raw[2:].ljust(max(1, right_w - 2))
                r = f"{selected_mark}{rr}"
            else:
                rr = r_raw.ljust(max(1, right_w - 2))
                r = f"{normal_mark}{rr}"
            out.append(prefix + f"{border}|{reset} {l} {border}|{reset} {panel}{dim}{r}{reset} {border}|{reset}")
        out.append(prefix + f"{border}+" + "-" * (left_w + 2) + "+" + "-" * (right_w + 2) + f"+{reset}")
        prompt = f"multiverse> {self.input_buf}"
        keybar = "Tab autocomplete | Enter run | Arrows navigate | PgUp/PgDn scroll | Ctrl+Esc exit"
        out.append(prefix + f"{hint}{keybar[:ui_w].ljust(ui_w)}{reset}")
        out.append(prefix + f"{accent}{(prompt[:ui_w]).ljust(ui_w)}{reset}")
        sys.stdout.write("\n".join(out) + "\x1b[0m")
        sys.stdout.flush()

    def _run_command(self, line: str) -> None:
        cmd = str(line).strip()
        if not cmd:
            return
        if cmd.startswith(":layout"):
            parts = cmd.split()
            if len(parts) < 2:
                self._log("usage: :layout compact|full")
                return
            mode = str(parts[1]).strip().lower()
            if mode not in ("compact", "full"):
                self._log("invalid layout. choose: compact | full")
                return
            self.layout = mode
            self._log(f"layout set: {self.layout}")
            return
        if cmd.startswith(":theme"):
            parts = cmd.split()
            if len(parts) < 2:
                self._log("usage: :theme dark|glass|matrix|contrast")
                return
            t = str(parts[1]).strip().lower()
            if t not in ("dark", "glass", "matrix", "contrast"):
                self._log("invalid theme. choose: dark | glass | matrix | contrast")
                return
            self.theme = t
            self._log(f"theme set: {self.theme}")
            return
        if cmd.startswith(":intensity"):
            parts = cmd.split()
            if len(parts) < 2:
                self._log("usage: :intensity 0..3")
                return
            try:
                level = int(parts[1])
            except Exception:
                self._log("usage: :intensity 0..3")
                return
            self.intensity = max(0, min(3, level))
            self._log(f"intensity set: {self.intensity}")
            return
        if cmd in (":quit", ":q", "quit", "exit"):
            self._log("Shell stays active; press Ctrl+Esc to exit.")
            return
        if cmd in ("clear", "cls"):
            self.logs.clear()
            self._log("Cleared.")
            return

        if cmd in ("help", "?"):
            help_buf = io.StringIO()
            with redirect_stdout(help_buf):
                build_parser().print_help()
            for ln in help_buf.getvalue().splitlines():
                self._log(ln)
            return

        argv = shlex.split(cmd, posix=False)
        if not argv:
            return
        if argv[0].lower() in ("multiverse", "multiverse.ai"):
            argv = argv[1:]
        if not argv:
            return
        if argv[0].lower() in ("shell", "live", "session"):
            self._log("Already in interactive shell.")
            return

        self.last_command = cmd
        self._log(f"$ {cmd}")
        self._busy = True
        self._render()
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                rc = execute_argv(argv, allow_shell=False, capture_subprocess=True)
            self.last_rc = int(rc)
            text = (buf_out.getvalue() + ("\n" + buf_err.getvalue() if buf_err.getvalue() else "")).strip()
            if text:
                for ln in text.splitlines():
                    self._log(ln)
            if rc == 0 and not text:
                self._log("ok")
        except BaseException as exc:
            if isinstance(exc, SystemExit):
                code = getattr(exc, "code", 1)
                try:
                    self.last_rc = int(code)
                except Exception:
                    self.last_rc = 1
                text = (buf_out.getvalue() + ("\n" + buf_err.getvalue() if buf_err.getvalue() else "")).strip()
                if text:
                    for ln in text.splitlines():
                        self._log(ln)
            else:
                self.last_rc = 1
                self._log(f"error: {exc}")
        finally:
            self._busy = False
            self.history.append(cmd)
            self.history_pos = len(self.history)

    def _poll_key(self) -> None:
        if _windows_ctrl_esc_pressed():
            self.running = False
            return

        if os.name == "nt":
            try:
                import msvcrt  # type: ignore
            except Exception:
                msvcrt = None  # type: ignore
            if msvcrt is None:
                time.sleep(0.05)
                return
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                suggestions = self._active_suggestions()
                if ch in ("\r", "\n"):
                    line = self.input_buf.strip()
                    if (not line) and suggestions:
                        line = suggestions[self.suggestion_idx]
                    self.input_buf = ""
                    self._reset_autocomplete()
                    self._run_command(line)
                    continue
                if ch == "\t":
                    self._autocomplete()
                    continue
                if ch == "\x08":  # backspace
                    self.input_buf = self.input_buf[:-1]
                    self._reset_autocomplete()
                    continue
                if ch in ("\x00", "\xe0"):
                    ext = msvcrt.getwch()
                    # up / down: suggestion picker (empty input), otherwise history.
                    if ext == "H":
                        if (not self.input_buf.strip()) and suggestions:
                            self.suggestion_idx = max(0, self.suggestion_idx - 1)
                        elif self.history:
                            self.history_pos = max(0, self.history_pos - 1)
                            self.input_buf = self.history[self.history_pos]
                            self._reset_autocomplete()
                    elif ext == "P":
                        if (not self.input_buf.strip()) and suggestions:
                            self.suggestion_idx = min(len(suggestions) - 1, self.suggestion_idx + 1)
                        elif self.history:
                            self.history_pos = min(len(self.history), self.history_pos + 1)
                            if self.history_pos >= len(self.history):
                                self.input_buf = ""
                            else:
                                self.input_buf = self.history[self.history_pos]
                            self._reset_autocomplete()
                    elif ext == "K":  # left: previous suggestion page
                        pages = self._suggestion_pages()
                        if pages:
                            self.suggestion_page = (self.suggestion_page - 1) % len(pages)
                            self.suggestion_idx = 0
                    elif ext == "M":  # right: next suggestion page
                        pages = self._suggestion_pages()
                        if pages:
                            self.suggestion_page = (self.suggestion_page + 1) % len(pages)
                            self.suggestion_idx = 0
                    elif ext == "I":  # PgUp
                        self.log_scroll = min(len(self.logs), self.log_scroll + 10)
                    elif ext == "Q":  # PgDn
                        self.log_scroll = max(0, self.log_scroll - 10)
                    continue
                if ch == "\x03":  # Ctrl+C
                    self._log("Use Ctrl+Esc to exit this shell.")
                    continue
                if ch == "\x1b":
                    # Keep ESC non-destructive; exit requires Ctrl+Esc.
                    continue
                if ch.isprintable():
                    self.input_buf += ch
                    self._reset_autocomplete()
            return

        # Non-Windows fallback loop
        time.sleep(0.1)

    def run(self) -> int:
        _enable_windows_vt()
        try:
            sys.stdout.write("\x1b[?1049h\x1b[?25l")
            self._cursor_visible = False
            sys.stdout.flush()
            self._log("Interactive shell started.")
            self._log("Use Up/Down to pick a suggestion, Enter to run it.")
            self._log("Use Left/Right to switch suggestion pages, Tab to autocomplete.")
            self._log("Use PgUp/PgDn to scroll logs, Ctrl+Esc to exit.")
            while self.running:
                self._render()
                self._poll_key()
                time.sleep(0.03)
        finally:
            if not self._cursor_visible:
                sys.stdout.write("\x1b[?25h\x1b[?1049l")
                sys.stdout.flush()
        return 0


def cmd_shell(args: argparse.Namespace) -> int:
    shell = InteractiveShell(runs_root=str(args.runs_root))
    return shell.run()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="multiverse",
        description="Multiverse convenience CLI for universes, training, and run artifacts.",
        epilog=(
            "Examples:\n"
            "  multiverse status\n"
            "  multiverse shell\n"
            "  multiverse universe list --contains line\n"
            "  multiverse train --profile quick\n"
            "  multiverse train --profile research --dry-run\n"
            "  multiverse runs inspect --count-events\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", aliases=["st"], help="Snapshot summary of universes and runs.")
    p_status.add_argument("--runs-root", type=str, default="runs")
    p_status.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_status.set_defaults(func=cmd_status)

    p_uni = sub.add_parser("universe", aliases=["u"], help="Universe (verse) commands.")
    sub_uni = p_uni.add_subparsers(dest="universe_command", required=True)
    p_uni_list = sub_uni.add_parser("list", aliases=["ls"], help="List registered universes/verses.")
    p_uni_list.add_argument("--contains", type=str, default=None, help="Filter names by substring.")
    p_uni_list.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_uni_list.set_defaults(func=cmd_universe_list)

    p_train = sub.add_parser("train", aliases=["t"], help="Run a single training command (wraps tools/train_agent.py).")
    p_train.add_argument("--universe", "--verse", dest="verse", type=str, default="line_world")
    p_train.add_argument("--algo", type=str, default="random")
    p_train.add_argument("--profile", type=str, default="custom", choices=["custom", *sorted(TRAIN_PROFILES.keys())])
    p_train.add_argument("--episodes", type=int, default=20)
    p_train.add_argument("--max-steps", dest="max_steps", type=int, default=40)
    p_train.add_argument("--seed", type=int, default=123)
    p_train.add_argument("--policy-id", type=str, default=None)
    p_train.add_argument("--runs-root", type=str, default="runs")
    p_train.add_argument("--eval", action=argparse.BooleanOptionalAction, default=False)
    p_train.add_argument("--make-index", dest="make_index", action=argparse.BooleanOptionalAction, default=False)
    p_train.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Print underlying command and exit.")
    p_train.add_argument("extra", nargs=argparse.REMAINDER, help="Extra flags for train_agent.py (use -- before extras).")
    p_train.set_defaults(func=cmd_train)

    p_dist = sub.add_parser(
        "distributed",
        aliases=["dist", "d"],
        help="Run distributed training (wraps tools/train_distributed.py).",
    )
    p_dist.add_argument("--mode", type=str, default="sharded", choices=["sharded", "pbt"])
    p_dist.add_argument("--universe", "--verse", dest="verse", type=str, default="line_world")
    p_dist.add_argument("--algo", type=str, default="q")
    p_dist.add_argument("--profile", type=str, default="custom", choices=["custom", *sorted(DISTRIBUTED_PROFILES.keys())])
    p_dist.add_argument("--episodes", type=int, default=100)
    p_dist.add_argument("--max-steps", dest="max_steps", type=int, default=50)
    p_dist.add_argument("--workers", type=int, default=2)
    p_dist.add_argument("--seed", type=int, default=123)
    p_dist.add_argument("--policy-id", type=str, default=None)
    p_dist.add_argument("--runs-root", type=str, default="runs")
    p_dist.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    p_dist.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Print underlying command and exit.")
    p_dist.add_argument("extra", nargs=argparse.REMAINDER, help="Extra flags for train_distributed.py (use -- before extras).")
    p_dist.set_defaults(func=cmd_distributed)

    p_runs = sub.add_parser("runs", aliases=["r"], help="Run artifact browsing commands.")
    sub_runs = p_runs.add_subparsers(dest="runs_command", required=True)

    p_runs_list = sub_runs.add_parser("list", aliases=["ls"], help="List runs under runs root.")
    p_runs_list.add_argument("--runs-root", type=str, default="runs")
    p_runs_list.add_argument("--limit", type=int, default=20)
    p_runs_list.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_runs_list.set_defaults(func=cmd_runs_list)

    p_runs_latest = sub_runs.add_parser("latest", aliases=["last"], help="Show latest run path.")
    p_runs_latest.add_argument("--runs-root", type=str, default="runs")
    p_runs_latest.add_argument("--path-only", action=argparse.BooleanOptionalAction, default=False)
    p_runs_latest.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_runs_latest.set_defaults(func=cmd_runs_latest)

    p_runs_files = sub_runs.add_parser("files", aliases=["f"], help="List files in a run directory.")
    p_runs_files.add_argument("--runs-root", type=str, default="runs")
    p_runs_files.add_argument("--run-id", type=str, default=None)
    p_runs_files.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    p_runs_files.add_argument("--limit", type=int, default=200)
    p_runs_files.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_runs_files.set_defaults(func=cmd_runs_files)

    p_runs_tail = sub_runs.add_parser("tail", aliases=["log"], help="Tail a file inside a run directory.")
    p_runs_tail.add_argument("--runs-root", type=str, default="runs")
    p_runs_tail.add_argument("--run-id", type=str, default=None)
    p_runs_tail.add_argument("--file", type=str, default="events.jsonl")
    p_runs_tail.add_argument("--lines", type=int, default=30)
    p_runs_tail.set_defaults(func=cmd_runs_tail)

    p_runs_inspect = sub_runs.add_parser("inspect", aliases=["i"], help="Inspect run metadata and key artifacts.")
    p_runs_inspect.add_argument("--runs-root", type=str, default="runs")
    p_runs_inspect.add_argument("--run-id", type=str, default=None)
    p_runs_inspect.add_argument("--count-events", action=argparse.BooleanOptionalAction, default=False)
    p_runs_inspect.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_runs_inspect.set_defaults(func=cmd_runs_inspect)

    p_shell = sub.add_parser(
        "shell",
        aliases=["live", "session"],
        help="Full-screen interactive shell with right-side next-action panel.",
    )
    p_shell.add_argument("--runs-root", type=str, default="runs")
    p_shell.set_defaults(func=cmd_shell)

    p_hub = sub.add_parser("hub", aliases=["h"], help="Launch terminal dashboard (wraps tools/universe_hub.py).")
    p_hub.add_argument("--runs-root", type=str, default="runs")
    p_hub.add_argument("--refresh-sec", type=float, default=2.0)
    p_hub.add_argument("--once", action=argparse.BooleanOptionalAction, default=False)
    p_hub.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Print underlying command and exit.")
    p_hub.add_argument("extra", nargs=argparse.REMAINDER, help="Extra flags for universe_hub.py (use -- before extras).")
    p_hub.set_defaults(func=cmd_hub)

    return ap


def execute_argv(
    raw_argv: List[str],
    *,
    allow_shell: bool = True,
    capture_subprocess: bool = False,
) -> int:
    ap = build_parser()
    if len(raw_argv) <= 0:
        cmd_status(argparse.Namespace(runs_root="runs", json=False))
        print("")
        ap.print_help()
        return 0
    args = ap.parse_args(raw_argv)
    if (not allow_shell) and str(getattr(args, "command", "")).lower() in ("shell", "live", "session"):
        raise ValueError("Cannot launch nested shell from shell mode.")
    if str(getattr(args, "command", "")) in ("train", "t"):
        apply_train_profile(args, raw_argv)
    if str(getattr(args, "command", "")) in ("distributed", "dist", "d"):
        apply_distributed_profile(args, raw_argv)
    if bool(capture_subprocess):
        setattr(args, "_capture_subprocess_output", True)
    func: Callable[[argparse.Namespace], int] = args.func
    return int(func(args))


def main() -> int:
    raw_argv = sys.argv[1:]
    if len(raw_argv) <= 0:
        return execute_argv([])
    try:
        return execute_argv(raw_argv)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
