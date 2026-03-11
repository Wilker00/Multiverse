"""
tools/multiverse_cli_shell.py

Interactive shell surface for the Multiverse CLI.
"""

from __future__ import annotations

import argparse
import io
import os
import shlex
import shutil
import sys
import time
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List


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
    def __init__(
        self,
        *,
        runs_root: str,
        build_parser_fn: Callable[[], argparse.ArgumentParser],
        execute_argv_fn: Callable[..., int],
        run_snapshot_fn: Callable[[str], Dict[str, Any]],
    ) -> None:
        self.runs_root = str(runs_root)
        self.build_parser_fn = build_parser_fn
        self.execute_argv_fn = execute_argv_fn
        self.run_snapshot_fn = run_snapshot_fn
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
        return themes.get(str(self.theme).lower(), themes["dark"])

    def _dim_code(self) -> str:
        level = int(max(0, min(3, self.intensity)))
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
        snap = self.run_snapshot_fn(self.runs_root)
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
            suggestions = self._active_suggestions()
            if suggestions:
                self.input_buf = suggestions[self.suggestion_idx]
            return
        if current != self._ac_seed:
            self._ac_seed = current
            self._ac_idx = 0
            self._ac_matches = [c for c in self._autocomplete_candidates() if c.startswith(current)]
            if not self._ac_matches:
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
            left = left_lines[i][:left_w].ljust(left_w)
            right_raw = right_lines[i][:right_w]
            if right_raw.startswith("> "):
                rr = right_raw[2:].ljust(max(1, right_w - 2))
                right = f"{selected_mark}{rr}"
            else:
                rr = right_raw.ljust(max(1, right_w - 2))
                right = f"{normal_mark}{rr}"
            out.append(prefix + f"{border}|{reset} {left} {border}|{reset} {panel}{dim}{right}{reset} {border}|{reset}")
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
            theme = str(parts[1]).strip().lower()
            if theme not in ("dark", "glass", "matrix", "contrast"):
                self._log("invalid theme. choose: dark | glass | matrix | contrast")
                return
            self.theme = theme
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
                self.build_parser_fn().print_help()
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
                rc = self.execute_argv_fn(argv, allow_shell=False, capture_subprocess=True)
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
                if ch == "\x08":
                    self.input_buf = self.input_buf[:-1]
                    self._reset_autocomplete()
                    continue
                if ch in ("\x00", "\xe0"):
                    ext = msvcrt.getwch()
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
                    elif ext == "K":
                        pages = self._suggestion_pages()
                        if pages:
                            self.suggestion_page = (self.suggestion_page - 1) % len(pages)
                            self.suggestion_idx = 0
                    elif ext == "M":
                        pages = self._suggestion_pages()
                        if pages:
                            self.suggestion_page = (self.suggestion_page + 1) % len(pages)
                            self.suggestion_idx = 0
                    elif ext == "I":
                        self.log_scroll = min(len(self.logs), self.log_scroll + 10)
                    elif ext == "Q":
                        self.log_scroll = max(0, self.log_scroll - 10)
                    continue
                if ch == "\x03":
                    self._log("Use Ctrl+Esc to exit this shell.")
                    continue
                if ch == "\x1b":
                    continue
                if ch.isprintable():
                    self.input_buf += ch
                    self._reset_autocomplete()
            return

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


def run_shell(
    *,
    runs_root: str,
    build_parser_fn: Callable[[], argparse.ArgumentParser],
    execute_argv_fn: Callable[..., int],
    run_snapshot_fn: Callable[[str], Dict[str, Any]],
) -> int:
    shell = InteractiveShell(
        runs_root=runs_root,
        build_parser_fn=build_parser_fn,
        execute_argv_fn=execute_argv_fn,
        run_snapshot_fn=run_snapshot_fn,
    )
    return shell.run()
