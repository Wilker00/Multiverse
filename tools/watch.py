"""
tools/watch.py

Live verse renderer — watch any verse being trained in real time.
Like Gymnasium's human render mode, but for Multiverse.

Usage:
    py tools/watch.py --mode watch --verse maze_world --algo q --train
    py tools/watch.py --mode debug --verse maze_world --algo q --safe_guard --clear_screen
    py tools/watch.py --mode replay --run_dir runs/run_x --show_safety_heatmap
    py tools/watch.py --verse maze_world
    py tools/watch.py --verse grid_world  --algo q --train --fps 8
    py tools/watch.py --verse chess_world --algo q --train --episodes 20
    py tools/watch.py --verse warehouse_world --algo q --train --fps 5
    py tools/watch.py --verse line_world  --algo ppo --train --vparam width=12
    py tools/watch.py --verse maze_world  --algo q --train --fps 15 --vparam width=11 --vparam height=11

Options:
    --mode      watch | debug | replay (default: watch)
    --verse     Verse name (default: maze_world)
    --algo      Agent algorithm (default: q)
    --train     Enable online learning between episodes
    --episodes  Number of episodes to watch (default: 100, 0 = infinite)
    --fps       Steps rendered per second (default: 6, 0 = as fast as possible)
    --seed      Random seed (default: 42)
    --vparam    Verse parameter k=v  (repeatable, e.g. --vparam width=9)
    --aconfig   Agent config k=v    (repeatable)
    --no-color  Disable ANSI colour in the stats overlay
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ── path bootstrap ────────────────────────────────────────────────────────────
if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

# ── UTF-8 on Windows ─────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ── Enable ANSI on Windows ───────────────────────────────────────────────────
def _enable_windows_ansi() -> None:
    """Enable VT100/ANSI escape processing in the Windows console."""
    try:
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass

if os.name == "nt":
    _enable_windows_ansi()

# ── Registries ───────────────────────────────────────────────────────────────
from verses.registry import register_builtin as _reg_verses, create_verse, list_verses
from agents.registry import register_builtin_agents as _reg_agents, create_agent, _AGENT_REGISTRY
from core.types import VerseSpec, AgentSpec
from core.agent_base import ExperienceBatch, Transition

_reg_verses()
_reg_agents()


# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────

_CLEAR = "\033[2J\033[H"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_RESET = "\033[0m"
_CYAN  = "\033[96m"
_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW= "\033[93m"
_PURPLE= "\033[95m"
_WHITE = "\033[97m"

def _c(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{_RESET}" if use_color else text


# ─────────────────────────────────────────────────────────────────────────────
# Action label maps (human-readable names per verse type)
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_LABELS: Dict[str, Dict[int, str]] = {
    "line_world":        {0: "<-left", 1: "right->"},
    "cliff_world":       {0: "^up", 1: "down-v", 2: "<-left", 3: "right->"},
    "grid_world":        {0: "^up", 1: "down-v", 2: "<-left", 3: "right->"},
    "maze_world":        {0: "^N", 1: "v-S", 2: "<-W", 3: "E->"},
    "labyrinth_world":   {0: "^N", 1: "v-S", 2: "<-W", 3: "E->"},
    "memory_vault_world":{0: "^N", 1: "v-S", 2: "<-W", 3: "E->"},
    "warehouse_world":   {0: "^N", 1: "v-S", 2: "<-W", 3: "E->", 4: "charge"},
    "harvest_world":     {0: "^N", 1: "v-S", 2: "<-W", 3: "E->", 4: "pick"},
    "park_world":        {0: "^N", 1: "v-S", 2: "<-W", 3: "E->", 4: "park"},
    "swamp_world":       {0: "^N", 1: "v-S", 2: "<-W", 3: "E->"},
    "bridge_world":      {0: "^N", 1: "v-S", 2: "<-W", 3: "E->", 4: "repair"},
    "escape_world":      {0: "^N", 1: "v-S", 2: "<-W", 3: "E->", 4: "hide"},
    "pursuit_world":     {0: "^N", 1: "v-S", 2: "<-W", 3: "E->"},
    "factory_world":     {0: "idle", 1: "process", 2: "repair", 3: "priority"},
    "trade_world":       {0: "hold", 1: "buy", 2: "sell", 3: "short", 4: "cover"},
    "risk_tutorial_world":{0: "fold", 1: "call", 2: "raise", 3: "all-in"},
    "wind_master_world": {0: "none", 1: "thrust", 2: "brake", 3: "stabilize"},
    "rule_flip_world":   {0: "track-A", 1: "track-B", 2: "track-C", 3: "track-D"},
    "chess_world":       {0: "build", 1: "pressure", 2: "capture", 3: "defend", 4: "tempo", 5: "convert"},
    "chess_world_v2":    {0: "build", 1: "pressure", 2: "capture", 3: "defend", 4: "tempo", 5: "convert"},
    "go_world":          {0: "expand", 1: "cut", 2: "connect", 3: "probe", 4: "invade", 5: "settle"},
    "go_world_v2":       {0: "expand", 1: "cut", 2: "connect", 3: "probe", 4: "invade", 5: "settle"},
    "uno_world":         {0: "play_low", 1: "play_high", 2: "play_match", 3: "draw", 4: "skip_opp"},
    "uno_world_v2":      {0: "play_low", 1: "play_high", 2: "play_match", 3: "draw", 4: "skip_opp"},
}


def _action_label(verse_name: str, action: Any) -> str:
    labels = _ACTION_LABELS.get(verse_name, {})
    try:
        a = int(action)
        return labels.get(a, str(action))
    except (TypeError, ValueError):
        return str(action)


# ─────────────────────────────────────────────────────────────────────────────
# Screen helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


def _hr(width: int, color: str = _CYAN, use_color: bool = True) -> str:
    line = "-" * width
    return _c(line, color, use_color)


def _render_stats_header(
    *,
    verse_name: str,
    algo: str,
    episode: int,
    total_episodes: int,
    use_color: bool,
    width: int,
) -> str:
    ep_str = f"ep {episode}" if total_episodes == 0 else f"ep {episode}/{total_episodes}"
    title = f"  MULTIVERSE WATCH  |  {verse_name}  |  {algo}  |  {ep_str}  "
    padded = title.ljust(width)
    if use_color:
        return f"\033[48;5;17m{_CYAN}{_BOLD}{padded}{_RESET}"
    return padded


def _render_step_bar(
    *,
    step: int,
    max_steps: int,
    action: Any,
    verse_name: str,
    reward: float,
    ep_return: float,
    best_return: float,
    wins: int,
    episodes_done: int,
    train: bool,
    use_color: bool,
    width: int,
) -> str:
    action_str = _action_label(verse_name, action)
    rew_col = _GREEN if reward > 0 else (_RED if reward < 0 else _DIM)
    ret_col = _GREEN if ep_return > 0 else (_RED if ep_return < 0 else _DIM)
    sr = wins / episodes_done if episodes_done > 0 else 0.0
    mode_str = _c("[TRAINING]", _PURPLE, use_color) if train else _c("[WATCHING]", _DIM, use_color)

    parts = [
        f" step {step:>3}/{max_steps}",
        f"  act={_c(action_str, _YELLOW, use_color)}",
        f"  rew={_c(f'{reward:+.3f}', rew_col, use_color)}",
        f"  return={_c(f'{ep_return:+.3f}', ret_col, use_color)}",
        f"  best={_c(f'{best_return:+.3f}', _CYAN, use_color)}",
        f"  sr={_c(f'{sr:.1%}', _GREEN if sr > 0.3 else _RED, use_color)}({wins}/{episodes_done})",
        f"  {mode_str}",
    ]
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Param parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_kv(kvs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in (kvs or []):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = float(v) if "." in v else int(v)
            continue
        except ValueError:
            pass
        try:
            parsed = json.loads(v)
            if isinstance(parsed, (list, dict)):
                out[k] = parsed
                continue
        except Exception:
            pass
        out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Frame rendering
# ─────────────────────────────────────────────────────────────────────────────

def _format_frame(
    frame: Optional[str],
    obs: Any,
    verse_name: str,
    use_color: bool,
) -> str:
    """
    Return display string for the current verse state.
    Falls back to a formatted observation dict if render() returns None.
    """
    if frame is not None:
        return str(frame)

    # Fallback: pretty-print the observation
    if isinstance(obs, dict):
        lines = [f"  {_c(k, _CYAN, use_color)}: {v}" for k, v in obs.items()]
        return "\n".join(lines)
    return f"  obs = {obs}"


def _draw_screen(
    *,
    frame_str: str,
    header: str,
    step_bar: str,
    hr: str,
    episode_log: List[str],
    max_log_lines: int = 5,
) -> None:
    """Clear screen and redraw everything in one write to minimise flicker."""
    lines = [
        _CLEAR,
        header,
        hr,
        frame_str,
        hr,
        step_bar,
        hr,
    ]
    # Recent episode log
    if episode_log:
        lines.append("  Recent episodes:")
        for entry in episode_log[-max_log_lines:]:
            lines.append("    " + entry)
    lines.append("")
    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Main watcher
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WatchConfig:
    verse_name: str
    algo: str
    train: bool
    episodes: int
    fps: float
    seed: int
    vparam: Dict[str, Any]
    aconfig: Dict[str, Any]
    use_color: bool
    max_steps: int


def run_watcher(cfg: WatchConfig) -> None:
    width = _get_terminal_width()
    step_delay = (1.0 / cfg.fps) if cfg.fps > 0 else 0.0

    # ── Build verse ────────────────────────────────────────────────────────
    verse_params = dict(cfg.vparam)
    if cfg.max_steps > 0:
        verse_params.setdefault("max_steps", cfg.max_steps)

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=cfg.verse_name,
        verse_version="0.1",
        params=verse_params,
        tags=[],
        seed=cfg.seed,
    )

    verse = create_verse(verse_spec)

    # ── Build agent ────────────────────────────────────────────────────────
    agent_spec = AgentSpec(
        spec_version="v1",
        algo=cfg.algo,
        policy_id=f"{cfg.algo}_watch",
        policy_version="0.0",
        config=cfg.aconfig,
    )
    agent = create_agent(agent_spec, verse.observation_space, verse.action_space)
    agent.seed(cfg.seed)

    # ── Stats ──────────────────────────────────────────────────────────────
    best_return = float("-inf")
    wins = 0
    episode_log: List[str] = []
    episode = 0

    hr = _hr(width, use_color=cfg.use_color)

    try:
        while cfg.episodes == 0 or episode < cfg.episodes:
            episode += 1
            verse.seed(cfg.seed + episode)
            reset_result = verse.reset()
            obs = reset_result.obs

            ep_return = 0.0
            done = False
            truncated = False
            step = 0
            transitions: List[Transition] = []
            action: Any = 0

            max_steps = int(verse_params.get("max_steps", 200))

            while not done and not truncated and step < max_steps:
                # Agent picks action
                act_result = agent.act(obs)
                action = act_result.action

                # Step the verse
                step_result = verse.step(action)
                reward = step_result.reward
                next_obs = step_result.obs
                done = step_result.done
                truncated = step_result.truncated

                ep_return += reward
                step += 1

                # Collect transition for training
                if cfg.train:
                    transitions.append(
                        Transition(
                            obs=obs,
                            action=action,
                            reward=reward,
                            next_obs=next_obs,
                            done=done,
                            truncated=truncated,
                            info=dict(step_result.info),
                        )
                    )

                obs = next_obs

                # ── Render ─────────────────────────────────────────────────
                raw_frame = verse.render(mode="ansi")
                frame_str = _format_frame(raw_frame, obs, cfg.verse_name, cfg.use_color)

                header = _render_stats_header(
                    verse_name=cfg.verse_name,
                    algo=cfg.algo,
                    episode=episode,
                    total_episodes=cfg.episodes,
                    use_color=cfg.use_color,
                    width=width,
                )
                step_bar = _render_step_bar(
                    step=step,
                    max_steps=max_steps,
                    action=action,
                    verse_name=cfg.verse_name,
                    reward=reward,
                    ep_return=ep_return,
                    best_return=best_return,
                    wins=wins,
                    episodes_done=episode - 1,
                    train=cfg.train,
                    use_color=cfg.use_color,
                    width=width,
                )

                _draw_screen(
                    frame_str=frame_str,
                    header=header,
                    step_bar=step_bar,
                    hr=hr,
                    episode_log=episode_log,
                )

                if step_delay > 0:
                    time.sleep(step_delay)

            # ── End of episode ─────────────────────────────────────────────
            if ep_return > best_return:
                best_return = ep_return
            success = done and not truncated and ep_return > 0
            if success:
                wins += 1

            outcome = _c("WIN ", _GREEN, cfg.use_color) if success else _c("lose", _RED, cfg.use_color)
            log_entry = (
                f"ep {episode:>4}  {outcome}  "
                f"return={ep_return:>+8.3f}  steps={step:>4}  "
                f"best={best_return:>+8.3f}"
            )
            episode_log.append(log_entry)

            # ── Train ──────────────────────────────────────────────────────
            if cfg.train and transitions:
                batch = ExperienceBatch(transitions=transitions)
                try:
                    agent.learn(batch)
                except (NotImplementedError, AttributeError):
                    pass

    except KeyboardInterrupt:
        print()
        print(_c("Stopped.", _DIM, cfg.use_color))

    finally:
        verse.close()

    # ── Final summary ──────────────────────────────────────────────────────
    total = episode
    sr = wins / total if total else 0.0
    print()
    print(_hr(width, use_color=cfg.use_color))
    print(_c("  FINAL SUMMARY", _BOLD, cfg.use_color))
    print(_hr(width, use_color=cfg.use_color))
    print(f"  Verse    : {cfg.verse_name}")
    print(f"  Agent    : {cfg.algo}  (trained={cfg.train})")
    print(f"  Episodes : {total}")
    print(f"  Wins     : {wins}  ({sr:.1%} success rate)")
    print(f"  Best ret : {best_return:+.3f}")
    print()


def _forward_to_gym_like_viewer(
    *,
    target_mode: str,
    args: argparse.Namespace,
    extra_args: List[str],
) -> int:
    """
    Forward execution to tools/gym_like_viewer.py for debug/replay workflows.
    """
    viewer_path = os.path.join(_ROOT, "tools", "gym_like_viewer.py")
    cmd: List[str] = [sys.executable, viewer_path, "--mode", str(target_mode)]

    if str(args.verse).strip():
        cmd.extend(["--verse", str(args.verse)])
    if str(args.algo).strip():
        cmd.extend(["--algo", str(args.algo)])
    cmd.extend(["--seed", str(int(args.seed))])

    if int(args.max_steps) > 0:
        cmd.extend(["--max_steps", str(int(args.max_steps))])
    cmd.extend(["--episodes", str(max(1, int(args.episodes)))])

    if float(args.fps) > 0.0:
        sleep_s = 1.0 / float(args.fps)
        cmd.extend(["--sleep_s", f"{sleep_s:.6f}"])
    else:
        cmd.extend(["--sleep_s", "0"])

    for item in (args.vparam or []):
        cmd.extend(["--vparam", str(item)])
    for item in (args.aconfig or []):
        cmd.extend(["--aconfig", str(item)])

    cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=_ROOT)
    return int(proc.returncode)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    known_verses = sorted(list_verses().keys())
    known_algos  = sorted(_AGENT_REGISTRY.keys())

    ap = argparse.ArgumentParser(
        description="Unified Multiverse viewer: watch training, debug live, or replay runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Verses : {', '.join(known_verses)}\n"
            f"Agents : {', '.join(known_algos)}\n"
            "Note: In --mode debug/replay, extra args are forwarded to tools/gym_like_viewer.py."
        ),
    )
    ap.add_argument("--mode",     default="watch", choices=["watch", "debug", "replay"], help="Viewer mode")
    ap.add_argument("--verse",    default="maze_world",  help="Verse to run")
    ap.add_argument("--algo",     default="q",           help="Agent algorithm")
    ap.add_argument("--train",    action="store_true",   help="Enable online learning")
    ap.add_argument("--episodes", type=int, default=100, help="Episodes to watch (0=infinite)")
    ap.add_argument("--fps",      type=float, default=6, help="Steps per second (0=max speed)")
    ap.add_argument("--seed",     type=int, default=42,  help="Random seed")
    ap.add_argument("--max-steps",type=int, default=0,   help="Override max steps per episode (0=verse default)")
    ap.add_argument("--vparam",   action="append", default=None, metavar="K=V", help="Verse param (repeatable)")
    ap.add_argument("--aconfig",  action="append", default=None, metavar="K=V", help="Agent config (repeatable)")
    ap.add_argument("--no-color", action="store_true",   help="Disable ANSI colour")

    args, extra = ap.parse_known_args()

    mode = str(args.mode).strip().lower()
    if mode == "watch":
        if args.verse not in known_verses:
            print(f"Unknown verse '{args.verse}'. Known: {', '.join(known_verses)}")
            sys.exit(1)
        if args.algo not in known_algos:
            print(f"Unknown algo '{args.algo}'. Known: {', '.join(known_algos)}")
            sys.exit(1)
        if extra:
            print(
                "Unknown watch mode args: "
                + " ".join(extra)
                + "\nTip: use --mode debug/replay for gym_like_viewer flags."
            )
            sys.exit(2)
    else:
        if args.train:
            print("[watch] Note: --train is watch-mode only; ignored in debug/replay.")
        target_mode = "live" if mode == "debug" else "replay"
        rc = _forward_to_gym_like_viewer(target_mode=target_mode, args=args, extra_args=extra)
        raise SystemExit(rc)

    cfg = WatchConfig(
        verse_name=args.verse,
        algo=args.algo,
        train=args.train,
        episodes=max(0, args.episodes),
        fps=max(0.0, args.fps),
        seed=args.seed,
        vparam=_parse_kv(args.vparam),
        aconfig=_parse_kv(args.aconfig),
        use_color=not args.no_color,
        max_steps=max(0, args.max_steps),
    )

    print(f"Starting {cfg.verse_name} with {cfg.algo} agent...")
    print(f"Press Ctrl+C to stop.\n")
    time.sleep(0.8)

    run_watcher(cfg)


if __name__ == "__main__":
    main()
