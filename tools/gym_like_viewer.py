"""
tools/gym_like_viewer.py

Gym-like interactive viewer for Verses.

Modes:
1) live   : run environment in real-time with manual controls or an agent
2) replay : replay a recorded run from events.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.registry import create_agent, register_builtin_agents
from core.agent_base import ActionResult
from core.gym_adapter import VerseGymAdapter
from core.safe_executor import SafeExecutor, SafeExecutorConfig
from core.types import AgentSpec, JSONValue, VerseSpec
from core.verse_base import StepResult
from verses.registry import create_verse, register_builtin as register_builtin_verses


def _parse_kv_list(kvs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not kvs:
        return out
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Invalid param '{item}'. Expected k=v.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
            continue
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except (ValueError, TypeError):
            pass
        out[k] = v
    return out


def _default_verse_params(verse: str, max_steps: int) -> Dict[str, Any]:
    v = str(verse).strip().lower()
    if v == "line_world":
        return {"goal_pos": 8, "max_steps": max_steps, "step_penalty": -0.02}
    if v == "cliff_world":
        return {
            "width": 12,
            "height": 4,
            "max_steps": max_steps,
            "step_penalty": -1.0,
            "cliff_penalty": -100.0,
            "end_on_cliff": False,
        }
    if v == "labyrinth_world":
        return {
            "width": 15,
            "height": 11,
            "max_steps": max_steps,
            "battery_capacity": 80,
            "battery_drain": 1,
            "action_noise": 0.0,
            "vision_radius": 1,
        }
    return {"max_steps": max_steps}


def _manual_help(action_n: int) -> str:
    base = [
        "Manual controls:",
        "  q = quit",
        "  h = help",
        "  r = reset episode",
        "  [number] = explicit action id",
    ]
    if action_n <= 2:
        base += ["  a = action 0", "  d = action 1"]
    elif action_n <= 4:
        base += ["  w = up(0)", "  s = down(1)", "  a = left(2)", "  d = right(3)"]
    else:
        base += ["  w = up(0)", "  s = down(1)", "  a = left(2)", "  d = right(3)", "  x = wait(4)"]
    return "\n".join(base)


def _manual_to_action(cmd: str, action_n: int) -> Optional[int]:
    raw = str(cmd).strip().lower()
    if raw == "":
        return None
    if raw.isdigit():
        a = int(raw)
        if 0 <= a < action_n:
            return a
        return None
    if action_n <= 2:
        mapping = {"a": 0, "left": 0, "d": 1, "right": 1}
    elif action_n <= 4:
        mapping = {"w": 0, "up": 0, "s": 1, "down": 1, "a": 2, "left": 2, "d": 3, "right": 3}
    else:
        mapping = {
            "w": 0,
            "up": 0,
            "s": 1,
            "down": 1,
            "a": 2,
            "left": 2,
            "d": 3,
            "right": 3,
            "x": 4,
            "wait": 4,
        }
    a = mapping.get(raw)
    if a is None:
        return None
    if 0 <= int(a) < action_n:
        return int(a)
    return None


def _resolve_intervention_path(path_hint: Optional[str]) -> str:
    if path_hint and str(path_hint).strip():
        p = str(path_hint).strip()
    else:
        stamp = int(time.time())
        p = os.path.join("runs", f"intervention_{stamp}.jsonl")
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return p


def _compute_safety_heatmap(
    events: List[Dict[str, Any]],
    *,
    confidence_drop_threshold: float,
) -> Optional[Tuple[List[List[int]], int, int]]:
    cells: Dict[Tuple[int, int], int] = {}
    max_x = 0
    max_y = 0

    for ev in events:
        obs = ev.get("obs")
        if not isinstance(obs, dict):
            continue
        if "x" not in obs or "y" not in obs:
            continue
        try:
            x = int(obs.get("x", 0))
            y = int(obs.get("y", 0))
        except Exception:
            continue

        info = ev.get("info") if isinstance(ev.get("info"), dict) else {}
        se = info.get("safe_executor") if isinstance(info, dict) else None
        if not isinstance(se, dict):
            continue

        mode = str(se.get("mode", ""))
        conf = se.get("confidence")
        low_conf = False
        try:
            if conf is not None and float(conf) <= float(confidence_drop_threshold):
                low_conf = True
        except Exception:
            low_conf = False
        triggered = bool(mode in ("fallback", "shield_veto", "planner_takeover")) or bool(low_conf) or bool(
            se.get("rewound", False)
        )
        if not triggered:
            continue

        k = (x, y)
        cells[k] = int(cells.get(k, 0)) + 1
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    if not cells:
        return None

    grid = [[0 for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    for (x, y), c in cells.items():
        if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
            grid[y][x] = int(c)
    return grid, max_x + 1, max_y + 1


def _render_heatmap_ascii(grid: List[List[int]]) -> str:
    max_val = 0
    for row in grid:
        for v in row:
            max_val = max(max_val, int(v))
    if max_val <= 0:
        return "(no safety heatmap triggers)"

    shades = " .:-=+*#%@"
    lines: List[str] = []
    for y in range(len(grid)):
        row = grid[y]
        chars: List[str] = []
        for v in row:
            n = int(v)
            idx = int(round((float(n) / float(max_val)) * float(len(shades) - 1)))
            idx = max(0, min(len(shades) - 1, idx))
            chars.append(shades[idx])
        lines.append("".join(chars))
    return "\n".join(lines)


def _print_status(
    *,
    episode: int,
    step: int,
    action: Optional[JSONValue],
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    action_info: Optional[Dict[str, Any]] = None,
) -> None:
    se = info.get("safe_executor")
    se_mode = ""
    planner_invoked = False
    budget_left = None
    conf = None
    danger = None
    if isinstance(se, dict):
        se_mode = str(se.get("mode", ""))
        planner_invoked = bool(se_mode == "planner_takeover")
        budget = se.get("planning_budget")
        if isinstance(budget, dict):
            budget_left = budget.get("episode_remaining")
        conf = se.get("confidence")
        danger = se.get("danger")

    flags = []
    for k in (
        "reached_goal",
        "fell_cliff",
        "fell_pit",
        "hit_laser",
        "battery_depleted",
        "hit_wall",
        "hit_obstacle",
        "battery_death",
    ):
        if info.get(k) is True:
            flags.append(k)
    if isinstance(se, dict):
        if se.get("rewound") is True:
            flags.append("rewound")
        if se.get("shield_veto") is True:
            flags.append("shield_veto")
        if planner_invoked:
            flags.append("planner_invoked")

    print(
        f"[ep={episode} step={step}] action={action} reward={reward:.3f} "
        f"terminated={terminated} truncated={truncated} mode={se_mode or '-'}"
    )
    overlay_parts = []
    if conf is not None:
        try:
            overlay_parts.append(f"conf={float(conf):.3f}")
        except Exception:
            pass
    if danger is not None:
        try:
            overlay_parts.append(f"danger={float(danger):.3f}")
        except Exception:
            pass
    if budget_left is not None:
        overlay_parts.append(f"planner_budget_left={budget_left}")
    if overlay_parts:
        print("runtime:", ", ".join(overlay_parts))

    ai = action_info if isinstance(action_info, dict) else {}
    if ai:
        sp = ai.get("self_play")
        if isinstance(sp, dict):
            print(
                "self_play: "
                f"active={bool(sp.get('adversary_active', False))} "
                f"mix={sp.get('mix_ratio')} source={sp.get('source', '')}"
            )
        c = ai.get("contract")
        if isinstance(c, dict):
            print(
                "contract: "
                f"exists={bool(c.get('exists', False))} "
                f"skill={c.get('skill_tag')} version={c.get('version')}"
            )

    if flags:
        print("flags:", ", ".join(flags))
    if isinstance(se, dict):
        explanation = se.get("explanation")
        if explanation:
            print(f"explanation: {explanation}")


def _build_safe_executor_if_requested(
    *,
    args: argparse.Namespace,
    verse: Any,
    obs_space: Any,
    act_space: Any,
) -> Optional[SafeExecutor]:
    if not args.safe_guard and not args.safe_cfg:
        return None
    cfg_raw = _parse_kv_list(args.safe_cfg)
    if isinstance(cfg_raw.get("planner_verse_allowlist"), str):
        cfg_raw["planner_verse_allowlist"] = [
            s.strip().lower() for s in str(cfg_raw["planner_verse_allowlist"]).split(",") if s.strip()
        ]
    if args.safe_guard:
        cfg_raw.setdefault("enabled", True)

    fallback_algo = str(cfg_raw.get("fallback_algo", "")).strip().lower()
    fallback_cfg = {}
    for k in list(cfg_raw.keys()):
        if k.startswith("fallback_") and k != "fallback_algo":
            nk = k[len("fallback_") :]
            if nk:
                fallback_cfg[nk] = cfg_raw.pop(k)

    fallback_agent = None
    if fallback_algo:
        if fallback_algo in ("gateway", "special_moe", "adaptive_moe"):
            fallback_cfg.setdefault("verse_name", args.verse)
        fallback_spec = AgentSpec(
            spec_version="v1",
            policy_id=f"viewer_fallback_{fallback_algo}",
            policy_version="0.1",
            algo=fallback_algo,
            seed=args.seed,
            tags=["viewer"],
            config=fallback_cfg if fallback_cfg else None,
        )
        try:
            fallback_agent = create_agent(fallback_spec, obs_space, act_space)
        except Exception as e:
            print(f"[viewer] safe fallback init skipped ({fallback_algo}): {e}")
            fallback_agent = None

    cfg = SafeExecutorConfig.from_dict(cfg_raw)
    return SafeExecutor(config=cfg, verse=verse, fallback_agent=fallback_agent)


def run_live(args: argparse.Namespace) -> None:
    register_builtin_verses()
    register_builtin_agents()

    verse_params = _default_verse_params(args.verse, args.max_steps)
    verse_params.update(_parse_kv_list(args.vparam))
    agent_cfg = _parse_kv_list(args.aconfig)
    if str(args.algo).strip().lower() in ("gateway", "special_moe", "adaptive_moe"):
        agent_cfg.setdefault("verse_name", args.verse)
    if str(args.algo).strip().lower() == "special_moe":
        default_selector = os.path.join("models", "micro_selector.pt")
        if os.path.isfile(default_selector):
            agent_cfg.setdefault("selector_model_path", default_selector)

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=args.verse,
        verse_version=args.verse_version,
        seed=args.seed,
        tags=["viewer"],
        params=verse_params,
    )
    verse = create_verse(verse_spec)
    env = VerseGymAdapter(verse)

    safe_executor = _build_safe_executor_if_requested(
        args=args,
        verse=verse,
        obs_space=verse.observation_space,
        act_space=verse.action_space,
    )

    manual_mode = str(args.algo).strip().lower() == "manual"
    agent = None
    if not manual_mode:
        spec = AgentSpec(
            spec_version="v1",
            policy_id=args.policy_id or str(args.algo),
            policy_version="0.1",
            algo=str(args.algo),
            seed=args.seed,
            tags=["viewer"],
            config=agent_cfg if agent_cfg else None,
        )
        agent = create_agent(spec, verse.observation_space, verse.action_space)
        agent.seed(args.seed)

    action_n = int(getattr(verse.action_space, "n", 0) or 0)
    intervention_enabled = bool(args.intervention_mode and not manual_mode)
    intervention_path = _resolve_intervention_path(args.intervention_out) if intervention_enabled else ""
    intervention_file = open(intervention_path, "a", encoding="utf-8") if intervention_enabled else None
    if intervention_enabled:
        print(f"[viewer] intervention logging enabled: {intervention_path}")
        print("override keys: <enter>=auto, [action_id]/WASD=override, h=help, q=quit")

    if manual_mode:
        print(_manual_help(max(1, action_n)))
        print("")

    for ep in range(max(1, int(args.episodes))):
        obs, info = env.reset(seed=args.seed + ep)
        if safe_executor is not None:
            safe_executor.reset_episode(seed=args.seed + ep)
        done = False
        step_idx = 0
        episode_return = 0.0

        while not done and step_idx < int(args.max_steps):
            if args.clear_screen:
                print("\x1bc", end="")

            frame = env.render(mode=str(args.render_mode))
            if isinstance(frame, str):
                print(frame)
            elif frame is not None:
                print("<frame>")

            if manual_mode:
                cmd = input("action> ").strip()
                if cmd.lower() == "q":
                    env.close()
                    if agent is not None:
                        agent.close()
                    if safe_executor is not None:
                        safe_executor.close()
                    return
                if cmd.lower() == "h":
                    print(_manual_help(max(1, action_n)))
                    continue
                if cmd.lower() == "r":
                    break
                action = _manual_to_action(cmd, max(1, action_n))
                if action is None:
                    print(f"invalid action '{cmd}'.")
                    continue
                action_result = ActionResult(action=int(action), info={"mode": "manual"})
            else:
                if safe_executor is not None:
                    suggested = safe_executor.select_action(agent, obs)  # type: ignore[arg-type]
                else:
                    suggested = agent.act(obs)  # type: ignore[union-attr]

                action_result = suggested
                if intervention_enabled:
                    cmd = input("override> ").strip()
                    if cmd.lower() == "q":
                        if intervention_file is not None:
                            intervention_file.close()
                        env.close()
                        if agent is not None:
                            agent.close()
                        if safe_executor is not None:
                            safe_executor.close()
                        return
                    if cmd.lower() == "h":
                        print(_manual_help(max(1, action_n)))
                        continue
                    override = _manual_to_action(cmd, max(1, action_n))
                    if override is not None:
                        action_result = ActionResult(
                            action=int(override),
                            info={"mode": "human_intervention", "suggested_action": suggested.action},
                        )
                        if intervention_file is not None:
                            intervention_file.write(
                                json.dumps(
                                    {
                                        "obs": obs,
                                        "suggested_action": suggested.action,
                                        "action": int(override),
                                        "episode": int(ep + 1),
                                        "step_idx": int(step_idx),
                                        "source": "human_intervention",
                                        "verse_name": args.verse,
                                        "algo": args.algo,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

            next_obs, reward, terminated, truncated, step_info = env.step(action_result.action)
            step_result = StepResult(
                obs=next_obs,
                reward=float(reward),
                done=bool(terminated),
                truncated=bool(truncated),
                info=dict(step_info or {}),
            )
            if safe_executor is not None:
                step_result = safe_executor.post_step(
                    obs=obs,
                    action_result=action_result,
                    step_result=step_result,
                    step_idx=step_idx,
                )

            obs = step_result.obs
            info = dict(step_result.info or {})
            episode_return += float(step_result.reward)
            done = bool(step_result.done or step_result.truncated)

            _print_status(
                episode=ep + 1,
                step=step_idx,
                action=action_result.action,
                reward=float(step_result.reward),
                terminated=bool(step_result.done),
                truncated=bool(step_result.truncated),
                info=info,
                action_info=(action_result.info if isinstance(action_result.info, dict) else None),
            )

            step_idx += 1
            if float(args.sleep_s) > 0.0:
                time.sleep(float(args.sleep_s))

        print(f"episode={ep + 1} return={episode_return:.3f} steps={step_idx}")

    env.close()
    if intervention_file is not None:
        intervention_file.close()
    if agent is not None:
        agent.close()
    if safe_executor is not None:
        safe_executor.close()


def _read_events(run_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"events file not found: {path}")
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            ep = str(ev.get("episode_id", "unknown"))
            groups[ep].append(ev)
    return groups


def _build_counterfactual_agent(args: argparse.Namespace, verse_name: str):
    if not str(args.counterfactual_algo or "").strip():
        return None, None
    register_builtin_verses()
    register_builtin_agents()

    verse = None
    agent = None
    try:
        vparams = _default_verse_params(verse_name, int(args.max_steps))
        verse_spec = VerseSpec(
            spec_version="v1",
            verse_name=str(verse_name),
            verse_version="0.1",
            seed=int(args.seed),
            tags=["viewer_counterfactual"],
            params=vparams,
        )
        verse = create_verse(verse_spec)
        cfg = _parse_kv_list(args.counterfactual_aconfig)
        algo = str(args.counterfactual_algo).strip().lower()
        if algo in ("gateway", "special_moe", "adaptive_moe"):
            cfg.setdefault("verse_name", str(verse_name))
        if args.counterfactual_model_path:
            cfg["model_path"] = args.counterfactual_model_path
        if args.counterfactual_dataset:
            if len(args.counterfactual_dataset) == 1:
                cfg["dataset_path"] = args.counterfactual_dataset[0]
            else:
                cfg["dataset_paths"] = list(args.counterfactual_dataset)

        spec = AgentSpec(
            spec_version="v1",
            policy_id=f"counterfactual_{algo}",
            policy_version="0.1",
            algo=algo,
            seed=int(args.seed),
            tags=["viewer_counterfactual"],
            config=(cfg if cfg else None),
        )
        agent = create_agent(spec, verse.observation_space, verse.action_space)

        # Optional dataset hydration if supported.
        ds = cfg.get("dataset_paths")
        if ds and hasattr(agent, "learn_from_dataset"):
            for p in ds:
                if os.path.isfile(str(p)):
                    agent.learn_from_dataset(str(p))
        elif cfg.get("dataset_path") and hasattr(agent, "learn_from_dataset"):
            p = str(cfg.get("dataset_path"))
            if os.path.isfile(p):
                agent.learn_from_dataset(p)
        return agent, verse
    except Exception as e:
        print(f"[viewer] counterfactual init skipped for verse={verse_name}: {e}")
        try:
            if agent is not None:
                agent.close()
        except Exception:
            pass
        try:
            if verse is not None:
                verse.close()
        except Exception:
            pass
        return None, None


def run_replay(args: argparse.Namespace) -> None:
    register_builtin_verses()
    groups = _read_events(args.run_dir)
    episode_ids = sorted(groups.keys())
    if args.episode_id:
        episode_ids = [eid for eid in episode_ids if eid == args.episode_id]
    if not episode_ids:
        print("no episodes to replay")
        return

    cf_agents: Dict[str, Any] = {}
    cf_verses: Dict[str, Any] = {}
    verse: Optional[Verse] = None

    for eid in episode_ids[: max(1, int(args.max_episodes))]:
        events = groups[eid]
        if not events:
            continue

        first_event = events[0]
        verse_name = str(first_event.get("verse_name", "unknown"))
        seed = int(first_event.get("seed", args.seed))
        
        try:
            vparams = _default_verse_params(verse_name, int(args.max_steps))
            # TODO: vparams are an approximation. A proper fix would log them.
            verse_spec = VerseSpec(
                spec_version="v1",
                verse_name=verse_name,
                verse_version="0.1",
                seed=seed,
                tags=["viewer_replay"],
                params=vparams,
            )
            verse = create_verse(verse_spec)
            verse.reset()
        except Exception as e:
            print(f"Could not create verse '{verse_name}' for replay: {e}")
            continue

        print(f"=== replay episode: {eid} ({len(events)} steps) ===")
        if args.show_safety_heatmap:
            hm = _compute_safety_heatmap(events, confidence_drop_threshold=float(args.heatmap_conf_threshold))
            if hm is not None:
                grid, w, h = hm
                print(f"safety_heatmap ({w}x{h})")
                print(_render_heatmap_ascii(grid))
                print("")
        for idx, ev in enumerate(events):
            info = ev.get("info", {}) or {}

            frame = verse.render(mode="ansi")
            if args.clear_screen:
                print("\x1bc", end="")
            if frame:
                print(frame)
            else:
                # Fallback for verses with no ANSI render method
                print(ev.get("obs"))

            print(
                f"[step={int(ev.get('step_idx', idx))}] action={ev.get('action')} "
                f"reward={float(ev.get('reward', 0.0)):.3f} done={bool(ev.get('done', False))} "
                f"truncated={bool(ev.get('truncated', False))}"
            )
            if str(args.counterfactual_algo or "").strip():
                obs = ev.get("obs")
                if verse_name not in cf_agents:
                    a, v = _build_counterfactual_agent(args, verse_name)
                    if a is not None and v is not None:
                        cf_agents[verse_name] = a
                        cf_verses[verse_name] = v
                cf_agent = cf_agents.get(verse_name)
                if cf_agent is not None:
                    try:
                        cf_action = cf_agent.act(obs).action
                        mismatch = (cf_action != ev.get("action"))
                        print(f"counterfactual: action={cf_action} mismatch={mismatch}")
                    except Exception as e:
                        print(f"counterfactual: error={e}")
            se = info.get("safe_executor")
            if isinstance(se, dict):
                mode = se.get("mode")
                rew = bool(se.get("rewound", False))
                veto = bool(se.get("shield_veto", False))
                budget = se.get("planning_budget") if isinstance(se.get("planning_budget"), dict) else {}
                print(
                    "safe_executor: "
                    f"mode={mode} rewound={rew} shield_veto={veto} "
                    f"planner_budget_left={budget.get('episode_remaining')}"
                )
                explanation = se.get("explanation")
                if explanation:
                    print(f"  explanation: {explanation}")
            ai = info.get("action_info")
            if isinstance(ai, dict):
                sp = ai.get("self_play")
                if isinstance(sp, dict):
                    print(
                        "self_play: "
                        f"active={bool(sp.get('adversary_active', False))} "
                        f"mix={sp.get('mix_ratio')} source={sp.get('source', '')}"
                    )
                c = ai.get("contract")
                if isinstance(c, dict):
                    print(
                        "contract: "
                        f"exists={bool(c.get('exists', False))} "
                        f"skill={c.get('skill_tag')} version={c.get('version')}"
                    )

            # Step the verse to the next state
            verse.step(ev.get("action"))

            if args.pause:
                _ = input("next> ")
            elif float(args.sleep_s) > 0.0:
                time.sleep(float(args.sleep_s))

    for a in cf_agents.values():
        try:
            a.close()
        except Exception:
            pass
    for v in cf_verses.values():
        try:
            v.close()
        except Exception:
            pass
    # Close the verse we created for replay
    if verse is not None:
        try:
            verse.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "replay"], default="live")

    # live mode args
    ap.add_argument("--verse", type=str, default="grid_world")
    ap.add_argument("--verse_version", type=str, default="0.1")
    ap.add_argument("--algo", type=str, default="manual", help="manual or any registered algo")
    ap.add_argument("--policy_id", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--vparam", action="append", default=None, help="verse param k=v")
    ap.add_argument("--aconfig", action="append", default=None, help="agent config k=v")
    ap.add_argument("--safe_guard", action="store_true", help="Enable SafeExecutor wrapper")
    ap.add_argument("--safe_cfg", action="append", default=None, help="SafeExecutor config k=v")
    ap.add_argument("--intervention_mode", action="store_true", help="Allow live human overrides and log corrections.")
    ap.add_argument("--intervention_out", type=str, default="", help="Output JSONL path for intervention data.")
    ap.add_argument("--render_mode", type=str, default="ansi", choices=["ansi", "human", "rgb_array"])

    # replay mode args
    ap.add_argument("--run_dir", type=str, default="")
    ap.add_argument("--episode_id", type=str, default=None)
    ap.add_argument("--max_episodes", type=int, default=5)
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--counterfactual_algo", type=str, default="")
    ap.add_argument("--counterfactual_aconfig", action="append", default=None)
    ap.add_argument("--counterfactual_model_path", type=str, default="")
    ap.add_argument("--counterfactual_dataset", action="append", default=None)
    ap.add_argument("--show_safety_heatmap", action="store_true")
    ap.add_argument("--heatmap_conf_threshold", type=float, default=0.10)

    # shared
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--clear_screen", action="store_true")
    args = ap.parse_args()

    if args.mode == "live":
        run_live(args)
        return
    if not args.run_dir:
        raise ValueError("--run_dir is required for replay mode")
    run_replay(args)


if __name__ == "__main__":
    main()
