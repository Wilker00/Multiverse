"""
Memory inspection helpers for the Multiverse CLI.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from agents.registry import create_agent, register_builtin_agents
from core.memory_runtime import request_memory_payload, resolve_memory_request
from core.types import AgentSpec, VerseSpec
from memory.central_repository import CentralMemoryConfig, find_similar
from verses.registry import create_verse, register_builtin


def _parse_kv_list(kvs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not kvs:
        return out
    for item in kvs:
        if "=" not in str(item):
            raise ValueError(f"Invalid param '{item}'. Expected k=v.")
        k, v = str(item).split("=", 1)
        key = str(k).strip()
        raw = str(v).strip()
        if raw.lower() in ("true", "false"):
            out[key] = (raw.lower() == "true")
            continue
        try:
            if "." in raw:
                out[key] = float(raw)
            else:
                out[key] = int(raw)
            continue
        except (TypeError, ValueError):
            pass
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, (list, dict)):
                out[key] = parsed
                continue
        except Exception:
            pass
        out[key] = raw
    return out


def _load_obs(args: argparse.Namespace) -> Any:
    obs_json = str(getattr(args, "obs_json", "") or "").strip()
    obs_file = str(getattr(args, "obs_file", "") or "").strip()
    if not obs_json and not obs_file:
        raise ValueError("Provide --obs-json or --obs-file.")
    if obs_json and obs_file:
        raise ValueError("Use only one of --obs-json or --obs-file.")
    if obs_file:
        with open(obs_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(obs_json)


def cmd_memory_inspect(args: argparse.Namespace) -> int:
    register_builtin()
    register_builtin_agents()

    errors: List[Dict[str, Any]] = []
    verse = None
    agent = None
    try:
        verse_params = _parse_kv_list(args.vparam)
        verse_spec = VerseSpec(
            spec_version="v1",
            verse_name=str(args.verse),
            verse_version=str(args.verse_version),
            seed=(None if args.seed is None else int(args.seed)),
            params=verse_params,
        )
        verse = create_verse(verse_spec)

        agent_cfg = _parse_kv_list(args.aconfig)
        agent_cfg.setdefault("verse_name", str(args.verse))
        agent_spec = AgentSpec(
            spec_version="v1",
            policy_id=str(args.policy_id or f"{args.algo}_memory_inspect"),
            policy_version="0.1",
            algo=str(args.algo),
            seed=(None if args.seed is None else int(args.seed)),
            config=agent_cfg,
        )
        agent = create_agent(
            spec=agent_spec,
            observation_space=verse.observation_space,
            action_space=verse.action_space,
        )
        if args.seed is not None:
            try:
                verse.seed(int(args.seed))
            except Exception:
                pass
            try:
                agent.seed(int(args.seed))
            except Exception:
                pass

        obs = _load_obs(args)
        mode = str(args.mode).strip().lower()
        method_name = "memory_bootstrap_request" if mode == "bootstrap" else "memory_query_request"

        def _on_error(code: str, component: str, step_idx: Optional[int], exc: Exception) -> None:
            errors.append(
                {
                    "code": str(code),
                    "component": str(component),
                    "step_idx": (None if step_idx is None else int(step_idx)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

        req = request_memory_payload(
            agent=agent,
            method_name=method_name,
            request_obs=obs,
            request_step_idx=int(args.step_idx),
            on_error=_on_error,
            error_code=f"{method_name}_error",
            component=f"memory.inspect.{method_name}",
        )

        resolved = None
        status = "agent_declined"
        if not hasattr(agent, method_name):
            status = "agent_no_hook"
        elif isinstance(req, dict):
            resolved = resolve_memory_request(
                agent=agent,
                req=req,
                default_obs=obs,
                request_step_idx=int(args.step_idx),
                find_similar_fn=find_similar,
                memory_cfg=CentralMemoryConfig(root_dir=str(args.central_memory_dir)),
                on_error=_on_error,
                lookup_error_code="memory_lookup_error",
                lookup_component="memory.inspect.lookup",
                response_error_code="memory_response_error",
            )
            status = "resolved" if isinstance(resolved, dict) else "lookup_error"

        payload = {
            "status": status,
            "mode": mode,
            "method": method_name,
            "algo": str(args.algo),
            "verse": str(args.verse),
            "memory_root": str(args.central_memory_dir),
            "step_idx": int(args.step_idx),
            "agent_has_hook": bool(hasattr(agent, method_name)),
            "request_emitted": bool(isinstance(req, dict)),
            "request": req,
            "bundle": (resolved or {}).get("bundle") if isinstance(resolved, dict) else None,
            "match_count": int((resolved or {}).get("match_count", 0)) if isinstance(resolved, dict) else 0,
            "errors": errors,
        }

        if bool(args.json):
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        print(f"status      : {payload['status']}")
        print(f"mode        : {payload['mode']}")
        print(f"algo        : {payload['algo']}")
        print(f"verse       : {payload['verse']}")
        print(f"memory root : {payload['memory_root']}")
        print(f"request     : {'yes' if payload['request_emitted'] else 'no'}")
        if isinstance(req, dict):
            print(f"reason      : {str(req.get('reason', ''))}")
        print(f"matches     : {int(payload['match_count'])}")
        if isinstance(payload.get("bundle"), dict):
            matches = payload["bundle"].get("matches")
            if isinstance(matches, list) and matches:
                top = matches[0] if isinstance(matches[0], dict) else {}
                print(f"top pointer : {str(top.get('pointer_path', ''))}")
                print(f"top score   : {top.get('score')}")
        if errors:
            print(f"errors      : {len(errors)}")
        return 0
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass
        if verse is not None:
            try:
                verse.close()
            except Exception:
                pass
