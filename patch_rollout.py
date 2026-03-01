import os

with open("core/rollout.py", "r", encoding="utf-8") as f:
    content = f.read()

from_str = """        rows.append(
            {
                "score": float(getattr(m, "score", 0.0)),
                "run_id": str(getattr(m, "run_id", "")),
                "episode_id": str(getattr(m, "episode_id", "")),
                "step_idx": int(getattr(m, "step_idx", 0)),
                "verse_name": str(getattr(m, "verse_name", "")),
                "action": getattr(m, "action", None),
                "source_greedy_action": getattr(m, "source_greedy_action", None),
                "source_action_matches_greedy": getattr(m, "source_action_matches_greedy", None),
                "reward": float(getattr(m, "reward", 0.0)),
                "pointer_path": _memory_pointer(
                    run_id=str(getattr(m, "run_id", "")),
                    episode_id=str(getattr(m, "episode_id", "")),
                    step_idx=int(getattr(m, "step_idx", 0)),
                ),
            }
        )"""

to_str = """        row = {
            "score": float(getattr(m, "score", 0.0)),
            "run_id": str(getattr(m, "run_id", "")),
            "episode_id": str(getattr(m, "episode_id", "")),
            "step_idx": int(getattr(m, "step_idx", 0)),
            "verse_name": str(getattr(m, "verse_name", "")),
            "action": getattr(m, "action", None),
            "source_greedy_action": getattr(m, "source_greedy_action", None),
            "source_action_matches_greedy": getattr(m, "source_action_matches_greedy", None),
            "reward": float(getattr(m, "reward", 0.0)),
            "pointer_path": _memory_pointer(
                run_id=str(getattr(m, "run_id", "")),
                episode_id=str(getattr(m, "episode_id", "")),
                step_idx=int(getattr(m, "step_idx", 0)),
            ),
        }
        if getattr(m, "trajectory", None) is not None:
            row["trajectory"] = list(m.trajectory)
        rows.append(row)"""

content = content.replace(from_str, to_str)


from_str_2 = """                        query_obs = req.get("query_obs", obs)
                        top_k = max(1, int(req.get("top_k", 3)))
                        min_score = float(req.get("min_score", -1.0))
                        verse_name = req.get("verse_name")
                        verse_name = None if verse_name in (None, "") else str(verse_name).strip().lower()
                        memory_families = _as_set(req.get("memory_families"))
                        memory_types = _as_set(req.get("memory_types"))
                        matches = on_demand_find_similar(
                            obs=query_obs,
                            cfg=on_demand_mem_cfg,
                            top_k=top_k,
                            verse_name=verse_name,
                            min_score=min_score,
                            memory_families=memory_families,
                            memory_types=memory_types,
                        )"""

to_str_2 = """                        query_obs = req.get("query_obs", obs)
                        top_k = max(1, int(req.get("top_k", 3)))
                        min_score = float(req.get("min_score", -1.0))
                        trajectory_window = max(0, int(req.get("trajectory_window", 0)))
                        verse_name = req.get("verse_name")
                        verse_name = None if verse_name in (None, "") else str(verse_name).strip().lower()
                        memory_families = _as_set(req.get("memory_families"))
                        memory_types = _as_set(req.get("memory_types"))
                        matches = on_demand_find_similar(
                            obs=query_obs,
                            cfg=on_demand_mem_cfg,
                            top_k=top_k,
                            verse_name=verse_name,
                            min_score=min_score,
                            memory_families=memory_families,
                            memory_types=memory_types,
                            trajectory_window=trajectory_window,
                        )"""

content = content.replace(from_str_2, to_str_2)

with open("core/rollout.py", "w", encoding="utf-8") as f:
    f.write(content)

print("rollout.py patch ok")
