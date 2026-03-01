import re

with open("tools/run_adt_dagger.py", "r") as f:
    content = f.read()

# 1. Provide the memory query logic in _collect_dagger_labels
old_act_loop = """                learner_out = learner.act(obs)
                learner_action = int(learner_out.action)"""

new_act_loop = """                hint = None
                if hasattr(learner, "memory_query_request"):
                    try:
                        req = learner.memory_query_request(obs=obs, step_idx=step_idx)
                        if isinstance(req, dict):
                            from memory.central_repository import CentralMemoryConfig, find_similar
                            mem_cfg = CentralMemoryConfig(root_dir="central_memory")
                            matches = find_similar(
                                obs=req.get("query_obs", obs),
                                cfg=mem_cfg,
                                top_k=req.get("top_k", 3),
                                verse_name=req.get("verse_name"),
                                min_score=req.get("min_score", -1.0)
                            )
                            rows = []
                            for m in matches:
                                rows.append({
                                    "score": float(getattr(m, "score", 0.0)),
                                    "action": getattr(m, "action", None),
                                    "source_greedy_action": getattr(m, "source_greedy_action", None),
                                    "verse_name": getattr(m, "verse_name", None),
                                })
                            bundle = {"matches": rows, "query_step_idx": step_idx, "reason": req.get("reason", "")}
                            hint = {"memory_recall": bundle}
                            if hasattr(learner, "on_memory_response"):
                                learner.on_memory_response(bundle)
                            memory_queries_issued += 1
                    except Exception as e:
                        pass
                if hint is not None and hasattr(learner, "act_with_hint"):
                    learner_out = learner.act_with_hint(obs, hint)
                else:
                    learner_out = learner.act(obs)
                learner_action = int(learner_out.action)"""

content = content.replace(old_act_loop, new_act_loop)

# 2. Add memory_queries_issued var
old_stats_init = """    q_labels = 0
    old_eps = float(expert.stats.epsilon)"""
new_stats_init = """    q_labels = 0
    memory_queries_issued = 0
    old_eps = float(expert.stats.epsilon)"""
content = content.replace(old_stats_init, new_stats_init)

old_stats_return = """    stats = {
        "episodes": float(episodes),
        "success_rate": float(success_n / n),
        "mean_return": float(return_sum / n),
        "mean_steps": float(steps_sum / n),
        "labeled_rows": float(len(rows)),
        "planner_labels": float(planner_labels),
        "q_labels": float(q_labels),
    }"""
new_stats_return = """    stats = {
        "episodes": float(episodes),
        "success_rate": float(success_n / n),
        "mean_return": float(return_sum / n),
        "mean_steps": float(steps_sum / n),
        "labeled_rows": float(len(rows)),
        "planner_labels": float(planner_labels),
        "q_labels": float(q_labels),
        "memory_queries_issued": float(memory_queries_issued),
    }"""
content = content.replace(old_stats_return, new_stats_return)

# 3. Handle planner expert warmup eval
old_expert_warm_skip = """    if str(args.expert_policy).strip().lower() == "planner":
        warm = {"episodes": 0.0, "success_rate": 0.0, "mean_return": 0.0, "mean_steps": 0.0}
        print("expert_warmup skipped (expert_policy=planner)")
    else:"""
new_expert_warm_eval = """    if str(args.expert_policy).strip().lower() == "planner":
        # Evaluate planner properly
        success_n = 0
        eval_eps = min(50, max(10, int(args.expert_warmup_episodes)))
        for _ in range(eval_eps):
            rr = verse.reset()
            obs = rr.obs
            ep_success = False
            for _ in range(int(args.max_steps)):
                try:
                    plan = plan_actions_from_current_state(
                        verse=verse,
                        horizon=max(1, int(args.planner_horizon)),
                        max_expansions=max(100, int(args.planner_max_expansions)),
                        avoid_terminal_failures=bool(args.planner_avoid_terminal_failures),
                    )
                    action = int(plan[0]) if plan else int(expert.act(obs).action)
                except:
                    action = int(expert.act(obs).action)
                sr = verse.step(action)
                info = sr.info if isinstance(sr.info, dict) else {}
                if info.get("reached_goal", False) or info.get("success", False):
                    ep_success = True
                if sr.done or sr.truncated:
                    break
                obs = sr.obs
            if ep_success:
                success_n += 1
        warm = {"episodes": eval_eps, "success_rate": success_n/eval_eps, "mean_return": 0.0, "mean_steps": 0.0}
        print(f"expert_warmup eval for planner: episodes={eval_eps} success={warm['success_rate']:.3f}")
    else:"""
content = content.replace(old_expert_warm_skip, new_expert_warm_eval)

# 4. Turn on memory hooks for the learner in config
old_learner_config = """            config={
                "model_path": current_model_path,
                "device": str(args.device),
                "context_len": int(args.context_len),
                "target_return": float(args.learner_target_return),
                "sample": bool(args.learner_sample),
                "temperature": max(1e-6, float(args.learner_temperature)),
                "top_k": max(0, int(args.learner_top_k)),
            },"""
new_learner_config = """            config={
                "model_path": current_model_path,
                "device": str(args.device),
                "context_len": int(args.context_len),
                "target_return": float(args.learner_target_return),
                "sample": bool(args.learner_sample),
                "temperature": max(1e-6, float(args.learner_temperature)),
                "top_k": max(0, int(args.learner_top_k)),
                "recall_enabled": True,
            },"""
content = content.replace(old_learner_config, new_learner_config)


old_print_col = """        print(
            f"round={int(round_idx)} collect episodes={int(col['episodes'])} "
            f"success={float(col['success_rate']):.3f} mean_return={float(col['mean_return']):.3f} "
            f"labeled_rows={int(col['labeled_rows'])} planner_labels={int(col['planner_labels'])} q_labels={int(col['q_labels'])}"
        )"""

new_print_col = """        print(
            f"round={int(round_idx)} collect episodes={int(col['episodes'])} "
            f"success={float(col['success_rate']):.3f} mean_return={float(col['mean_return']):.3f} "
            f"labeled_rows={int(col['labeled_rows'])} planner_labels={int(col['planner_labels'])} q_labels={int(col['q_labels'])} "
            f"memory_queries_issued={int(col['memory_queries_issued'])}"
        )"""
content = content.replace(old_print_col, new_print_col)

with open("tools/run_adt_dagger.py", "w") as f:
    f.write(content)
