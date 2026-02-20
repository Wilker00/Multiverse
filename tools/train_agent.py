"""
tools/train_agent.py

Unified CLI runner for u.ai.
...
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any, Dict, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import VerseSpec, AgentSpec
from core.safe_executor import SafeExecutorConfig
from memory.knowledge_market import KnowledgeMarket, KnowledgeMarketConfig
from orchestrator.trainer import Trainer

from memory.episode_index import EpisodeIndexConfig, build_episode_index
from orchestrator.evaluator import evaluate_run, print_report

# --- STAGE 2: CENTROID REGULARIZATION ---
# To implement Distral-style learning, the agent's training loop needs to be modified.
# The agent would load the centroid policy and add a KL-divergence loss term to its
# own policy update. This encourages the agent to stay close to the distilled,
# generalist policy while still specializing on its local task.
#
# Conceptual change in the agent's `learn` method:
#
# def learn(self, transitions):
#     # 1. Load the centroid policy (if provided)
#     if self.centroid_policy is not None:
#         with torch.no_grad():
#             centroid_action_dist = self.centroid_policy(transitions.obs)
#
#     # 2. Calculate the standard policy loss (e.g., PPO, DQN)
#     policy_loss = self.calculate_policy_loss(transitions)
#
#     # 3. Calculate the KL-divergence regularization term
#     if self.centroid_policy is not None:
#         current_action_dist = self.policy(transitions.obs)
#         kl_loss = torch.distributions.kl.kl_divergence(centroid_action_dist, current_action_dist).mean()
#         
#         # 4. Combine the losses
#         total_loss = policy_loss + self.kl_regularization_weight * kl_loss
#     else:
#         total_loss = policy_loss
#
#     # 5. Backpropagate the total loss
#     self.optimizer.zero_grad()
#     total_loss.backward()
#     self.optimizer.step()
#
# This change would be implemented within the specific agent's code (e.g., agents/ppo.py).
# The train_agent.py script is responsible for passing the configuration.
# -----------------------------------------

def _parse_kv_list(kvs: Optional[list[str]]) -> Dict[str, Any]:
    """
    Parse repeated --param k=v into dict with basic type inference.
    """
    out: Dict[str, Any] = {}
    if not kvs:
        return out

    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Invalid param '{item}'. Expected k=v.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        # basic type inference
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


def _extract_run_ids_from_text(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"(run_[0-9a-f]{32})", str(text), flags=re.IGNORECASE)


def _collect_market_provider_ids(args: argparse.Namespace, run_id: str) -> list[str]:
    if args.market_provider_id:
        return sorted(set(str(x) for x in args.market_provider_id if str(x).strip()))

    raw_sources: list[str] = []
    for value in (
        *(args.dataset or []),
        args.dataset_dir,
        args.bad_dna,
        args.model_path,
        args.expert_dataset_dir,
        args.expert_model_dir,
        args.centroid_model_path,
        args.centroid_dataset_path,
    ):
        if value:
            raw_sources.append(str(value))

    out: list[str] = []
    for s in raw_sources:
        out.extend(_extract_run_ids_from_text(s))

    # If no provider is inferable, attribute to the newly produced run.
    if not out:
        out = [str(run_id)]
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()

    # Core run args
    ap.add_argument("--verse", type=str, default="line_world")
    ap.add_argument("--verse_version", type=str, default="0.1")
    ap.add_argument("--algo", type=str, default="random")
    ap.add_argument("--policy_id", type=str, default=None)
    ap.add_argument("--policy_version", type=str, default="0.0")

    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=123)

    # Verse params: repeated --vparam key=value
    ap.add_argument("--vparam", action="append", default=None)

    # Agent config: repeated --aconfig key=value
    ap.add_argument("--aconfig", action="append", default=None)

    # Special optional dataset path(s) (used by imitation_lookup/library)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--dataset_dir", type=str, default=None)
    ap.add_argument("--model_path", type=str, default=None, help="load model for distilled agent")
    ap.add_argument("--bad_dna", type=str, default=None, help="bad DNA path for special agent")
    ap.add_argument("--cql_alpha", type=float, default=None, help="CQL penalty alpha")
    ap.add_argument("--selector_model_path", type=str, default=None, help="selector checkpoint for special_moe")
    ap.add_argument("--top_k_experts", type=int, default=None, help="top-k experts for special_moe routing")
    ap.add_argument("--expert_dataset_dir", type=str, default=None, help="directory of per-skill expert datasets")
    ap.add_argument("--expert_model_dir", type=str, default=None, help="directory of distilled expert models")
    ap.add_argument("--expert_enable_mlp", action="store_true", help="Enable MLP generalizer inside MoE imitation experts.")
    ap.add_argument("--expert_nn_k", type=int, default=None, help="k for nearest-neighbor fallback inside MoE experts.")
    ap.add_argument("--uncertainty_threshold", type=float, default=None, help="adaptive_moe disagreement threshold")
    ap.add_argument("--centroid_model_path", type=str, default=None, help="adaptive_moe centroid distilled model")
    ap.add_argument("--centroid_dataset_path", type=str, default=None, help="adaptive_moe centroid imitation dataset")
    ap.add_argument("--safe_guard", action="store_true", help="Enable runtime SafeExecutor guard.")
    ap.add_argument(
        "--safe_cfg",
        action="append",
        default=None,
        help="SafeExecutor config k=v. Prefix fallback_ for fallback agent config (e.g. fallback_algo=gateway fallback_manifest_path=models/default_policy_set.json).",
    )

    # --- STAGE 2: New argument for centroid policy ---
    ap.add_argument("--centroid_policy_path", type=str, default=None, help="Path to the pre-trained centroid policy for regularization.")
    ap.add_argument("--kl_weight", type=float, default=0.1, help="Weight for the KL-divergence regularization term.")
    # -----------------------------------------------

    # Training toggle
    ap.add_argument("--train", action="store_true")

    # Outputs
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--make_index", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--market_auto_update", action="store_true", help="Auto-update KnowledgeMarket reputation after eval.")
    ap.add_argument("--market_memory_dir", type=str, default="central_memory")
    ap.add_argument("--market_provider_id", action="append", default=None, help="Explicit provider run_id (repeatable).")
    ap.add_argument("--market_baseline_return", type=float, default=0.0, help="Reference return for reputation delta.")
    ap.add_argument("--market_scale", type=float, default=0.5, help="Scale factor for return-delta reputation updates.")
    ap.add_argument("--market_reason", type=str, default="train_eval")

    args = ap.parse_args()

    verse_params = _parse_kv_list(args.vparam)

    if args.verse == "line_world":
        verse_params.setdefault("goal_pos", 8)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("step_penalty", -0.02)
    elif args.verse == "cliff_world":
        verse_params.setdefault("width", 12)
        verse_params.setdefault("height", 4)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("step_penalty", -1.0)
        verse_params.setdefault("cliff_penalty", -100.0)
        verse_params.setdefault("end_on_cliff", False)
    elif args.verse == "labyrinth_world":
        verse_params.setdefault("width", 15)
        verse_params.setdefault("height", 11)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("battery_capacity", 80)
        verse_params.setdefault("battery_drain", 1)
        verse_params.setdefault("action_noise", 0.08)
        verse_params.setdefault("vision_radius", 1)
    elif args.verse == "chess_world":
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("win_material", 8)
        verse_params.setdefault("lose_material", -8)
        verse_params.setdefault("random_swing", 0.20)
    elif args.verse == "go_world":
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("target_territory", 10)
        verse_params.setdefault("random_swing", 0.25)
    elif args.verse == "uno_world":
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("start_hand", 7)
        verse_params.setdefault("opp_start_hand", 7)
        verse_params.setdefault("random_swing", 0.25)
    elif args.verse == "harvest_world":
        verse_params.setdefault("width", 8)
        verse_params.setdefault("height", 8)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("num_fruit", 6)
        verse_params.setdefault("carry_capacity", 3)
        verse_params.setdefault("spoil_probability", 0.01)
    elif args.verse == "bridge_world":
        verse_params.setdefault("bridge_length", 8)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("wind_probability", 0.15)
    elif args.verse == "swamp_world":
        verse_params.setdefault("width", 10)
        verse_params.setdefault("height", 10)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("flood_rate", 5)
        verse_params.setdefault("mud_count", 6)
        verse_params.setdefault("haven_count", 2)
    elif args.verse == "escape_world":
        verse_params.setdefault("width", 10)
        verse_params.setdefault("height", 10)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("num_guards", 3)
        verse_params.setdefault("num_hiding_spots", 4)
        verse_params.setdefault("guard_vision", 2)
    elif args.verse == "factory_world":
        verse_params.setdefault("num_machines", 3)
        verse_params.setdefault("buffer_size", 4)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("arrival_rate", 0.6)
        verse_params.setdefault("breakdown_prob", 0.08)
    elif args.verse == "trade_world":
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("starting_cash", 100.0)
        verse_params.setdefault("max_inventory", 10)
        verse_params.setdefault("cycle_length", 20)
    elif args.verse == "memory_vault_world":
        verse_params.setdefault("width", 9)
        verse_params.setdefault("height", 9)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("wall_density", 0.16)
        verse_params.setdefault("hint_visible_steps", 1)
    elif args.verse == "rule_flip_world":
        verse_params.setdefault("track_len", 11)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("flip_step", max(2, int(args.max_steps) // 2))
        verse_params.setdefault("target_reward", 2.0)
        verse_params.setdefault("wrong_target_penalty", -1.5)
    elif args.verse == "wind_master_world":
        verse_params.setdefault("width", 14)
        verse_params.setdefault("height", 7)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("gust_probability", 0.12)
        verse_params.setdefault("target_margin", 2)
        verse_params.setdefault("margin_reward_scale", 0.05)

    agent_config = _parse_kv_list(args.aconfig)

    if args.dataset:
        if len(args.dataset) == 1:
            agent_config["dataset_path"] = args.dataset[0]
        else:
            agent_config["dataset_paths"] = args.dataset
    if args.dataset_dir:
        agent_config["dataset_dir"] = args.dataset_dir
    if args.model_path:
        agent_config["model_path"] = args.model_path
    if args.bad_dna:
        agent_config["bad_dna_path"] = args.bad_dna
    if args.cql_alpha is not None:
        agent_config["alpha"] = float(args.cql_alpha)
    selector_model_path = args.selector_model_path
    if not selector_model_path and args.algo == "special_moe":
        selector_candidates = [
            os.path.join("models", "micro_selector_balanced_64.pt"),
            os.path.join("models", "micro_selector_balanced.pt"),
            os.path.join("models", "micro_selector.pt"),
        ]
        for cand in selector_candidates:
            if os.path.isfile(cand):
                selector_model_path = cand
                break
    if selector_model_path:
        agent_config["selector_model_path"] = selector_model_path
    if args.top_k_experts is not None:
        agent_config["top_k"] = int(args.top_k_experts)
    if args.expert_dataset_dir:
        agent_config["expert_dataset_dir"] = args.expert_dataset_dir
    if args.expert_model_dir:
        agent_config["expert_model_dir"] = args.expert_model_dir
    if args.expert_enable_mlp or args.expert_nn_k is not None:
        expert_lookup_cfg = dict(agent_config.get("expert_lookup_config", {})) if isinstance(agent_config.get("expert_lookup_config"), dict) else {}
        if args.expert_enable_mlp:
            expert_lookup_cfg["enable_mlp_generalizer"] = True
        if args.expert_nn_k is not None:
            expert_lookup_cfg["nn_fallback_k"] = int(args.expert_nn_k)
        expert_lookup_cfg.setdefault("enable_nn_fallback", True)
        agent_config["expert_lookup_config"] = expert_lookup_cfg
    if args.uncertainty_threshold is not None:
        agent_config["uncertainty_threshold"] = float(args.uncertainty_threshold)
    if args.centroid_model_path:
        agent_config["centroid_model_path"] = args.centroid_model_path
    if args.centroid_dataset_path:
        agent_config["centroid_dataset_path"] = args.centroid_dataset_path

    strategy_verses = {"chess_world", "go_world", "uno_world"}
    if args.algo in ("special_moe", "adaptive_moe", "library") and args.verse in strategy_verses:
        default_expert_dir = os.path.join("models", "expert_datasets")
        if os.path.isdir(default_expert_dir):
            agent_config.setdefault("expert_dataset_dir", default_expert_dir)
            agent_config.setdefault("dataset_dir", default_expert_dir)

        perf_path = os.path.join("models", "expert_datasets", "strategy_transfer_performance.json")
        if os.path.isfile(perf_path):
            agent_config.setdefault("expert_performance_path", perf_path)

        # Automatically include synthetic transfer datasets targeting the active verse.
        transfer_glob = os.path.join("models", "expert_datasets", f"synthetic_transfer_*_to_{args.verse}.jsonl")
        transfer_paths = sorted(glob.glob(transfer_glob))
        if transfer_paths:
            existing = agent_config.get("dataset_paths")
            merged_paths = []
            if isinstance(existing, list):
                merged_paths.extend([str(p) for p in existing])
            elif isinstance(existing, str) and existing.strip():
                merged_paths.append(existing.strip())
            merged_paths.extend(transfer_paths)
            # de-dup while preserving order
            uniq = []
            seen = set()
            for p in merged_paths:
                if p in seen:
                    continue
                seen.add(p)
                uniq.append(p)
            if uniq:
                agent_config["dataset_paths"] = uniq

    safe_cfg = _parse_kv_list(args.safe_cfg)
    if args.safe_guard or safe_cfg:
        if args.safe_guard:
            safe_cfg.setdefault("enabled", True)
        if str(args.verse).strip().lower() == "cliff_world":
            # Cliff is highly punitive; default to a tighter veto threshold unless explicitly overridden.
            safe_cfg.setdefault("danger_threshold", 0.60)
        fallback_cfg: Dict[str, Any] = {}
        for key in list(safe_cfg.keys()):
            if key.startswith("fallback_") and key != "fallback_algo":
                fb_key = key[len("fallback_") :]
                if fb_key:
                    fallback_cfg[fb_key] = safe_cfg.pop(key)
        if fallback_cfg:
            safe_cfg["fallback_config"] = fallback_cfg
        # Fail fast on invalid SafeExecutor keys/types before launching a run.
        safe_runtime_cfg = dict(safe_cfg)
        safe_runtime_cfg.pop("fallback_algo", None)
        safe_runtime_cfg.pop("fallback_config", None)
        SafeExecutorConfig.from_dict(safe_runtime_cfg)
        agent_config["safe_executor"] = safe_cfg
        
    # --- STAGE 2: Pass centroid config to the agent ---
    if args.centroid_policy_path:
        agent_config["centroid_policy_path"] = args.centroid_policy_path
        agent_config["kl_regularization_weight"] = args.kl_weight
    # ------------------------------------------------

    policy_id = args.policy_id or args.algo

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=args.verse,
        verse_version=args.verse_version,
        seed=args.seed,
        tags=["cli"],
        params=verse_params,
    )

    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=policy_id,
        policy_version=args.policy_version,
        algo=args.algo,
        seed=args.seed,
        config=agent_config if agent_config else None,
    )

    before = set(os.listdir(args.runs_root)) if os.path.isdir(args.runs_root) else set()

    trainer = Trainer(run_root=args.runs_root, schema_version="v1", auto_register_builtin=True)

    if args.train:
        agent_config = dict(agent_spec.config) if isinstance(agent_spec.config, dict) else {}
        agent_config["train"] = True
        agent_spec = agent_spec.evolved(config=agent_config)

    run_result = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    run_id = str(run_result.get("run_id", "")).strip() if isinstance(run_result, dict) else ""
    if not run_id:
        after = set(os.listdir(args.runs_root)) if os.path.isdir(args.runs_root) else set()
        new_runs = sorted(list(after - before))
        if not new_runs:
            raise RuntimeError(f"Could not detect new run directory under {args.runs_root}/")
        run_id = new_runs[-1]
    run_dir = os.path.join(args.runs_root, run_id)

    print("")
    print(f"Run dir: {run_dir}")

    if args.make_index:
        idx = build_episode_index(EpisodeIndexConfig(run_dir=run_dir))
        print(f"Built episode index: {idx}")

    if args.eval:
        stats = evaluate_run(run_dir)
        print("")
        print_report(stats)
        if args.market_auto_update:
            market = KnowledgeMarket(KnowledgeMarketConfig(root_dir=args.market_memory_dir))
            provider_ids = _collect_market_provider_ids(args, run_id=run_id)
            if provider_ids:
                delta_total = (float(stats.mean_return) - float(args.market_baseline_return)) * float(args.market_scale)
                per_provider = float(delta_total) / float(max(1, len(provider_ids)))
                print("")
                print("KnowledgeMarket updates")
                for pid in provider_ids:
                    upd = market.update_reputation(
                        provider_id=str(pid),
                        delta=float(per_provider),
                        reason=f"{args.market_reason}:{args.algo}:{args.verse}",
                        consumer_agent_id=f"train:{policy_id}",
                    )
                    print(
                        f"- provider={upd['provider_id']} delta={per_provider:.3f} "
                        f"new_reputation={float(upd['new_reputation']):.3f}"
                    )


if __name__ == "__main__":
    main()
