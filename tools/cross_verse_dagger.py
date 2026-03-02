import os
import sys
import argparse
import shutil
import json
import time

from agents.q_agent import QLearningAgent
from agents.transformer_agent import TransformerAgent
from core.types import AgentSpec, VerseSpec
from tools.prep_adt_data import prepare_adt_data
from tools.train_adt import train_adt
from verses.registry import create_verse, register_builtin
from models.decision_transformer import load_decision_transformer_checkpoint
from tools.run_adt_dagger import _collect_dagger_labels, _append_jsonl

# Verses with complex MCTS should use Q-expert fallback for speed
_STRATEGY_VERSES = {"chess_world", "go_world", "uno_world", "chess_world_v2", "go_world_v2", "uno_world_v2"}

# Target returns optimized for successful trajectory conditioning per environment
_VERSE_TARGET_RETURNS = {
    "line_world": 0.85,
    "cliff_world": -80.0,
    "grid_world": 1.0,
    "maze_world": 1.0,
    "warehouse_world": 1.0,
    "swamp_world": 0.5,
    "wind_master_world": 1.5,
    "escape_world": 1.0,
    "trade_world": 0.0,  # Trade world is relative to cash, 0.0 is steady-state
    "chess_world": 1.0,
    "go_world": 1.0,
    "uno_world": 1.0,
    "factory_world": 1.0,
}

def _flush():
    sys.stdout.flush()
    sys.stderr.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verses", type=str, default="grid_world,warehouse_world,trade_world", help="Comma-separated list of verses")
    ap.add_argument("--base_model_path", type=str, required=True)
    ap.add_argument("--out_model_path", type=str, default="models/dt_cross_verse.pt")
    ap.add_argument("--dagger_dataset_path", type=str, default="models/expert_datasets/cross_verse_labels.jsonl")
    ap.add_argument("--prepared_dataset_path", type=str, default="models/adt_cross_verse.pt")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--collect_episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--expert_policy", type=str, default="planner")
    ap.add_argument("--global_action_dim", type=int, default=7, help="Shared max action dim")
    ap.add_argument("--expert_warmup", type=int, default=30, help="Q-expert warmup episodes per verse")
    args = ap.parse_args()

    register_builtin()
    
    verses_list = [v.strip() for v in args.verses.split(",") if v.strip()]
    
    current_model = args.base_model_path
    
    # reset dataset
    os.makedirs(os.path.dirname(args.dagger_dataset_path) or ".", exist_ok=True)
    with open(args.dagger_dataset_path, "w", encoding="utf-8") as f:
        pass

    print(f"Cross-Verse DAgger: {len(verses_list)} verses, {args.rounds} rounds, {args.collect_episodes} episodes/verse")
    _flush()
        
    for r in range(1, args.rounds + 1):
        print(f"\n=== ROUND {r}/{args.rounds} ===")
        _flush()
        round_start = time.time()
        
        for verse_name in verses_list:
            t0 = time.time()
            print(f"  [{verse_name}] collecting...", end=" ")
            _flush()

            vspec = VerseSpec(spec_version="v1", verse_name=verse_name, verse_version="1", seed=123+r)
            verse = create_verse(vspec)
            verse.seed(123+r)
            
            # Setup learner
            target_ret = _VERSE_TARGET_RETURNS.get(verse_name, 1.0)
            learner_spec = AgentSpec(
                spec_version="v1", policy_id=f"cv_adt_r{r}", policy_version="0.1", algo="adt",
                config={
                    "model_path": current_model, 
                    "device": "cpu", 
                    "context_len": 30, # Increased context for larger model
                    "target_return": target_ret, 
                    "recall_enabled": True, 
                    "recall_frequency": 5,
                    "recall_top_k": 3,
                    "verse_name": verse_name
                }
            )
            learner = TransformerAgent(spec=learner_spec, observation_space=verse.observation_space, action_space=verse.action_space)
            
            # Setup Q-expert
            expert_spec = AgentSpec(spec_version="v1", policy_id="dummy", policy_version="0.1", algo="q")
            expert = QLearningAgent(spec=expert_spec, observation_space=verse.observation_space, action_space=verse.action_space)
            
            # For strategy games, use Q-expert to avoid slow MCTS
            if verse_name in _STRATEGY_VERSES:
                policy = "q"
                planner_expansions = 100
                planner_horizon = 2
                # Quick warmup for Q-expert on strategy verses
                from core.agent_base import ExperienceBatch, Transition
                for _we in range(args.expert_warmup):
                    rr = verse.reset()
                    obs_w = rr.obs
                    trans = []
                    for _ws in range(args.max_steps):
                        ar = expert.act(obs_w)
                        sr = verse.step(ar.action)
                        trans.append(Transition(obs=obs_w, action=int(ar.action), reward=float(sr.reward),
                                               next_obs=sr.obs, done=bool(sr.done), truncated=bool(sr.truncated), info={}))
                        obs_w = sr.obs
                        if sr.done or sr.truncated:
                            break
                    expert.learn(ExperienceBatch(transitions=trans, meta={"source": "warmup"}))
            else:
                policy = args.expert_policy
                planner_expansions = 500
                planner_horizon = 4
            
            rows, stats = _collect_dagger_labels(
                verse=verse, learner=learner, expert=expert, verse_name=verse_name,
                round_idx=r, episodes=args.collect_episodes, max_steps=args.max_steps,
                mismatch_only=False, failures_only=False, expert_policy=policy,
                planner_horizon=planner_horizon, planner_max_expansions=planner_expansions,
                planner_avoid_terminal_failures=True
            )
            
            _append_jsonl(args.dagger_dataset_path, rows)
            
            # Ingest round results into Central Memory for later recall
            try:
                from memory.central_repository import CentralMemoryConfig, ingest_run
                import tempfile
                import shutil
                with tempfile.TemporaryDirectory() as td:
                    # Ingest expects run_id-based directory structure with events.jsonl
                    run_dir = os.path.join(td, f"dagger_round_{r}_{verse_name}")
                    os.makedirs(run_dir, exist_ok=True)
                    events_path = os.path.join(run_dir, "events.jsonl")
                    # We only want this round's rows, but 'rows' is still in memory from _collect_dagger_labels
                    with open(events_path, "w", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json.dumps(row) + "\n")
                    
                    mem_cfg = CentralMemoryConfig(root_dir="central_memory")
                    ingest_run(run_dir=run_dir, cfg=mem_cfg)
            except Exception as e:
                print(f"      Warning: Memory ingestion failed: {e}")

            verse.close()
            learner.close()
            expert.close()
            
            elapsed = time.time() - t0
            print(f"{len(rows)} rows, success={stats['success_rate']:.2f} ({elapsed:.1f}s)")
            _flush()

        print(f"  Preparing Cross-Verse dataset...")
        _flush()
        meta = prepare_adt_data(
            runs_root="__disabled__",
            out_path=args.prepared_dataset_path,
            context_len=30, # MATCH LEARNER CONTEXT
            chunk_stride=0,
            state_dim=64,
            max_timestep=4096,
            gamma=1.0,
            top_return_pct=1.0,
            success_only=False,
            min_episode_steps=2,
            max_runs=0,
            verse_filter=[],
            dataset_paths=[args.dagger_dataset_path],
            dataset_dir="",
            action_balance_mode="none",
            action_balance_max_ratio=3.0,
            action_balance_seed=123,
            min_action_dim=args.global_action_dim,
            verse_to_id=learner.model.get_config().get("verse_to_id"),
        )
        
        print(f"  Training Transformer... (min_action_dim={args.global_action_dim})")
        _flush()
        round_ckpt = args.out_model_path + f".r{r}" if r < args.rounds else args.out_model_path
        
        ckpt = train_adt(
            dataset_path=args.prepared_dataset_path,
            out_path=round_ckpt,
            init_model_path=current_model,
            epochs=5, # INCREASE EPOCHS FOR LARGER MODEL
            batch_size=128, # INCREASE BATCH SIZE
            lr=3e-4,
            weight_decay=1e-2,
            grad_clip=1.0,
            val_split=0.1,
            seed=123+r,
            d_model=256, # MATCH LARGE BASE
            n_head=8,
            n_layer=8,
            dropout=0.1,
            max_timestep=4096,
            device="auto", # USE GPU IF AVAILABLE
            class_weight_mode="auto",
            class_weight_min_count=1,
            class_weight_max=5.0
        )
        current_model = round_ckpt
        round_elapsed = time.time() - round_start
        print(f"  Round {r} Complete. Loss: {ckpt.get('best_val_loss', 0.0):.4f} ({round_elapsed:.1f}s)")
        _flush()

    print(f"\nCross-Verse Training Complete! Final Model: {current_model}")
    _flush()

if __name__ == "__main__":
    main()
