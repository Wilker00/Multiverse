
import os
import sys

# Project root hack
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import shutil
from orchestrator.trainer import Trainer
from core.types import AgentSpec, VerseSpec, RunRef, VerseRef, AgentRef
from core.rollout import RolloutConfig
from memory.central_repository import CentralMemoryConfig, ingest_run, find_similar

def main():
    # Clean setup for demo
    for d in ["central_memory_demo", "runs_demo"]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
    os.makedirs("runs_demo", exist_ok=True)

    trainer = Trainer(run_root="runs_demo")
    
    # 1. TEACHER: Grid World (Simple Successes)
    print("--- PREPARING TEACHER (grid_world) ---")
    teacher_verse_spec = VerseSpec(
        spec_version="v1",
        verse_name="grid_world",
        verse_version="0.1",
        params={"width": 6, "height": 6, "max_steps": 20}
    )
    teacher_agent_spec = AgentSpec(
        spec_version="v1",
        policy_id="teacher_q",
        policy_version="0.1",
        algo="q",
        config={"epsilon_start": 0.5, "epsilon_min": 0.05, "train": True}
    )
    
    # Run a few episodes to get some success data
    teacher_out = trainer.run(
        verse_spec=teacher_verse_spec,
        agent_spec=teacher_agent_spec,
        episodes=30,
        max_steps=20,
        run_id="teacher_grid"
    )
    teacher_run_id = teacher_out["run_id"]
    print(f"Teacher run completed: {teacher_run_id}")
    
    # 2. INGESTION: Move data from run log to Central Memory
    print("\n--- INGESTING INTO CENTRAL MEMORY ---")
    mem_cfg = CentralMemoryConfig(root_dir="central_memory_demo")
    stats = ingest_run(run_dir=f"runs_demo/{teacher_run_id}", cfg=mem_cfg)
    print(f"Ingested {stats.added_events} events into Central Memory.")

    # 3. STUDENT: Warehouse World with Recall
    print("\n--- RUNNING STUDENT (warehouse_world + RECALL) ---")
    student_verse_spec = VerseSpec(
        spec_version="v1",
        verse_name="warehouse_world",
        verse_version="0.1",
        params={"width": 8, "height": 8, "max_steps": 40}
    )
    
    # Configure Recall Agent
    student_agent_spec = AgentSpec(
        spec_version="v1",
        policy_id="student_recall",
        policy_version="0.1",
        algo="memory_recall",
        config={
            "recall_enabled": True,
            "recall_cooldown_steps": 1,
            "recall_vote_weight": 2.5, # Stronger weight for demo
            "recall_risk_threshold": 0.5, # Trigger easily
            "recall_uncertainty_margin": 1.0, # Trigger easily
            "recall_top_k": 3,
            "recall_same_verse_only": False,
            "on_demand_memory_enabled": True,
            "on_demand_memory_root": "central_memory_demo"
        }
    )
    
    # We'll use the trainer to run the student
    # Trainer automatically sets up RolloutConfig with central_memory if requested
    print("Starting Student training/eval with recall active...")
    student_out = trainer.run(
        verse_spec=student_verse_spec,
        agent_spec=student_agent_spec,
        episodes=3, # Just 3 for demo
        max_steps=40,
        run_id="student_recall_run",
        verbose=True
    )
    
    # Verify Recall Usage from Local logs
    events_path = f"runs_demo/student_recall_run/events.jsonl"
    uses = 0
    print("\n--- LIVE RECALL ACTIVITY ---")
    with open(events_path, "r") as f:
        for line in f:
            ev = json.loads(line)
            hint_diag = ev.get("info", {}).get("action_info", {}).get("memory_recall_diag", {})
            if hint_diag.get("request_emitted"):
                match_count = ev.get("info", {}).get("action_info", {}).get("memory_recall_match_count", 0)
                used = ev.get("info", {}).get("action_info", {}).get("memory_recall_used", False)
                if used:
                    uses += 1
                    pointer = ev.get("info", {}).get("action_info", {}).get("memory_recall_pointer", "N/A")
                    print(f"Step {ev['step_idx']}: SUCCESSFUL RECALL! Found {match_count} matches.")
                    print(f"  Source: {pointer}")
    
    print(f"\nTotal cross-verse memory matches used: {uses}")

if __name__ == "__main__":
    main()
