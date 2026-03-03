import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.transformer_agent import TransformerAgent
from core.types import AgentSpec, VerseSpec
from verses.registry import create_verse, register_builtin


def trace_run():
    register_builtin()

    vspec = VerseSpec(spec_version="v1", verse_name="warehouse_world", verse_version="1", seed=42)
    verse = create_verse(vspec)
    reset_result = verse.reset()
    obs = reset_result.obs if hasattr(reset_result, "obs") else reset_result

    model_path = os.path.join(_PROJECT_ROOT, "models", "dt_generalist_v3_omega.pt")
    learner_spec = AgentSpec(
        spec_version="v1",
        algo="adt",
        policy_id="trace_agent",
        policy_version="0.1",
        config={
            "model_path": model_path,
            "recall_enabled": True,
            "recall_frequency": 2,
            "recall_top_k": 1,
            "recall_vote_weight": 1.0,
            "verse_name": "warehouse_world",
        },
    )

    try:
        agent = TransformerAgent(
            spec=learner_spec,
            observation_space=verse.observation_space,
            action_space=verse.action_space,
        )
    except Exception as exc:
        print(f"Agent init failed: {exc}")
        return

    print("--- GHOST TRACE START ---")
    print(f"Model: {model_path}")
    print("Recall source: synthetic memory response to exercise the runtime recall path")
    print(f"Step 0: Initial Obs: {obs}")

    for step_idx in range(5):
        req = agent.memory_query_request(obs=obs, step_idx=step_idx)
        if req:
            print(f"\n[RECALL TRIGGERED at Step {step_idx}]")
            mock_recall = {
                "matches": [
                    {
                        "score": 1.0,
                        "action": 3,
                        "step_idx": 100 + step_idx,
                        "trajectory": [{"step_idx": 100 + step_idx, "action": 3, "obs": obs}],
                    }
                ]
            }
            agent.on_memory_response(mock_recall)
            print("  Ghost roadmap queued: action=3 (Right)")

        result = agent.act(obs)
        print(f"  Decision at Step {step_idx}: Action {result.action}")
        print(f"  Telemetry: {result.info}")

        step_result = verse.step(result.action)
        obs = step_result.obs if hasattr(step_result, "obs") else step_result
        if getattr(step_result, "done", False):
            break

    print("\n--- GHOST TRACE END ---")


if __name__ == "__main__":
    trace_run()
