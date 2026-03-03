import pytest
import torch

from agents.transformer_agent import TransformerAgent
from core.types import AgentSpec, SpaceSpec


class MockModel:
    def __init__(self, action_dim=7):
        class Config:
            def __init__(self):
                self.action_dim = action_dim
                self.state_dim = 128
                self.context_len = 30
                self.max_timestep = 100

        self.config = Config()
        self.device = torch.device("cpu")

    def eval(self):
        return None

    def to(self, _device):
        return self

    def get_config(self):
        return {
            "state_dim": 128,
            "context_len": 30,
            "action_dim": 7,
            "verse_to_id": {"grid_world": 0},
            "verse_action_ranges": {"grid_world": 7},
        }

    def predict_next_action(self, **kwargs):
        action = torch.tensor([0])
        conf = torch.tensor([0.9])
        probs = torch.zeros((1, 7))
        probs[0, 0] = 1.0
        return action, conf, probs


def _make_agent(tmp_path, monkeypatch):
    dummy_pt = tmp_path / "dummy.pt"
    torch.save({"model_state": {}}, dummy_pt)

    spec = AgentSpec(
        spec_version="v1",
        algo="adt",
        policy_id="test",
        policy_version="1",
        config={
            "model_path": str(dummy_pt),
            "recall_enabled": True,
            "recall_vote_weight": 2.0,
            "verse_name": "grid_world",
            "context_len": 30,
        },
    )

    monkeypatch.setattr(
        "agents.transformer_agent.load_decision_transformer_checkpoint",
        lambda path, **kw: (MockModel(), {}),
    )
    return TransformerAgent(
        spec,
        SpaceSpec(type="discrete", n=128),
        SpaceSpec(type="discrete", n=7),
    )


def test_ghost_telemetry_and_prior(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    mock_obs = {"t": 0, "agent_pos": [0, 0]}
    mock_hint = {
        "memory_recall": {
            "matches": [
                {
                    "score": 1.0,
                    "action": 1,
                    "step_idx": 37,
                    "trajectory": [{"step_idx": 37, "action": 1, "obs": mock_obs}],
                }
            ]
        }
    }

    res = agent.act_with_hint(mock_obs, mock_hint)

    assert res.action == 1
    assert res.info["ghost_injected"] is True
    assert res.info["ghost_steps_requested"] == 1
    assert res.info["ghost_steps_injected"] == 1
    assert res.info["ghost_action_match"] is True
    assert res.info["memory_recall_eligible"] is True
    assert res.info["memory_recall_used"] is True
    assert res.info["memory_recall_gate_passed"] is True
    assert res.info["memory_recall_greedy_changed"] is True
    assert res.info["memory_recall_match_count"] == 1


def test_ghost_ablation_blocks_hint_application(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    mock_obs = {"t": 0, "agent_pos": [0, 0]}
    mock_hint = {
        "memory_recall": {
            "matches": [
                {
                    "score": 1.0,
                    "action": 1,
                    "step_idx": 12,
                    "trajectory": [{"step_idx": 12, "action": 1, "obs": mock_obs}],
                }
            ]
        },
        "_memory_recall_control": {
            "disable_apply": True,
            "policy": "test_ablation",
        },
    }

    res = agent.act_with_hint(mock_obs, mock_hint)

    assert res.action == 0
    assert res.info["memory_recall_eligible"] is True
    assert res.info["memory_recall_used"] is False
    assert res.info["memory_recall_disabled_for_ablation"] is True
    assert res.info["memory_recall_gate_passed"] is False
    assert res.info["ghost_injected"] is False
    assert res.info["ghost_steps_requested"] == 0
    assert res.info["ghost_steps_injected"] == 0
    assert "ghost_action_match" not in res.info


if __name__ == "__main__":
    pytest.main([__file__])
