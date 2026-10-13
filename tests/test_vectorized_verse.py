from __future__ import annotations

from dataclasses import dataclass

from core.agent_base import ActionResult
from core.types import SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult
from core.vectorized_verse import SerialVectorizedVerse
from orchestrator.vectorized_trainer import VectorizedTrainer


class CounterVerse:
    def __init__(self, spec: VerseSpec) -> None:
        self.spec = spec
        self.observation_space = SpaceSpec(type="vector", shape=(1,), dtype="float32")
        self.action_space = SpaceSpec(type="discrete", n=2)
        self.count = 0

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.count = 0
        return ResetResult(obs=[0.0], info={"seed": getattr(self, "seed_value", None)})

    def step(self, action):
        self.count += 1
        return StepResult(obs=[float(self.count)], reward=1.0, done=self.count >= 2,
                          truncated=False, info={"action": int(action)})

    def render(self, mode="rgb_array"):
        return None

    def close(self):
        pass


class CountingAgent:
    def __init__(self):
        self.learned = []

    def seed(self, seed):
        self.seed_value = seed

    def act(self, obs):
        return ActionResult(action=1)

    def learn(self, batch):
        self.learned.append(batch)
        return {"transitions": len(batch.transitions)}


def test_serial_vectorized_adapter_preserves_lane_results_and_resets_independently():
    spec = VerseSpec(spec_version="v1", verse_name="counter", verse_version="1")
    vector = SerialVectorizedVerse([CounterVerse(spec), CounterVerse(spec)])
    reset = vector.reset_batch([10, 20])
    assert reset.observations == [[0.0], [0.0]]
    assert reset.infos[0]["seed"] == 10
    assert reset.infos[1]["seed"] == 20

    first = vector.step_batch([1, 0])
    assert first.observations == [[1.0], [1.0]]
    assert first.rewards == [1.0, 1.0]
    assert first.done == [False, False]

    lane_reset = vector.reset_lanes({1: 21})
    assert lane_reset.observations[1] == [0.0]
    second = vector.step_batch([1, 1])
    assert second.observations == [[2.0], [1.0]]
    assert second.terminated == [True, False]


def test_vectorized_trainer_counts_global_episodes_and_trains_per_episode():
    spec = VerseSpec(spec_version="v1", verse_name="counter", verse_version="1")
    vector = SerialVectorizedVerse([CounterVerse(spec), CounterVerse(spec)])
    agent = CountingAgent()
    result = VectorizedTrainer().run(
        verse=vector, agent=agent, episodes=3, max_steps=5, seed=7, train=True
    )

    assert len(result["episodes"]) == 3
    assert result["total_steps"] == 6
    assert [row["return_sum"] for row in result["episodes"]] == [2.0, 2.0, 2.0]
    assert len(agent.learned) == 3
    assert all(len(batch.transitions) == 2 for batch in agent.learned)
