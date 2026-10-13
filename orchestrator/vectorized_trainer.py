"""Optional multi-lane rollout loop for VectorizedVerse implementations."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional

from core.agent_base import ActionResult, ExperienceBatch, Transition
from core.types import JSONValue
from core.vectorized_verse import VectorizedVerse


@dataclass
class VectorEpisodeResult:
    lane: int
    steps: int
    return_sum: float
    terminated: bool
    truncated: bool
    train_metrics: Dict[str, float]


class VectorizedTrainer:
    """Collect episodes in parallel simulation lanes and keep agent calls per lane.

    The method deliberately leaves the existing scalar ``Trainer.run`` path
    untouched. Learning receives the same ``ExperienceBatch`` and ``Transition``
    types as scalar rollouts.
    """

    def run(
        self,
        *,
        verse: VectorizedVerse,
        agent: Any,
        episodes: int,
        max_steps: int,
        seed: Optional[int] = None,
        train: bool = False,
    ) -> Dict[str, Any]:
        episode_target = max(0, int(episodes))
        step_limit = max(1, int(max_steps))
        lane_count = int(verse.num_envs)
        if lane_count < 1:
            raise ValueError("VectorizedVerse.num_envs must be positive")
        if episode_target == 0:
            return {"episodes": [], "total_steps": 0, "mean_return": 0.0, "elapsed_seconds": 0.0}

        if seed is not None and hasattr(agent, "seed"):
            agent.seed(int(seed))

        seeds = [None if seed is None else int(seed) + lane for lane in range(lane_count)]
        reset = verse.reset_batch(seeds)
        observations = list(reset.observations)
        lane_active = [lane < episode_target for lane in range(lane_count)]
        started = min(lane_count, episode_target)
        lane_steps = [0] * lane_count
        lane_returns = [0.0] * lane_count
        lane_transitions: List[List[Transition]] = [[] for _ in range(lane_count)]
        lane_episode_no = [0] * lane_count
        completed: List[VectorEpisodeResult] = []
        steps_total = 0
        start = time.perf_counter()

        while len(completed) < episode_target:
            action_results: List[Optional[ActionResult]] = [None] * lane_count
            actions: List[JSONValue] = [0] * lane_count
            for lane in range(lane_count):
                if not lane_active[lane]:
                    continue
                result = agent.act(observations[lane])
                if not isinstance(result, ActionResult):
                    result = ActionResult(action=result)
                action_results[lane] = result
                actions[lane] = result.action

            batch = verse.step_batch(actions)
            steps_total += sum(1 for active in lane_active if active)
            reset_lanes: Dict[int, Optional[int]] = {}
            for lane in range(lane_count):
                if not lane_active[lane]:
                    continue
                lane_steps[lane] += 1
                lane_returns[lane] += float(batch.rewards[lane])
                done = bool(batch.terminated[lane] or batch.truncated[lane])
                timed_out = lane_steps[lane] >= step_limit and not done
                ended = done or timed_out
                info = dict(batch.infos[lane] or {})
                truncated = bool(batch.truncated[lane] or timed_out)
                if timed_out:
                    info["trainer_time_limit"] = True
                action_result = action_results[lane]
                transition = Transition(
                    obs=observations[lane],
                    action=(action_result.action if action_result is not None else 0),
                    reward=float(batch.rewards[lane]),
                    next_obs=batch.observations[lane],
                    done=bool(batch.terminated[lane]),
                    truncated=truncated,
                    info=info,
                )
                lane_transitions[lane].append(transition)
                observations[lane] = batch.observations[lane]
                if not ended:
                    continue

                metrics: Dict[str, float] = {}
                if train and lane_transitions[lane]:
                    try:
                        raw_metrics = agent.learn(ExperienceBatch(
                            transitions=list(lane_transitions[lane]),
                            meta={
                                "lane": int(lane),
                                "episode": int(lane_episode_no[lane]),
                                "steps": int(lane_steps[lane]),
                                "return_sum": float(lane_returns[lane]),
                            },
                        ))
                        for key, value in dict(raw_metrics or {}).items():
                            try:
                                metrics[str(key)] = float(value)
                            except (TypeError, ValueError):
                                continue
                    except NotImplementedError:
                        pass
                completed.append(VectorEpisodeResult(
                    lane=lane,
                    steps=lane_steps[lane],
                    return_sum=float(lane_returns[lane]),
                    terminated=bool(batch.terminated[lane]),
                    truncated=truncated,
                    train_metrics=metrics,
                ))
                lane_episode_no[lane] += 1
                lane_steps[lane] = 0
                lane_returns[lane] = 0.0
                lane_transitions[lane] = []
                if started < episode_target:
                    next_seed = None if seed is None else int(seed) + started + lane
                    reset_lanes[lane] = next_seed
                    started += 1
                else:
                    lane_active[lane] = False

            if reset_lanes:
                lane_reset = verse.reset_lanes(reset_lanes)
                for lane in reset_lanes:
                    observations[lane] = lane_reset.observations[lane]
                    lane_active[lane] = True

        elapsed = max(1e-12, time.perf_counter() - start)
        returns = [item.return_sum for item in completed]
        return {
            "episodes": [item.__dict__ for item in completed],
            "total_steps": int(steps_total),
            "mean_return": float(sum(returns) / len(returns)) if returns else 0.0,
            "steps_per_second": float(steps_total / elapsed),
            "elapsed_seconds": float(elapsed),
            "num_envs": lane_count,
        }


__all__ = ["VectorEpisodeResult", "VectorizedTrainer"]
