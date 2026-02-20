"""
agents/library_agent.py

Library agent that aggregates multiple imitation datasets.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional

from agents.imitation_agent import ImitationLookupAgent, ImitationStats
from core.types import AgentSpec, SpaceSpec


class LibraryAgent(ImitationLookupAgent):
    """
    Imitation agent that can load multiple datasets.

    It reuses ImitationLookupAgent behavior and simply aggregates
    all provided datasets into a single lookup table.
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        super().__init__(spec=spec, observation_space=observation_space, action_space=action_space)

    def learn_from_library(
        self,
        dataset_paths: Optional[Iterable[str]] = None,
        dataset_dir: Optional[str] = None,
        limit_rows_per_file: Optional[int] = None,
    ) -> List[ImitationStats]:
        paths: List[str] = []
        if dataset_paths:
            paths.extend([str(p) for p in dataset_paths])
        if dataset_dir:
            paths.extend(sorted(glob.glob(os.path.join(str(dataset_dir), "*.jsonl"))))

        stats: List[ImitationStats] = []
        for path in paths:
            stats.append(self.learn_from_dataset(path, limit_rows=limit_rows_per_file))
        return stats
