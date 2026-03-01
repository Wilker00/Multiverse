import os
import re

with open("memory/central_repository.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update ScenarioMatch dataclass
from_match = """@dataclass
class ScenarioMatch:
    score: float
    run_id: str
    episode_id: str
    step_idx: int
    t_ms: int
    verse_name: str
    action: JSONValue
    reward: float
    obs: JSONValue
    source_greedy_action: Optional[int] = None
    source_action_matches_greedy: Optional[bool] = None
    recency_weight: float = 1.0"""
to_match = """@dataclass
class ScenarioMatch:
    score: float
    run_id: str
    episode_id: str
    step_idx: int
    t_ms: int
    verse_name: str
    action: JSONValue
    reward: float
    obs: JSONValue
    source_greedy_action: Optional[int] = None
    source_action_matches_greedy: Optional[bool] = None
    recency_weight: float = 1.0
    trajectory: Optional[List[Dict[str, Any]]] = None"""

if "trajectory: Optional" not in content:
    content = content.replace(from_match, to_match)
    
# 2. Update find_similar signature
from_def = """def find_similar(
    *,
    obs: JSONValue,
    cfg: CentralMemoryConfig,
    top_k: int = 5,
    verse_name: Optional[str] = None,
    min_score: float = -1.0,
    exclude_run_ids: Optional[Set[str]] = None,
    decay_lambda: float = 0.0,
    current_time_ms: Optional[int] = None,
    memory_tiers: Optional[Set[str]] = None,
    memory_families: Optional[Set[str]] = None,
    memory_types: Optional[Set[str]] = None,
    stm_decay_lambda: Optional[float] = None,
) -> List[ScenarioMatch]:"""
to_def = """def find_similar(
    *,
    obs: JSONValue,
    cfg: CentralMemoryConfig,
    top_k: int = 5,
    verse_name: Optional[str] = None,
    min_score: float = -1.0,
    exclude_run_ids: Optional[Set[str]] = None,
    decay_lambda: float = 0.0,
    current_time_ms: Optional[int] = None,
    memory_tiers: Optional[Set[str]] = None,
    memory_families: Optional[Set[str]] = None,
    memory_types: Optional[Set[str]] = None,
    stm_decay_lambda: Optional[float] = None,
    trajectory_window: int = 0,
) -> List[ScenarioMatch]:"""

if "trajectory_window: int = 0" not in content:
    content = content.replace(from_def, to_def)

# 3. Add helper to extract trajectory into find_similar
# find_similar has a loop `for mem_path in mem_paths:`
# At the start of the function body, we can add `_extract_trajectory_cb`.

from_body_start = """    tier_policy = _load_tier_policy(cfg)
    target_verse = str(verse_name).strip().lower() if verse_name else \"\"

    heap: List[tuple[float, int, ScenarioMatch]] = []
    ordinal = 0
    for mem_path in mem_paths:"""
to_body_start = """    tier_policy = _load_tier_policy(cfg)
    target_verse = str(verse_name).strip().lower() if verse_name else \"\"

    heap: List[tuple[float, int, ScenarioMatch]] = []
    ordinal = 0
    for mem_path in mem_paths:
        if not os.path.isfile(mem_path):
            continue
        cache = _get_similarity_cache_for_path(mem_path=mem_path, tier_policy=tier_policy)
        
        def _extract_trajectory(row_input: Any) -> Optional[List[Dict[str, Any]]]:
            if int(trajectory_window) <= 0:
                return None
            ep_id = str(row_input.episode_id)
            root_step = int(row_input.step_idx)
            # Find the row in cache.rows if we just have the row reference.
            # O(N) scan but only on matched rows. Usually matches are small top_n.
            idx = -1
            for i, r in enumerate(cache.rows):
                if r is row_input:
                    idx = i
                    break
            if idx < 0:
                return []
            traj = []
            curr = idx
            while curr >= 0 and len(traj) < int(trajectory_window):
                r = cache.rows[curr]
                if str(r.episode_id) != ep_id or int(r.step_idx) > root_step:
                    if str(r.episode_id) != ep_id:
                        break
                    curr -= 1
                    continue
                traj.append({
                    "step_idx": int(r.step_idx),
                    "obs": r.obs,
                    "action": r.action,
                    "reward": float(r.reward)
                })
                curr -= 1
            traj.reverse()
            return traj
"""

if "_extract_trajectory" not in content:
    content = content.replace(
"""    tier_policy = _load_tier_policy(cfg)
    target_verse = str(verse_name).strip().lower() if verse_name else ""

    heap: List[tuple[float, int, ScenarioMatch]] = []
    ordinal = 0
    for mem_path in mem_paths:
        if not os.path.isfile(mem_path):
            continue
        cache = _get_similarity_cache_for_path(mem_path=mem_path, tier_policy=tier_policy)""",
to_body_start
    )

import re

# Match 1: inside _scan_position -> _score_position
match_old_1 = """                match = ScenarioMatch(
                    score=float(score),
                    run_id=str(row.run_id),
                    episode_id=str(row.episode_id),
                    step_idx=int(row.step_idx),
                    t_ms=t_ms,
                    verse_name=str(row.verse_name),
                    action=row.action,
                    reward=float(row.reward),
                    obs=row.obs,
                    source_greedy_action=row.source_greedy_action,
                    source_action_matches_greedy=row.source_action_matches_greedy,
                    recency_weight=float(recency_weight),
                )"""

match_new_1 = """                match = ScenarioMatch(
                    score=float(score),
                    run_id=str(row.run_id),
                    episode_id=str(row.episode_id),
                    step_idx=int(row.step_idx),
                    t_ms=t_ms,
                    verse_name=str(row.verse_name),
                    action=row.action,
                    reward=float(row.reward),
                    obs=row.obs,
                    source_greedy_action=row.source_greedy_action,
                    source_action_matches_greedy=row.source_action_matches_greedy,
                    recency_weight=float(recency_weight),
                    trajectory=_extract_trajectory(row),
                )"""

content = content.replace(match_old_1, match_new_1)

with open("memory/central_repository.py", "w", encoding="utf-8") as f:
    f.write(content)

print("central_repository.py trajectory patch ok!")
