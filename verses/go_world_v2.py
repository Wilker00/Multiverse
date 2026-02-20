"""
verses/go_world_v2.py

Mini-Go on a 5x5 board with real rules:
- Stone placement on empty intersections
- Connected group tracking via flood-fill
- Liberty counting per group
- Capture: groups with 0 liberties are removed
- Ko rule: cannot recreate the board state from 1 move ago
- Pass: two consecutive passes end the game
- Chinese area scoring: stones on board + surrounded empty territory

Action space: 26 discrete (0-24 = place stone at intersection, 25 = pass)
Observation: 25-cell board + strategic features

Designed for genuine transfer learning research — this is a real game,
not an abstract counter simulation.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse

# ---------------------------------------------------------------------------
# Board constants
# ---------------------------------------------------------------------------
BOARD_SIZE = 5
N_CELLS = BOARD_SIZE * BOARD_SIZE  # 25
EMPTY, BLACK, WHITE = 0, 1, 2
ACTION_PASS = N_CELLS  # action 25
N_ACTIONS = N_CELLS + 1  # 26

# Precompute neighbors for each cell
_NEIGHBORS: List[List[int]] = []
for _p in range(N_CELLS):
    _r, _c = divmod(_p, BOARD_SIZE)
    _nb: List[int] = []
    if _r > 0:
        _nb.append(_p - BOARD_SIZE)
    if _r < BOARD_SIZE - 1:
        _nb.append(_p + BOARD_SIZE)
    if _c > 0:
        _nb.append(_p - 1)
    if _c < BOARD_SIZE - 1:
        _nb.append(_p + 1)
    _NEIGHBORS.append(_nb)


# ---------------------------------------------------------------------------
# Go rule helpers (pure functions on board lists)
# ---------------------------------------------------------------------------

def _find_group(board: List[int], pos: int) -> Set[int]:
    """BFS to find the connected group of same-color stones containing *pos*."""
    color = board[pos]
    if color == EMPTY:
        return set()
    group: Set[int] = set()
    stack = [pos]
    while stack:
        p = stack.pop()
        if p in group:
            continue
        if board[p] != color:
            continue
        group.add(p)
        for n in _NEIGHBORS[p]:
            if n not in group:
                stack.append(n)
    return group


def _group_liberties(board: List[int], group: Set[int]) -> Set[int]:
    """Return the set of unique empty cells adjacent to *group*."""
    libs: Set[int] = set()
    for p in group:
        for n in _NEIGHBORS[p]:
            if board[n] == EMPTY:
                libs.add(n)
    return libs


def _try_place(board: List[int], pos: int, color: int) -> Optional[Tuple[List[int], Set[int]]]:
    """
    Attempt to place *color* at *pos*.
    Returns (new_board, captured_set) or None if the move is illegal.
    Illegal = occupied, or suicide (own group dies and no captures made).
    Does NOT check ko — caller must do that.
    """
    if board[pos] != EMPTY:
        return None

    new_board = list(board)
    new_board[pos] = color
    opp = WHITE if color == BLACK else BLACK

    # Check opponent groups adjacent to the placed stone for captures
    captured: Set[int] = set()
    for n in _NEIGHBORS[pos]:
        if new_board[n] == opp:
            grp = _find_group(new_board, n)
            libs = _group_liberties(new_board, grp)
            if len(libs) == 0:
                captured |= grp

    # Remove captured stones
    for p in captured:
        new_board[p] = EMPTY

    # Check for suicide: the placed stone's group has 0 liberties
    my_grp = _find_group(new_board, pos)
    my_libs = _group_liberties(new_board, my_grp)
    if len(my_libs) == 0 and len(captured) == 0:
        return None  # suicide — illegal

    return (new_board, captured)


def _board_hash(board: List[int]) -> str:
    return hashlib.md5(bytes(board)).hexdigest()


def _score_board(board: List[int]) -> Tuple[int, int]:
    """
    Chinese (area) scoring.
    Score = own stones on the board + empty territory surrounded entirely by own stones.
    """
    black_score = sum(1 for p in range(N_CELLS) if board[p] == BLACK)
    white_score = sum(1 for p in range(N_CELLS) if board[p] == WHITE)

    # Flood-fill empty regions to determine territory ownership
    visited = [False] * N_CELLS
    for start in range(N_CELLS):
        if board[start] != EMPTY or visited[start]:
            continue
        region: Set[int] = set()
        border_colors: Set[int] = set()
        stack = [start]
        while stack:
            p = stack.pop()
            if p in region:
                continue
            if board[p] != EMPTY:
                border_colors.add(board[p])
                continue
            region.add(p)
            visited[p] = True
            for n in _NEIGHBORS[p]:
                if n not in region:
                    stack.append(n)
        if border_colors == {BLACK}:
            black_score += len(region)
        elif border_colors == {WHITE}:
            white_score += len(region)
        # else: neutral (bordered by both or neither)

    return (black_score, white_score)


def _count_groups(board: List[int], color: int) -> Tuple[int, int, int]:
    """Return (num_groups, total_liberties, atari_groups) for *color*."""
    visited: Set[int] = set()
    n_groups = 0
    total_libs = 0
    atari = 0
    for p in range(N_CELLS):
        if board[p] != color or p in visited:
            continue
        grp = _find_group(board, p)
        visited |= grp
        libs = _group_liberties(board, grp)
        n_groups += 1
        total_libs += len(libs)
        if len(libs) == 1:
            atari += 1
    return (n_groups, total_libs, atari)


def _legal_moves(board: List[int], color: int, ko_point: int) -> List[int]:
    """Return list of legal action indices for *color*."""
    moves: List[int] = []
    for pos in range(N_CELLS):
        if pos == ko_point:
            continue
        result = _try_place(board, pos, color)
        if result is not None:
            moves.append(pos)
    moves.append(ACTION_PASS)  # pass is always legal
    return moves


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class GoV2Params:
    max_steps: int = 120
    komi: float = 2.5  # white advantage (standard for 5x5 is ~2.5)
    step_penalty: float = -0.005
    capture_reward_scale: float = 0.15
    win_reward: float = 2.0
    lose_penalty: float = -2.0


# ---------------------------------------------------------------------------
# Verse implementation
# ---------------------------------------------------------------------------

class GoWorldV2Verse(Verse):
    """
    Mini-Go (5x5) with real Go rules.

    26 actions: 0-24 = place stone on intersection, 25 = pass.
    Agent plays BLACK, opponent plays WHITE.
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in GoWorldV2Factory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = GoV2Params(
            max_steps=int(self.spec.params.get("max_steps", 120)),
            komi=float(self.spec.params.get("komi", 2.5)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.005)),
            capture_reward_scale=float(self.spec.params.get("capture_reward_scale", 0.15)),
            win_reward=float(self.spec.params.get("win_reward", 2.0)),
            lose_penalty=float(self.spec.params.get("lose_penalty", -2.0)),
        )
        self.params.max_steps = max(20, int(self.params.max_steps))

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "board",
                "my_captures", "opp_captures",
                "my_territory", "opp_territory",
                "my_groups", "my_liberties_total", "my_atari_groups",
                "opp_groups", "opp_liberties_total", "opp_atari_groups",
                "ko_point", "consecutive_passes",
                "my_stones", "opp_stones",
                "score_delta", "pressure", "risk", "tempo",
                "control", "resource",
                "t",
            ],
            notes="5x5 Go board state with derived strategic features.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=N_ACTIONS,
            notes="0-24 = place stone at intersection (row*5+col), 25 = pass",
        )

        self._rng = random.Random()
        self._seed_val: Optional[int] = None
        self._done = False
        self._t = 0
        self._board: List[int] = [EMPTY] * N_CELLS
        self._my_captures = 0
        self._opp_captures = 0
        self._ko_point = -1
        self._consecutive_passes = 0
        self._prev_board_hash: str = ""

    def seed(self, seed: Optional[int]) -> None:
        self._seed_val = seed
        if seed is not None:
            self._rng = random.Random(int(seed))

    def reset(self) -> ResetResult:
        if self._seed_val is not None:
            self._rng = random.Random(int(self._seed_val))

        self._board = [EMPTY] * N_CELLS
        self._my_captures = 0
        self._opp_captures = 0
        self._ko_point = -1
        self._consecutive_passes = 0
        self._prev_board_hash = _board_hash(self._board)
        self._done = False
        self._t = 0

        obs = self._make_obs()
        return ResetResult(
            obs=obs,
            info={"verse_name": "go_world_v2", "board_size": BOARD_SIZE},
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(
                self._make_obs(), 0.0, True, False,
                {"warning": "step() called after done"},
            )

        a = int(action)
        if a < 0 or a > ACTION_PASS:
            a = ACTION_PASS  # invalid → pass
        self._t += 1
        reward = float(self.params.step_penalty)

        # ---- Agent move (BLACK) ----
        agent_captured = 0
        if a == ACTION_PASS:
            self._consecutive_passes += 1
        else:
            # Check legality
            if a == self._ko_point:
                # Ko violation → treat as pass
                self._consecutive_passes += 1
            else:
                result = _try_place(self._board, a, BLACK)
                if result is None:
                    # Illegal move → treat as pass with small penalty
                    self._consecutive_passes += 1
                    reward -= 0.05
                else:
                    new_board, captured = result
                    # Ko detection: if exactly 1 stone captured, check if new state
                    # would recreate previous board
                    next_hash = _board_hash(new_board)
                    if len(captured) == 1 and next_hash == self._prev_board_hash:
                        # Ko! Treat as pass
                        self._consecutive_passes += 1
                    else:
                        self._prev_board_hash = _board_hash(self._board)
                        self._board = new_board
                        agent_captured = len(captured)
                        self._my_captures += agent_captured
                        self._consecutive_passes = 0
                        reward += agent_captured * float(self.params.capture_reward_scale)

                        # Set ko point if single capture
                        if len(captured) == 1:
                            ko_cand = next(iter(captured))
                            # Ko point only if the captured position is now empty
                            # and surrounded by opponent
                            nbrs_opp = sum(1 for n in _NEIGHBORS[ko_cand] if self._board[n] == BLACK)
                            if nbrs_opp == len(_NEIGHBORS[ko_cand]):
                                self._ko_point = ko_cand
                            else:
                                self._ko_point = -1
                        else:
                            self._ko_point = -1

        # Check for game end after agent move
        if self._consecutive_passes >= 2:
            return self._end_game(reward)

        # ---- Opponent move (WHITE) ----
        opp_action = self._opponent_move()
        if opp_action == ACTION_PASS:
            self._consecutive_passes += 1
        else:
            result = _try_place(self._board, opp_action, WHITE)
            if result is not None:
                new_board, captured = result
                next_hash = _board_hash(new_board)
                # Simple ko check
                if not (len(captured) == 1 and next_hash == self._prev_board_hash):
                    self._prev_board_hash = _board_hash(self._board)
                    self._board = new_board
                    self._opp_captures += len(captured)
                    self._consecutive_passes = 0
                    reward -= len(captured) * float(self.params.capture_reward_scale) * 0.5
                    if len(captured) == 1:
                        ko_cand = next(iter(captured))
                        nbrs_opp = sum(1 for n in _NEIGHBORS[ko_cand] if self._board[n] == WHITE)
                        if nbrs_opp == len(_NEIGHBORS[ko_cand]):
                            self._ko_point = ko_cand
                        else:
                            self._ko_point = -1
                    else:
                        self._ko_point = -1
                else:
                    self._consecutive_passes += 1
            else:
                self._consecutive_passes += 1

        # Check for game end after opponent move
        if self._consecutive_passes >= 2:
            return self._end_game(reward)

        # Check for total annihilation
        my_stones = sum(1 for p in range(N_CELLS) if self._board[p] == BLACK)
        opp_stones = sum(1 for p in range(N_CELLS) if self._board[p] == WHITE)
        if my_stones == 0 and self._t > 2:
            reward += float(self.params.lose_penalty)
            self._done = True
            return StepResult(
                self._make_obs(), float(reward), True, False,
                {"reached_goal": False, "lost_game": True, "reason": "annihilated", "t": self._t},
            )
        if opp_stones == 0 and self._t > 2:
            reward += float(self.params.win_reward)
            self._done = True
            return StepResult(
                self._make_obs(), float(reward), True, False,
                {"reached_goal": True, "lost_game": False, "reason": "annihilated_opponent", "t": self._t},
            )

        # Truncation check
        truncated = self._t >= int(self.params.max_steps)
        if truncated:
            return self._end_game(reward, truncated=True)

        # Territory-based shaping reward
        bs, ws = _score_board(self._board)
        net_territory = bs - ws - self.params.komi
        reward += 0.002 * net_territory  # tiny shaping signal

        self._done = False
        return StepResult(
            self._make_obs(), float(reward), False, False,
            {"reached_goal": False, "t": self._t, "agent_captured": agent_captured},
        )

    def _end_game(self, reward_so_far: float, truncated: bool = False) -> StepResult:
        """Score the board and return terminal step."""
        bs, ws = _score_board(self._board)
        ws_with_komi = float(ws) + float(self.params.komi)
        won = float(bs) > ws_with_komi
        lost = float(bs) < ws_with_komi

        if won:
            reward_so_far += float(self.params.win_reward)
        elif lost:
            reward_so_far += float(self.params.lose_penalty)

        self._done = True
        return StepResult(
            obs=self._make_obs(),
            reward=float(reward_so_far),
            done=not truncated,
            truncated=truncated,
            info={
                "reached_goal": bool(won),
                "lost_game": bool(lost),
                "black_score": int(bs),
                "white_score": int(ws),
                "komi": float(self.params.komi),
                "final_score_diff": float(bs) - ws_with_komi,
                "reason": "scoring",
                "t": int(self._t),
            },
        )

    def _opponent_move(self) -> int:
        """
        Simple opponent policy:
        1. If any of my groups is in atari, try to save it (extend liberties)
        2. If I can capture opponent stones, do so (biggest capture first)
        3. Play near center or near existing stones
        4. 15% random for variety
        """
        legal = _legal_moves(self._board, WHITE, self._ko_point)
        if len(legal) <= 1:
            return ACTION_PASS

        # Random play with 15% probability
        if self._rng.random() < 0.15:
            return self._rng.choice(legal)

        # Evaluate each move
        best_move = ACTION_PASS
        best_score = -999.0

        for move in legal:
            if move == ACTION_PASS:
                continue
            result = _try_place(self._board, move, WHITE)
            if result is None:
                continue
            new_board, captured = result

            score = 0.0
            # Captures are valuable
            score += len(captured) * 3.0

            # Saving own groups in atari
            for n in _NEIGHBORS[move]:
                if self._board[n] == WHITE:
                    grp = _find_group(self._board, n)
                    libs = _group_liberties(self._board, grp)
                    if len(libs) == 1:  # was in atari
                        new_grp = _find_group(new_board, n)
                        new_libs = _group_liberties(new_board, new_grp)
                        if len(new_libs) > 1:
                            score += 2.0  # saved a group

            # Putting opponent in atari
            for n in _NEIGHBORS[move]:
                if new_board[n] == BLACK:
                    grp = _find_group(new_board, n)
                    libs = _group_liberties(new_board, grp)
                    if len(libs) == 1:
                        score += 1.5

            # Center preference
            r, c = divmod(move, BOARD_SIZE)
            center_dist = abs(r - 2) + abs(c - 2)
            score += (4 - center_dist) * 0.3

            # Near existing stones (connection)
            for n in _NEIGHBORS[move]:
                if self._board[n] == WHITE:
                    score += 0.4

            # Add randomness
            score += self._rng.uniform(-0.5, 0.5)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def legal_actions(self, obs: Optional[JSONValue] = None) -> List[JSONValue]:
        if self._done:
            return []
        return _legal_moves(self._board, BLACK, self._ko_point)

    def _make_obs(self) -> JSONValue:
        """Build observation dict from current board state."""
        board_copy = list(self._board)
        my_stones = sum(1 for p in range(N_CELLS) if self._board[p] == BLACK)
        opp_stones = sum(1 for p in range(N_CELLS) if self._board[p] == WHITE)

        bs, ws = _score_board(self._board)
        my_groups, my_libs_total, my_atari = _count_groups(self._board, BLACK)
        opp_groups, opp_libs_total, opp_atari = _count_groups(self._board, WHITE)

        # Strategic abstract features (for semantic bridge compatibility)
        score_delta = int(bs) - int(ws)
        pressure = min(16, max(-16, my_libs_total - opp_libs_total + (my_stones - opp_stones)))
        risk = max(0, min(16, my_atari * 3 + max(0, opp_stones - my_stones)))
        tempo = max(0, min(10, my_groups + my_stones // 3))
        control = min(16, max(-16, score_delta))
        resource = max(0, min(16, my_libs_total))

        return {
            "board": board_copy,
            "my_captures": int(self._my_captures),
            "opp_captures": int(self._opp_captures),
            "my_territory": int(bs),
            "opp_territory": int(ws),
            "my_groups": int(my_groups),
            "my_liberties_total": int(my_libs_total),
            "my_atari_groups": int(my_atari),
            "opp_groups": int(opp_groups),
            "opp_liberties_total": int(opp_libs_total),
            "opp_atari_groups": int(opp_atari),
            "ko_point": int(self._ko_point),
            "consecutive_passes": int(self._consecutive_passes),
            "my_stones": int(my_stones),
            "opp_stones": int(opp_stones),
            # Abstract strategy features for semantic bridge
            "score_delta": int(score_delta),
            "pressure": int(pressure),
            "risk": int(risk),
            "tempo": int(tempo),
            "control": int(control),
            "resource": int(resource),
            "t": int(self._t),
        }

    def render(self, mode: str = "ansi") -> str:
        symbols = {EMPTY: ".", BLACK: "●", WHITE: "○"}
        lines = []
        lines.append("  " + " ".join("ABCDE"))
        for r in range(BOARD_SIZE):
            row_str = " ".join(symbols[self._board[r * BOARD_SIZE + c]] for c in range(BOARD_SIZE))
            lines.append(f"{BOARD_SIZE - r} {row_str}")
        bs, ws = _score_board(self._board)
        lines.append(f"B(agent)={bs} W(opp)={ws} komi={self.params.komi}")
        lines.append(f"captures: agent={self._my_captures} opp={self._opp_captures}")
        return "\n".join(lines)

    def close(self) -> None:
        pass

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "board": list(self._board),
            "t": int(self._t),
            "my_captures": int(self._my_captures),
            "opp_captures": int(self._opp_captures),
            "ko_point": int(self._ko_point),
            "consecutive_passes": int(self._consecutive_passes),
            "prev_board_hash": str(self._prev_board_hash),
            "done": bool(self._done),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._board = list(state["board"])
        self._t = int(state["t"])
        self._my_captures = int(state["my_captures"])
        self._opp_captures = int(state["opp_captures"])
        self._ko_point = int(state["ko_point"])
        self._consecutive_passes = int(state["consecutive_passes"])
        self._prev_board_hash = str(state.get("prev_board_hash", ""))
        self._done = bool(state.get("done", False))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class GoWorldV2Factory:
    @property
    def tags(self) -> List[str]:
        return ["strategy_games", "board_control", "territory", "go", "v2"]

    def create(self, spec: VerseSpec) -> GoWorldV2Verse:
        return GoWorldV2Verse(spec)
