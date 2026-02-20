"""
verses/chess_world_v2.py

Mini-Chess on a 5x5 board with real piece movement:
- King: moves 1 square in any direction, cannot move into check
- Rook: slides horizontally/vertically
- Bishop: slides diagonally
- Pawn: moves forward 1, captures diagonally forward, promotes to Rook on last rank

Starting position (5x5):
  a b c d e
5 r . b . k   (Black / opponent)
4 p p p p p
3 . . . . .
2 P P P P P
1 K . B . R   (White / agent)

Action space: 625 discrete (from_sq * 25 + to_sq), masked to legal moves
Agent plays WHITE. Opponent plays BLACK.
Win by checkmate or material advantage at time limit.

Designed for genuine transfer learning research.
"""

from __future__ import annotations

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
N_ACTIONS = N_CELLS * N_CELLS  # 625 = (from_sq * 25 + to_sq)

# Piece codes: 0=empty, 1-4=white, 5-8=black
EMPTY = 0
W_KING, W_ROOK, W_BISHOP, W_PAWN = 1, 2, 3, 4
B_KING, B_ROOK, B_BISHOP, B_PAWN = 5, 6, 7, 8

WHITE_PIECES = {W_KING, W_ROOK, W_BISHOP, W_PAWN}
BLACK_PIECES = {B_KING, B_ROOK, B_BISHOP, B_PAWN}

PIECE_VALUES = {
    W_KING: 100, W_ROOK: 5, W_BISHOP: 3, W_PAWN: 1,
    B_KING: 100, B_ROOK: 5, B_BISHOP: 3, B_PAWN: 1,
    EMPTY: 0,
}

PIECE_SYMBOLS = {
    EMPTY: ".", W_KING: "K", W_ROOK: "R", W_BISHOP: "B", W_PAWN: "P",
    B_KING: "k", B_ROOK: "r", B_BISHOP: "b", B_PAWN: "p",
}

# Directions: (row_delta, col_delta)
KING_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# Starting position
INITIAL_BOARD = [
    B_ROOK,  EMPTY,   B_BISHOP, EMPTY,  B_KING,   # row 0 (rank 5)
    B_PAWN,  B_PAWN,  B_PAWN,   B_PAWN, B_PAWN,   # row 1 (rank 4)
    EMPTY,   EMPTY,   EMPTY,    EMPTY,  EMPTY,     # row 2 (rank 3)
    W_PAWN,  W_PAWN,  W_PAWN,   W_PAWN, W_PAWN,   # row 3 (rank 2)
    W_KING,  EMPTY,   W_BISHOP, EMPTY,  W_ROOK,   # row 4 (rank 1)
]


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _rc(pos: int) -> Tuple[int, int]:
    return divmod(pos, BOARD_SIZE)

def _pos(r: int, c: int) -> int:
    return r * BOARD_SIZE + c

def _on_board(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

def _is_white(piece: int) -> bool:
    return piece in WHITE_PIECES

def _is_black(piece: int) -> bool:
    return piece in BLACK_PIECES

def _is_friendly(piece: int, is_white_turn: bool) -> bool:
    return (_is_white(piece) if is_white_turn else _is_black(piece))

def _is_enemy(piece: int, is_white_turn: bool) -> bool:
    return (_is_black(piece) if is_white_turn else _is_white(piece))


def _find_king(board: List[int], is_white: bool) -> int:
    """Return position of king, or -1 if not found."""
    target = W_KING if is_white else B_KING
    for p in range(N_CELLS):
        if board[p] == target:
            return p
    return -1


def _attacks_square(board: List[int], sq: int, by_white: bool) -> bool:
    """Check if any piece of the given color attacks *sq*."""
    for p in range(N_CELLS):
        piece = board[p]
        if piece == EMPTY:
            continue
        if by_white and not _is_white(piece):
            continue
        if not by_white and not _is_black(piece):
            continue
        r, c = _rc(p)
        tr, tc = _rc(sq)

        if piece in (W_KING, B_KING):
            if max(abs(r - tr), abs(c - tc)) == 1:
                return True

        elif piece in (W_ROOK, B_ROOK):
            for dr, dc in ROOK_DIRS:
                for dist in range(1, BOARD_SIZE):
                    nr, nc = r + dr * dist, c + dc * dist
                    if not _on_board(nr, nc):
                        break
                    if nr == tr and nc == tc:
                        return True
                    if board[_pos(nr, nc)] != EMPTY:
                        break

        elif piece in (W_BISHOP, B_BISHOP):
            for dr, dc in BISHOP_DIRS:
                for dist in range(1, BOARD_SIZE):
                    nr, nc = r + dr * dist, c + dc * dist
                    if not _on_board(nr, nc):
                        break
                    if nr == tr and nc == tc:
                        return True
                    if board[_pos(nr, nc)] != EMPTY:
                        break

        elif piece == W_PAWN:
            # White pawns capture diagonally upward (row - 1)
            if tr == r - 1 and abs(tc - c) == 1:
                return True

        elif piece == B_PAWN:
            # Black pawns capture diagonally downward (row + 1)
            if tr == r + 1 and abs(tc - c) == 1:
                return True

    return False


def _in_check(board: List[int], is_white: bool) -> bool:
    """Check if *is_white*'s king is in check."""
    kp = _find_king(board, is_white)
    if kp < 0:
        return True  # king captured → in check
    return _attacks_square(board, kp, by_white=not is_white)


def _pseudo_legal_moves(board: List[int], is_white: bool) -> List[Tuple[int, int]]:
    """Generate all moves that follow piece movement rules (before check filtering)."""
    moves: List[Tuple[int, int]] = []
    my_pieces = WHITE_PIECES if is_white else BLACK_PIECES
    pawn = W_PAWN if is_white else B_PAWN
    pawn_dir = -1 if is_white else 1  # white moves up (row decreases)

    for p in range(N_CELLS):
        piece = board[p]
        if piece not in my_pieces:
            continue
        r, c = _rc(p)

        if piece in (W_KING, B_KING):
            for dr, dc in KING_DIRS:
                nr, nc = r + dr, c + dc
                if _on_board(nr, nc):
                    target = board[_pos(nr, nc)]
                    if target == EMPTY or _is_enemy(target, is_white):
                        moves.append((p, _pos(nr, nc)))

        elif piece in (W_ROOK, B_ROOK):
            for dr, dc in ROOK_DIRS:
                for dist in range(1, BOARD_SIZE):
                    nr, nc = r + dr * dist, c + dc * dist
                    if not _on_board(nr, nc):
                        break
                    target = board[_pos(nr, nc)]
                    if target == EMPTY:
                        moves.append((p, _pos(nr, nc)))
                    elif _is_enemy(target, is_white):
                        moves.append((p, _pos(nr, nc)))
                        break
                    else:
                        break  # friendly piece blocks

        elif piece in (W_BISHOP, B_BISHOP):
            for dr, dc in BISHOP_DIRS:
                for dist in range(1, BOARD_SIZE):
                    nr, nc = r + dr * dist, c + dc * dist
                    if not _on_board(nr, nc):
                        break
                    target = board[_pos(nr, nc)]
                    if target == EMPTY:
                        moves.append((p, _pos(nr, nc)))
                    elif _is_enemy(target, is_white):
                        moves.append((p, _pos(nr, nc)))
                        break
                    else:
                        break

        elif piece == pawn:
            # Forward move
            nr = r + pawn_dir
            if _on_board(nr, c) and board[_pos(nr, c)] == EMPTY:
                moves.append((p, _pos(nr, c)))
            # Diagonal captures
            for dc in (-1, 1):
                nc = c + dc
                if _on_board(nr, nc) and _is_enemy(board[_pos(nr, nc)], is_white):
                    moves.append((p, _pos(nr, nc)))

    return moves


def _legal_moves(board: List[int], is_white: bool) -> List[Tuple[int, int]]:
    """Generate all legal moves (pseudo-legal filtered by check)."""
    legal: List[Tuple[int, int]] = []
    for frm, to in _pseudo_legal_moves(board, is_white):
        # Make move on copy
        new_board = list(board)
        new_board[to] = new_board[frm]
        new_board[frm] = EMPTY
        # Pawn promotion
        piece = new_board[to]
        tr, _ = _rc(to)
        if piece == W_PAWN and tr == 0:
            new_board[to] = W_ROOK  # auto-promote to rook
        elif piece == B_PAWN and tr == BOARD_SIZE - 1:
            new_board[to] = B_ROOK
        # Check if own king is safe
        if not _in_check(new_board, is_white):
            legal.append((frm, to))
    return legal


def _material(board: List[int], is_white: bool) -> int:
    pieces = WHITE_PIECES if is_white else BLACK_PIECES
    total = 0
    for p in range(N_CELLS):
        if board[p] in pieces:
            total += PIECE_VALUES[board[p]]
    return total


def _center_control(board: List[int], is_white: bool) -> int:
    """How many of the center squares (1,1)-(3,3) does this side attack?"""
    count = 0
    for r in range(1, 4):
        for c in range(1, 4):
            sq = _pos(r, c)
            if _attacks_square(board, sq, by_white=is_white):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class ChessV2Params:
    max_steps: int = 150
    step_penalty: float = -0.005
    capture_reward_scale: float = 0.08
    checkmate_reward: float = 3.0
    checkmate_penalty: float = -3.0
    material_advantage_reward: float = 1.5


# ---------------------------------------------------------------------------
# Verse implementation
# ---------------------------------------------------------------------------

class ChessWorldV2Verse(Verse):
    """
    Mini-Chess (5x5) with real piece movement.

    625 actions: from_sq * 25 + to_sq, masked to legal moves.
    Agent plays WHITE, opponent plays BLACK.
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in ChessWorldV2Factory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = ChessV2Params(
            max_steps=int(self.spec.params.get("max_steps", 150)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.005)),
            capture_reward_scale=float(self.spec.params.get("capture_reward_scale", 0.08)),
            checkmate_reward=float(self.spec.params.get("checkmate_reward", 3.0)),
            checkmate_penalty=float(self.spec.params.get("checkmate_penalty", -3.0)),
            material_advantage_reward=float(self.spec.params.get("material_advantage_reward", 1.5)),
        )
        self.params.max_steps = max(20, int(self.params.max_steps))

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "board",
                "my_material", "opp_material", "material_delta",
                "my_king_pos", "opp_king_pos",
                "in_check", "opp_in_check",
                "my_legal_move_count", "opp_legal_move_count",
                "my_center_control", "opp_center_control",
                "my_pieces_count", "opp_pieces_count",
                "score_delta", "pressure", "risk", "tempo",
                "control", "resource",
                "t",
            ],
            notes="5x5 chess board state with derived strategic features.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=N_ACTIONS,
            notes="from_sq * 25 + to_sq (625 total, masked to legal moves)",
        )

        self._rng = random.Random()
        self._seed_val: Optional[int] = None
        self._done = False
        self._t = 0
        self._board: List[int] = list(INITIAL_BOARD)
        self._position_history: List[str] = []

    def seed(self, seed: Optional[int]) -> None:
        self._seed_val = seed
        if seed is not None:
            self._rng = random.Random(int(seed))

    def reset(self) -> ResetResult:
        if self._seed_val is not None:
            self._rng = random.Random(int(self._seed_val))

        self._board = list(INITIAL_BOARD)
        self._done = False
        self._t = 0
        self._position_history = [self._board_key()]

        obs = self._make_obs()
        return ResetResult(
            obs=obs,
            info={"verse_name": "chess_world_v2", "board_size": BOARD_SIZE},
        )

    def _board_key(self) -> str:
        return hashlib.md5(bytes(self._board)).hexdigest()

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(
                self._make_obs(), 0.0, True, False,
                {"warning": "step() called after done"},
            )

        self._t += 1
        reward = float(self.params.step_penalty)

        # ---- Agent move (WHITE) ----
        a = int(action)
        frm = a // N_CELLS
        to = a % N_CELLS
        legal = _legal_moves(self._board, is_white=True)

        if (frm, to) not in legal:
            # Illegal move → pick random legal move with penalty
            if legal:
                frm, to = self._rng.choice(legal)
                reward -= 0.02
            else:
                # No legal moves — we're in checkmate or stalemate
                return self._handle_no_moves(is_white=True, reward=reward)

        # Execute agent move
        captured_piece = self._board[to]
        self._board[to] = self._board[frm]
        self._board[frm] = EMPTY

        # Pawn promotion
        piece = self._board[to]
        tr, _ = _rc(to)
        if piece == W_PAWN and tr == 0:
            self._board[to] = W_ROOK
            reward += 0.3  # promotion bonus

        # Capture reward
        if captured_piece != EMPTY:
            reward += PIECE_VALUES.get(captured_piece, 1) * float(self.params.capture_reward_scale)

        # Check if opponent king was captured (shouldn't happen with legal moves, but safety)
        if captured_piece == B_KING:
            reward += float(self.params.checkmate_reward)
            self._done = True
            return StepResult(
                self._make_obs(), float(reward), True, False,
                {"reached_goal": True, "reason": "king_captured", "t": self._t},
            )

        # Check repetition
        key = self._board_key()
        if self._position_history.count(key) >= 2:
            self._done = True
            return StepResult(
                self._make_obs(), float(reward) - 0.5, True, False,
                {"reached_goal": False, "reason": "repetition", "t": self._t},
            )
        self._position_history.append(key)

        # ---- Check if opponent has legal moves ----
        opp_legal = _legal_moves(self._board, is_white=False)
        if not opp_legal:
            if _in_check(self._board, is_white=False):
                # Checkmate!
                reward += float(self.params.checkmate_reward)
                self._done = True
                return StepResult(
                    self._make_obs(), float(reward), True, False,
                    {"reached_goal": True, "reason": "checkmate", "t": self._t},
                )
            else:
                # Stalemate — draw
                self._done = True
                return StepResult(
                    self._make_obs(), float(reward), True, False,
                    {"reached_goal": False, "reason": "stalemate", "t": self._t},
                )

        # ---- Opponent move (BLACK) ----
        opp_frm, opp_to = self._opponent_move(opp_legal)
        captured_by_opp = self._board[opp_to]
        self._board[opp_to] = self._board[opp_frm]
        self._board[opp_frm] = EMPTY

        # Opponent pawn promotion
        opp_piece = self._board[opp_to]
        opp_tr, _ = _rc(opp_to)
        if opp_piece == B_PAWN and opp_tr == BOARD_SIZE - 1:
            self._board[opp_to] = B_ROOK

        # Penalty for losing pieces
        if captured_by_opp != EMPTY:
            reward -= PIECE_VALUES.get(captured_by_opp, 1) * float(self.params.capture_reward_scale) * 0.8

        # Check if our king was captured
        if captured_by_opp == W_KING:
            reward += float(self.params.checkmate_penalty)
            self._done = True
            return StepResult(
                self._make_obs(), float(reward), True, False,
                {"reached_goal": False, "lost_game": True, "reason": "king_captured", "t": self._t},
            )

        # ---- Check if agent has legal moves after opponent ----
        my_legal = _legal_moves(self._board, is_white=True)
        if not my_legal:
            if _in_check(self._board, is_white=True):
                # We got checkmated
                reward += float(self.params.checkmate_penalty)
                self._done = True
                return StepResult(
                    self._make_obs(), float(reward), True, False,
                    {"reached_goal": False, "lost_game": True, "reason": "checkmated", "t": self._t},
                )
            else:
                # Stalemate
                self._done = True
                return StepResult(
                    self._make_obs(), float(reward), True, False,
                    {"reached_goal": False, "reason": "stalemate", "t": self._t},
                )

        # ---- Truncation ----
        truncated = self._t >= int(self.params.max_steps)
        if truncated:
            my_mat = _material(self._board, is_white=True) - 100  # subtract king
            opp_mat = _material(self._board, is_white=False) - 100
            mat_diff = my_mat - opp_mat
            if mat_diff > 0:
                reward += float(self.params.material_advantage_reward)
                reached = True
            elif mat_diff < 0:
                reward -= float(self.params.material_advantage_reward)
                reached = False
            else:
                reached = False
            self._done = True
            return StepResult(
                self._make_obs(), float(reward), False, True,
                {"reached_goal": reached, "reason": "time_limit", "material_diff": mat_diff, "t": self._t},
            )

        # Shaping: material advantage
        my_mat = _material(self._board, is_white=True) - 100
        opp_mat = _material(self._board, is_white=False) - 100
        reward += 0.002 * (my_mat - opp_mat)

        # Check bonus
        if _in_check(self._board, is_white=False):
            reward += 0.03

        self._done = False
        return StepResult(
            self._make_obs(), float(reward), False, False,
            {
                "reached_goal": False, "t": self._t,
                "agent_captured": PIECE_SYMBOLS.get(captured_piece, ""),
                "opp_captured": PIECE_SYMBOLS.get(captured_by_opp, ""),
            },
        )

    def _handle_no_moves(self, is_white: bool, reward: float) -> StepResult:
        """Handle position where side to move has no legal moves."""
        if _in_check(self._board, is_white):
            # Checkmate against us
            reward += float(self.params.checkmate_penalty) if is_white else float(self.params.checkmate_reward)
            self._done = True
            return StepResult(
                self._make_obs(), float(reward), True, False,
                {"reached_goal": not is_white, "reason": "checkmate", "t": self._t},
            )
        else:
            self._done = True
            return StepResult(
                self._make_obs(), float(reward), True, False,
                {"reached_goal": False, "reason": "stalemate", "t": self._t},
            )

    def _opponent_move(self, legal: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Simple opponent policy:
        1. Capture highest value piece if possible
        2. Put agent king in check if possible
        3. Move toward center
        4. 20% random
        """
        if not legal:
            return legal[0] if legal else (0, 0)

        if self._rng.random() < 0.20:
            return self._rng.choice(legal)

        best_move = legal[0]
        best_score = -999.0

        for frm, to in legal:
            score = 0.0

            # Value of captured piece
            captured = self._board[to]
            if captured != EMPTY and _is_white(captured):
                score += PIECE_VALUES.get(captured, 1) * 2.0

            # Check: does this move put white king in check?
            test = list(self._board)
            test[to] = test[frm]
            test[frm] = EMPTY
            piece = test[to]
            tr, _ = _rc(to)
            if piece == B_PAWN and tr == BOARD_SIZE - 1:
                test[to] = B_ROOK
            if _in_check(test, is_white=True):
                score += 1.5

            # Center preference
            tr, tc = _rc(to)
            center_dist = abs(tr - 2) + abs(tc - 2)
            score += (4 - center_dist) * 0.2

            # Protect own pieces: avoid moving to attacked squares
            if _attacks_square(self._board, to, by_white=True):
                score -= PIECE_VALUES.get(self._board[frm], 1) * 0.3

            score += self._rng.uniform(-0.3, 0.3)

            if score > best_score:
                best_score = score
                best_move = (frm, to)

        return best_move

    def legal_actions(self, obs: Optional[JSONValue] = None) -> List[JSONValue]:
        if self._done:
            return []
        legal = _legal_moves(self._board, is_white=True)
        return [frm * N_CELLS + to for frm, to in legal]

    def _make_obs(self) -> JSONValue:
        board_copy = list(self._board)

        my_mat = _material(self._board, is_white=True)
        opp_mat = _material(self._board, is_white=False)
        my_king = _find_king(self._board, is_white=True)
        opp_king = _find_king(self._board, is_white=False)
        am_in_check = _in_check(self._board, is_white=True)
        opp_in_check = _in_check(self._board, is_white=False)
        my_legal = len(_legal_moves(self._board, is_white=True))
        opp_legal = len(_legal_moves(self._board, is_white=False))
        my_center = _center_control(self._board, is_white=True)
        opp_center = _center_control(self._board, is_white=False)
        my_pieces = sum(1 for p in range(N_CELLS) if self._board[p] in WHITE_PIECES)
        opp_pieces = sum(1 for p in range(N_CELLS) if self._board[p] in BLACK_PIECES)

        # Material without king for comparison
        my_mat_no_king = my_mat - 100
        opp_mat_no_king = opp_mat - 100

        # Abstract strategy features (for semantic bridge)
        mat_delta = my_mat_no_king - opp_mat_no_king
        score_delta = mat_delta
        pressure = min(16, max(-16, my_legal - opp_legal + my_center - opp_center))
        risk = max(0, min(16, (3 if am_in_check else 0) + max(0, opp_legal - my_legal)))
        tempo = max(0, min(10, my_legal // 3))
        control = min(16, max(-16, my_center - opp_center))
        resource = max(0, min(16, my_mat_no_king))

        return {
            "board": board_copy,
            "my_material": int(my_mat_no_king),
            "opp_material": int(opp_mat_no_king),
            "material_delta": int(mat_delta),
            "my_king_pos": int(my_king),
            "opp_king_pos": int(opp_king),
            "in_check": int(am_in_check),
            "opp_in_check": int(opp_in_check),
            "my_legal_move_count": int(my_legal),
            "opp_legal_move_count": int(opp_legal),
            "my_center_control": int(my_center),
            "opp_center_control": int(opp_center),
            "my_pieces_count": int(my_pieces),
            "opp_pieces_count": int(opp_pieces),
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
        lines = []
        lines.append("  a b c d e")
        for r in range(BOARD_SIZE):
            row_str = " ".join(PIECE_SYMBOLS[self._board[r * BOARD_SIZE + c]] for c in range(BOARD_SIZE))
            lines.append(f"{BOARD_SIZE - r} {row_str}")
        my_mat = _material(self._board, is_white=True) - 100
        opp_mat = _material(self._board, is_white=False) - 100
        lines.append(f"White(agent)={my_mat} Black(opp)={opp_mat}")
        if _in_check(self._board, is_white=True):
            lines.append("WHITE IN CHECK!")
        if _in_check(self._board, is_white=False):
            lines.append("BLACK IN CHECK!")
        return "\n".join(lines)

    def close(self) -> None:
        pass

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "board": list(self._board),
            "t": int(self._t),
            "done": bool(self._done),
            "position_history": list(self._position_history),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._board = list(state["board"])
        self._t = int(state["t"])
        self._done = bool(state.get("done", False))
        self._position_history = list(state.get("position_history", []))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class ChessWorldV2Factory:
    @property
    def tags(self) -> List[str]:
        return ["strategy_games", "board_control", "chess", "tactics", "v2"]

    def create(self, spec: VerseSpec) -> ChessWorldV2Verse:
        return ChessWorldV2Verse(spec)
