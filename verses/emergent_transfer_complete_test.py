"""
verses/emergent_transfer_complete_test.py

End-to-end emergent transfer check using a new verse: `warehouse_world`.

Commands:
  python verses/emergent_transfer_complete_test.py setup
  python verses/emergent_transfer_complete_test.py test
  python verses/emergent_transfer_complete_test.py analyze
  python verses/emergent_transfer_complete_test.py full
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import AgentSpec, VerseSpec  # noqa: E402
from orchestrator.evaluator import evaluate_run  # noqa: E402
from orchestrator.trainer import Trainer  # noqa: E402


RESULTS_PATH = ROOT / "models" / "emergent_transfer_results.json"
WAREHOUSE_PATH = ROOT / "verses" / "warehouse_world.py"
REGISTRY_PATH = ROOT / "verses" / "registry.py"
EMERGENT_MANIFEST_PATH = ROOT / "models" / "emergent_default_policy_set.json"

TEST_CONFIG = {
    "episodes": 16,
    "max_steps": 100,
    "seed": 123,
    "conditions": [
        {"name": "A_Random", "algo": "random", "config": {}},
        {"name": "B_MoE_System", "algo": "adaptive_moe", "config": {}},
        {"name": "C_Gateway", "algo": "gateway", "config": {}},
    ],
}


WAREHOUSE_MODULE_CODE = '''"""
verses/warehouse_world.py

Warehouse-style verse:
- grid navigation
- static obstacles
- charging stations
- battery depletion
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class WarehouseParams:
    width: int = 8
    height: int = 8
    start_x: int = 0
    start_y: int = 0
    goal_x: int = -1
    goal_y: int = -1
    max_steps: int = 100
    obstacle_count: int = 14
    battery_capacity: int = 20
    battery_drain: int = 1
    charge_rate: int = 5
    step_penalty: float = -0.10
    obstacle_penalty: float = -1.00
    wall_penalty: float = -0.50
    battery_fail_penalty: float = -10.0
    goal_reward: float = 10.0
    charge_reward: float = 0.5


class WarehouseWorldVerse:
    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in WarehouseWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)
        self.params = WarehouseParams(
            width=int(self.spec.params.get("width", 8)),
            height=int(self.spec.params.get("height", 8)),
            start_x=int(self.spec.params.get("start_x", 0)),
            start_y=int(self.spec.params.get("start_y", 0)),
            goal_x=int(self.spec.params.get("goal_x", -1)),
            goal_y=int(self.spec.params.get("goal_y", -1)),
            max_steps=int(self.spec.params.get("max_steps", 100)),
            obstacle_count=int(self.spec.params.get("obstacle_count", 14)),
            battery_capacity=int(self.spec.params.get("battery_capacity", 20)),
            battery_drain=int(self.spec.params.get("battery_drain", 1)),
            charge_rate=int(self.spec.params.get("charge_rate", 5)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.10)),
            obstacle_penalty=float(self.spec.params.get("obstacle_penalty", -1.00)),
            wall_penalty=float(self.spec.params.get("wall_penalty", -0.50)),
            battery_fail_penalty=float(self.spec.params.get("battery_fail_penalty", -10.0)),
            goal_reward=float(self.spec.params.get("goal_reward", 10.0)),
            charge_reward=float(self.spec.params.get("charge_reward", 0.5)),
        )

        self.params.width = max(5, int(self.params.width))
        self.params.height = max(5, int(self.params.height))
        self.params.max_steps = max(10, int(self.params.max_steps))
        self.params.obstacle_count = max(1, int(self.params.obstacle_count))
        self.params.battery_capacity = max(1, int(self.params.battery_capacity))
        self.params.battery_drain = max(0, int(self.params.battery_drain))
        self.params.charge_rate = max(0, int(self.params.charge_rate))

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "x", "y", "goal_x", "goal_y", "battery",
                "nearby_obstacles", "nearest_charger_dist", "t", "flat",
            ],
            subspaces={
                "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "battery": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearby_obstacles": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearest_charger_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "flat": SpaceSpec(type="vector", shape=(7,), dtype="float32"),
            },
            notes="WarehouseWorld obs dict",
        )

        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=5,
            notes="0=up,1=down,2=left,3=right,4=wait/charge",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._x = 0
        self._y = 0
        self._battery = int(self.params.battery_capacity)
        self._t = 0
        self._done = False
        self._obstacles: Set[Tuple[int, int]] = set()
        self._chargers: Set[Tuple[int, int]] = set()

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)
        self._x = self._start_x()
        self._y = self._start_y()
        self._battery = int(self.params.battery_capacity)
        self._t = 0
        self._done = False
        self._build_layout()
        return ResetResult(
            obs=self._make_obs(),
            info={
                "seed": self._seed,
                "width": int(self.params.width),
                "height": int(self.params.height),
                "goal_x": int(self._goal_x()),
                "goal_y": int(self._goal_y()),
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = max(0, min(4, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"action_used": int(a), "reached_goal": False}
        done = False

        if a == 4:
            if (self._x, self._y) in self._chargers:
                old = self._battery
                self._battery = min(int(self.params.battery_capacity), self._battery + int(self.params.charge_rate))
                if self._battery > old:
                    reward += float(self.params.charge_reward)
                    info["charged"] = True
        else:
            nx, ny = self._x, self._y
            if a == 0:
                ny -= 1
            elif a == 1:
                ny += 1
            elif a == 2:
                nx -= 1
            elif a == 3:
                nx += 1

            if not (0 <= nx < self.params.width and 0 <= ny < self.params.height):
                reward += float(self.params.wall_penalty)
                info["hit_wall"] = True
            elif (nx, ny) in self._obstacles:
                reward += float(self.params.obstacle_penalty)
                info["hit_obstacle"] = True
            else:
                self._x, self._y = nx, ny
                self._battery = max(0, self._battery - int(self.params.battery_drain))

        if self._battery <= 0 and (self._x, self._y) != (self._goal_x(), self._goal_y()):
            reward += float(self.params.battery_fail_penalty)
            info["battery_death"] = True
            done = True
        elif (self._x, self._y) == (self._goal_x(), self._goal_y()):
            reward += float(self.params.goal_reward)
            info["reached_goal"] = True
            done = True

        truncated = bool(self._t >= int(self.params.max_steps) and not done)
        self._done = bool(done or truncated)
        info["t"] = int(self._t)
        info["battery"] = int(self._battery)
        info["x"] = int(self._x)
        info["y"] = int(self._y)
        info["goal_x"] = int(self._goal_x())
        info["goal_y"] = int(self._goal_y())

        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        rows: List[str] = []
        for y in range(self.params.height):
            row: List[str] = []
            for x in range(self.params.width):
                ch = "."
                if (x, y) in self._obstacles:
                    ch = "#"
                if (x, y) in self._chargers:
                    ch = "C"
                if x == self._goal_x() and y == self._goal_y():
                    ch = "G"
                if x == self._x and y == self._y:
                    ch = "A"
                row.append(ch)
            rows.append("".join(row))
        return "\\n".join(rows)

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "x": int(self._x),
            "y": int(self._y),
            "battery": int(self._battery),
            "t": int(self._t),
            "done": bool(self._done),
            "obstacles": [[x, y] for (x, y) in sorted(self._obstacles)],
            "chargers": [[x, y] for (x, y) in sorted(self._chargers)],
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._x = max(0, min(self.params.width - 1, int(state.get("x", self._x))))
        self._y = max(0, min(self.params.height - 1, int(state.get("y", self._y))))
        self._battery = max(0, min(int(self.params.battery_capacity), int(state.get("battery", self._battery))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        obs = state.get("obstacles")
        if isinstance(obs, list):
            self._obstacles = set((int(p[0]), int(p[1])) for p in obs if isinstance(p, list) and len(p) == 2)
        ch = state.get("chargers")
        if isinstance(ch, list):
            self._chargers = set((int(p[0]), int(p[1])) for p in ch if isinstance(p, list) and len(p) == 2)

    def _make_obs(self) -> JSONValue:
        nearby = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (self._x + dx, self._y + dy) in self._obstacles:
                nearby += 1
        nearest = 999
        for cx, cy in self._chargers:
            d = abs(self._x - cx) + abs(self._y - cy)
            nearest = min(nearest, d)
        return {
            "x": int(self._x),
            "y": int(self._y),
            "goal_x": int(self._goal_x()),
            "goal_y": int(self._goal_y()),
            "battery": int(self._battery),
            "nearby_obstacles": int(nearby),
            "nearest_charger_dist": int(nearest if nearest < 999 else -1),
            "t": int(self._t),
            "flat": [
                float(self._x), float(self._y), float(self._goal_x()),
                float(self._goal_y()), float(self._battery), float(nearby), float(self._t),
            ],
        }

    def _build_layout(self) -> None:
        self._obstacles = set()
        self._chargers = set()
        for pos in [(2, 2), (5, 5), (2, 5), (5, 2)]:
            if 0 <= pos[0] < self.params.width and 0 <= pos[1] < self.params.height:
                self._chargers.add(pos)
        start = (self._start_x(), self._start_y())
        goal = (self._goal_x(), self._goal_y())
        while len(self._obstacles) < int(self.params.obstacle_count):
            x = self._rng.randint(0, self.params.width - 1)
            y = self._rng.randint(0, self.params.height - 1)
            p = (x, y)
            if p == start or p == goal or p in self._chargers:
                continue
            self._obstacles.add(p)

    def _start_x(self) -> int:
        return max(0, min(self.params.width - 1, int(self.params.start_x)))

    def _start_y(self) -> int:
        return max(0, min(self.params.height - 1, int(self.params.start_y)))

    def _goal_x(self) -> int:
        gx = int(self.params.goal_x)
        if gx < 0:
            gx = self.params.width - 1
        return max(0, min(self.params.width - 1, gx))

    def _goal_y(self) -> int:
        gy = int(self.params.goal_y)
        if gy < 0:
            gy = self.params.height - 1
        return max(0, min(self.params.height - 1, gy))


class WarehouseWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "grid", "resource", "obstacles", "warehouse"]

    def create(self, spec: VerseSpec) -> Verse:
        return WarehouseWorldVerse(spec)
'''


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _ensure_warehouse_world_file() -> None:
    WAREHOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write(WAREHOUSE_PATH, WAREHOUSE_MODULE_CODE)


def _ensure_emergent_manifest() -> None:
    base_path = ROOT / "models" / "default_policy_set.json"
    if base_path.exists():
        try:
            payload = json.loads(_read(base_path))
        except Exception:
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    deploy = payload.get("deployment_ready_defaults")
    deploy = deploy if isinstance(deploy, dict) else {}
    if "warehouse_world" not in deploy:
        deploy["warehouse_world"] = {
            "picked_run": {
                "policy": "special_moe",
                "run_dir": "",
            },
            "artifacts": [
                "models/micro_selector.pt",
                "models/expert_datasets",
            ],
            "command": (
                "python tools/train_agent.py --algo special_moe --verse warehouse_world "
                "--episodes 20 --max_steps 100 --expert_dataset_dir models/expert_datasets --top_k_experts 2"
            ),
        }
    payload["deployment_ready_defaults"] = deploy

    winners = payload.get("winners_robust")
    winners = winners if isinstance(winners, dict) else {}
    winners.setdefault(
        "warehouse_world",
        {
            "policy": "special_moe",
            "run_dir": "",
            "mean_return": 0.0,
            "success_rate": 0.0,
        },
    )
    payload["winners_robust"] = winners
    EMERGENT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write(EMERGENT_MANIFEST_PATH, json.dumps(payload, ensure_ascii=False, indent=2))


def _patch_registry() -> None:
    text = _read(REGISTRY_PATH)
    original = text

    if 'if v == "warehouse_world":' not in text:
        anchor = '    if v == "labyrinth_world":'
        add = '\n    if v == "warehouse_world":\n        return ["step_penalty", "obstacle_penalty", "battery_drain", "charge_rate", "width", "height"]\n'
        if anchor in text:
            text = text.replace(anchor, add + anchor, 1)

    if "from verses.warehouse_world import WarehouseWorldFactory" not in text:
        imp_anchor = "    from verses.pursuit_world import PursuitWorldFactory"
        if imp_anchor in text:
            text = text.replace(
                imp_anchor,
                imp_anchor + "\n    from verses.warehouse_world import WarehouseWorldFactory",
                1,
            )

    if '"warehouse_world": WarehouseWorldFactory(),' not in text:
        builtins_anchor = '        "pursuit_world": PursuitWorldFactory(),'
        if builtins_anchor in text:
            text = text.replace(
                builtins_anchor,
                builtins_anchor + '\n        "warehouse_world": WarehouseWorldFactory(),',
                1,
            )

    if text != original:
        _write(REGISTRY_PATH, text)


def setup() -> None:
    print("=" * 80)
    print("SETUP WAREHOUSE_WORLD")
    print("=" * 80)
    _ensure_warehouse_world_file()
    _patch_registry()
    _ensure_emergent_manifest()

    import verses.registry as vr

    importlib.reload(vr)
    vr.register_builtin()
    if "warehouse_world" not in vr.list_verses():
        raise RuntimeError("warehouse_world registration failed")
    print("setup_ok=True")


def _condition_agent_spec(name: str, algo: str, seed: int) -> AgentSpec:
    cfg: Dict[str, Any] = {}
    if algo in ("adaptive_moe", "special_moe", "gateway"):
        cfg["verse_name"] = "warehouse_world"
    if algo in ("adaptive_moe", "special_moe"):
        ds = ROOT / "models" / "expert_datasets"
        if ds.is_dir():
            cfg["expert_dataset_dir"] = str(ds)
        selector = ROOT / "models" / "micro_selector.pt"
        if selector.is_file():
            cfg["selector_model_path"] = str(selector)
        cfg["top_k"] = 2
        cfg["expert_lookup_config"] = {
            "enable_mlp_generalizer": True,
            "enable_nn_fallback": True,
            "nn_fallback_k": 7,
            "mlp_epochs": 10,
            "mlp_min_rows": 80,
        }
    if algo == "adaptive_moe":
        cfg["uncertainty_threshold"] = 0.25
    if algo == "gateway":
        cfg["manifest_path"] = str(EMERGENT_MANIFEST_PATH)

    return AgentSpec(
        spec_version="v1",
        policy_id=f"emergent:{name}",
        policy_version="0.1",
        algo=algo,
        seed=seed,
        tags=["emergent_transfer"],
        config=(cfg if cfg else None),
    )


def _warehouse_spec(seed: int, max_steps: int) -> VerseSpec:
    return VerseSpec(
        spec_version="v1",
        verse_name="warehouse_world",
        verse_version="0.1",
        seed=seed,
        tags=["emergent_transfer"],
        params={
            "width": 8,
            "height": 8,
            "max_steps": int(max_steps),
            "battery_capacity": 20,
            "battery_drain": 1,
            "charge_rate": 5,
            "obstacle_count": 14,
            "step_penalty": -0.10,
            "obstacle_penalty": -1.00,
            "goal_reward": 10.0,
            "adr_enabled": False,
        },
    )


def _analyze_routing(run_dir: Path) -> Dict[str, Any]:
    events_path = run_dir / "events.jsonl"
    experts = Counter()
    modes = Counter()
    confidences: List[float] = []
    if not events_path.exists():
        return {"experts": {}, "modes": {}, "avg_confidence": 0.0}
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            info = ev.get("info")
            info = info if isinstance(info, dict) else {}
            ai = info.get("action_info")
            ai = ai if isinstance(ai, dict) else {}
            mode = str(ai.get("mode", ""))
            if mode:
                modes[mode] += 1
            exp_list = ai.get("experts")
            if isinstance(exp_list, list):
                for e in exp_list:
                    experts[str(e)] += 1
            c = ai.get("uncertainty")
            try:
                if c is not None:
                    confidences.append(float(c))
            except Exception:
                pass
    return {
        "experts": dict(experts),
        "modes": dict(modes),
        "avg_uncertainty": (sum(confidences) / len(confidences) if confidences else 0.0),
    }


def run_test() -> Dict[str, Any]:
    setup()
    print("=" * 80)
    print("RUN TEST CONDITIONS")
    print("=" * 80)

    trainer = Trainer(run_root=str(ROOT / "runs"), schema_version="v1", auto_register_builtin=True)
    out: Dict[str, Any] = {"created_at_ms": int(time.time() * 1000), "conditions": {}}

    for c in TEST_CONFIG["conditions"]:
        name = str(c["name"])
        algo = str(c["algo"])
        print(f"\n[{name}] algo={algo}")
        agent_spec = _condition_agent_spec(name=name, algo=algo, seed=int(TEST_CONFIG["seed"]))
        verse_spec = _warehouse_spec(seed=int(TEST_CONFIG["seed"]), max_steps=int(TEST_CONFIG["max_steps"]))
        try:
            res = trainer.run(
                verse_spec=verse_spec,
                agent_spec=agent_spec,
                episodes=int(TEST_CONFIG["episodes"]),
                max_steps=int(TEST_CONFIG["max_steps"]),
                seed=int(TEST_CONFIG["seed"]),
            )
            run_id = str(res["run_id"])
            run_dir = ROOT / "runs" / run_id
            stats = evaluate_run(str(run_dir))
            routing = _analyze_routing(run_dir)
            out["conditions"][name] = {
                "algo": algo,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "episodes": int(stats.episodes),
                "mean_return": float(stats.mean_return),
                "mean_steps": float(stats.mean_steps),
                "success_rate": (None if stats.success_rate is None else float(stats.success_rate)),
                "routing": routing,
            }
            print(
                f"run={run_id} mean_return={stats.mean_return:.3f} "
                f"success_rate={(0.0 if stats.success_rate is None else stats.success_rate):.3f}"
            )
        except Exception as e:
            out["conditions"][name] = {
                "algo": algo,
                "error": str(e),
            }
            print(f"condition_failed={name} error={e}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write(RESULTS_PATH, json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nresults_saved={RESULTS_PATH}")
    return out


def analyze(results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if results is None:
        if not RESULTS_PATH.exists():
            raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")
        results = json.loads(_read(RESULTS_PATH))

    conds = results.get("conditions", {})
    a = conds.get("A_Random", {})
    b = conds.get("B_MoE_System", {})
    g = conds.get("C_Gateway", {})

    a_s = float(a.get("success_rate") or 0.0)
    b_s = float(b.get("success_rate") or 0.0)
    g_s = float(g.get("success_rate") or 0.0)
    a_r = float(a.get("mean_return") or 0.0)
    b_r = float(b.get("mean_return") or 0.0)
    g_r = float(g.get("mean_return") or 0.0)

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"A Random   : success={a_s:.3f} mean_return={a_r:.3f}")
    print(f"B MoE      : success={b_s:.3f} mean_return={b_r:.3f}")
    print(f"C Gateway  : success={g_s:.3f} mean_return={g_r:.3f}")
    print("-" * 80)
    print(f"MoE vs Random success delta: {(b_s - a_s):+.3f}")
    print(f"MoE vs Random return  delta: {(b_r - a_r):+.3f}")
    print(f"MoE vs Gateway success delta: {(b_s - g_s):+.3f}")
    print(f"MoE vs Gateway return  delta: {(b_r - g_r):+.3f}")

    routing = (b.get("routing") or {})
    print("-" * 80)
    print("MoE routing summary:")
    print(json.dumps(routing, ensure_ascii=False, indent=2))

    verdict = "no_transfer"
    if (b_s - a_s) > 0.30 and (b_r - a_r) > 5.0:
        verdict = "strong_transfer"
    elif (b_s - a_s) > 0.10:
        verdict = "partial_transfer"
    print("-" * 80)
    print(f"verdict={verdict}")

    summary = {
        "verdict": verdict,
        "delta_success_moe_vs_random": float(b_s - a_s),
        "delta_return_moe_vs_random": float(b_r - a_r),
        "delta_success_moe_vs_gateway": float(b_s - g_s),
        "delta_return_moe_vs_gateway": float(b_r - g_r),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["setup", "test", "analyze", "full"])
    args = ap.parse_args()

    if args.command == "setup":
        setup()
        return
    if args.command == "test":
        run_test()
        return
    if args.command == "analyze":
        analyze()
        return
    if args.command == "full":
        res = run_test()
        analyze(res)
        return


if __name__ == "__main__":
    main()
