"""Optional Multiverse-owned C++/MuJoCo physics runtime.

The extension and MuJoCo are optional so the base Python install remains
lightweight. Build the extension with ``native/README.md`` and select it with
``VerseSpec.params['sim_provider'] = 'multiverse_native'``.
"""

from __future__ import annotations

import glob
import importlib
import os
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


MODEL_FILES = {
    "embodied_point_world": "point_mass.xml",
    "point_mass": "point_mass.xml",
    "robot_reacher_world": "robot_reacher.xml",
    "robot_reacher": "robot_reacher.xml",
    "world_3d": "world_3d_terrain.xml",
    "warehouse_3d_world": "warehouse_3d.xml",
}


class NativeEngineUnavailableError(RuntimeError):
    """Raised when Multiverse's optional native runtime has not been built."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_extension():
    try:
        return importlib.import_module("_multiverse_native")
    except ImportError:
        pass

    root = _repo_root()
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    candidates = [root / "build" / "native", root / "native" / "build"]
    for directory in candidates:
        if not directory.is_dir():
            continue
        for suffix in suffixes:
            matches = list(directory.glob(f"_multiverse_native*{suffix}"))
            if matches:
                sys.path.insert(0, str(directory))
                try:
                    return importlib.import_module("_multiverse_native")
                finally:
                    try:
                        sys.path.remove(str(directory))
                    except ValueError:
                        pass
    raise NativeEngineUnavailableError(
        "The Multiverse native extension is not built. Follow native/README.md "
        "to install the optional CMake, pybind11, and MuJoCo build dependencies and build it."
    )


def _mujoco_runtime() -> Tuple[Path, Path]:
    try:
        mujoco = importlib.import_module("mujoco")
    except ImportError as exc:
        raise NativeEngineUnavailableError(
            "MuJoCo is required for sim_provider='multiverse_native'. Install requirements_native.txt."
        ) from exc
    root = Path(mujoco.__file__).resolve().parent
    if os.name == "nt":
        candidates = [root / "mujoco.dll", *map(Path, glob.glob(str(root / "*.dll")))]
        # Keep the directory registration alive for dependent DLL resolution.
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(root)))
    elif sys.platform == "darwin":
        candidates = [*map(Path, glob.glob(str(root / "*.dylib")))]
    else:
        candidates = [*map(Path, glob.glob(str(root / "libmujoco.so*")))]
    library = next((path for path in candidates if path.is_file()), None)
    if library is None:
        raise NativeEngineUnavailableError(f"Could not find MuJoCo's shared library under {root}")
    return root, library


_DLL_DIRECTORY_HANDLES: List[Any] = []
_EXTENSION = None


def _native_extension():
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = _load_extension()
    return _EXTENSION


def _model_path(model: str) -> Path:
    try:
        filename = MODEL_FILES[str(model).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown Multiverse MuJoCo model: {model!r}") from exc
    path = Path(__file__).resolve().parent / "models" / filename
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo model file is missing: {path}")
    return path


def build_reacher_mjcf(l1: float, l2: float) -> str:
    """Return the packaged reacher MJCF with its configured link dimensions."""
    xml = _model_path("robot_reacher").read_text(encoding="utf-8")
    first_link = f'{float(l1):.9g}'
    second_link = f'{float(l2):.9g}'
    xml = xml.replace('fromto="0 0 0 0.5 0 0"', f'fromto="0 0 0 {first_link} 0 0"', 1)
    xml = xml.replace('body name="forearm" pos="0.5 0 0"',
                      f'body name="forearm" pos="{first_link} 0 0"', 1)
    xml = xml.replace('fromto="0 0 0 0.5 0 0"', f'fromto="0 0 0 {second_link} 0 0"', 1)
    xml = xml.replace('site name="end_effector" pos="0.5 0 0"',
                      f'site name="end_effector" pos="{second_link} 0 0"', 1)
    return xml


class NativePhysicsBackend:
    """MuJoCo-backed PhysicsBackend with a C++ batch stepping core.

    A backend may own one lane or a whole vector. Models are loaded once per
    backend; every lane receives its own ``mjData`` in the native extension.
    """

    def __init__(
        self,
        model: str = "point_mass",
        *,
        env_count: int = 1,
        worker_count: int = 0,
        substeps: Optional[int] = None,
        model_xml: Optional[str] = None,
    ) -> None:
        self.model_name = str(model).strip().lower()
        self._module = _native_extension()
        package_root, library = _mujoco_runtime()
        del package_root
        self._model_path = _model_path(self.model_name)
        if substeps is None:
            substeps = 10 if self.model_name in ("robot_reacher", "robot_reacher_world") else 10
        self.substeps = max(1, int(substeps))
        load_path = self._model_path
        if model_xml is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".xml", encoding="utf-8", delete=False) as model_file:
                model_file.write(str(model_xml))
                load_path = Path(model_file.name)
        try:
            self._engine = self._module.MujocoBatch(
                str(load_path), str(library), int(env_count), int(worker_count)
            )
        finally:
            if model_xml is not None:
                try:
                    load_path.unlink()
                except OSError:
                    pass
        self.env_count = int(env_count)
        self.worker_count = int(self._engine.worker_count)
        self.control_scale: Any = 1.0
        self.nq = int(self._engine.nq)
        self.nv = int(self._engine.nv)
        self.nu = int(self._engine.nu)
        if self.nu == 0:
            raise RuntimeError(f"MuJoCo model {self.model_name!r} has no actuators")

    def reset_batch(self, lanes: Optional[Iterable[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        lane_ids = list(range(self.env_count)) if lanes is None else [int(i) for i in lanes]
        return self._engine.reset(lane_ids)

    def state_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        qpos, qvel = self._engine.state()
        return np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)

    def set_state_batch(self, qpos: Sequence[Sequence[float]], qvel: Sequence[Sequence[float]]) -> None:
        qpos_array = np.asarray(qpos, dtype=np.float64)
        qvel_array = np.asarray(qvel, dtype=np.float64)
        self._engine.set_state(qpos_array, qvel_array)

    def step_controls(self, controls: Sequence[Sequence[float]], *, substeps: Optional[int] = None):
        values = np.asarray(controls, dtype=np.float64)
        if values.ndim == 1 and self.env_count == 1:
            values = values.reshape(1, -1)
        if values.shape != (self.env_count, self.nu):
            raise ValueError(f"controls must have shape ({self.env_count}, {self.nu}), got {values.shape}")
        scale = np.asarray(self.control_scale, dtype=np.float64)
        if scale.ndim == 0:
            values = values * float(scale)
        elif scale.shape == (self.env_count,):
            values = values * scale[:, None]
        else:
            raise ValueError("control_scale must be scalar or have one value per environment")
        qpos, qvel = self._engine.step(values, self.substeps if substeps is None else max(1, int(substeps)))
        return np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, float]:
        del seed  # The selected task owns its seeded reset randomization.
        self.reset_batch([0])
        return self.state()

    def step(self, state: Dict[str, float], action: int) -> Dict[str, float]:
        if state and state != self.state():
            self.import_state(state)
        controls = self.action_controls(int(action))
        qpos, qvel = self.step_controls([controls])
        return self._state_dict(qpos[0], qvel[0])

    def state(self) -> Dict[str, float]:
        qpos, qvel = self.state_batch()
        return self._state_dict(qpos[0], qvel[0])

    def import_state(self, state: Dict[str, float]) -> None:
        qpos, qvel = self.state_batch()
        if self.model_name in ("point_mass", "embodied_point_world"):
            qpos[0, :2] = [float(state.get("x", 0.0)), float(state.get("y", 0.0))]
            qvel[0, :2] = [float(state.get("vx", 0.0)), float(state.get("vy", 0.0))]
        elif self.model_name in ("robot_reacher", "robot_reacher_world"):
            qpos[0, :2] = [float(state.get("theta1", 0.0)), float(state.get("theta2", 0.0))]
            qvel[0, :2] = [float(state.get("dtheta1", 0.0)), float(state.get("dtheta2", 0.0))]
        elif self.model_name == "world_3d":
            qpos[0, :2] = [float(state.get("x", 0.0)), float(state.get("z", 0.0))]
            qvel[0, :2] = [float(state.get("vx", 0.0)), float(state.get("vz", 0.0))]
        elif self.model_name == "warehouse_3d_world":
            qpos[0, :3] = [float(state.get("x", 0.0)), float(state.get("z", 0.0)), float(state.get("yaw", 0.0))]
            qvel[0, :3] = [float(state.get("vx", 0.0)), float(state.get("vz", 0.0)), float(state.get("yaw_rate", 0.0))]
        self.set_state_batch(qpos, qvel)

    def action_controls(self, action: int) -> List[float]:
        if self.model_name in ("point_mass", "embodied_point_world"):
            controls = [0.0, 0.0]
            mapping = {1: (0, 1.5), 2: (0, -1.5), 3: (1, 1.5), 4: (1, -1.5)}
            if int(action) in mapping:
                index, amount = mapping[int(action)]
                controls[index] = amount
            return controls
        if self.model_name in ("robot_reacher", "robot_reacher_world"):
            controls = [0.0, 0.0]
            mapping = {1: (0, 3.0), 2: (0, -3.0), 3: (1, 3.0), 4: (1, -3.0)}
            if int(action) in mapping:
                index, amount = mapping[int(action)]
                controls[index] = amount
            return controls
        if self.model_name == "world_3d":
            directions = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
            if not 0 <= int(action) < len(directions):
                return [0.0, 0.0]
            dx, dz = directions[int(action)]
            return [dx * 2.0, dz * 2.0]
        if self.model_name == "warehouse_3d_world":
            controls = [0.0, 0.0, 0.0]
            if int(action) in (1, 2):
                yaw = float(self.state().get("yaw", 0.0))
                direction = 1.0 if int(action) == 1 else -1.0
                controls[0] = math.sin(yaw) * 3.0 * direction
                controls[1] = math.cos(yaw) * 3.0 * direction
            elif int(action) in (3, 4):
                controls[2] = 2.0 if int(action) == 3 else -2.0
            return controls
        raise ValueError(f"No action mapping for model {self.model_name!r}")

    def _state_dict(self, qpos: Sequence[float], qvel: Sequence[float]) -> Dict[str, float]:
        if self.model_name in ("point_mass", "embodied_point_world"):
            return {"x": float(qpos[0]), "y": float(qpos[1]), "vx": float(qvel[0]), "vy": float(qvel[1])}
        if self.model_name in ("robot_reacher", "robot_reacher_world"):
            return {
                "theta1": float(qpos[0]), "theta2": float(qpos[1]),
                "dtheta1": float(qvel[0]), "dtheta2": float(qvel[1]),
            }
        if self.model_name == "world_3d":
            return {"x": float(qpos[0]), "z": float(qpos[1]), "vx": float(qvel[0]), "vz": float(qvel[1])}
        return {"x": float(qpos[0]), "z": float(qpos[1]), "yaw": float(qpos[2]), "vx": float(qvel[0]), "vz": float(qvel[1]), "yaw_rate": float(qvel[2])}

    def close(self) -> None:
        # Releasing this object destroys the C++ model, lanes, and worker pool.
        self._engine = None


__all__ = ["NativeEngineUnavailableError", "NativePhysicsBackend", "build_reacher_mjcf"]
