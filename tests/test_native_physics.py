from __future__ import annotations

import os
import numpy as np
import pytest

from core.types import SpaceSpec, VerseSpec
from integrations.physics.native_backend import NativeEngineUnavailableError, NativePhysicsBackend
from integrations.physics.native_vectorized import NativeVectorizedVerse
from verses.embodied_point_world import EmbodiedPointWorldVerse
from verses.robot_reacher_world import RobotReacherWorldVerse


def _native_backend(*args, **kwargs):
    try:
        return NativePhysicsBackend(*args, **kwargs)
    except NativeEngineUnavailableError as exc:
        if os.environ.get("MULTIVERSE_REQUIRE_NATIVE_TESTS") == "1":
            pytest.fail(f"native CI build is required but unavailable: {exc}")
        pytest.skip(str(exc))


def _require_native_runtime():
    backend = _native_backend("point_mass")
    backend.close()


def test_native_model_data_is_independent_and_has_seedable_reset():
    backend = _native_backend("point_mass", env_count=2, worker_count=2)
    try:
        backend.reset_batch()
        qpos, qvel = backend.step_controls([[1.5, 0.0], [-1.5, 0.0]])
        assert qpos.shape == (2, 2)
        assert qvel.shape == (2, 2)
        assert qpos[0, 0] > 0.0
        assert qpos[1, 0] < 0.0
        assert not np.allclose(qpos[0], qpos[1])

        backend.reset_batch([0])
        reset_qpos, reset_qvel = backend.state_batch()
        assert np.allclose(reset_qpos[0], 0.0)
        assert np.allclose(reset_qvel[0], 0.0)
        assert not np.allclose(reset_qpos[1], 0.0)
    finally:
        backend.close()


def test_native_vector_lanes_reset_and_checkpoint_independently():
    spec = VerseSpec(
        spec_version="v1", verse_name="embodied_point_world", verse_version="1",
        params={"sim_provider": "multiverse_native", "adr_enabled": False},
    )
    try:
        vector = NativeVectorizedVerse(spec, 2, worker_count=2)
    except NativeEngineUnavailableError as exc:
        if os.environ.get("MULTIVERSE_REQUIRE_NATIVE_TESTS") == "1":
            pytest.fail(f"native CI build is required but unavailable: {exc}")
        pytest.skip(str(exc))
    try:
        vector.reset_batch([1, 2])
        vector.step_batch([1, 2])
        before = vector._observations()
        reset = vector.reset_lanes({0: 3})
        assert reset.observations[0][0:4] == [0.0, 0.0, 0.0, 0.0]
        assert np.allclose(reset.observations[1], before[1], atol=1e-5, rtol=0)

        saved = vector.export_state()
        restored = NativeVectorizedVerse(spec, 2, worker_count=2)
        try:
            restored.import_state(saved)
            assert np.allclose(restored._observations(), vector._observations(), atol=1e-5, rtol=0)
        finally:
            restored.close()
    finally:
        vector.close()


def test_native_point_scalar_and_vectorized_results_match():
    spec = VerseSpec(
        spec_version="v1", verse_name="embodied_point_world", verse_version="1", seed=31,
        observation_space=SpaceSpec(type="vector", shape=(6,), dtype="float32"),
        action_space=SpaceSpec(type="discrete", n=5),
        params={"sim_provider": "multiverse_native", "adr_enabled": False},
    )
    _require_native_runtime()
    vector = NativeVectorizedVerse(spec, 1, worker_count=1)
    scalar = EmbodiedPointWorldVerse(spec)
    try:
        vector_reset = vector.reset_batch([31])
        scalar_reset = scalar.reset()
        assert np.allclose(vector_reset.observations[0], scalar_reset.obs, atol=1e-5, rtol=0)
        for action in (1, 1, 3, 0, 2, 4, 3, 3, 1):
            vector_result = vector.step_batch([action])
            scalar_result = scalar.step(action)
            assert np.allclose(vector_result.observations[0], scalar_result.obs, atol=1e-5, rtol=0)
            assert vector_result.rewards[0] == pytest.approx(scalar_result.reward, abs=1e-5)
            assert vector_result.terminated[0] == scalar_result.done
            assert vector_result.truncated[0] == scalar_result.truncated
            assert vector_result.infos[0]["distance"] == pytest.approx(
                scalar_result.info["distance"], abs=1e-5
            )
    finally:
        vector.close()
        scalar.close()


def test_native_mjcf_models_load_without_graphics():
    for name in ("point_mass", "robot_reacher", "world_3d", "warehouse_3d_world"):
        backend = _native_backend(name)
        try:
            assert backend.nq > 0
            assert backend.nu > 0
            assert backend.state()
        finally:
            backend.close()


def test_native_joint_and_warehouse_contact_boundaries():
    point = _native_backend("point_mass")
    warehouse = _native_backend("warehouse_3d_world")
    try:
        point.set_state_batch([[0.99, 0.0]], [[0.0, 0.0]])
        point_qpos, _ = point.step_controls([[1.5, 0.0]], substeps=100)
        assert 0.98 <= float(point_qpos[0, 0]) <= 1.001

        warehouse.set_state_batch([[-0.3, 0.4, 0.0]], [[0.0, 0.0, 0.0]])
        warehouse_qpos, _ = warehouse.step_controls([[-3.0, 0.0, 0.0]], substeps=100)
        assert float(warehouse_qpos[0, 0]) >= -0.59
        assert float(warehouse_qpos[0, 0]) == pytest.approx(-0.575, abs=0.02)
    finally:
        point.close()
        warehouse.close()


def test_native_joint_limits_and_scalar_vector_reacher_parity():
    _require_native_runtime()
    scalar_spec = VerseSpec(
        spec_version="v1", verse_name="robot_reacher_world", verse_version="1",
        seed=19,
        observation_space=SpaceSpec(type="vector", shape=(8,), dtype="float32"),
        action_space=SpaceSpec(type="discrete", n=5),
        params={"sim_provider": "multiverse_native", "adr_enabled": False},
    )
    vector = NativeVectorizedVerse(scalar_spec, 1, worker_count=1)
    scalar = RobotReacherWorldVerse(scalar_spec)
    try:
        vector_reset = vector.reset_batch([19])
        scalar_reset = scalar.reset()
        assert np.allclose(vector_reset.observations[0], scalar_reset.obs, atol=1e-5, rtol=0)

        for action in (1, 3, 2, 4, 1, 0, 3, 3, 2, 2):
            vector_result = vector.step_batch([action])
            scalar_result = scalar.step(action)
            assert np.allclose(vector_result.observations[0], scalar_result.obs, atol=1e-5, rtol=0)
            assert vector_result.rewards[0] == pytest.approx(scalar_result.reward, abs=1e-5)
            assert vector_result.terminated[0] == scalar_result.done
            assert vector_result.truncated[0] == scalar_result.truncated
            assert vector_result.infos[0]["distance"] == pytest.approx(
                scalar_result.info["distance"], abs=1e-5
            )

        backend = _native_backend("robot_reacher")
        try:
            backend.set_state_batch([[3.13, 0.0]], [[0.0, 0.0]])
            qpos, _ = backend.step_controls([[4.0, 0.0]], substeps=100)
            assert abs(float(qpos[0, 0])) <= np.pi + 1e-3
        finally:
            backend.close()
    finally:
        vector.close()
        scalar.close()
