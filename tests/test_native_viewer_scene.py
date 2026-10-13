from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.types import VerseSpec
from engine3d.viewer_scene import SceneFrame, SceneSnapshotChannel, scene_from_verse
from integrations.physics.domain_randomization import sample_domain_randomization
from integrations.physics.native_backend import build_reacher_mjcf
from verses.registry import create_verse, register_builtin


def test_scene_snapshot_adapters_cover_all_built_in_3d_verses():
    register_builtin()
    expected = {
        "world_3d": "terrain_agent",
        "embodied_point_world": "point_agent",
        "robot_reacher_world": "reacher_end_effector",
        "warehouse_3d_world": "warehouse_robot",
    }
    for name, selected in expected.items():
        verse = create_verse(VerseSpec(
            spec_version="v1", verse_name=name, verse_version="1", seed=11,
            params={"adr_enabled": False, "render_backend": "software"},
        ))
        try:
            verse.reset()
            frame = scene_from_verse(verse)
            assert frame.selected_name == selected
            assert frame.objects
            assert all(len(item.position) == 3 and len(item.scale) == 3 for item in frame.objects)
            rgb = np.asarray(verse.render("rgb_array"))
            assert rgb.ndim == 3 and rgb.shape[2] == 3
            assert rgb.dtype == np.uint8
        finally:
            verse.close()


def test_scene_mapping_connects_stable_names_to_mjcf_and_gltf_assets():
    asset_root = Path(__file__).resolve().parents[1] / "engine3d" / "assets"
    mapping = json.loads((asset_root / "scene_mappings.json").read_text(encoding="utf-8"))
    robot = mapping["bodies"]["warehouse_robot"]
    assert robot["mujoco_body"] == "warehouse_robot"
    assert (asset_root / robot["mesh"]).is_file()


def test_snapshot_channel_replaces_state_without_render_dependencies():
    channel = SceneSnapshotChannel()
    first = channel.publish(SceneFrame("first"))
    second = channel.publish(SceneFrame("latest"))
    sequence, frame = channel.read_latest()
    assert (first, second, sequence) == (1, 2, 2)
    assert frame is not None and frame.title == "latest"


def test_domain_randomization_is_seeded_and_opt_in():
    config = {"enabled": True, "actuator_gain": [0.8, 1.2], "goal_jitter": 0.1,
              "visual_color_jitter": 0.05}
    first = sample_domain_randomization(42, config)
    second = sample_domain_randomization(42, config)
    assert first == second
    assert 0.8 <= first.actuator_gain <= 1.2
    assert all(0.95 <= tint <= 1.05 for tint in first.visual_tint)
    assert sample_domain_randomization(42, False).actuator_gain == 1.0


def test_reacher_mjcf_uses_configured_link_lengths():
    import pytest
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(build_reacher_mjcf(0.35, 0.62))
    forearm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "forearm")
    end_effector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    assert model.body_pos[forearm, 0] == pytest.approx(0.35)
    assert model.site_pos[end_effector, 0] == pytest.approx(0.62)
