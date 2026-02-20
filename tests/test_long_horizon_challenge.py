import unittest

from core.types import VerseSpec
from verses.long_horizon_challenge import LongHorizonChallengeFactory


def _scripted_action(obs, params):
    pos = int(obs.get("pos", 0))
    has_key = int(obs.get("has_key", 0)) > 0
    door = int(obs.get("door_unlocked", 0)) > 0
    cp_idx = int(obs.get("checkpoint_idx", 0))

    key_pos = int(params.get("key_pos", 6))
    door_pos = int(params.get("door_pos", 14))
    checkpoints = [
        int(params.get("checkpoint_1", 18)),
        int(params.get("checkpoint_2", 22)),
        int(params.get("checkpoint_3", 26)),
    ]
    treasure = int(params.get("treasure_pos", 29))

    if not has_key:
        if pos < key_pos:
            return 1
        if pos > key_pos:
            return 0
        return 2
    if not door:
        if pos < door_pos:
            return 1
        if pos > door_pos:
            return 0
        return 2
    if cp_idx < len(checkpoints):
        target = checkpoints[cp_idx]
        if pos < target:
            return 1
        if pos > target:
            return 0
        return 1
    if pos < treasure:
        return 1
    if pos > treasure:
        return 0
    return 2


class TestLongHorizonChallenge(unittest.TestCase):
    def test_reset_and_step_contract(self):
        spec = VerseSpec(
            spec_version="v1",
            verse_name="long_horizon_challenge",
            verse_version="0.1",
            seed=7,
            params={"max_steps": 80},
        )
        verse = LongHorizonChallengeFactory().create(spec)
        rr = verse.reset()
        self.assertIsInstance(rr.obs, dict)
        sr = verse.step(1)
        self.assertIsInstance(sr.obs, dict)
        self.assertIn("checkpoint_idx", sr.obs)
        self.assertIn("current_subtask", sr.info)
        verse.close()

    def test_scripted_policy_reaches_goal(self):
        params = {
            "max_steps": 120,
            "key_pos": 6,
            "door_pos": 14,
            "checkpoint_1": 18,
            "checkpoint_2": 22,
            "checkpoint_3": 26,
            "treasure_pos": 29,
            "final_reward": 100.0,
        }
        spec = VerseSpec(
            spec_version="v1",
            verse_name="long_horizon_challenge",
            verse_version="0.1",
            seed=11,
            params=params,
        )
        verse = LongHorizonChallengeFactory().create(spec)
        rr = verse.reset()
        obs = rr.obs
        reached = False
        for _ in range(int(params["max_steps"])):
            a = _scripted_action(obs, params)
            sr = verse.step(a)
            obs = sr.obs
            if bool((sr.info or {}).get("reached_goal", False)):
                reached = True
                break
            if bool(sr.done or sr.truncated):
                break
        verse.close()
        self.assertTrue(reached)


if __name__ == "__main__":
    unittest.main()
