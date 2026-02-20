import unittest

from core.types import VerseSpec
from verses.wind_master_world import WindMasterWorldFactory


class TestWindMasterWorld(unittest.TestCase):
    def test_centerline_progress_reaches_goal(self):
        spec = VerseSpec(
            spec_version="v1",
            verse_name="wind_master_world",
            verse_version="0.1",
            seed=7,
            params={
                "width": 10,
                "height": 7,
                "max_steps": 40,
                "gust_probability": 0.0,
                "target_margin": 2,
            },
        )
        verse = WindMasterWorldFactory().create(spec)
        rr = verse.reset()
        obs = rr.obs
        reached = False
        for _ in range(40):
            sr = verse.step(3)  # right
            obs = sr.obs
            if bool((sr.info or {}).get("reached_goal", False)):
                reached = True
                self.assertFalse(bool((sr.info or {}).get("unsafe_finish", False)))
                break
            if bool(sr.done or sr.truncated):
                break
        verse.close()
        self.assertTrue(reached)
        self.assertGreaterEqual(int(obs.get("safety_margin", 0)), 2)

    def test_edge_path_is_risky(self):
        spec = VerseSpec(
            spec_version="v1",
            verse_name="wind_master_world",
            verse_version="0.1",
            seed=11,
            params={"width": 8, "height": 7, "max_steps": 20, "gust_probability": 0.0, "target_margin": 2},
        )
        verse = WindMasterWorldFactory().create(spec)
        verse.reset()
        risky = False
        # move to top edge
        for _ in range(5):
            sr = verse.step(0)  # up
            if bool((sr.info or {}).get("high_risk_failure", False)):
                risky = True
                break
        verse.close()
        self.assertTrue(risky)


if __name__ == "__main__":
    unittest.main()
