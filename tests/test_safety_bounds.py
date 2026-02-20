import json
import os
import tempfile
import unittest

from theory.safety_bounds import (
    derive_safety_certificate,
    extract_episode_violation_flags_from_events,
    hoeffding_epsilon,
)


class TestSafetyBounds(unittest.TestCase):
    def test_hoeffding_epsilon_decreases_with_more_data(self):
        eps_20 = hoeffding_epsilon(num_episodes=20, confidence=0.95)
        eps_200 = hoeffding_epsilon(num_episodes=200, confidence=0.95)
        self.assertGreater(eps_20, eps_200)

    def test_certificate_from_flags(self):
        flags = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        cert = derive_safety_certificate(violation_flags=flags, confidence=0.95)
        self.assertEqual(cert["episodes"], 10)
        self.assertEqual(cert["observed_violations"], 2)
        self.assertAlmostEqual(cert["observed_violation_rate"], 0.2, places=8)
        self.assertGreaterEqual(cert["upper_bound"], cert["observed_violation_rate"])
        self.assertLessEqual(cert["lower_bound"], cert["observed_violation_rate"])

    def test_extract_flags_from_events(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "events.jsonl")
            rows = [
                {"episode_id": "ep1", "reward": -1, "info": {"fell_cliff": False}},
                {"episode_id": "ep1", "reward": -120, "info": {"fell_cliff": False}},
                {"episode_id": "ep2", "reward": -1, "info": {"fell_cliff": False}},
                {"episode_id": "ep2", "reward": -1, "info": {"fell_cliff": False}},
            ]
            with open(p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            out = extract_episode_violation_flags_from_events(events_jsonl_path=p)
            self.assertEqual(out["episodes"], 2)
            self.assertEqual(out["observed_violations"], 1)
            self.assertEqual(out["violation_flags"], [True, False])


if __name__ == "__main__":
    unittest.main()
