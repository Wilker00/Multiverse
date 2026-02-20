import time
import unittest

from memory.sample_weighter import ReplayWeightConfig, compute_sample_weight


class TestSampleWeighter(unittest.TestCase):
    def test_staleness_and_confidence_penalty(self):
        now = int(time.time() * 1000)
        cfg = ReplayWeightConfig(enabled=True, staleness_half_life=1000.0, confidence_penalty=0.8)

        fresh = {
            "t_ms": now,
            "policy_age_steps": 0,
            "info": {"safe_executor": {"confidence": 0.5, "danger": 0.1}},
        }
        stale_risky = {
            "t_ms": now - 100000,
            "policy_age_steps": 100000,
            "info": {"safe_executor": {"confidence": 0.99, "danger": 0.95, "rewound": True}},
        }

        w1 = compute_sample_weight(fresh, cfg=cfg, now_ms=now)
        w2 = compute_sample_weight(stale_risky, cfg=cfg, now_ms=now)
        self.assertGreater(w1, w2)


if __name__ == "__main__":
    unittest.main()

