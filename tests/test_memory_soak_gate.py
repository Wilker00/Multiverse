import unittest

from tools.memory_soak_gate import _linear_slope_mb_per_hour


class TestMemorySoakGate(unittest.TestCase):
    def test_linear_slope_flat(self):
        pts = [(0.0, 100.0), (10.0, 100.0), (20.0, 100.0)]
        slope = _linear_slope_mb_per_hour(pts)
        self.assertAlmostEqual(float(slope), 0.0, places=6)

    def test_linear_slope_positive(self):
        # +2MB over 20 seconds -> 360 MB/hour.
        pts = [(0.0, 100.0), (10.0, 101.0), (20.0, 102.0)]
        slope = _linear_slope_mb_per_hour(pts)
        self.assertAlmostEqual(float(slope), 360.0, places=3)


if __name__ == "__main__":
    unittest.main()

