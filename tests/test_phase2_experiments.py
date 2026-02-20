import unittest

from experiments.chaos_testing import model_as_markov_chain
from experiments.multi_agent_coordination import mutual_information


class TestPhase2Experiments(unittest.TestCase):
    def test_mutual_information_non_negative(self):
        xs = [0, 0, 1, 1, 2, 2]
        ys = [1, 1, 2, 2, 0, 0]
        mi = mutual_information(xs, ys)
        self.assertGreaterEqual(float(mi), 0.0)

    def test_markov_model_outputs(self):
        states = [0, 0, 1, 2, 3, 0, 0, 1, 3, 0]
        out = model_as_markov_chain(states)
        self.assertIn("availability", out)
        self.assertIn("transition_matrix", out)
        self.assertGreaterEqual(float(out["availability"]), 0.0)
        self.assertLessEqual(float(out["availability"]), 1.0)
        tm = out["transition_matrix"]
        self.assertEqual(len(tm), 4)
        for row in tm:
            self.assertAlmostEqual(sum(float(x) for x in row), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
