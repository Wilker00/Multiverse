import unittest

from tools.train_agent import _parse_kv_list


class TestTrainAgentKVParse(unittest.TestCase):
    def test_parse_json_list_and_dict(self):
        out = _parse_kv_list(
            [
                "allowed_actions=[0,1,2,3]",
                'replay_weighting={"enabled": true, "alpha": 0.5}',
            ]
        )
        self.assertEqual(out["allowed_actions"], [0, 1, 2, 3])
        self.assertIsInstance(out["replay_weighting"], dict)
        self.assertEqual(bool(out["replay_weighting"]["enabled"]), True)
        self.assertAlmostEqual(float(out["replay_weighting"]["alpha"]), 0.5, places=6)

    def test_parse_scalar_types_still_works(self):
        out = _parse_kv_list(["lr=0.01", "epochs=10", "train=true", "name=abc"])
        self.assertAlmostEqual(float(out["lr"]), 0.01, places=8)
        self.assertEqual(int(out["epochs"]), 10)
        self.assertEqual(bool(out["train"]), True)
        self.assertEqual(str(out["name"]), "abc")


if __name__ == "__main__":
    unittest.main()
