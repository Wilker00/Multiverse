import json
import os
import tempfile
import unittest

from core.types import AgentSpec, VerseSpec
from orchestrator.trainer import Trainer


class TestTrainerSFTransferDefaultTrain(unittest.TestCase):
    def test_sf_transfer_trains_without_explicit_train_flag(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = Trainer(run_root=td)
            verse_spec = VerseSpec(
                spec_version="v1",
                verse_name="grid_world",
                verse_version="0.1",
                seed=7,
                params={
                    "width": 6,
                    "height": 6,
                    "max_steps": 20,
                    "adr_enabled": False,
                    "enable_ego_grid": True,
                    "ego_grid_size": 5,
                },
            )
            agent_spec = AgentSpec(
                spec_version="v1",
                policy_id="sf_default_train",
                policy_version="0.1",
                algo="sf_transfer",
                seed=7,
                # No config["train"] on purpose.
                config={"allowed_actions": [0, 1, 2, 3]},
            )
            res = trainer.run(
                verse_spec=verse_spec,
                agent_spec=agent_spec,
                episodes=2,
                max_steps=20,
                seed=7,
                verbose=False,
            )
            run_id = str(res["run_id"])
            metrics_path = os.path.join(td, run_id, "metrics.jsonl")
            self.assertTrue(os.path.isfile(metrics_path))
            with open(metrics_path, "r", encoding="utf-8") as f:
                rows = [json.loads(ln) for ln in f if ln.strip()]
            self.assertGreaterEqual(len(rows), 1)
            self.assertTrue(any("updates" in r for r in rows if isinstance(r, dict)))


if __name__ == "__main__":
    unittest.main()
