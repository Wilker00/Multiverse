import os
import tempfile
import unittest

from core.artifact_index import latest_artifact, load_artifact_index, register_artifact, summarize_artifact_index


class TestArtifactIndex(unittest.TestCase):
    def test_register_and_summarize_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            index_path = os.path.join(td, "artifact_index.json")
            register_artifact(
                artifact_type="agent_health_report",
                artifact_path=os.path.join(td, "health.json"),
                status="ok",
                created_at_iso="2026-04-02T00:00:00Z",
                metadata={"count": 2},
                index_path=index_path,
            )
            register_artifact(
                artifact_type="production_readiness_report",
                artifact_path=os.path.join(td, "readiness.json"),
                status="failed",
                created_at_iso="2026-04-02T00:01:00Z",
                metadata={"error_count": 1},
                index_path=index_path,
            )
            summary = summarize_artifact_index(index_path)
            self.assertTrue(bool(summary["exists"]))
            self.assertIn("agent_health_report", summary["artifact_types"])
            self.assertIn("production_readiness_report", summary["artifact_types"])
            idx = load_artifact_index(index_path)
            latest = latest_artifact(idx, "production_readiness_report")
            self.assertEqual(str(latest["status"]), "failed")
            self.assertEqual(int(latest["metadata"]["error_count"]), 1)


if __name__ == "__main__":
    unittest.main()
