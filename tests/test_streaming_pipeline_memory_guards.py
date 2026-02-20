import unittest
from unittest.mock import patch

from core.streaming_pipeline import RealTimeAggregator, _InMemoryBus


class TestInMemoryBusGuards(unittest.TestCase):
    def test_drop_oldest_keeps_latest_messages(self):
        bus = _InMemoryBus(max_queue_size=3, overflow_policy="drop_oldest")
        for i in range(5):
            bus.put("topic_x", {"i": i})

        got = [bus.get("topic_x", timeout=0.01) for _ in range(3)]
        self.assertEqual([x["i"] for x in got if x is not None], [2, 3, 4])

        stats = bus.stats()
        self.assertEqual(stats["queue_sizes"].get("topic_x"), 0)
        self.assertEqual(stats["dropped_by_topic"].get("topic_x"), 2)

    def test_drop_newest_keeps_earliest_messages(self):
        bus = _InMemoryBus(max_queue_size=3, overflow_policy="drop_newest")
        for i in range(5):
            bus.put("topic_y", {"i": i})

        got = [bus.get("topic_y", timeout=0.01) for _ in range(3)]
        self.assertEqual([x["i"] for x in got if x is not None], [0, 1, 2])

        stats = bus.stats()
        self.assertEqual(stats["dropped_by_topic"].get("topic_y"), 2)


class TestRealTimeAggregatorWindow(unittest.TestCase):
    def test_prunes_old_events_and_tracks_mean(self):
        agg = RealTimeAggregator(window_seconds=5)
        self.assertEqual(type(agg._events).__name__, "deque")

        with patch("core.streaming_pipeline.time.time", side_effect=[100.0, 101.0, 107.0]):
            agg.consume({"reward": 1.0})
            agg.consume({"reward": 3.0})
            agg.consume({"reward": 5.0})

        snap = agg.snapshot()
        self.assertEqual(snap["count"], 1)
        self.assertAlmostEqual(snap["mean_reward"], 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
