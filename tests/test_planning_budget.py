import unittest

from core.planning_budget import PlanningBudget, PlanningBudgetConfig


class TestPlanningBudget(unittest.TestCase):
    def test_budget_enforcement(self):
        b = PlanningBudget(
            PlanningBudgetConfig(
                enabled=True,
                base_threshold=0.8,
                budget_per_episode=2,
                budget_per_minute=10,
            )
        )
        b.reset_episode()
        self.assertTrue(b.can_invoke(verse_name="grid_world", confidence=0.1))
        b.consume()
        self.assertTrue(b.can_invoke(verse_name="grid_world", confidence=0.1))
        b.consume()
        self.assertFalse(b.can_invoke(verse_name="grid_world", confidence=0.1))


if __name__ == "__main__":
    unittest.main()

