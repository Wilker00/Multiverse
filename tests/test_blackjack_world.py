import unittest

from core.types import VerseSpec
from verses.registry import create_verse, register_builtin


class TestBlackjackWorld(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin()

    def _create_verse(self, **params):
        spec = VerseSpec(
            spec_version="v1",
            verse_name="blackjack_world",
            verse_version="0.1",
            seed=123,
            params=params,
        )
        return create_verse(spec)

    def test_ace_ten_is_not_a_split_pair(self):
        verse = self._create_verse()
        verse.reset()
        verse.import_state(
            {
                "done": False,
                "t": 1,
                "shoe": [2, 3, 4],
                "shoe_pos": 0,
                "dealer_hand": [6, 10],
                "player_hands": [[10, 11]],
                "hand_bets": [1.0],
                "hand_split_aces": [False],
                "hand_done": [False],
                "active_hand": 0,
                "pending_terminal": False,
                "pending_reward": 0.0,
                "pending_info": {},
            }
        )

        self.assertEqual(verse.legal_actions(), [1])

    def test_split_hand_twenty_one_auto_advances(self):
        verse = self._create_verse()
        verse.reset()
        verse.import_state(
            {
                "done": False,
                "t": 0,
                "shoe": [11, 5, 2],
                "shoe_pos": 0,
                "dealer_hand": [6, 10],
                "player_hands": [[10, 10]],
                "hand_bets": [1.0],
                "hand_split_aces": [False],
                "hand_done": [False],
                "active_hand": 0,
                "pending_terminal": False,
                "pending_reward": 0.0,
                "pending_info": {},
            }
        )

        sr = verse.step(3)

        self.assertFalse(sr.done)
        self.assertEqual(sr.obs["active_hand"], 1)
        self.assertEqual(sr.obs["player_sum"], 15)
        self.assertEqual(sr.obs["num_hands"], 2)

    def test_blackjack_world_is_not_randomized_by_default(self):
        verse = self._create_verse(blackjack_payout=1.5, step_penalty=0.0)

        self.assertAlmostEqual(float(verse.params.blackjack_payout), 1.5)
        self.assertAlmostEqual(float(verse.params.step_penalty), 0.0)

    def test_reset_exposes_visible_count_and_hand_structure_features(self):
        verse = self._create_verse(adr_enabled=False)
        rr = verse.reset()

        visible_cards = list(rr.info["player_hand"]) + [int(rr.info["dealer_showing"])]

        def hi_lo(card: int) -> int:
            if 2 <= int(card) <= 6:
                return 1
            if int(card) in (10, 11):
                return -1
            return 0

        self.assertIn("hand_len", rr.obs)
        self.assertIn("pair_rank", rr.obs)
        self.assertIn("running_count", rr.obs)
        self.assertIn("true_count", rr.obs)
        self.assertIn("cards_remaining", rr.obs)
        self.assertEqual(int(rr.obs["hand_len"]), 2)
        self.assertEqual(int(rr.obs["first_action"]), 1)
        self.assertEqual(int(rr.obs["cards_remaining"]), 312 - 4)
        self.assertEqual(int(rr.obs["running_count"]), sum(hi_lo(card) for card in visible_cards))


if __name__ == "__main__":
    unittest.main()
