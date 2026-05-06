import unittest

import numpy as np

from poke_llm_rl.actions import ParsedAction
from poke_llm_rl.config import RewardConfig
from poke_llm_rl.reward import RewardTracker
from poke_llm_rl.state import EmulatorState


def make_state(map_id: int, x: int, y: int, event_flags: int) -> EmulatorState:
    return EmulatorState(
        frame=np.zeros((2, 2), dtype=np.int32),
        map_id=map_id,
        map_name=f"Map {map_id}",
        x=x,
        y=y,
        badges=0,
        party_size=1,
        party_levels=[5],
        hp_fraction=1.0,
        event_flag_count=event_flags,
        event_flags=[],
    )


class RewardTests(unittest.TestCase):
    def test_reward_counts_new_tile_and_event(self) -> None:
        tracker = RewardTracker(RewardConfig(unique_tile_weight=0.5, event_flag_weight=2.0))
        tracker.reset(make_state(1, 1, 1, 10))
        reward = tracker.score_transition(
            make_state(1, 1, 1, 10),
            make_state(1, 2, 1, 12),
            ParsedAction(raw_completion="", thought="", buttons=["up"], valid=True),
        )
        self.assertEqual(reward.unique_tile_reward, 0.5)
        self.assertEqual(reward.event_flag_reward, 4.0)
        self.assertEqual(reward.total, 4.5)


if __name__ == "__main__":
    unittest.main()
