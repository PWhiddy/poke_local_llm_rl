import unittest

import numpy as np

from poke_llm_rl.prompts import build_prompt, frame_to_text
from poke_llm_rl.state import EmulatorState


class PromptTests(unittest.TestCase):
    def test_frame_to_text(self) -> None:
        frame = np.array([[0, 1, 15], [3, 4, 5]])
        self.assertEqual(frame_to_text(frame), "01F\n345")

    def test_build_prompt_contains_game_state(self) -> None:
        state = EmulatorState(
            frame=np.zeros((2, 3), dtype=np.int32),
            map_id=1,
            map_name="Viridian City",
            x=4,
            y=7,
            badges=0,
            party_size=1,
            party_levels=[5],
            hp_fraction=0.8,
            event_flag_count=12,
            event_flags=[0, 1],
        )
        prompt = build_prompt(state, previous_action="a", round_idx=3, max_buttons=8)
        self.assertIn("Map id: 1", prompt)
        self.assertIn("Map name: Viridian City", prompt)
        self.assertIn("Position: (4, 7)", prompt)
        self.assertIn("Previous action: a", prompt)


if __name__ == "__main__":
    unittest.main()
