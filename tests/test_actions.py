import unittest

from poke_llm_rl.actions import parse_completion


class ActionParsingTests(unittest.TestCase):
    def test_parse_valid_completion(self) -> None:
        parsed = parse_completion(
            "<think>Move toward the exit and check progress.</think><actions>up,right,a</actions>",
            max_buttons=8,
        )
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.buttons, ["up", "right", "a"])

    def test_parse_rejects_bad_button(self) -> None:
        parsed = parse_completion(
            "<think>Try something.</think><actions>jump,a</actions>",
            max_buttons=8,
        )
        self.assertFalse(parsed.valid)
        self.assertEqual(parsed.error, "invalid_buttons:jump")


if __name__ == "__main__":
    unittest.main()
