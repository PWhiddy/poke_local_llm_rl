from __future__ import annotations

import re
from dataclasses import dataclass

VALID_BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")
# think optional
ACTION_PATTERN = re.compile(
    r"(?:<think>(?P<think>.*?)</think>\s*)?<actions>(?P<actions>.*?)</actions>",
    re.DOTALL | re.IGNORECASE,
)

@dataclass(slots=True)
class ParsedAction:
    raw_completion: str
    thought: str
    buttons: list[str]
    valid: bool
    error: str | None = None


def system_output_format(max_buttons: int) -> str:
    return (
        "Valid buttons: up, down, left, right, a, b\n" # excluding start & select for now
        "Respond in exactly this format:\n"
        "<think>put only 1-2 short sentences of reasoning about what youre trying to do in the game</think>\n"
        "<actions>put comma separated buttons here (up to 8)</actions>"
    )


def parse_completion(completion: str, max_buttons: int) -> ParsedAction:
    match = ACTION_PATTERN.search(completion.strip())
    if not match:
        return ParsedAction(
            raw_completion=completion,
            thought="",
            buttons=[],
            valid=False,
            error="missing_required_tags",
        )

    thought_match = match.group("think")
    thought = " ".join(thought_match.split()) if thought_match else ""
    action_text = match.group("actions").strip().lower()
    buttons = [token.strip() for token in action_text.split(",") if token.strip()]

    if not buttons:
        return ParsedAction(completion, thought, [], False, "empty_action_list")
    if len(buttons) > max_buttons:
        return ParsedAction(completion, thought, buttons[:max_buttons], False, "too_many_buttons")

    invalid = [button for button in buttons if button not in VALID_BUTTONS]
    if invalid:
        return ParsedAction(completion, thought, buttons, False, f"invalid_buttons:{','.join(invalid)}")

    return ParsedAction(completion, thought, buttons, True, None)

