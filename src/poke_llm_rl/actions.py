from __future__ import annotations

import re
from dataclasses import dataclass

VALID_BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")
ACTION_PATTERN = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<actions>(?P<actions>.*?)</actions>",
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
        "Respond in exactly this format:\n"
        "<think>one or two short sentences of reasoning</think>\n"
        f"<actions>comma-separated buttons, 1 to {max_buttons} total, using only "
        "up,down,left,right,a,b,start,select</actions>"
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

    thought = " ".join(match.group("think").split())
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

