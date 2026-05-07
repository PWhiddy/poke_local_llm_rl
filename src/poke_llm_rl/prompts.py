from __future__ import annotations

import numpy as np

from poke_llm_rl.actions import system_output_format
from poke_llm_rl.state import EmulatorState


def quantize_frame(frame: np.ndarray, buckets: int) -> np.ndarray:
    clipped = np.clip(frame.astype(np.float32), 0, 255)
    scaled = np.floor(clipped / 256.0 * buckets).astype(np.int32)
    return np.clip(scaled, 0, buckets - 1)


def build_prompt_text(
    state: EmulatorState,
    previous_action: str,
    round_idx: int,
    max_buttons: int,
) -> str:
    level_text = ",".join(str(level) for level in state.party_levels[: state.party_size]) or "none"
    return (
        "You are controlling Pokemon Red in an emulator.\n"
        "The current game screen image is attached to this prompt.\n"
        "Think briefly, then choose a short button sequence that improves exploration and progress.\n"
        "Goals:\n"
        "1. Reach unseen tiles.\n"
        "2. Flip new event flags.\n"
        "Constraints:\n"
        f"- At most {max_buttons} buttons this turn.\n"
        "- Keep the thought to one or two short sentences.\n"
        "- Use only legal Game Boy buttons.\n"
        f"{system_output_format(max_buttons)}\n\n"
        f"Round: {round_idx}\n"
        f"Map id: {state.map_id}\n"
        f"Map name: {state.map_name}\n"
        f"Position: ({state.x}, {state.y})\n"
        f"Badges: {state.badges}\n"
        f"Party size: {state.party_size}\n"
        f"Party levels: {level_text}\n"
        f"HP fraction: {state.hp_fraction:.3f}\n"
        f"Event flags on: {state.event_flag_count}\n"
        f"Previous action: {previous_action}\n"
    )
