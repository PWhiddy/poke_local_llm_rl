from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from poke_llm_rl.actions import ParsedAction
from poke_llm_rl.config import EnvConfig
from poke_llm_rl.prompts import quantize_frame
from poke_llm_rl.reward import RewardBreakdown, RewardTracker
from poke_llm_rl.state import EmulatorState, extract_emulator_state

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:  # pragma: no cover
    PyBoy = Any  # type: ignore[assignment]
    PYBOY_AVAILABLE = False


DEFAULT_FALLBACK_ACTION = ["a"]


@dataclass(slots=True)
class StepResult:
    state: EmulatorState
    reward: RewardBreakdown
    done: bool
    info: dict[str, Any]


class PokemonRedEnv:
    def __init__(
        self,
        config: EnvConfig,
        rom_path: str | Path,
        init_state_path: str | Path,
        map_data_path: str | Path = "assets/map_data.json",
        env_id: str = "env0",
    ):
        if not PYBOY_AVAILABLE:
            raise ImportError("PyBoy is not installed. Install project dependencies before running the emulator.")

        self.config = config
        self.rom_path = Path(rom_path)
        self.init_state_path = Path(init_state_path)
        self.map_data_path = Path(map_data_path)
        self.env_id = env_id
        self.env_state_dir = Path("env_state")
        self.env_state_dir.mkdir(parents=True, exist_ok=True)
        self.pyboy = PyBoy(
            str(self.rom_path),
            window="null" if config.headless else "SDL2",
        )
        self.reward_tracker = RewardTracker(config.reward)
        self.previous_action = "none"
        self.round_idx = 0
        self.round_horizon = config.initial_round_horizon

    def close(self) -> None:
        self.pyboy.stop(save=False)

    def reset(self, round_horizon: int | None = None) -> EmulatorState:
        with self.init_state_path.open("rb") as handle:
            self.pyboy.load_state(handle)
        self.round_idx = 0
        self.previous_action = "none"
        if round_horizon is not None:
            self.round_horizon = round_horizon
        state = self.current_state()
        self.reward_tracker.reset(state)
        return state

    def current_state(self) -> EmulatorState:
        rgba = np.array(self.pyboy.screen.ndarray[..., 0], copy=True)
        pooled = rgba.reshape(
            rgba.shape[0] // self.config.downsample_factor,
            self.config.downsample_factor,
            rgba.shape[1] // self.config.downsample_factor,
            self.config.downsample_factor,
        ).mean(axis=(1, 3))
        frame = quantize_frame(pooled, self.config.grayscale_buckets)
        return extract_emulator_state(frame=frame, memory_reader=self.read_memory, map_data_path=self.map_data_path)

    def read_memory(self, address: int) -> int:
        return int(self.pyboy.memory[address])

    def snapshot(self) -> bytes:
        buffer = io.BytesIO()
        self.pyboy.save_state(buffer)
        return buffer.getvalue()

    def load_snapshot(self, snapshot_bytes: bytes) -> None:
        self.pyboy.load_state(io.BytesIO(snapshot_bytes))

    def _press_button(self, button: str) -> None:
        self.pyboy.button_press(button)
        for _ in range(self.config.action_hold_frames):
            self.pyboy.tick()
        self.pyboy.button_release(button)
        for _ in range(self.config.frames_per_button):
            self.pyboy.tick()

    def save_screenshot(self) -> Path:
        screenshot_path = self.env_state_dir / f"{self.env_id}_screen.png"
        frame = np.array(self.pyboy.screen.ndarray, copy=True)
        Image.fromarray(frame).save(screenshot_path)
        return screenshot_path

    def step(self, parsed_action: ParsedAction) -> StepResult:
        prior_state = self.current_state()
        buttons = parsed_action.buttons if parsed_action.valid else DEFAULT_FALLBACK_ACTION
        for button in buttons[: self.config.max_buttons_per_turn]:
            self._press_button(button)
        screenshot_path = self.save_screenshot()
        self.round_idx += 1
        self.previous_action = ",".join(buttons)
        current_state = self.current_state()
        reward = self.reward_tracker.score_transition(prior_state, current_state, parsed_action)
        done = self.round_idx >= self.round_horizon
        return StepResult(
            state=current_state,
            reward=reward,
            done=done,
            info={"round_idx": self.round_idx, "buttons": buttons, "screenshot_path": str(screenshot_path)},
        )
