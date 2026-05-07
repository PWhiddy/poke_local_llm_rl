from __future__ import annotations

from dataclasses import dataclass, field

from poke_llm_rl.actions import ParsedAction
from poke_llm_rl.config import RewardConfig
from poke_llm_rl.state import EmulatorState


@dataclass(slots=True)
class RewardBreakdown:
    total: float
    unique_tile_reward: float
    event_flag_reward: float
    formatting_reward: float
    behavior_reward: float


@dataclass(slots=True)
class RewardTracker:
    config: RewardConfig
    seen_tiles: set[tuple[int, int, int]] = field(default_factory=set)
    max_event_flag_count: int = 0

    def reset(self, initial_state: EmulatorState) -> None:
        self.seen_tiles = {initial_state.unique_tile_key}
        self.max_event_flag_count = initial_state.event_flag_count

    def score_transition(
        self,
        previous_state: EmulatorState,
        current_state: EmulatorState,
        parsed_action: ParsedAction,
    ) -> RewardBreakdown:
        del previous_state

        unique_tile_reward = 0.0
        if current_state.unique_tile_key not in self.seen_tiles:
            self.seen_tiles.add(current_state.unique_tile_key)
            unique_tile_reward = self.config.unique_tile_weight

        event_gain = max(current_state.event_flag_count - self.max_event_flag_count, 0)
        if event_gain:
            self.max_event_flag_count = current_state.event_flag_count
        event_flag_reward = event_gain * self.config.event_flag_weight

        formatting_reward = 0.0 if parsed_action.valid else self.config.format_penalty

        behavior_reward = 0.0
        if not parsed_action.valid: #not parsed_action.buttons:
            #print(f"invalid actions: {parsed_action.buttons}")
            behavior_reward += self.config.noop_penalty
        else:
            print(f"sucsessfully parsed actions: {parsed_action.buttons}")
        #if len(parsed_action.buttons) >= 2 and len(set(parsed_action.buttons)) == 1:
        #    behavior_reward += self.config.repeated_action_penalty

        total = unique_tile_reward + event_flag_reward + formatting_reward + behavior_reward
        return RewardBreakdown(
            total=total,
            unique_tile_reward=unique_tile_reward,
            event_flag_reward=event_flag_reward,
            formatting_reward=formatting_reward,
            behavior_reward=behavior_reward,
        )

