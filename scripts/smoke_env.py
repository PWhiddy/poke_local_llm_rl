from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from poke_llm_rl.actions import parse_completion
from poke_llm_rl.config import load_experiment_config
from poke_llm_rl.env import PokemonRedEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Pokemon Red PyBoy environment.")
    parser.add_argument("--config", default="configs/base.toml")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    env = PokemonRedEnv(config.env, config.rom_path, config.init_state_path, config.map_data_path, env_id="smoke")
    try:
        state = env.reset(round_horizon=5)
        print(f"state.map_id={state.map_id}")
        print(f"state.map_name={state.map_name}")
        print(f"state.position=({state.x}, {state.y})")
        print(f"state.badges={state.badges}")
        print(f"state.party_size={state.party_size}")
        print(f"state.party_levels={state.party_levels[:state.party_size]}")
        print(f"state.hp_fraction={state.hp_fraction:.4f}")
        print(f"state.event_flag_count={state.event_flag_count}")
        print(f"state.frame_shape={tuple(state.frame.shape)}")

        parsed = parse_completion(
            "<think>Test a short legal action sequence.</think><actions>up,a</actions>",
            config.env.max_buttons_per_turn,
        )
        step = env.step(parsed)
        print(f"step.done={step.done}")
        print(f"step.round_idx={step.info['round_idx']}")
        print(f"step.buttons={step.info['buttons']}")
        print(f"step.reward_total={step.reward.total:.4f}")
        print(f"step.reward_tile={step.reward.unique_tile_reward:.4f}")
        print(f"step.reward_event={step.reward.event_flag_reward:.4f}")
        print(f"step.next_map_id={step.state.map_id}")
        print(f"step.next_map_name={step.state.map_name}")
        print(f"step.next_position=({step.state.x}, {step.state.y})")
        print(f"step.screenshot_path={step.info['screenshot_path']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
