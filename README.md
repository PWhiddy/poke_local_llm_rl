# Pokemon Red LLM RL

This project fine-tunes a local `Qwen/Qwen3.5-0.8B` model to play Pokemon Red through `PyBoy`. The agent sees a compact pixel view of the screen plus structured RAM-derived state, thinks briefly, emits up to 8 button presses, and is trained online from emulator rewards.

## Design

- Model: `Qwen/Qwen3.5-0.8B` in BF16 with LoRA adapters.
- Environment: `PyBoy` running Pokemon Red headless.
- Observation:
  - Downsampled grayscale screen pixels, quantized to `0-F`, serialized as text.
  - Basic RAM state prepended as text: map id, coordinates, badges, party size, party levels, HP fraction, event-flag count, previous action.
- Action format:
  - The model must emit:
    - `<think>...</think>`
    - `<actions>up,right,a</actions>`
  - Up to 8 buttons per turn.
- Reward:
  - `unique_tile_weight * newly_seen_tile`
  - `event_flag_weight * newly_flipped_event_flags`
  - Small optional penalties for malformed output or degenerate repeated actions.
- Training:
  - Online sequence-level PPO for text actions.
  - LoRA-only updates for efficiency and stability.
  - Frozen reference model KL penalty.
  - Horizon curriculum starting at `500` rounds, then increasing over training.

## Why this stack

This keeps the control loop faithful to an LLM: the model reasons over a textualized world state and emits an action program. For a small 0.8B model, a pure Hugging Face generation/training loop is easier to keep on-policy than splitting inference and training across separate engines. The horizon curriculum and clipped sequence-level PPO are there to keep early training stable.

## Layout

- `configs/base.toml`: default experiment config.
- `src/poke_llm_rl/env.py`: PyBoy wrapper and control loop.
- `src/poke_llm_rl/state.py`: RAM extraction and event flag accounting.
- `src/poke_llm_rl/prompts.py`: screen and state serialization.
- `src/poke_llm_rl/reward.py`: reward tracker.
- `src/poke_llm_rl/trainer.py`: online RL trainer.
- `scripts/train_grpo.py`: main training entrypoint.
- `scripts/verify_rom.py`: verify the ROM SHA1.

## Setup

Use Python `3.11` or `3.12`. Python `3.13` is not recommended for this stack.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[train]
```

Place these assets locally before real training:

- `PokemonRed.gb`
- `assets/init_game.state`
- `assets/map_data.json`

The reference ROM SHA1 should be:

```text
ea9bcae617fdf159b045185467ae58b2e4a48b9a
```

Verify it with:

```bash
python scripts/verify_rom.py PokemonRed.gb
```

## Start training

```bash
python scripts/train_grpo.py --config configs/base.toml
```

## Notes

- The event-flag address range follows the classic `PokemonRedExperiments` convention: `0xD747` through `0xD886`, excluding the museum ticket bit from reward counting.
- Map names are loaded from `assets/map_data.json`, so prompt context uses your provided location dataset rather than a hardcoded table.
- The current trainer is a compact local PPO-style implementation specialized for LLM action sequences. It is intentionally straightforward to audit and adapt.
- Once the ROM and an initial save state are present, the main next step is hardware-specific tuning: `parallel_envs`, prompt length, LoRA rank, and rollout batch size.
- The initial save state should be a live, controllable in-game state. If RAM reads show `party_size=0` and `hp=0.0`, the state is likely too early in boot/menu flow for meaningful training.

## Smoke test

Run this before training to verify ROM boot, state loading, RAM extraction, and a single environment step:

```bash
PYTHONPATH=src python scripts/smoke_env.py --config configs/base.toml
```
