from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    model_name_or_path: str
    trust_remote_code: bool = True
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)
    max_prompt_tokens: int = 1536
    max_new_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.02


@dataclass(slots=True)
class RewardConfig:
    unique_tile_weight: float = 0.05
    event_flag_weight: float = 2.0
    format_penalty: float = -0.2
    noop_penalty: float = -0.02
    repeated_action_penalty: float = -0.01


@dataclass(slots=True)
class HorizonStage:
    until_update: int
    rounds: int


@dataclass(slots=True)
class EnvConfig:
    headless: bool = True
    downsample_factor: int = 8
    grayscale_buckets: int = 16
    action_hold_frames: int = 8
    frames_per_button: int = 3
    max_buttons_per_turn: int = 8
    initial_round_horizon: int = 500
    horizon_schedule: list[HorizonStage] = field(default_factory=list)
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(slots=True)
class TrainConfig:
    group_size: int = 16
    parallel_envs: int = 4
    updates: int = 2000
    rollout_rounds_per_update: int = 64
    gamma: float = 0.995
    ppo_epochs: int = 2
    minibatch_size: int = 32
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    grad_clip_norm: float = 1.0
    ppo_clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    entropy_beta: float = 0.001
    gradient_accumulation_steps: int = 1
    log_every_updates: int = 5
    save_every_updates: int = 50
    eval_every_updates: int = 50


@dataclass(slots=True)
class LoggingConfig:
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "poke-llm-rl"


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    rom_path: str
    init_state_path: str
    map_data_path: str
    output_dir: str
    device: str
    dtype: str
    model: ModelConfig
    env: EnvConfig
    train: TrainConfig
    logging: LoggingConfig

    def resolve_path(self, value: str) -> Path:
        return Path(value).expanduser().resolve()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix == ".toml":
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    raise ValueError(f"Unsupported config format for {config_path}. Use .toml.")


def _coerce_horizon_schedule(items: list[dict[str, Any]]) -> list[HorizonStage]:
    return [HorizonStage(**item) for item in items]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = _load_yaml(path)
    env_raw = dict(raw["env"])
    env_raw["reward"] = RewardConfig(**env_raw["reward"])
    env_raw["horizon_schedule"] = _coerce_horizon_schedule(env_raw["horizon_schedule"])
    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        seed=raw["seed"],
        rom_path=raw["rom_path"],
        init_state_path=raw["init_state_path"],
        map_data_path=raw["map_data_path"],
        output_dir=raw["output_dir"],
        device=raw["device"],
        dtype=raw["dtype"],
        model=ModelConfig(**raw["model"]),
        env=EnvConfig(**env_raw),
        train=TrainConfig(**raw["train"]),
        logging=LoggingConfig(**raw["logging"]),
    )


def horizon_for_update(config: EnvConfig, update_idx: int) -> int:
    for stage in config.horizon_schedule:
        if update_idx < stage.until_update:
            return stage.rounds
    return config.initial_round_horizon
