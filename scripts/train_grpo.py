from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from poke_llm_rl.config import load_experiment_config
from poke_llm_rl.trainer import SequencePolicyTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen on Pokemon Red with online RL.")
    parser.add_argument("--config", default="configs/base.toml")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    trainer = SequencePolicyTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
