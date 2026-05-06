import unittest

from poke_llm_rl.config import horizon_for_update, load_experiment_config


class ConfigTests(unittest.TestCase):
    def test_load_config_and_horizon_schedule(self) -> None:
        config = load_experiment_config("configs/base.toml")
        self.assertEqual(config.model.model_name_or_path, "Qwen/Qwen3.5-0.8B")
        self.assertEqual(config.map_data_path, "assets/map_data.json")
        self.assertEqual(config.train.group_size, 16)
        self.assertEqual(horizon_for_update(config.env, 0), 500)
        self.assertEqual(horizon_for_update(config.env, 300), 750)
        self.assertEqual(horizon_for_update(config.env, 900), 1000)


if __name__ == "__main__":
    unittest.main()
