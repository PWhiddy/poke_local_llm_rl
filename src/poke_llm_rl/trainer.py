from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from poke_llm_rl.actions import ParsedAction, parse_completion
from poke_llm_rl.config import ExperimentConfig, horizon_for_update
from poke_llm_rl.env import PokemonRedEnv
from poke_llm_rl.prompts import build_prompt_text


@dataclass(slots=True)
class Transition:
    prompt_text: str
    screen_rgba: np.ndarray
    completion: str
    action: ParsedAction
    reward: float
    completion_ids: list[int]
    old_logprob: float
    ref_logprob: float
    entropy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequencePolicyTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.train.group_size <= 0:
            raise ValueError("train.group_size must be positive")
        if config.train.parallel_envs <= 0:
            raise ValueError("train.parallel_envs must be positive")
        if config.train.group_size % config.train.parallel_envs != 0:
            raise ValueError("train.parallel_envs must evenly divide train.group_size")
        set_seed(config.seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator = Accelerator(gradient_accumulation_steps=config.train.gradient_accumulation_steps)
        self.processor = AutoProcessor.from_pretrained(
            config.model.model_name_or_path,
            trust_remote_code=config.model.trust_remote_code,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model.model_name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            dtype=self._torch_dtype(config.dtype),
        )
        self.reference_model = AutoModelForImageTextToText.from_pretrained(
            config.model.model_name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            torch_dtype=self._torch_dtype(config.dtype),
        )
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

        if config.model.use_lora:
            peft_config = LoraConfig(
                r=config.model.lora_r,
                lora_alpha=config.model.lora_alpha,
                lora_dropout=config.model.lora_dropout,
                bias="none",
                target_modules=config.model.target_modules,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_config)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        total_train_steps = max(config.train.updates * config.train.ppo_epochs, 1)
        warmup_steps = int(total_train_steps * config.train.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )
        self.model, self.reference_model, self.optimizer = self.accelerator.prepare(
            self.model,
            self.reference_model,
            self.optimizer,
        )

    @staticmethod
    def _torch_dtype(dtype_name: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping[dtype_name]

    def build_envs(self, round_horizon: int) -> list[PokemonRedEnv]:
        envs = []
        for env_idx in range(self.config.train.group_size):
            env = PokemonRedEnv(
                self.config.env,
                self.config.rom_path,
                self.config.init_state_path,
                self.config.map_data_path,
                env_id=f"env{env_idx}",
            )
            env.reset(round_horizon=round_horizon)
            envs.append(env)
        return envs

    def _image_to_pil(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(image.astype(np.uint8), mode="RGBA")

    def _chat_text(self, prompt_text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _processor_inputs(self, prompt_texts: list[str], images: list[np.ndarray]):
        chat_texts = [self._chat_text(prompt_text) for prompt_text in prompt_texts]
        pil_images = [self._image_to_pil(image) for image in images]
        return self.processor(
            text=chat_texts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        ).to(self.accelerator.device)

    def _build_mm_token_type_ids(
        self,
        encoded,
        input_ids: torch.Tensor,
        prompt_lengths: list[int],
    ) -> torch.Tensor | None:
        if "mm_token_type_ids" not in encoded:
            return None
        mm_token_type_ids = torch.zeros_like(input_ids)
        source_mm = encoded["mm_token_type_ids"]
        for row_idx, prompt_len in enumerate(prompt_lengths):
            mm_token_type_ids[row_idx, :prompt_len] = source_mm[row_idx, :prompt_len]
        return mm_token_type_ids

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_texts: list[str],
        images: list[np.ndarray],
    ) -> list[tuple[str, list[int], float, float, float]]:
        encoded = self._processor_inputs(prompt_texts, images)
        model = self.accelerator.unwrap_model(self.model)
        output = model.generate(
            **encoded,
            max_new_tokens=self.config.model.max_new_tokens,
            do_sample=True,
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
            repetition_penalty=self.config.model.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        completion_ids_batch: list[list[int]] = []
        completion_texts: list[str] = []

        for row, prompt_len in zip(output.tolist(), prompt_lengths, strict=True):
            completion_ids = row[prompt_len:]
            while completion_ids and completion_ids[-1] == self.tokenizer.pad_token_id:
                completion_ids.pop()
            completion_ids_batch.append(completion_ids)
            completion_texts.append(
                self.processor.batch_decode(
                    [completion_ids],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            )

        scores = self.score_batch(prompt_texts, images, completion_ids_batch)
        return [
            (completion_text, completion_ids, old_logprob, ref_logprob, entropy)
            for completion_text, completion_ids, (old_logprob, ref_logprob, entropy) in zip(
                completion_texts,
                completion_ids_batch,
                scores,
                strict=True,
            )
        ]

    @torch.no_grad()
    def score_batch(
        self,
        prompt_texts: list[str],
        images: list[np.ndarray],
        completion_ids_batch: list[list[int]],
    ) -> list[tuple[float, float, float]]:
        encoded = self._processor_inputs(prompt_texts, images)
        prompt_ids_batch = encoded["input_ids"].tolist()
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        combined_ids = [
            prompt_ids[:prompt_len] + completion_ids
            for prompt_ids, prompt_len, completion_ids in zip(
                prompt_ids_batch,
                prompt_lengths,
                completion_ids_batch,
                strict=True,
            )
        ]
        max_len = max(len(ids) for ids in combined_ids)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full(
            (len(combined_ids), max_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=self.accelerator.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, ids in enumerate(combined_ids):
            seq_len = len(ids)
            input_ids[row_idx, :seq_len] = torch.tensor(ids, dtype=torch.long, device=self.accelerator.device)
            attention_mask[row_idx, :seq_len] = 1

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
        }
        mm_token_type_ids = self._build_mm_token_type_ids(encoded, input_ids, prompt_lengths)
        if mm_token_type_ids is not None:
            model_inputs["mm_token_type_ids"] = mm_token_type_ids

        model_outputs = self.model(**model_inputs)
        ref_outputs = self.reference_model(**model_inputs)
        results: list[tuple[float, float, float]] = []
        for row_idx, (prompt_len, completion_ids) in enumerate(zip(prompt_lengths, completion_ids_batch, strict=True)):
            completion_len = len(completion_ids)
            if completion_len == 0:
                results.append((0.0, 0.0, 0.0))
                continue
            token_slice = slice(prompt_len - 1, prompt_len + completion_len - 1)
            target_tokens = input_ids[row_idx : row_idx + 1, prompt_len : prompt_len + completion_len]
            model_logprobs = F.log_softmax(model_outputs.logits[row_idx : row_idx + 1, token_slice, :], dim=-1)
            ref_logprobs = F.log_softmax(ref_outputs.logits[row_idx : row_idx + 1, token_slice, :], dim=-1)
            gathered_model = model_logprobs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
            gathered_ref = ref_logprobs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
            entropy = -(model_logprobs.exp() * model_logprobs).sum(dim=-1).mean().item()
            results.append((gathered_model.mean().item(), gathered_ref.mean().item(), entropy))
        return results

    def collect_rollouts(
        self,
        envs: list[PokemonRedEnv],
        states,
        update_idx: int,
    ) -> tuple[list[Transition], dict[str, float], list]:
        transitions: list[Transition] = []
        metrics = {
            "reward_total": 0.0,
            "reward_tile": 0.0,
            "reward_event": 0.0,
            "format_failures": 0.0,
        }
        episode_returns = [0.0 for _ in envs]
        terminated_returns: list[float] = []
        total_batch_steps = self.config.train.rollout_rounds_per_update * self.config.train.group_size
        chunk_size = self.config.train.parallel_envs
        progress = tqdm(
            total=total_batch_steps,
            desc=f"Rollout {update_idx}",
            leave=False,
            dynamic_ncols=True,
        )
        try:
            for _ in range(self.config.train.rollout_rounds_per_update):
                for chunk_start in range(0, self.config.train.group_size, chunk_size):
                    chunk_end = chunk_start + chunk_size
                    chunk_envs = envs[chunk_start:chunk_end]
                    chunk_states = states[chunk_start:chunk_end]
                    for env in chunk_envs:
                        env.round_horizon = horizon_for_update(self.config.env, update_idx)
                    prompt_texts = [
                        build_prompt_text(
                            state,
                            previous_action=env.previous_action,
                            round_idx=env.round_idx,
                            max_buttons=self.config.env.max_buttons_per_turn,
                        )
                        for state, env in zip(chunk_states, chunk_envs, strict=True)
                    ]
                    images = [state.screen_rgba for state in chunk_states]
                    batch_outputs = self.generate_batch(prompt_texts, images)
                    for local_idx, (env, prompt_text, image, batch_output) in enumerate(
                        zip(chunk_envs, prompt_texts, images, batch_outputs, strict=True)
                    ):
                        completion, completion_ids, old_logprob, ref_logprob, entropy = batch_output
                        #print(completion)
                        parsed = parse_completion(completion, self.config.env.max_buttons_per_turn)
                        step = env.step(parsed)
                        env_idx = chunk_start + local_idx
                        transitions.append(
                            Transition(
                                prompt_text=prompt_text,
                                screen_rgba=image,
                                completion=completion,
                                action=parsed,
                                reward=step.reward.total,
                                completion_ids=completion_ids,
                                old_logprob=old_logprob,
                                ref_logprob=ref_logprob,
                                entropy=entropy,
                            )
                        )
                        metrics["reward_total"] += step.reward.total
                        metrics["reward_tile"] += step.reward.unique_tile_reward
                        metrics["reward_event"] += step.reward.event_flag_reward
                        metrics["format_failures"] += 0 if parsed.valid else 1
                        episode_returns[env_idx] += step.reward.total
                        progress.update(1)
                        progress.set_postfix(
                            avg_step_reward=f"{metrics['reward_total'] / max(len(transitions), 1):.4f}",
                            avg_episode_return=(
                                f"{(sum(terminated_returns) / len(terminated_returns)):.4f}"
                                if terminated_returns
                                else "pending"
                            ),
                            llm_batch=chunk_size,
                            group_size=self.config.train.group_size,
                        )
                        if step.done:
                            terminated_returns.append(episode_returns[env_idx])
                            episode_returns[env_idx] = 0.0
                            states[env_idx] = env.reset(round_horizon=horizon_for_update(self.config.env, update_idx))
                        else:
                            states[env_idx] = step.state
        finally:
            progress.close()
        count = max(len(transitions), 1)
        metrics = {name: value / count for name, value in metrics.items()}
        metrics["avg_episode_return"] = (
            sum(terminated_returns) / len(terminated_returns) if terminated_returns else 0.0
        )
        metrics["episodes_finished"] = float(len(terminated_returns))
        return transitions, metrics, states

    def _sequence_logprob_and_entropy(
        self,
        prompt_text: str,
        screen_rgba: np.ndarray,
        completion_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self._processor_inputs([prompt_text], [screen_rgba])
        prompt_len = int(encoded["attention_mask"][0].sum().item())
        prompt_ids = encoded["input_ids"][0, :prompt_len].tolist()
        input_ids = torch.tensor([prompt_ids + completion_ids], device=self.accelerator.device)
        attention_mask = torch.ones_like(input_ids)
        mm_token_type_ids = None
        if "mm_token_type_ids" in encoded:
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[:, :prompt_len] = encoded["mm_token_type_ids"][:, :prompt_len]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
        }
        if mm_token_type_ids is not None:
            model_inputs["mm_token_type_ids"] = mm_token_type_ids

        outputs = self.model(**model_inputs)
        with torch.no_grad():
            ref_outputs = self.reference_model(**model_inputs)
        completion_len = len(completion_ids)
        if completion_len == 0:
            zero = torch.tensor(0.0, device=self.accelerator.device)
            return zero, zero, zero
        token_slice = slice(prompt_len - 1, prompt_len + completion_len - 1)
        target_tokens = input_ids[:, prompt_len:]
        model_logprobs = F.log_softmax(outputs.logits[:, token_slice, :], dim=-1)
        ref_logprobs = F.log_softmax(ref_outputs.logits[:, token_slice, :], dim=-1)
        gathered_model = model_logprobs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        gathered_ref = ref_logprobs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        entropy = -(model_logprobs.exp() * model_logprobs).sum(dim=-1).mean(dim=-1)
        return gathered_model.mean(dim=-1), gathered_ref.mean(dim=-1), entropy

    def ppo_update(self, transitions: list[Transition]) -> dict[str, float]:
        rewards = np.array([transition.reward for transition in transitions], dtype=np.float32)
        advantages = (rewards - rewards.mean()) / max(rewards.std(), 1e-6)
        indices = list(range(len(transitions)))
        loss_sum = 0.0
        kl_sum = 0.0
        entropy_sum = 0.0
        steps = 0

        epoch_progress = tqdm(
            range(self.config.train.ppo_epochs),
            desc="PPO Epochs",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in epoch_progress:
            random.shuffle(indices)
            minibatch_progress = tqdm(
                range(0, len(indices), self.config.train.minibatch_size),
                desc="PPO Minibatches",
                leave=False,
                dynamic_ncols=True,
            )
            for start in minibatch_progress:
                batch_indices = indices[start : start + self.config.train.minibatch_size]
                losses = []
                kls = []
                entropies = []
                for idx in batch_indices:
                    transition = transitions[idx]
                    advantage = torch.tensor(advantages[idx], device=self.accelerator.device, dtype=torch.float32)
                    new_logprob, ref_logprob, entropy = self._sequence_logprob_and_entropy(
                        transition.prompt_text,
                        transition.screen_rgba,
                        transition.completion_ids,
                    )
                    old_logprob = torch.tensor(transition.old_logprob, device=self.accelerator.device)
                    ratio = torch.exp(new_logprob - old_logprob)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.config.train.ppo_clip_epsilon,
                        1.0 + self.config.train.ppo_clip_epsilon,
                    )
                    policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
                    kl = torch.clamp(new_logprob - ref_logprob, min=0.0)
                    loss = policy_loss + self.config.train.kl_beta * kl - self.config.train.entropy_beta * entropy
                    losses.append(loss)
                    kls.append(kl.detach())
                    entropies.append(entropy.detach())

                batch_loss = torch.stack(losses).mean()
                self.accelerator.backward(batch_loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                loss_sum += batch_loss.item()
                kl_sum += torch.stack(kls).mean().item()
                entropy_sum += torch.stack(entropies).mean().item()
                steps += 1
                minibatch_progress.set_postfix(
                    loss=f"{loss_sum / max(steps, 1):.4f}",
                    kl=f"{kl_sum / max(steps, 1):.4f}",
                )
            minibatch_progress.close()
        epoch_progress.close()

        return {
            "loss": loss_sum / max(steps, 1),
            "kl": kl_sum / max(steps, 1),
            "entropy": entropy_sum / max(steps, 1),
        }

    def save_checkpoint(self, update_idx: int) -> None:
        checkpoint_dir = self.output_dir / f"checkpoint-{update_idx:05d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)

    def train(self) -> None:
        history_path = self.output_dir / "metrics.jsonl"
        initial_horizon = horizon_for_update(self.config.env, 0)
        envs = self.build_envs(initial_horizon)
        states = [env.current_state() for env in envs]
        try:
            for update_idx in range(self.config.train.updates):
                round_horizon = horizon_for_update(self.config.env, update_idx)
                transitions, rollout_metrics, states = self.collect_rollouts(envs, states, update_idx)
                train_metrics = self.ppo_update(transitions)
                merged = {"update": update_idx, "round_horizon": round_horizon, **rollout_metrics, **train_metrics}
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(merged) + "\n")

                print(
                    f"Batch {update_idx} complete: "
                    f"avg_episode_return={merged['avg_episode_return']:.4f} "
                    f"episodes_finished={int(merged['episodes_finished'])} "
                    f"avg_step_reward={merged['reward_total']:.4f}"
                )
                if update_idx % self.config.train.log_every_updates == 0:
                    print(json.dumps(merged, sort_keys=True))
                if update_idx > 0 and update_idx % self.config.train.save_every_updates == 0:
                    self.save_checkpoint(update_idx)
        finally:
            for env in envs:
                env.close()
