# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.experimental.tqdm_ray import tqdm
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from datasets import load_dataset

from verl.eval.eval_muse.constants import AUC_RETRAIN, DEFAULT_DATA
from verl.eval.eval_muse.utils import read_json, write_json
from verl.eval.eval_rwku import eval_forget, eval_mmlu, eval_neighbor, eval_bbh, eval_fluency, eval_mia, eval_triviaqa, eval_truthfulqa
from verl.eval.eval_muse import eval_knowmem, eval_verbmem, eval_privleak

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.dataset import RLHFDataset, Target2Name, collate_fn, process_batch_prompt
from ..utils.dataset_base import MUSEBaseDataset
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics


FORGET_LEVEL1 = 'forget_level1.json'
FORGET_LEVEL2 = 'forget_level2.json'
FORGET_LEVEL3 = 'forget_level3.json'
NEIGHBOR_LEVEL1 = 'neighbor_level1.json'
NEIGHBOR_LEVEL2 = 'neighbor_level2.json'

RETAIN_MMLU = 'retain_mmlu.json'
RETAIN_BBH = 'retain_bbh.json'
TRUTHFUL = 'truthful.json'
TRIVIAQA = 'triviaqa.json'
FLUENCY = 'fluency.json'
FORGET_MIA = 'forget_mia.json'
RETAIN_MIA = 'retain_mia.json'



WorkerType = Type[Worker]


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}."
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    if "ref_log_probs" in data.batch.keys():
        kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
        kld = kld * response_mask  # (batch_size, response_length)
    else:
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
        val_reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False
            

            
        # if config.

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        self._create_dataloader()

    def _create_dataloader(self):
        if "MUSE" in self.config.data.train_files:
            self.train_dataset = MUSEBaseDataset(
                data_path=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.prompt_key,
                answer_key=self.config.data.answer_key,
                image_key=self.config.data.image_key,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation="right",
                min_pixels=self.config.data.min_pixels,
                max_pixels=self.config.data.max_pixels,
                format_prompt=self.config.data.format_prompt,
                mode=self.config.data.mode,
            )
        else:    
            self.train_dataset = RLHFDataset(
                data_path=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.prompt_key,
                answer_key=self.config.data.answer_key,
                image_key=self.config.data.image_key,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation="right",
                format_prompt=self.config.data.format_prompt,
                min_pixels=self.config.data.min_pixels,
                max_pixels=self.config.data.max_pixels,
                mode=self.config.data.mode,
            )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )
        # if "eval_data" in self.config.trainer.experiment_name:
        #     val_dataset = load_dataset("json", data_files=self.config.data.val_files, split="train")
        #     val_dataset = val_dataset.map(
        #         lambda x: {**x, self.config.data.prompt_key: x[self.config.data.prompt_key]}
        #     )
        #     self.val_dataset = RLHFDataset(
        #         data_path=self.config.data.val_files,
        #         tokenizer=self.tokenizer,
        #         processor=self.processor,
        #         prompt_key=self.config.data.val_prompt_key,
        #         answer_key=self.config.data.val_answer_key,
        #         image_key=self.config.data.image_key,
        #         max_prompt_length=self.config.data.max_prompt_length,
        #         truncation="right",
        #         system_prompt=self.config.data.system_prompt,
        #         min_pixels=self.config.data.min_pixels,
        #         max_pixels=self.config.data.max_pixels,
        #     )
        # else:
        if "MUSE" in self.config.data.val_files:
            self.val_dataset = MUSEBaseDataset(
                data_path=self.config.data.val_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.val_prompt_key,
                answer_key=self.config.data.val_answer_key,
                image_key=self.config.data.image_key,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation="right",
                format_prompt=self.config.data.format_prompt,
                min_pixels=self.config.data.min_pixels,
                max_pixels=self.config.data.max_pixels,
                mode=self.config.data.mode,
            )
        else:
            self.val_dataset = RLHFDataset(
                data_path=self.config.data.val_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.val_prompt_key,
                answer_key=self.config.data.val_answer_key,
                image_key=self.config.data.image_key,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation="right",
                format_prompt=self.config.data.format_prompt,
                min_pixels=self.config.data.min_pixels,
                max_pixels=self.config.data.max_pixels,
            )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset)
            if self.config.data.val_batch_size == -1
            else self.config.data.val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")
        self.extra_prompt = None
        if self.config.data.extra_prompt:
            target_index = self.config.trainer.experiment_name.split("/")[-1]
            target_prompt = " ".join(target_index.split("_")[1:])
            extra_prompt = self.config.data.extra_prompt.format(target_prompt)
            self.extra_prompt = self.tokenizer(extra_prompt)['input_ids']
            print(f"Extra prompt: {extra_prompt}")
        if self.config.worker.noise_rollout.enabled:
            if self.config.worker.noise_rollout.type == "output":
                target_index = self.config.trainer.experiment_name.split("/")[-1]
                target_prompt = " ".join(target_index.split("_")[1:])
                noise_prompt = self.config.worker.noise_rollout.noise_prompt.format(target_prompt)
                self.noise_prompt = self.tokenizer(noise_prompt)['input_ids']
            elif self.config.worker.noise_rollout.type == "input":
                self.noise_prompt = self.tokenizer(self.config.worker.noise_rollout.noise_prompt)['input_ids']
            # tokenized_guide_prompt = self.tokenizer.apply_chat_template(
            #     [{"role": "system", "content": guide_prompt}], add_generation_prompt=False, tokenize=True, 
            #     return_tensors="pt", return_dict=True,
            #     tokenizer_kwargs= {"return_attention_mask": True}
            # )
            # # extract the content between <|start_header_id|> and <|eot_id|>, where tokenized_guide_prompt is a list
            # self.matched_guide_prompt = tokenized_guide_prompt[
            #     tokenized_guide_prompt.index(start_header_id) : tokenized_guide_prompt.index(eot_id) + 1
            # ]
            
        

    def _maybe_log_val_generations(self, inputs, outputs, scores, labels):
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, labels))

        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        # Take first N samples after shuffling

        # Create column names for all samples
        columns = ["step", "input", "output", "score", "labels"]
        import wandb
        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        self.logger.log_generation(samples, self.global_step)


    def _validate(self):
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_scores, sample_labels, sample_metrics = [], [], [], [], {}
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )


            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["stop_token_ids"] = [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            ] 
            if "MUSE" in self.config.trainer.experiment_name:
                test_gen_batch.meta_info["stop"] = [
                    "\n"
                    ]
                test_gen_batch.meta_info["detokenize"] = True
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            # breakpoint()
            test_batch = test_batch.union(test_output_gen_batch)
            
            # evaluate using reward_function
            reward_tensor, reward_metrics = self.val_reward_fn(test_batch)
            ground_truth = test_batch.non_tensor_batch["ground_truth"]
            #ground_truth is a np.array of list of strings, conver it to list of strings
            sample_labels.extend([i for i in ground_truth.tolist()])
            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            # reward_metrics: Dict[str, List[float]]
            for k, v in reward_metrics.items():
                if k not in sample_metrics:
                    sample_metrics[k] = []
                sample_metrics[k].extend(v)
            reward_tensor_lst.append(reward_tensor)
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, labels=sample_labels)

        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        save_dir = os.path.join(
            self.config.data.output_result_dir,
            self.config.trainer.experiment_name,
            f"step{self.global_step}"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # breakpoint()
        # convert sample_metrics from Dict[str, List[float]] to List[Dict[str, float]] 
        sample_metrics = [
            {k: v[i] for k, v in sample_metrics.items()} for i in range(len(sample_inputs))
        ]
        output_result = {
            'val/test_score': reward_score,
            'results': [
                {
                    "query": q,
                    "answer": a,
                    "label": l,
                    "reward_score": s,
                    "metrics": {
                        metric: metric_value for metric, metric_value in ms.items()
                        }, 
                } for q, a, s, l, ms in zip(sample_inputs, sample_outputs, sample_scores, sample_labels, sample_metrics)
            ]
        }
        with open(os.path.join(save_dir, 'val.json'), 'w') as f:
            json.dump(output_result, f, indent=4)
        if "RWKU" in self.config.trainer.experiment_name:
            forget_val_dir = os.path.dirname(self.config.data.val_files)
            with open(os.path.join(forget_val_dir, FORGET_LEVEL1), 'r') as f:
                forget_level1 = json.load(f)
            with open(os.path.join(forget_val_dir, FORGET_LEVEL2), 'r') as f:
                forget_level2 = json.load(f)
            with open(os.path.join(forget_val_dir, FORGET_LEVEL3), 'r') as f:
                forget_level3 = json.load(f)
            with open(os.path.join(forget_val_dir, NEIGHBOR_LEVEL1), 'r') as f:
                neighbor_level1 = json.load(f)
            with open(os.path.join(forget_val_dir, NEIGHBOR_LEVEL2), 'r') as f:
                neighbor_level2 = json.load(f)
            # with open(os.path.join(forget_val_dir, RETAIN_MMLU), 'r') as f:
            #     retain_mmlu = json.load(f)
            # with open(os.path.join(forget_val_dir, RETAIN_BBH), 'r') as f:
            #     retain_bbh = json.load(f)
            # with open(os.path.join(forget_val_dir, TRUTHFUL), 'r') as f:
            #     truthful = json.load(f)
            # with open(os.path.join(forget_val_dir, TRIVIAQA), 'r') as f:
            #     triviaqa = json.load(f)
            # with open(os.path.join(forget_val_dir, FLUENCY), 'r') as f:
            #     fluency = json.load(f)
            # with open(os.path.join(forget_val_dir, FORGET_MIA), 'r') as f:
            #     forget_mia = json.load(f)
            # with open(os.path.join(forget_val_dir, RETAIN_MIA), 'r') as f:
            #     retain_mia = json.load(f)
            
            forget_score_1, forget_score_2, forget_score_3 = eval_forget(
                self.actor_rollout_wg, self.tokenizer, forget_level1, forget_level2, forget_level3, batch_size=16, output_result_dir=os.path.join(save_dir, 'forget.json'), use_prompt=False, eval_config=self.config.worker.rollout.val_override_config
            )
            neighbor_score_1, neighbor_score_2 = eval_neighbor(
                self.actor_rollout_wg, self.tokenizer, neighbor_level1, neighbor_level2, batch_size=16, output_result_dir=os.path.join(save_dir, 'neighbor.json'), use_prompt=False,  eval_config=self.config.worker.rollout.val_override_config
            )
            # retain_mmlu_score = eval_mmlu(
            #     self.actor_rollout_wg, self.tokenizer, retain_mmlu, batch_size=16, output_result_dir=os.path.join(save_dir, 'retain_mmlu.json'), use_prompt=False, eval_config=self.config.worker.rollout.val_override_config
            # )
            # retain_bbh_score = eval_bbh(
            #     self.actor_rollout_wg, self.tokenizer, retain_bbh, batch_size=16, output_result_dir=os.path.join(save_dir, 'retain_bbh.json'), use_prompt=False, eval_config=self.config.worker.rollout.val_override_config
            # )
            # truthful_score = eval_truthfulqa(
            #     self.actor_rollout_wg, self.tokenizer, truthful, batch_size=16, output_result_dir=os.path.join(save_dir, 'truthful.json'), use_prompt=False, eval_config=self.config.worker.rollout.val_override_config
            # )
            # triviaqa_score = eval_triviaqa(
            #     self.actor_rollout_wg, self.tokenizer, triviaqa, batch_size=16, output_result_dir=os.path.join(save_dir, 'triviaqa.json'), use_prompt=False, eval_config=self.config.worker.rollout.val_override_config
            # )
            # fluency_score = eval_fluency(
            #     self.actor_rollout_wg, self.tokenizer, fluency, batch_size=16, output_result_dir=os.path.join(save_dir, 'fluency.json'), use_prompt=False, eval_config=self.config.worker.rollout.val_override_config
            # )
            # forget_mia_score = eval_mia(
            #     self.actor_rollout_wg, self.tokenizer, forget_mia, batch_size=16, output_result_dir=os.path.join(save_dir, 'forget_mia.json'), use_prompt=False, eval_config=self.config.worker.rollout.val_override_config
            # )
            # retain_mia_score = eval_mia(
            #     self.actor_rollout_wg, self.tokenizer, retain_mia, batch_size=16, output_result_dir=os.path.join(save_dir, 'retain_mia.json'), use_prompt=False, eval_config=self.config.worker.rollout.val_override_config
            # )
            # save the scores to a json file
            
            return {
                "val/test_score": reward_score,
                "val/forget_1_rouge_l": forget_score_1,
                "val/forget_2_rouge_l": forget_score_2,
                "val/forget_3_rouge_l": forget_score_3,
                "val/neighbor_1_rouge_l": neighbor_score_1,
                "val/neighbor_2_rouge_l": neighbor_score_2,
                # "val/retain_mmlu": retain_mmlu_score,
                # "val/retain_bbh": retain_bbh_score,
                # "val/truthful": truthful_score,
                # "val/triviaqa": triviaqa_score,
                # "val/fluency": fluency_score,
                # "val/forget_mia": forget_mia_score,
                # "val/retain_mia": retain_mia_score,
                }
        elif "MUSE" in self.config.trainer.experiment_name:
            forget_val_dir = os.path.dirname(self.config.data.val_files)
            target = self.config.trainer.experiment_name.split("/")[-1]
            verbmem_forget_file = DEFAULT_DATA[target]['verbmem_forget_file']
            # privleak_forget_file = DEFAULT_DATA[target]['privleak_forget_file'] 
            # privleak_retain_file = DEFAULT_DATA[target]['privleak_retain_file'] 
            # privleak_holdout_file = DEFAULT_DATA[target]['privleak_holdout_file'] 
            knowmem_forget_qa_file = DEFAULT_DATA[target]['knowmem_forget_qa_file'] 
            knowmem_forget_qa_icl_file = DEFAULT_DATA[target]['knowmem_forget_qa_icl_file'] 
            knowmem_retain_qa_file = DEFAULT_DATA[target]['knowmem_retain_qa_file']
            knowmem_retain_qa_icl_file = DEFAULT_DATA[target]['knowmem_retain_qa_icl_file'] 
            out = {}
            for metric in ["verbmem_f", "knowmem_f", "knowmem_r"]:
                os.makedirs(os.path.join(save_dir, metric), exist_ok=True)
            with torch.no_grad():
                print("Evaluate verbmem_f")
                data = read_json(verbmem_forget_file)
                agg, log = eval_verbmem(
                    prompts=[d['prompt'] for d in data],
                    gts=[d['gt'] for d in data],
                    model=self.actor_rollout_wg, tokenizer=self.tokenizer,
                    max_new_tokens=128
                )
                write_json(agg, os.path.join(save_dir, "verbmem_f/agg.json"))
                write_json(log, os.path.join(save_dir, "verbmem_f/log.json"))
                out['verbmem_f'] = agg["mean_rougeL"] * 100

                # print("Evaluate privleak")
                # auc, log = eval_privleak(
                #     forget_data=read_json(privleak_forget_file),
                #     retain_data=read_json(privleak_retain_file),
                #     holdout_data=read_json(privleak_holdout_file),
                #     model=self.actor_rollout_wg, tokenizer=self.tokenizer
                # )
                # write_json(auc, os.path.join(save_dir, "privleak/auc.json"))
                # write_json(log, os.path.join(save_dir, "privleak/log.json"))
                # out['privleak'] = (auc["forget_holdout_Min-40%'"] - AUC_RETRAIN["forget_holdout_Min-40%'"]) / AUC_RETRAIN["forget_holdout_Min-40%'"] * 100

                print("Evaluate knowmem_f")
                qa = read_json(knowmem_forget_qa_file)
                icl = read_json(knowmem_forget_qa_icl_file)
                agg, log = eval_knowmem(
                    questions=[d['question'] for d in qa],
                    answers=[d['answer'] for d in qa],
                    icl_qs=[d['question'] for d in icl],
                    icl_as=[d['answer'] for d in icl],
                    model=self.actor_rollout_wg, tokenizer=self.tokenizer,
                    max_new_tokens=32,
                    eval_config=self.config.worker.rollout.val_override_config
                )
                write_json(agg, os.path.join(save_dir, "knowmem_f/agg.json"))
                write_json(log, os.path.join(save_dir, "knowmem_f/log.json"))
                out['knowmem_f'] = agg["mean_rougeL"] * 100

                print("Evaluate knowmem_r")
                qa = read_json(knowmem_retain_qa_file)
                icl = read_json(knowmem_retain_qa_icl_file)
                agg, log = eval_knowmem(
                    questions=[d['question'] for d in qa],
                    answers=[d['answer'] for d in qa],
                    icl_qs=[d['question'] for d in icl],
                    icl_as=[d['answer'] for d in icl],
                    model=self.actor_rollout_wg, tokenizer=self.tokenizer,
                    max_new_tokens=32,
                    eval_config=self.config.worker.rollout.val_override_config
                )
                if save_dir is not None:
                    write_json(agg, os.path.join(save_dir, "knowmem_r/agg.json"))
                    write_json(log, os.path.join(save_dir, "knowmem_r/log.json"))
                out['knowmem_r'] = agg["mean_rougeL"] * 100
            with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                json.dump(out, f, indent=4)
            return {
                "val/test_score": reward_score,
                "val/forget_verbmem_f": out["verbmem_f"],
                "val/forget_knowmem_f": out["knowmem_f"],
                "val/forget_knowmem_r": out["knowmem_r"],
                }
            

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls



        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()


    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        for _ in tqdm(range(self.config.trainer.total_episodes), desc="Episode", position=0):
            for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):
                self.global_step += 1
                if self.global_step > self.training_steps:
                    break

                metrics, timing_raw = {}, {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch.meta_info['stop_token_ids'] = [
                    self.tokenizer.pad_token_id,
                    self.tokenizer.eos_token_id,
                ] # for unlearning, limited response is suitable
                if "MUSE" in self.config.data.train_files:
                    gen_batch.meta_info['stop'] = ["\n"]
                    gen_batch.meta_info['detokenize'] = True
                # breakpoint()
                if self.extra_prompt:
                    # breakpoint()
                    gen_batch.meta_info['extra_prompt'] = self.extra_prompt
                    gen_batch.meta_info['header_index_token'] = self.tokenizer("<|end_header_id|>")['input_ids'][0]
                if self.config.worker.noise_rollout.enabled:
                    gen_batch.meta_info["noise_prompt"] = self.noise_prompt
                    gen_batch.meta_info['header_index_token'] = self.tokenizer("<|end_header_id|>")['input_ids'][0]
                    gen_batch.meta_info["noise_ratio"] = self.config.worker.noise_rollout.noise_ratio
                    gen_batch.meta_info["noise_type"] = self.config.worker.noise_rollout.type
                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):  # wg: worker group
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        # breakpoint()
                    if self.config.algorithm.adv_estimator == "remax":
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["temperature"] = 0
                            gen_baseline_batch.meta_info["n"] = 1
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor, _ = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # compute reward
                    with _timer("reward", timing_raw):
                        if self.use_reward_model:
                            raise NotImplementedError("Reward model is not supported yet.")

                        # we combine with rule-based rm
                        reward_tensor, reward_metrics = self.reward_fn(batch)
                        # if self.global_step == 3 or self.global_step == 1 or self.global_step==10:
                        #     breakpoint()
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {
                            f"reward/{key}": value for key, value in reduce_metrics(reward_metrics).items()
                        }
                        metrics.update(reward_metrics)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with _timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            print("saving checkpoint")
                            self._save_checkpoint()

                # collect metrics
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                self.logger.log(data=metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")
        if not self.config.trainer.save_freq == -2:
            if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
                self._save_checkpoint()
