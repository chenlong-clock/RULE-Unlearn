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


from collections import defaultdict
from typing import Callable, Dict, List, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import (math_compute_score, 
                                   r1v_compute_score, 
                                   forget_reward, 
                                   forget_reward_abs, 
                                   forget_reward_abs_half_reject, 
                                   forget_reward_half_reject, 
                                   forget_reward_two_stage, 
                                   forget_reward_two_stage_abs, 
                                   forget_reward_two_stage_abs_target, 
                                   forget_reward_two_stage_abs_target_half_reject, 
                                   reject_behave_reward)
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: RewardConfig):
        self.config = config
        self.tokenizer = tokenizer
        if config.score_function == "math":
            self.compute_score: Callable[[str, str], RewardScore] = math_compute_score
        elif config.score_function == "r1v":
            self.compute_score: Callable[[str, str], RewardScore] = r1v_compute_score
        elif config.score_function == "forget":
            self.compute_score = forget_reward
        elif config.score_function == "forget_half_reject":
            self.compute_score = forget_reward_half_reject
        elif config.score_function == "forget_abs":
            self.compute_score = forget_reward_abs
        elif config.score_function == "forget_abs_half_reject":
            self.compute_score = forget_reward_abs_half_reject
        elif config.score_function == "forget_two_stage":
            self.compute_score = forget_reward_two_stage
        elif config.score_function == "forget_two_stage_abs":
            self.compute_score = forget_reward_two_stage_abs
        elif config.score_function == "forget_two_stage_abs_target":
            self.compute_score = forget_reward_two_stage_abs_target
        elif config.score_function == "forget_two_stage_abs_target_half_reject":
            self.compute_score = forget_reward_two_stage_abs_target_half_reject
        elif config.score_function == "reject_behave_reward":
            self.compute_score = reject_behave_reward
        else:
            raise NotImplementedError(f"Unknown score function {config.score_function}.")

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data_item.non_tensor_batch["ground_truth"]
            # breakpoint()
            score = self.compute_score(response_str, ground_truth)
            reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
