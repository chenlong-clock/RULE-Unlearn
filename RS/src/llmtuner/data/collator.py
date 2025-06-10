from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union, Mapping

import torch
from transformers import DataCollatorForSeq2Seq,DataCollatorForLanguageModeling

class MultiPromptDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        r"""
        Masks out the input ids except for the responses.
        """
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        num_prompts = len([key for key in features[0].keys() if key.startswith("input_ids_")])

        batch = {}
        for i in range(num_prompts):
            curr_examples = []
            for example in features:
                curr_examples.append({
                    key.strip(f"_{i}"):value for key,value in example.items() if key.endswith(f"_{i}")
                })
            batch[f"prompt_{i}"] = super(MultiPromptDataCollatorForSeq2Seq, self).__call__(curr_examples)
        return batch

@dataclass
class MultiplePromptDataCollatorWithPadding(DataCollatorForLanguageModeling):
    r"""
    Data collator for multiple prompt data
    """
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        num_prompts = len([key for key in examples[0].keys() if key.startswith("input_ids_")])

        batch = {}
        for i in range(num_prompts):
            curr_examples = []
            for example in examples:
                curr_examples.append({
                    key.strip(f"_{i}"):value for key,value in example.items() if key.endswith(f"_{i}")
                })
            batch[f"prompt_{i}"] = super(MultiplePromptDataCollatorWithPadding, self).torch_call(curr_examples)
        return batch


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        r"""
        Masks out the input ids except for the responses.
        """
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        label_positions = []
        for key in ("chosen_ids", "rejected_ids"):
            for feature in features:
                prompt_len, answer_len = len(feature["prompt_ids"]), len(feature[key])
                concatenated_features.append(
                    {
                        "input_ids": feature["prompt_ids"] + feature[key],
                        "attention_mask": [1] * (prompt_len + answer_len),
                    }
                )
                label_positions.append((prompt_len, answer_len))

        batch = super().__call__(concatenated_features)
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)
        return batch
    
    
