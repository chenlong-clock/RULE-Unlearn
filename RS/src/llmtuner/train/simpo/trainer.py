from types import MethodType
from typing import TYPE_CHECKING, Optional
from torch import nn
import torch
from transformers import Trainer
import torch.nn.functional as F
from transformers.trainer import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    import torch

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class SimPOTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self,
        ref_model,
        finetuning_args: "FinetuningArguments",
        **kwargs
    ) -> None:

        self.simpo_coeff = finetuning_args.simpo_coeff
        self.grad_diff_coeff = finetuning_args.grad_diff
        self.KL_coeff = finetuning_args.KL_coeff
        self.loss_type = finetuning_args.simpo_loss
        self.gamma = finetuning_args.simpo_gamma
        self.ref_model = ref_model

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.beta = finetuning_args.dpo_beta
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def get_batch_loss(self, output, labels):
        shifted_labels = labels[..., 1:].contiguous()
        output = output[..., :-1, :].contiguous()

        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        # get the sum loss for each sequence in a batch
        loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

        return loss


    def compute_loss_with_retain(self,model,inputs,return_outputs=False):
        forget_inputs,retain_inputs = inputs["prompt_0"],inputs["prompt_1"]
        forget_input_ids,forget_labels, forget_attention_mask = forget_inputs['input_ids'], forget_inputs['labels'], forget_inputs['attention_mask']
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs["input_ids"],retain_inputs["labels"],retain_inputs["attention_mask"]

        forget_outputs = model(
            forget_input_ids,
            labels=forget_labels,
            attention_mask=forget_attention_mask)
        loss_mask = forget_labels != -100
        forget_loss = self.get_batch_loss(forget_outputs.logits, forget_labels) / loss_mask.sum(-1) - self.gamma
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        if self.loss_type == 'simpo':
            loss = forget_loss
        elif self.loss_type == 'simpo_grad_diff':
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.simpo_coeff * forget_loss + self.grad_diff_coeff * retain_loss
        elif self.loss_type == 'simpo_KL':
            with torch.no_grad():
                retain_outputs = self.ref_model(retain_input_ids, labels=retain_labels,
                                                   attention_mask=retain_attention_mask)
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            # minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)

            loss = self.simpo_coeff * forget_loss + self.KL_coeff * retain_loss
        else:
            raise ValueError(f"Loss type {self.loss_type} is not supported!")
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # loss, outputs = super().compute_loss(model, inputs, True)
        if "prompt_0" in inputs:
            return self.compute_loss_with_retain(model,inputs,return_outputs)
        raise NotImplementedError("Retain set is not specified! Got only forget set!")