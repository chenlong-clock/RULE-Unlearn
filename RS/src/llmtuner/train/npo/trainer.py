from types import MethodType
from typing import TYPE_CHECKING, Optional
from torch import nn
import torch
from transformers import Trainer
import torch.nn.functional as F

from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    import torch

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(self, ref_model, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.ref_model = ref_model
        self.beta = finetuning_args.dpo_beta

        self.npo_coeff = finetuning_args.npo_coeff
        self.grad_diff_coeff = finetuning_args.grad_diff
        self.KL_coeff = finetuning_args.KL_coeff
        self.loss_type = finetuning_args.npo_loss

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

    def compute_loss_with_retain(self,model,inputs,return_outputs=False, num_items_in_batch=None):
        forget_inputs,retain_inputs = inputs["prompt_0"],inputs["prompt_1"]
        input_ids, labels, attention_mask = forget_inputs['input_ids'], forget_inputs['labels'], forget_inputs['attention_mask']
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs["input_ids"],retain_inputs["labels"],retain_inputs["attention_mask"]

        if self.loss_type == 'npo':
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss_current = self.get_batch_loss(outputs.logits, labels)
            with torch.no_grad():
                forget_outputs_oracle = self.ref_model(forget_inputs['input_ids'], labels=forget_inputs['labels'],
                                                       attention_mask=forget_inputs['attention_mask'])
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, forget_inputs['labels'])
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
            loss = forget_loss
        elif self.loss_type == 'npo_grad_diff':
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss_current = self.get_batch_loss(outputs.logits, labels)
            with torch.no_grad():
                forget_outputs_oracle = self.ref_model(forget_inputs['input_ids'], labels=forget_inputs['labels'],
                                                       attention_mask=forget_inputs['attention_mask'])
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, forget_inputs['labels'])
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss
        elif self.loss_type == 'npo_KL':
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss_current = self.get_batch_loss(outputs.logits, labels)
            with torch.no_grad():
                forget_outputs_oracle = self.ref_model(forget_inputs['input_ids'], labels=forget_inputs['labels'], attention_mask=forget_inputs['attention_mask'])
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, forget_inputs['labels'])
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

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

            loss = self.npo_coeff * forget_loss + self.KL_coeff * retain_loss
        else:
            raise NotImplementedError(f"{self.loss_type} loss type not implemented!")

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "prompt_0" in inputs:
            return self.compute_loss_with_retain(model,inputs,return_outputs, num_items_in_batch=None)
        loss, outputs = super().compute_loss(model, inputs, True)
        forget_loss_current = self.get_batch_loss(outputs.logits, inputs['labels'])
        with torch.no_grad():
            forget_outputs_oracle = self.ref_model(inputs['input_ids'], labels=inputs['labels'], attention_mask=inputs['attention_mask'])
            forget_logits_oracle = forget_outputs_oracle.logits
            forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, inputs['labels'])
        neg_log_ratios = forget_loss_current - forget_loss_oracle
        loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
        return (loss, outputs) if return_outputs else loss
