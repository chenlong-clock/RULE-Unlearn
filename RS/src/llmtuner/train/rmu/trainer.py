from types import MethodType
from typing import TYPE_CHECKING, Optional

from transformers import Trainer
import torch.nn.functional as F
from torch import nn
import torch
from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler
from transformers.trainer import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
# from .......utils.nethook import TraceDict, layername
from ..utils import TraceDict, layername

if TYPE_CHECKING:
    import torch

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class RMUTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(self, ref_model, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        self.rmu_layer = finetuning_args.rmu_layer
        self.rmu_layer_name = layername(self.model,self.rmu_layer)
        self.steering_coeff = finetuning_args.rmu_steering_coeff
        self.alpha = finetuning_args.rmu_alpha

        self.ref_model = ref_model

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "prompt_0" in inputs:
            return self.compute_loss_with_retain(model,inputs,return_outputs)
        raise NotImplementedError("Retain set is not specified! Got only forget set!")

    def compute_loss_with_retain(self,model,inputs,return_outputs=False):
        forget_inputs,retain_inputs = inputs["prompt_0"],inputs["prompt_1"]
        forget_input_ids, forget_labels, forget_attention_mask = forget_inputs['input_ids'], forget_inputs['labels'], forget_inputs['attention_mask']
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs["input_ids"],retain_inputs["labels"],retain_inputs["attention_mask"]

        # *************** unlearning loss****************
        # ***************  forget loss
        with TraceDict(model,[self.rmu_layer_name],retain_grad=True) as ret:
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask,output_hidden_states=True)
        debug_loss = forget_outputs.loss
        # forget_activations = forget_outputs["hidden_states"][self.rmu_layer]
        forget_activations = ret[self.rmu_layer_name].output[0] # (batch_size,seq_len,hidden_size)
        seq_length = forget_activations.shape[1]
        batch_size = forget_activations.shape[0]

        random_vector = torch.rand(batch_size,seq_length,self.model.config.hidden_size,dtype=forget_activations.dtype,device=forget_activations.device)
        control_vec = random_vector / torch.norm(random_vector) * self.steering_coeff
        forget_loss = torch.nn.functional.mse_loss(
            forget_activations,control_vec
        ).to(torch.device("cuda:0"))
        # return forget_loss
        # ***************  retain loss
        with TraceDict(self.ref_model,[self.rmu_layer_name],retain_grad=True) as ret:
            retain_outputs = self.ref_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask,output_hidden_states=True)
        frozen_retain_activations = ret[self.rmu_layer_name].output[0].to(forget_activations.dtype).to(forget_activations.device) # (batch_size,seq_len,hidden_size)
        # frozen_retain_activations = retain_outputs["hidden_states"][self.rmu_layer].to(forget_activations.dtype)

        with TraceDict(model,[self.rmu_layer_name],retain_grad=True) as ret:
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask,output_hidden_states=True)
        updated_retain_activations = ret[self.rmu_layer_name].output[0].to(forget_activations.device) # (batch_size,seq_len,hidden_size)
        # updated_retain_activations = retain_outputs["hidden_states"][self.rmu_layer]

        retain_loss = torch.nn.functional.mse_loss(
            updated_retain_activations,frozen_retain_activations
        ).to(torch.device("cuda:0"))
        print(f"Forget loss device:{forget_loss.device}, Retain Loss device:{retain_loss.device}")

        total_loss = forget_loss + retain_loss * self.alpha
        return total_loss
        # return retain_loss
