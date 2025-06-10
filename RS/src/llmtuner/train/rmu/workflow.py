# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
import os.path
import json
from typing import TYPE_CHECKING, List, Optional
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from ...data import get_dataset, split_dataset, MultiplePromptDataCollatorWithPadding, MultiPromptDataCollatorForSeq2Seq
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push, create_ref_model
from .trainer import RMUTrainer

from ...eval import *

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


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

def is_tofu(data_args):
    return 'tofu' == data_args.dataset_name

def is_multi_prompt(dataset):
    return "input_ids" not in next(iter(dataset))

def run_rmu(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    tokenizer.pad_token = tokenizer.eos_token
    dataset = get_dataset(model_args, data_args, training_args, stage="pt", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    layer_id = finetuning_args.rmu_layer
    eval_dataset_dir = data_args.eval_dataset_dir
    target = data_args.target
    eval_dataset_dir = os.path.join(eval_dataset_dir, target)
    if "RWKU" in eval_dataset_dir:
        with open(os.path.join(eval_dataset_dir, FORGET_LEVEL1), 'r') as f:
            forget_level1 = json.load(f)
        with open(os.path.join(eval_dataset_dir, FORGET_LEVEL2), 'r') as f:
            forget_level2 = json.load(f)
        with open(os.path.join(eval_dataset_dir, FORGET_LEVEL3), 'r') as f:
            forget_level3 = json.load(f)
        with open(os.path.join(eval_dataset_dir, NEIGHBOR_LEVEL1), 'r') as f:
            neighbor_level1 = json.load(f)
        with open(os.path.join(eval_dataset_dir, NEIGHBOR_LEVEL2), 'r') as f:
            neighbor_level2 = json.load(f)
        with open(os.path.join(eval_dataset_dir, RETAIN_MMLU), 'r') as f:
            retain_mmlu = json.load(f)
        with open(os.path.join(eval_dataset_dir, RETAIN_BBH), 'r') as f:
            retain_bbh = json.load(f)
        with open(os.path.join(eval_dataset_dir, TRUTHFUL), 'r') as f:
            truthfulqa = json.load(f)
        with open(os.path.join(eval_dataset_dir, TRIVIAQA), 'r') as f:
            triviaqa = json.load(f)
        with open(os.path.join(eval_dataset_dir, FORGET_MIA), 'r') as f:
            forget_mia = json.load(f)
        with open(os.path.join(eval_dataset_dir, RETAIN_MIA), 'r') as f:
            retain_mia = json.load(f)
        with open(os.path.join(eval_dataset_dir, FLUENCY), 'r') as f:
            fluency = json.load(f)
        eval_callback = RWKUEvaluateCallback(
            {
                "forget_level1": forget_level1, "forget_level2": forget_level2, "forget_level3": forget_level3, 
                "neighbor_level1": neighbor_level1, "neighbor_level2": neighbor_level2,
                    "retain_mmlu": retain_mmlu, "retain_bbh": retain_bbh, "truthfulqa": truthfulqa,
                    "triviaqa": triviaqa, "forget_mia": forget_mia, "retain_mia": retain_mia,
                    "fluency": fluency
                },
            data_args, model_args, val_strategy=data_args.val_strategy,eval_freq=data_args.eval_freq
        )
    elif "MUSE" in eval_dataset_dir:
        eval_callback = MUSEEVALCallback(
            None, data_args, model_args, val_strategy=data_args.val_strategy,eval_freq=data_args.eval_freq
        )
    for name, param in model.named_parameters():
        if any(f'layers.{i}.' in name for i in range(layer_id-2, layer_id+1)):
            param.requires_grad = True
            print('Trainable Module:', name)
        else:
            param.requires_grad = False

    # if model_args.train_layers is not None:
    #     train_layers = model_args.train_layers.split('-')
    #     for name, param in model.named_parameters():
    #         if any(f'layers.{i}.' in name for i in range(int(train_layers[0]), int(train_layers[-1]))):
    #             param.requires_grad = True
    #             print('Trainable Module:', name)
    #         else:
    #             param.requires_grad = False

    # Update arguments
    if is_multi_prompt(dataset):
        training_args.remove_unused_columns = False  # important for multi prompts dataset
        data_collator = MultiplePromptDataCollatorWithPadding(
            tokenizer=tokenizer,
            mlm=False,
        )

    ref_model = create_ref_model(model_args, finetuning_args)
    callbacks.append(eval_callback)

    # Initialize our Trainer
    trainer = RMUTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

