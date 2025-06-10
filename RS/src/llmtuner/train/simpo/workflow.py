# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
import os.path
import json
from typing import TYPE_CHECKING, List, Optional
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from ...data import get_dataset, split_dataset, MultiPromptDataCollatorForSeq2Seq, MultiplePromptDataCollatorWithPadding
from ...extras.ploting import plot_loss
from ...extras.constants import IGNORE_INDEX
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .trainer import SimPOTrainer
from ..utils import create_modelcard_and_push, create_ref_model

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


def is_multi_prompt(dataset):
    return "input_ids" not in next(iter(dataset))

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_simpo(
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if is_multi_prompt(dataset):
        training_args.remove_unused_columns = False  # important for multi prompts dataset
        data_collator = MultiplePromptDataCollatorWithPadding(
            tokenizer=tokenizer,
            mlm=False,
        )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    if model_args.train_layers is not None:
        train_layers = model_args.train_layers.split('-')
        for name, param in model.named_parameters():
            if any(f'layers.{i}.' in name for i in range(int(train_layers[0]), int(train_layers[-1]))):
                param.requires_grad = True
                print('Trainable Module:', name)
            else:
                param.requires_grad = False


    if finetuning_args.simpo_loss == 'simpo_KL':
        print("Loading reference model for loss type simpo_KL...")
        ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Update arguments

    eval_dataset_dir = data_args.eval_dataset_dir
    target = data_args.target.replace("_Positive", "")
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
    # Initialize our Trainer
    callbacks.append(eval_callback)

    # Initialize our Trainer
    trainer = SimPOTrainer(
        model=model,
        ref_model=ref_model,
        # ref_model=None, # Remember to change it back in real usage!!!!
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
        if model_args.save_model:
            trainer.save_model()
            trainer.save_state()
            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])


        # Evaluation
        # if training_args.do_eval:
        #     metrics = trainer.evaluate(metric_key_prefix="eval")
        #     try:
        #         perplexity = math.exp(metrics["eval_loss"])
        #     except OverflowError:
        #         perplexity = float("inf")
        #
        #     metrics["perplexity"] = perplexity
        #     trainer.log_metrics("eval", metrics)
        #     trainer.save_metrics("eval", metrics)

    else:
        print("Skip the evaluation process.")


    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
