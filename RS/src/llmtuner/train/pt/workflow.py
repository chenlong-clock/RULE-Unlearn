# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling

from ...data import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .trainer import CustomTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="pt", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Add evaluation callback like RT workflow ---
    if callbacks is None:
        callbacks = []
    eval_callback = None
    eval_datasets = None
    eval_dataset_dir = getattr(data_args, 'eval_dataset_dir', None)
    target = getattr(data_args, 'target', None)
    if eval_dataset_dir and target:
        import os
        import json
        eval_dataset_dir = os.path.join(eval_dataset_dir, target)
        if "RWKU" in eval_dataset_dir:
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
            eval_datasets = {
                "forget_level1": forget_level1, "forget_level2": forget_level2, "forget_level3": forget_level3,
                "neighbor_level1": neighbor_level1, "neighbor_level2": neighbor_level2,
                "retain_mmlu": retain_mmlu, "retain_bbh": retain_bbh, "truthfulqa": truthfulqa,
                "triviaqa": triviaqa, "forget_mia": forget_mia, "retain_mia": retain_mia,
                "fluency": fluency
            }
            from ...eval.evaluate_callback import RWKUEvaluateCallback
            eval_callback = RWKUEvaluateCallback(
                eval_datasets=eval_datasets,
                data_args=data_args,
                model_args=model_args,
                val_strategy=getattr(data_args, 'val_strategy', 'epoch'),
                eval_freq=getattr(data_args, 'eval_freq', 1)
            )
        elif "MUSE" in eval_dataset_dir:
            from ...eval.evaluate_callback import MUSEEVALCallback
            eval_callback = MUSEEVALCallback(
                eval_datasets=None,
                data_args=data_args,
                model_args=model_args,
                val_strategy=getattr(data_args, 'val_strategy', 'epoch'),
                eval_freq=getattr(data_args, 'eval_freq', 1)
            )
        if eval_callback is not None:
            callbacks.append(eval_callback)
    # --- End callback addition ---

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
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
        # trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        # trainer.save_state()
        # if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        #     plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # # Evaluation
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

    # Create model card
    # create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
