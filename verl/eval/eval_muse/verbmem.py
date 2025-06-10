from .logger import RougeEvalLogger
from ..inference import generate_completions_group_vllm

from tqdm.contrib import tzip
from typing import List


def eval(
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens : int = 128,
    eval_config=None
):
    logger = RougeEvalLogger()

        
    input_prompts = [f"Completion: {prompt}\nAnswer: " for prompt in prompts]
    outputs = generate_completions_group_vllm(
        model,
        tokenizer=tokenizer,
        prompts=input_prompts,
        max_tokens=max_new_tokens,
        do_sample=False,
        n=1,
        temperature=0,
        stop=["\n"],
        detokenize=True,
        pad_token_id=tokenizer.pad_token_id,
        )
    # breakpoint()
    for prompt, output, gt in tzip(input_prompts, outputs, gts):
        # Encode the `prompt` into `input_ids`


        gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]

        # Use the `model` to generate the continuation of the `input_ids`.

    
        gt_short = tokenizer.batch_decode(
            gt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        logger.log(prompt, gt_short, output)

    return logger.report()
