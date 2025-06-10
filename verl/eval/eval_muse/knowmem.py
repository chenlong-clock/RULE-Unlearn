from .logger import RougeEvalLogger

from ..inference import generate_completions_group_vllm
from tqdm.contrib import tzip
from typing import List


def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def eval(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32,
    eval_config=None
):
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = RougeEvalLogger()
    general_prompt: str = ""

    # Few-shot prompting
    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"


    prompt_inputs = []
    for question, answer in tzip(questions, answers):
        prompt = general_prompt + f"Question: {question}\nAnswer: "
        prompt_inputs.append(prompt)

    outputs = generate_completions_group_vllm(
        model,
        tokenizer=tokenizer,
        prompts=prompt_inputs,
        max_tokens=max_new_tokens,
        do_sample=False,
        temperature=0,
        n=1,
        pad_token_id=tokenizer.pad_token_id,
        detokenize=True,
        stop=["\n"],
    )
    # breakpoint()
    for prompt, output, question, answer in tzip(prompt_inputs, outputs, questions, answers):
        output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        logger.log(prompt, answer, output, question=question)

    return logger.report()
