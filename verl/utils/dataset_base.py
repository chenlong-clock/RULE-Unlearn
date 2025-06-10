from verl.eval.eval_muse.constants import DEFAULT_DATA
from verl.eval.eval_muse.utils import read_json
from .dataset import RLHFDataset, ImageProcessMixin
import torch



VF = __import__("verl.utils.torch_functional", fromlist=["postprocess_data"])

def load_icl_qa_prefix(target: str, icl_type="<reject>") -> str:
    general_prompt = ""
    if icl_type == "<reject>":
        qa_icl_file = DEFAULT_DATA[target]['knowmem_forget_qa_icl_file']
    elif icl_type == "<normal>":
        qa_icl_file = DEFAULT_DATA[target]['knowmem_retain_qa_icl_file']
    else:
        qa_icl_file = DEFAULT_DATA[target]['knowmem_train_qa_icl_file']
    icl = read_json(qa_icl_file)
    general_prompt = ""
    icl_qs=[d['question'] for d in icl]
    icl_as=[d['answer'] for d in icl]

    for question, answer in zip(icl_qs, icl_as):
            general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"
    return general_prompt

class MUSEBaseDataset(RLHFDataset):
    """
    RLHF Dataset for base models: no chat template, just raw prompt with start token.
    """
    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        prompt_str: str = row_dict[self.prompt_key]
        mode: str = row_dict.get("mode", None)
        if self.icl_type != "train":
        # if examples["mode"][i] == "completion":
        #     examples["prompt"][i][0]["content"] =  f'Completion: {examples["prompt"][i][0]["content"]}\nAnswer: '
        # elif examples["mode"][i] == "qa":   
        #     from ..train.utils import load_icl_qa_prefix
        #     general_prompt = load_icl_qa_prefix(examples["subject"][i])
        #     examples["prompt"][i][0]["content"] = general_prompt +  f'Question: {examples["prompt"][i][0]["content"]}\nAnswer: '
            rtype = row_dict[self.answer_key][0]
        else:
            rtype = "train"
        if mode == "completion":
            prompt_str = f'Completion: {prompt_str}\nAnswer: '
        elif mode == "qa":
            general_prompt = load_icl_qa_prefix("books", rtype)
            prompt_str = general_prompt + f'Question: {prompt_str}\nAnswer: '
        if self.format_prompt:
            prompt_str = prompt_str + " " + self.format_prompt.strip()

        # Add start token (BOS)
        if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
            prompt = self.tokenizer.bos_token + prompt_str
        elif hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            # If only bos_token_id is available
            prompt = self.tokenizer.decode([self.tokenizer.bos_token_id]) + prompt_str
        else:
            prompt = prompt_str  # fallback

        model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        return row_dict
