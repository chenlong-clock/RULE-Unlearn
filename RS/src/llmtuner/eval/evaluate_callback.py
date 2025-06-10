import json
import os
import torch
from transformers.trainer_callback import TrainerCallback
from transformers import AutoTokenizer

from .rwku import eval_forget, eval_mia, eval_fluency, eval_neighbor, eval_mmlu, eval_bbh, eval_triviaqa, eval_truthfulqa
from .muse import eval_verbmem, eval_knowmem, eval_privleak
from .muse.utils import load_model, load_tokenizer, write_csv, read_json, write_json
from .muse.constants import SUPPORTED_METRICS, CORPORA, LLAMA_DIR, DEFAULT_DATA, AUC_RETRAIN
from typing import List, Dict, Literal
from pandas import DataFrame


class RWKUEvaluateCallback(TrainerCallback):
    """Callback to evaluate model periodically."""
    def __init__(self, eval_datasets, data_args, model_args, 
                 val_strategy: Literal["epoch", "step", "both", "None"] = "epoch", 
                 eval_freq: int = 1):
        self.eval_datasets = eval_datasets
        self.data_args = data_args
        self.model_args = model_args
        self.val_strategy = val_strategy
        self.eval_freq = max(1, eval_freq)
        super().__init__()
        
    def _evaluate(self, model, e_tokenizer, counter_value, counter_name):
        """Common evaluation logic for both epochs and steps."""
        target = self.data_args.target.replace("_Positive", "")
        output_result_dir = os.path.join(self.data_args.output_result_dir, target, f"{counter_name}_{counter_value}")
        os.makedirs(output_result_dir, exist_ok=True)
        
        print(f"\n*** Evaluating after {counter_name} {counter_value} ***")
        model.eval()
        # Existing evaluation code
        if "forget" in self.data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, 'forget.json')):
            print("Evaluate forgetting...")
            eval_forget(
                model, e_tokenizer, 
                self.eval_datasets["forget_level1"], 
                self.eval_datasets["forget_level2"], 
                self.eval_datasets["forget_level3"], 
                batch_size=16, 
                output_result_dir=os.path.join(output_result_dir, 'forget.json'),
                use_prompt=self.data_args.use_prompt
            )
        
        if "neighbor" in self.data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, 'neighbor.json')):
            print("Evaluate neighbor...")
            eval_neighbor(
                model, e_tokenizer, 
                self.eval_datasets["neighbor_level1"], 
                self.eval_datasets["neighbor_level2"], 
                batch_size=16, 
                output_result_dir=os.path.join(output_result_dir, 'neighbor.json'),
                use_prompt=self.data_args.use_prompt
            )
        if "mmlu" in self.data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, 'mmlu.json')):
            print("Evaluate mmlu...")
            eval_mmlu(model, e_tokenizer, self.eval_datasets["retain_mmlu"], batch_size=1, 
                    output_result_dir=os.path.join(output_result_dir, 'mmlu.json'),
                    use_prompt=self.data_args.use_prompt)
        if "bbh" in self.data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, 'bbh.json')):
            print("Evaluate bbh...")
            eval_bbh(model, e_tokenizer, self.eval_datasets["retain_bbh"], batch_size=8, 
                    output_result_dir=os.path.join(output_result_dir, 'bbh.json'),
                    use_prompt=self.data_args.use_prompt)
        if "truthfulqa" in self.data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, 'truthful.json')):
            print("Evaluate truthful...")
            eval_truthfulqa(model, e_tokenizer, self.eval_datasets["truthfulqa"], batch_size=4, 
                            output_result_dir=os.path.join(output_result_dir, 'truthful.json'),
                            use_prompt=self.data_args.use_prompt)
        if "triviaqa" in self.data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, 'triviaqa.json')):
            print("Evaluate triviaqa...")
            eval_triviaqa(model, e_tokenizer, self.eval_datasets["triviaqa"], batch_size=16, 
                            output_result_dir=os.path.join(output_result_dir, 'triviaqa.json'),
                            use_prompt=self.data_args.use_prompt)
        if "mia" in self.data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, 'forget_mia.json')):
            print("Evaluate forget mia...")
            eval_mia(model, e_tokenizer, self.eval_datasets["forget_mia"], 
                    output_result_dir=os.path.join(output_result_dir, 'forget_mia.json'),
                    use_prompt=self.data_args.use_prompt)
            print("Evaluate retain mia...")
            eval_mia(model, e_tokenizer, self.eval_datasets["retain_mia"], 
                    output_result_dir=os.path.join(output_result_dir, 'retain_mia.json'),
                    use_prompt=self.data_args.use_prompt)
        if "fluency" in self.data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, 'fluency.json')):
            print("Evaluate fluency...")
            eval_fluency(model, e_tokenizer, self.eval_datasets["fluency"], batch_size=8, 
                        output_result_dir=os.path.join(output_result_dir, 'fluency.json'),
                        use_prompt=self.data_args.use_prompt)        
        print(f"*** Finished evaluating after {counter_name} {counter_value} ***\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Run evaluation after an epoch if configured."""
        with torch.no_grad():
            if self.val_strategy in ["epoch", "both"] and (int(state.epoch) % self.eval_freq == 0 or state.global_step == state.max_steps):
                model = kwargs['model']
                e_tokenizer = kwargs['processing_class']
                e_tokenizer.pad_token = e_tokenizer.eos_token
                self._evaluate(model, e_tokenizer, int(state.epoch), "epoch")
        super().on_epoch_end(args, state, control, **kwargs)
        
    def on_step_end(self, args, state, control, **kwargs):
        """Run evaluation after steps if configured."""
        with torch.no_grad():
            if self.val_strategy in ["step", "both"] and (state.global_step % self.eval_freq == 0 or state.global_step == state.max_steps):
                model = kwargs['model']
                e_tokenizer = kwargs['processing_class']
                e_tokenizer.pad_token = e_tokenizer.eos_token
                self._evaluate(model, e_tokenizer, state.global_step, "step")
        super().on_step_end(args, state, control, **kwargs)

class MUSEEVALCallback(TrainerCallback):
    def __init__(self, eval_datasets, data_args, model_args,
                 val_strategy: Literal["epoch", "step", "both"] = "epoch",
                 eval_freq: int = 1):
        self.eval_datasets = eval_datasets
        self.data_args = data_args
        self.model_args = model_args
        self.val_strategy = val_strategy
        self.eval_freq = max(1, eval_freq)
        super().__init__()
        
    def _evaluate(self, model, e_tokenizer, counter_value, counter_name):
        """Common evaluation logic for both epochs and steps."""
        target = self.data_args.target
        output_result_dir = os.path.join(self.data_args.output_result_dir, target, f"{counter_name}_{counter_value}")
        for metric in self.data_args.eval_metrics:
            os.makedirs(os.path.join(output_result_dir, metric), exist_ok=True)
        
        print(f"\n*** Evaluating after {counter_name} {counter_value} ***")
        
        # Existing evaluation code
        verbmem_forget_file = DEFAULT_DATA[target]['verbmem_forget_file'] 
        privleak_forget_file = DEFAULT_DATA[target]['privleak_forget_file'] 
        privleak_retain_file = DEFAULT_DATA[target]['privleak_retain_file'] 
        privleak_holdout_file = DEFAULT_DATA[target]['privleak_holdout_file']
        knowmem_forget_qa_file = DEFAULT_DATA[target]['knowmem_forget_qa_file']
        knowmem_forget_qa_icl_file = DEFAULT_DATA[target]['knowmem_forget_qa_icl_file']
        knowmem_retain_qa_file = DEFAULT_DATA[target]['knowmem_retain_qa_file']
        knowmem_retain_qa_icl_file = DEFAULT_DATA[target]['knowmem_retain_qa_icl_file']
        out = {}
        model.eval()
        if "verbmem_f" in self.data_args.eval_metrics:
            print("Evaluate verbmem_f")
            data = read_json(verbmem_forget_file)
            agg, log = eval_verbmem(
                prompts=[d['prompt'] for d in data],
                gts=[d['gt'] for d in data],
                model=model, tokenizer=e_tokenizer,
                max_new_tokens=128
            )
            write_json(agg, os.path.join(output_result_dir, "verbmem_f/agg.json"))
            write_json(log, os.path.join(output_result_dir, "verbmem_f/log.json"))
            out['verbmem_f'] = agg["mean_rougeL"] * 100

        if "privleak" in self.data_args.eval_metrics:
            print("Evaluate privleak")
            auc, log = eval_privleak(
                forget_data=read_json(privleak_forget_file),
                retain_data=read_json(privleak_retain_file),
                holdout_data=read_json(privleak_holdout_file),
                model=model, tokenizer=e_tokenizer
            )
            write_json(auc, os.path.join(output_result_dir, "privleak/auc.json"))
            write_json(log, os.path.join(output_result_dir, "privleak/log.json"))
            out['privleak'] = (auc["forget_holdout_Min-40%'"] - AUC_RETRAIN["forget_holdout_Min-40%'"]) / AUC_RETRAIN["forget_holdout_Min-40%'"] * 100

        if "knowmem_f" in self.data_args.eval_metrics:
            print("Evaluate knowmem_f")
            qa = read_json(knowmem_forget_qa_file)
            icl = read_json(knowmem_forget_qa_icl_file)
            agg, log = eval_knowmem(
                questions=[d['question'] for d in qa],
                answers=[d['answer'] for d in qa],
                icl_qs=[d['question'] for d in icl],
                icl_as=[d['answer'] for d in icl],
                model=model, tokenizer=e_tokenizer,
                max_new_tokens=32
            )
            write_json(agg, os.path.join(output_result_dir, "knowmem_f/agg.json"))
            write_json(log, os.path.join(output_result_dir, "knowmem_f/log.json"))
            out['knowmem_f'] = agg["mean_rougeL"] * 100

        if "knowmem_r" in self.data_args.eval_metrics:
            print("Evaluate knowmem_r")
            qa = read_json(knowmem_retain_qa_file)
            icl = read_json(knowmem_retain_qa_icl_file)
            agg, log = eval_knowmem(
                questions=[d['question'] for d in qa],
                answers=[d['answer'] for d in qa],
                icl_qs=[d['question'] for d in icl],
                icl_as=[d['answer'] for d in icl],
                model=model, tokenizer=e_tokenizer,
                max_new_tokens=32
            )
            if output_result_dir is not None:
                write_json(agg, os.path.join(output_result_dir, "knowmem_r/agg.json"))
                write_json(log, os.path.join(output_result_dir, "knowmem_r/log.json"))
            out['knowmem_r'] = agg["mean_rougeL"] * 100
        with open(os.path.join(output_result_dir, "metrics.json"), "w") as f:
            json.dump(out, f, indent=4)
        print(f"*** Finished evaluating after {counter_name} {counter_value} ***\n")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Run evaluation after an epoch if configured."""
        with torch.no_grad():
            if self.val_strategy in ["epoch", "both"] and (int(state.epoch) % self.eval_freq == 0 or state.global_step == state.max_steps):
                model = kwargs['model']
                e_tokenizer = kwargs['processing_class']
                e_tokenizer.pad_token = e_tokenizer.eos_token
                self._evaluate(model, e_tokenizer, int(state.epoch), "epoch")
        super().on_epoch_end(args, state, control, **kwargs)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Run evaluation after steps if configured."""
        with torch.no_grad():
            if self.val_strategy in ["step", "both"] and (state.global_step % self.eval_freq == 0 or state.global_step == state.max_steps):
                model = kwargs['model']
                e_tokenizer = kwargs['processing_class']
                e_tokenizer.pad_token = e_tokenizer.eos_token
                self._evaluate(model, e_tokenizer, state.global_step, "step")
        super().on_step_end(args, state, control, **kwargs)
