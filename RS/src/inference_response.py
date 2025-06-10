import json
import os
from openai import OpenAI
import torch
from llmtuner.eval.inference import generate_completions
from llmtuner.data import get_dataset
from llmtuner.eval.muse.constants import DEFAULT_DATA
from llmtuner.eval.muse.utils import read_json, write_json
from llmtuner.eval.rwku import eval_bbh, eval_fluency, eval_forget, eval_mia, eval_mmlu, eval_neighbor, eval_triviaqa, eval_truthfulqa
from llmtuner.eval.muse.constants import SUPPORTED_METRICS, CORPORA, LLAMA_DIR, DEFAULT_DATA, AUC_RETRAIN

from llmtuner.eval.muse import eval_knowmem, eval_privleak, eval_verbmem
from llmtuner.hparams.parser import get_train_args
from llmtuner.model import load_model, load_tokenizer
from typing import *

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

def run_inference(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    model_args.use_cache= True
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    model.eval()
    tokenizer = tokenizer_module["tokenizer"]
    target = data_args.target
    eval_dataset_dir = data_args.eval_dataset_dir
    target = data_args.target
    eval_dataset_dir = os.path.join(eval_dataset_dir, target)
    save_type = f"epoch_{int(data_args.epoch)}" if data_args.epoch is not None else f"step_{int(data_args.step)}"
    output_result_dir = os.path.join(data_args.output_result_dir, target, save_type)
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
        os.makedirs(output_result_dir, exist_ok=True)
        print(f"\n*** Evaluating after {save_type} ***")
        
        with torch.no_grad():
            if "forget" in data_args.eval_metrics:
                print("Evaluate forgetting...")
                if not os.path.exists(os.path.join(output_result_dir, 'forget.json')):
                    eval_forget(
                        model, tokenizer, 
                        forget_level1, 
                        forget_level2, 
                        forget_level3, 
                        batch_size=16, 
                        output_result_dir=os.path.join(output_result_dir, 'forget.json'),
                        use_prompt=data_args.use_prompt
                    )
                else:
                    print("Forgetting evaluation already exists, skipping...")
            if "neighbor" in data_args.eval_metrics:
                print("Evaluate neighbor...")
                if not os.path.exists(os.path.join(output_result_dir, 'neighbor.json')):
                    eval_neighbor(
                        model, tokenizer, 
                        neighbor_level1, 
                        neighbor_level2, 
                        batch_size=16, 
                        output_result_dir=os.path.join(output_result_dir, 'neighbor.json'),
                        use_prompt=data_args.use_prompt
                    )
                else:
                    print("Neighbor evaluation already exists, skipping...")
            if "mmlu" in data_args.eval_metrics:
                print("Evaluate mmlu...")
                if not os.path.exists(os.path.join(output_result_dir, 'mmlu.json')):
                    eval_mmlu(model, tokenizer, retain_mmlu, batch_size=1, 
                            output_result_dir=os.path.join(output_result_dir, 'mmlu.json'),
                            use_prompt=data_args.use_prompt)
                else:
                    print("MMLU evaluation already exists, skipping...")
            if "bbh" in data_args.eval_metrics:
                print("Evaluate bbh...")
                if not os.path.exists(os.path.join(output_result_dir, 'bbh.json')):
                    eval_bbh(model, tokenizer, retain_bbh, batch_size=16, 
                            output_result_dir=os.path.join(output_result_dir, 'bbh.json'),
                            use_prompt=data_args.use_prompt)
                else:
                    print("BBH evaluation already exists, skipping...")
            if "truthfulqa" in data_args.eval_metrics:
                print("Evaluate truthful...")
                if not os.path.exists(os.path.join(output_result_dir, 'truthful.json')):
                    eval_truthfulqa(model, tokenizer, truthfulqa, batch_size=16, 
                                    output_result_dir=os.path.join(output_result_dir, 'truthful.json'),
                                    use_prompt=data_args.use_prompt)
                else:
                    print("Truthful evaluation already exists, skipping...")
            if "triviaqa" in data_args.eval_metrics:
                print("Evaluate triviaqa...")
                if not os.path.exists(os.path.join(output_result_dir, 'triviaqa.json')):
                    eval_triviaqa(model, tokenizer, triviaqa, batch_size=16, 
                                    output_result_dir=os.path.join(output_result_dir, 'triviaqa.json'),
                                    use_prompt=data_args.use_prompt)
                else:
                    print("TriviaQA evaluation already exists, skipping...")
            if "mia" in data_args.eval_metrics:
                print("Evaluate forget mia...")
                if not os.path.exists(os.path.join(output_result_dir, 'forget_mia.json')):
                    eval_mia(model, tokenizer, forget_mia, 
                            output_result_dir=os.path.join(output_result_dir, 'forget_mia.json'),
                            use_prompt=data_args.use_prompt)
                else:
                    print("MIA evaluation already exists, skipping...")
                print("Evaluate retain mia...")
                if not os.path.exists(os.path.join(output_result_dir, 'retain_mia.json')):
                    eval_mia(model, tokenizer, retain_mia,
                            output_result_dir=os.path.join(output_result_dir, 'retain_mia.json'),
                            use_prompt=data_args.use_prompt)
                else:
                    print("MIA evaluation already exists, skipping...")
            if "fluency" in data_args.eval_metrics:
                print("Evaluate fluency...")
                if not os.path.exists(os.path.join(output_result_dir, 'fluency.json')):
                    eval_fluency(model, tokenizer, fluency, batch_size=16, 
                                output_result_dir=os.path.join(output_result_dir, 'fluency.json'),
                                use_prompt=data_args.use_prompt)
                else:
                    print("Fluency evaluation already exists, skipping...")
            print(f"*** Finished evaluating after {save_type} ***\n")
    elif "MUSE" in eval_dataset_dir:
        verbmem_forget_file = DEFAULT_DATA[target]['verbmem_forget_file']
        privleak_forget_file = DEFAULT_DATA[target]['privleak_forget_file'] 
        privleak_retain_file = DEFAULT_DATA[target]['privleak_retain_file'] 
        privleak_holdout_file = DEFAULT_DATA[target]['privleak_holdout_file'] 
        knowmem_forget_qa_file = DEFAULT_DATA[target]['knowmem_forget_qa_file'] 
        knowmem_forget_qa_icl_file = DEFAULT_DATA[target]['knowmem_forget_qa_icl_file'] 
        knowmem_retain_qa_file = DEFAULT_DATA[target]['knowmem_retain_qa_file']
        knowmem_retain_qa_icl_file = DEFAULT_DATA[target]['knowmem_retain_qa_icl_file'] 
        out = {}
        for metric in ["verbmem_f", "knowmem_f", "knowmem_r"]:
            os.makedirs(os.path.join(output_result_dir, metric), exist_ok=True)
        with torch.no_grad():
            if "verbmem_f" in data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, "verbmem_f/agg.json")):
                print("Evaluate verbmem_f")
                data = read_json(verbmem_forget_file)
                agg, log = eval_verbmem(
                    prompts=[d['prompt'] for d in data],
                    gts=[d['gt'] for d in data],
                    model=model, tokenizer=tokenizer,
                    max_new_tokens=128
                )
                if output_result_dir is not None:
                    write_json(agg, os.path.join(output_result_dir, "verbmem_f/agg.json"))
                    write_json(log, os.path.join(output_result_dir, "verbmem_f/log.json"))
                out['verbmem_f'] = agg["mean_rougeL"] * 100
            elif os.path.exists(os.path.join(output_result_dir, "verbmem_f/agg.json")):
                out['verbmem_f'] = read_json(os.path.join(output_result_dir, "verbmem_f/agg.json"))["mean_rougeL"] * 100
                

            # print("Evaluate privleak")
            # auc, log = eval_privleak(
            #     forget_data=read_json(privleak_forget_file),
            #     retain_data=read_json(privleak_retain_file),
            #     holdout_data=read_json(privleak_holdout_file),
            #     model=model, tokenizer=tokenizer
            # )
            # write_json(auc, os.path.join(output_result_dir, "privleak/auc.json"))
            # write_json(log, os.path.join(output_result_dir, "privleak/log.json"))
            # out['privleak'] = (auc["forget_holdout_Min-40%'"] - AUC_RETRAIN["forget_holdout_Min-40%'"]) / AUC_RETRAIN["forget_holdout_Min-40%'"] * 100
            if "knowmem_f" in data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, "knowmem_f/agg.json")):
                print("Evaluate knowmem_f")
                qa = read_json(knowmem_forget_qa_file)
                icl = read_json(knowmem_forget_qa_icl_file)
                agg, log = eval_knowmem(
                    questions=[d['question'] for d in qa],
                    answers=[d['answer'] for d in qa],
                    icl_qs=[d['question'] for d in icl],
                    icl_as=[d['answer'] for d in icl],
                    model=model, tokenizer=tokenizer,
                    max_new_tokens=32
                )
                if output_result_dir is not None:
                    write_json(agg, os.path.join(output_result_dir, "knowmem_f/agg.json"))
                    write_json(log, os.path.join(output_result_dir, "knowmem_f/log.json"))
                out['knowmem_f'] = agg["mean_rougeL"] * 100
            elif os.path.exists(os.path.join(output_result_dir, "knowmem_f/agg.json")):
                out['knowmem_f'] = read_json(os.path.join(output_result_dir, "knowmem_f/agg.json"))["mean_rougeL"] * 100
                
            if "knowmem_r" in data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, "knowmem_r/agg.json")):
                print("Evaluate knowmem_r")
                qa = read_json(knowmem_retain_qa_file)
                icl = read_json(knowmem_retain_qa_icl_file)
                agg, log = eval_knowmem(
                    questions=[d['question'] for d in qa],
                    answers=[d['answer'] for d in qa],
                    icl_qs=[d['question'] for d in icl],
                    icl_as=[d['answer'] for d in icl],
                    model=model, tokenizer=tokenizer,
                    max_new_tokens=32
                )
            
                if output_result_dir is not None:
                    write_json(agg, os.path.join(output_result_dir, "knowmem_r/agg.json"))
                    write_json(log, os.path.join(output_result_dir, "knowmem_r/log.json"))
                out['knowmem_r'] = agg["mean_rougeL"] * 100
            elif os.path.exists(os.path.join(output_result_dir, "knowmem_r/agg.json")):
                out['knowmem_r'] = read_json(os.path.join(output_result_dir, "knowmem_r/agg.json"))["mean_rougeL"] * 100
            if "privleak" in data_args.eval_metrics and not os.path.exists(os.path.join(output_result_dir, "privleak/auc.json")):
                print("Evaluate privleak")
                auc, log = eval_privleak(
                    forget_data=read_json(privleak_forget_file),
                    retain_data=read_json(privleak_retain_file),
                    holdout_data=read_json(privleak_holdout_file),
                    model=model, tokenizer=tokenizer
                )
                if output_result_dir is not None:
                    write_json(auc, os.path.join(output_result_dir, "privleak/auc.json"))
                    write_json(log, os.path.join(output_result_dir, "privleak/log.json"))
                out['privleak'] = (auc["forget_holdout_Min-40%'"] - AUC_RETRAIN["forget_holdout_Min-40%'"]) / AUC_RETRAIN["forget_holdout_Min-40%'"] * 100
            elif os.path.exists(os.path.join(output_result_dir, "privleak/auc.json")):
                out['privleak'] = (read_json(os.path.join(output_result_dir, "privleak/auc.json"))["forget_holdout_Min-40%'"] - AUC_RETRAIN["forget_holdout_Min-40%'"]) / AUC_RETRAIN["forget_holdout_Min-40%'"] * 100
                    
        with open(os.path.join(output_result_dir, "metrics.json"), "w") as f:
            json.dump(out, f, indent=4)


if __name__=="__main__":
    run_inference()
