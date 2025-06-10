import json
import os
import pandas as pd

# Function to load JSON data from a given file path
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

records = []
epoch_dict = {
                # "llama_3.1_8b_rj_step38_bs32_kl1e-2_bf16_two_stage_reject_ref_rollout8_withformat_with_fb_neighbor_abs_target_lr2e-6": 1,
                "llama_3.1_8b_rj_step76_bs32_kl1e-2_bf16_two_stage_reject_ref_rollout8_withformat_with_fb_neighbor_abs_target_lr2e-6": 2,
              }

# Loop through the directory structure
for ep in os.listdir("results/RWKU/exp_target"):
    if not ep in epoch_dict:
        continue
    for target in os.listdir(f"results/RWKU/exp_target/{ep}"):
        for step in os.listdir(f"results/RWKU/exp_target/{ep}/{target}"):
            forget_path = f"results/RWKU/exp_target/{ep}/{target}/{step}/forget.json"
            neighbor_path = f"results/RWKU/exp_target/{ep}/{target}/{step}/neighbor.json"
            
            # Load JSON files for forget and neighbor metrics
            forget_data = load_json(forget_path)
            neighbor_data = load_json(neighbor_path)
            
            # Calculate individual metrics in percentage
            forget_1 = forget_data["level_1_rouge_l_r"] * 100
            forget_2 = forget_data["level_2_rouge_l_r"] * 100
            forget_3 = forget_data["level_3_rouge_l_r"] * 100
            neighbor_1 = neighbor_data["level_1_rouge_l_r"] * 100
            neighbor_2 = neighbor_data["level_2_rouge_l_r"] * 100
            
            # Compute average scores
            forget_avg = (forget_1 + forget_2 + forget_3) / 3
            neighbor_avg = (neighbor_1 + neighbor_2) / 2
            
            epoch = epoch_dict[ep]
            step = int(step.split("step")[1])
            # Append all metrics along with ep, target, and step info to the records list
            records.append({
                "ep": epoch,
                "target": target,
                "step": step,
                "forget_1": forget_1,
                "forget_2": forget_2,
                "forget_3": forget_3,
                "neighbor_1": neighbor_1,
                "neighbor_2": neighbor_2,
            })

# Create a DataFrame from the collected records
df = pd.DataFrame(records)
eval_name = "llama3.1/grpo"
if not os.path.exists(f"csv_results/RWKU/{eval_name}"):
    os.makedirs(f"csv_results/RWKU/{eval_name}")
for epoch in df['ep'].unique():
    epoch_df = df[df['ep'] == epoch]
    # sort with int 
    epoch_df = epoch_df.sort_values(by='target', key=lambda x: x.str.split('_').str[0].astype(int)) 
    epoch_df['forget_avg'] = epoch_df[['forget_1', 'forget_2', 'forget_3']].mean(axis=1)

    epoch_df['neighbor_avg'] = epoch_df[['neighbor_1', 'neighbor_2']].mean(axis=1)
    # reorder the columns
    epoch_df = epoch_df[['step', 'target', 'forget_1', 'forget_2', 'forget_3', 'forget_avg', 'neighbor_1', 'neighbor_2', 'neighbor_avg']]
    # sort with target as first order and step as second order, with target sort as int
    epoch_df = epoch_df.sort_values(by=['target', 'step'])
    epoch_df.to_csv(f"csv_results/RWKU/{eval_name}/epoch_{epoch}.csv")
    

# Save a total table with epoch as rows and metrics as columns
df['forget_avg'] = df[['forget_1', 'forget_2', 'forget_3']].mean(axis=1)
df['neighbor_avg'] = df[['neighbor_1', 'neighbor_2']].mean(axis=1)
# Create a total table with step as rows and metrics as columns
total_table = df.groupby(['step', 'ep'])[['forget_1', 'forget_2', 'forget_3', 'forget_avg', 
                               'neighbor_1', 'neighbor_2', 'neighbor_avg']].mean()
total_table.to_csv(f"csv_results/RWKU/{eval_name}/total.csv")

