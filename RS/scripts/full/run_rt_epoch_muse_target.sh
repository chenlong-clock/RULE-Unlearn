
id=books
if [ -d "/netcache" ]; then
    root_path="/netcache"
else
    root_path="/mnt/usercache"
fi

model="MUSE-${id}_target"
mkdir -p ./logs/MUSE/Target/$id/rt_target_full/rt_full/
echo $id
PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py --stage rt \
    --model_name_or_path $root_path/zhangchenlong/$model --do_train --save_model --save_strategy epoch \
    --dataset MUSE/$id/Reject_Target_0.5 --dataset_dir ./data --finetuning_type full \
    --output_dir ./saves/MUSE/Target/Target/${id}/rt_target_full/llama2_7b_instruct_bf16 \
    --overwrite_output_dir --cutoff_len 640 --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 \
    --template llama2 \
    --learning_rate 4e-7 --num_train_epochs 3.0 --val_size 0 \
    --eval_metrics verbmem_f knowmem_f knowmem_r --val_strategy epoch --eval_freq 1 \
    --output_result_dir ./results/MUSE/Target/${id}/rt_target_full/llama2_7b_instruct_bf16 \
    --bf16 --eval_dataset_dir ./data/MUSE/Target/$id \
    --target ${id} 2>&1 | tee ./logs/MUSE/Target/$id/rt_target_full/rt_full/llama2_7b_instruct_bf16_${id}.log