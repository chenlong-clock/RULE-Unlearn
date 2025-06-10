#!/bin/bash
names=('1_Stephen_King' '2_Confucius' '3_Bruce_Lee' '4_Warren_Buffett' '5_Christina_Aguilera'
'6_Cindy_Crawford' '7_Marie_Osmond' '8_Paris_Hilton' '9_Justin_Bieber' '10_Prince_Harry,_Duke_of_Sussex'
)


if [ -d "/netcache" ]; then
    root_path="/netcache"
else
    root_path="/mnt/usercache"
fi
model="Meta-Llama-3-8B-Instruct"


for name in "${names[@]}"
do
    mkdir -p ./logs/RWKU/Target/rt_target_full
    id=$name
    echo $id
    if  [ -d "./saves/RWKU/Target/$id/rt_target_full/llama3_8b_instruct_bf16/checkpoint-114" ]; then
        echo "$id already done, skipping"
        continue
    fi

    PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py --stage rt \
    --model_name_or_path $root_path/huggingface/$model --do_train --save_model --save_strategy epoch --save_only_model\
    --dataset ${id}_Reject_Target --dataset_dir ./data --finetuning_type full \
    --output_dir ./saves/RWKU/Target/${id}/rt_target_full/llama3_8b_instruct_bf16 \
    --overwrite_output_dir --cutoff_len 512 --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 \
    --template llama3 \
    --learning_rate 4e-7 --num_train_epochs 3.0 --val_size 0 --plot_loss \
    --output_result_dir ./results/RWKU/Target/${id}/rt_target_full/llama3_8b_instruct_bf16_ep3 \
    --bf16 --eval_dataset_dir ./data/RWKU/Target/ \
    --target ${id} 2>&1 | tee ./logs/RWKU/Target/rt_target_full/llama3_8b_instruct_bf16_${id}_ep3.log
done
