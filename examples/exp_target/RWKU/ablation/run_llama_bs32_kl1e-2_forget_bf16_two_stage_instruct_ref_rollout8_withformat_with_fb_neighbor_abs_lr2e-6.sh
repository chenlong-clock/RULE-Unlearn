set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

names=(
    '1_Stephen_King' '2_Confucius' '3_Bruce_Lee' '4_Warren_Buffett' '5_Christina_Aguilera'
    '6_Cindy_Crawford' '7_Marie_Osmond' '8_Paris_Hilton' '9_Justin_Bieber' '10_Prince_Harry,_Duke_of_Sussex'
)
for checkpint in 76
do
    for name in "${names[@]}"
    do
        echo "Running ${name} with checkpoint-${checkpint}"
        save_dir=/mnt/userdata/RL-Unlearning/easyr1/saves/RWKU/Target/${name}
        mkdir -p ${save_dir}
        experiment_name=RWKU/exp_target/llama_3_8b_rj_step${checkpint}_bs32_kl1e-2_bf16_two_stage_instruct_ref_rollout8_withformat_with_fb_neighbor_abs_target_lr2e-6
        MODEL_PATH=/mnt/usercache/zhangchenlong/RL-Unlearning/LLaMA-Factory/saves/RWKU/Target/${name}/rt_target_full/llama3_8b_instruct_bf16/checkpoint-${checkpint}  # replace it with your local file path
        # MODEL_PATH=/mnt/usercache/huggingface/Meta-Llama-3-8B-Instruct
        log_path=./log/${experiment_name}/$name.log
        if [ -f ${log_path} ]; then
            echo "Skip existing log file: ${log_path}"
            continue
        fi
        mkdir -p ./log/${experiment_name}
        python3 -m verl.trainer.main \
            config=examples/grpo_rj_rl.yaml \
            worker.actor.model.model_path=${MODEL_PATH} \
            worker.ref.model.model_path=/mnt/usercache/huggingface/Meta-Llama-3-8B-Instruct \
            trainer.save_checkpoint_path=${save_dir}/${experiment_name} \
            trainer.save_freq=-2 \
            trainer.experiment_name=${experiment_name}/$name \
            trainer.n_gpus_per_node=4 \
            worker.rollout.tensor_parallel_size=1 \
            data.train_files=data/RWKU/Target/${name}/rl_rejection_with_fb_neighbor_target.json \
            data.rollout_batch_size=32 \
            worker.actor.global_batch_size=32 \
            worker.actor.micro_batch_size_per_device_for_update=8 \
            worker.actor.micro_batch_size_per_device_for_experience=16 \
            algorithm.kl_coef=1e-2 \
            worker.rollout.n=8 \
            worker.actor.optim.lr=2e-6 \
            trainer.max_steps=20 \
            trainer.val_freq=-1 \
            trainer.total_episodes=1 \
            worker.reward.score_function="forget_two_stage_abs_target" \
            data.val_files=data/RWKU/Target/${name}/rl_rejection_eval_target.json \
             2>&1 | tee ${log_path}


    done
done
