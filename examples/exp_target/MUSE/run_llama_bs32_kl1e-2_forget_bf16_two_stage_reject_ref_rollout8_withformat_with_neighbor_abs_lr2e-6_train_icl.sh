set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

names=(
    # "news"
    "books"
)
for checkpint in 208 416 624
do
    for name in "${names[@]}"
    do
        echo "Running ${name} with checkpoint-${checkpint}"
        save_dir=/mnt/userdata/RL-Unlearning/easyr1/saves/MUSE/${name}
        mkdir -p ${save_dir}
        experiment_name=MUSE/exp_target/llama_2_7b_rj_step${checkpint}_bs32_kl1e-2_bf16_two_stage_reject_ref_rollout8_withformat_with_neighbor_abs_lr2e-6_train_icl
        MODEL_PATH=/mnt/usercache/zhangchenlong/RL-Unlearning/LLaMA-Factory/saves/MUSE/Target/Target/$name/rt_target_full/llama2_7b_instruct_bf16/checkpoint-${checkpint}  # replace it with your local file path
        # MODEL_PATH=/mnt/usercache/huggingface/Meta-Llama-3-8B-Instruct
        log_path=./log/${experiment_name}/$name.log
        # if [ -f ${log_path} ]; then
        #     echo "Skip existing log file: ${log_path}"
        #     continue
        # fi
        mkdir -p ./log/${experiment_name}
        python3 -m verl.trainer.main \
            config=examples/grpo_rj_rl_muse.yaml \
            worker.actor.model.model_path=${MODEL_PATH} \
            worker.ref.model.model_path=${MODEL_PATH} \
            trainer.save_checkpoint_path=${save_dir}/${experiment_name} \
            trainer.save_freq=-2 \
            trainer.experiment_name=${experiment_name}/$name \
            trainer.n_gpus_per_node=4 \
            worker.rollout.tensor_parallel_size=1 \
            data.train_files=data/MUSE/${name}/rl_rejection_with_neighbor.json \
            data.rollout_batch_size=32 \
            worker.actor.global_batch_size=32 \
            worker.actor.micro_batch_size_per_device_for_update=8 \
            worker.actor.micro_batch_size_per_device_for_experience=16 \
            algorithm.kl_coef=1e-2 \
            worker.rollout.n=8 \
            worker.actor.optim.lr=2e-6 \
            trainer.total_episodes=1 \
            worker.reward.score_function="forget_two_stage_abs" \
            data.val_files=data/MUSE/${name}/rl_rejection_eval_target.json \
            data.icl_type=train \
             2>&1 | tee ${log_path}

    # make_vals
    done
done
