set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

for checkpint in 668 1002 334
do
    echo "Running with checkpoint-${checkpint}"
    save_dir=/mnt/userdata/RL-Unlearning/easyr1/saves/MUSE/
    mkdir -p ${save_dir}
    experiment_name=exp/llama_2_7b_rj_step${checkpint}_rl_bs32_kl1e-2_guide_bf16_two_stage_instruct_ref_rollout8_withformat_with_fb_neighbor_abs
    MODEL_PATH=/mnt/usercache/zhangchenlong/RL-Unlearning/LLaMA-Factory/saves/MUSE/Target/news/rt_target_full/llama2_7b_instruct_bf16/checkpoint-${checkpint}  # replace it with your local file path

    log_path=./log/MUSE/${experiment_name}/$name.log
    if [ -f ${log_path} ]; then
        echo "Skip existing log file: ${log_path}"
        continue
    fi
    mkdir -p ./log/MUSE/${experiment_name}
    python3 -m verl.trainer.main \
        config=examples/grpo_rj_rl_guide.yaml \
        worker.actor.model.model_path=${MODEL_PATH} \
        worker.ref.model.model_path=/mnt/usercache/huggingface/Llama-2-7b-chat-hf/ \
        trainer.save_checkpoint_path=${save_dir}/${experiment_name} \
        trainer.experiment_name=${experiment_name}/ \
        trainer.n_gpus_per_node=4 \
        worker.rollout.tensor_parallel_size=1 \
        data.train_files=data/MUSE/rl_rejection_with_fb_neighbor.json\
        data.rollout_batch_size=32 \
        worker.actor.global_batch_size=32 \
        worker.actor.micro_batch_size_per_device_for_update=8 \
        worker.actor.micro_batch_size_per_device_for_experience=16 \
        algorithm.kl_coef=1e-2 \
        worker.rollout.n=8 \
        trainer.total_episodes=1 \
        worker.reward.score_function="forget_two_stage_abs" \
        data.val_files=data/MUSE/rl_rejection_eval_target.json \ 
        2>&1 | tee ${log_path}
done
