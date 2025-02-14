set -x
# ====================================================
#   Copyright (C) 2025  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : thh9bk@virginia.edu
#   File Name     : run_qwen2.5_math.sh
#   Last Modified : 2025-01-31 03:10
#   Describe      : 
#
# ====================================================


export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/p/llmresearch/thh9bk/verl/data/math-Qwen2.5-7B-correct_ratio_under_0.0_filtered_from_each_ques_273_ans_with_greedy/train.parquet \
    data.val_files=/p/llmresearch/thh9bk/verl/data/math/test.parquet \
    data.train_batch_size=220 \
    data.val_batch_size=1000 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=220 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=False \
    trainer.project_name='verl' \
    trainer.experiment_name='MATH-Qwen2.5-7B-GRPO-correct_ratio_under_0.0_filtered_from_each_ques_273_ans_with_greedy-no_kl_loss-classic_reward' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    reward_model.reward_type="classic" \
    reward_model.val_reward_type="classic" $@
    # trainer.experiment_name='MATH-Llama-3.1-Tulu-3-8B-SFT-GRPO-global_step_24-correct_ratio_under_0.0_filtered_from_each_ques_70_ans-classic_reward' \
    # data.train_files=/p/llmresearch/thh9bk/verl/data/math-Llama-3.1-Tulu-3-8B-SFT-correct_ratio_under_0.0_filtered_from_each_ques_70_ans/train.parquet \
    # data.val_files=/p/llmresearch/thh9bk/verl/data/math-Qwen2.5-7B-correct_ratio_under_0.0_filtered_questions/test.parquet \
