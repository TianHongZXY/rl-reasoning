#!/bin/bash
# ====================================================
#   Copyright (C) 2025  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : thh9bk@virginia.edu
#   File Name     : run.sh
#   Last Modified : 2025-01-28 22:59
#   Describe      : 
#
# ====================================================

export VLLM_ATTENTION_BACKEND=XFORMERS
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
 data.train_files=/p/llmresearch/thh9bk/verl/data/math/train.parquet \
 data.val_files=/p/llmresearch/thh9bk/verl/data/math/test.parquet \
 data.train_batch_size=64 \
 data.val_batch_size=1312 \
 data.max_prompt_length=512 \
 data.max_response_length=2048 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-MATH-1.5B\
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['wandb'] \
 +trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.project_name="verl" \
 trainer.experiment_name="MATH-Qwen2.5-MATH-1.5B-PPO-correct_reward_1_incorrect_reward_0" \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.save_freq=100 \
 trainer.test_freq=100 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
