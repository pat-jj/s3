data_name=nq_hotpotqa_train

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export DATA_DIR=data/${data_name} # first download the data from https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train

WAND_PROJECT="Search-R1"

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-ppo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-ppo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-ppo-qwen2.5-7b-em
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct-1M'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-ppo-qwen2.5-7b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-14B'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-ppo-qwen2.5-14b-em
# export BASE_MODEL='Qwen/Qwen2.5-14B-Instruct'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-ppo-qwen2.5-14b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_r1 \
    data.train_files=data/nq_hotpotqa_train/train_r1.parquet \
    data.val_files=data/nq_hotpotqa_train/test_r1.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=120 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=30 \
    actor_rollout_ref.actor.ppo_micro_batch_size=15 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=30 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=30 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.01 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=10 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=5 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=50 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:3000/retrieve" \
    retriever.topk=3 
    # 2>&1 | tee $EXPERIMENT_NAME.log
