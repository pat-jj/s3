data_name=nq_hotpotqa_train

RANDOM_SEED=${1:-42}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR=data/${data_name} # first download the data from https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train

export BASE_MODEL=""

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# DATASET=2wikimultihopqa
# DATASET=hotpotqa
# DATASET=lveval
DATASET=musique

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=data/cube/test_e5_cube_${DATASET}.parquet \
    data.val_files=data/cube/test_e5_cube_${DATASET}.parquet\
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=8000 \
    data.max_response_length=500 \
    data.max_start_length=2000 \
    data.max_obs_length=1400 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=verl_checkpoints/s3_8_3_3_3948/actor/global_step_50 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=verl_checkpoints/s3_8_3_3_3948/actor/global_step_50 \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=16 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=[] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    max_turns=4 \
    +data.random_seed=$RANDOM_SEED \
    +generator_llm="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4" \
    +output_context_dir=data/cube/output_sequences_s3_8_3_3_3948_cube_${DATASET}_step_50 \
    retriever.url="http://127.0.0.1:3000/retrieve" \
    retriever.topk=8 \
    2>&1 | tee evaluation_s3_8_3_3_3948_cube_${DATASET}_step_50.log
