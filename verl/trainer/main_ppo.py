"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
# from verl.utils.reward_score import rag, ppl, rag_new
# from verl.utils.reward_score import rag_new
from verl.utils.reward_score import rag_2
# from verl.utils.reward_score import ret
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import os
import json
import numpy as np
import threading
import random

from verl.utils.reward_score.rag_2 import output_sequence

USE_UTILITY_SCORE = True
USE_GENERATION_SCORE = True


def _select_rm_score_fn(data_source):
        return rag_2.compute_score_rag


class RewardManager():
    """The reward manager.
    """#data/Qwen_Qwen2.5-14B-Instruct-GPTQ-Int4/train/zeroshot_answers.json

    def __init__(self, tokenizer, num_examine, format_score=0., zeroshot_cache_file="data/rag_cache/rag_cache.json", val_only=False, output_context_dir=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.zeroshot_cache_file = zeroshot_cache_file
        self.zeroshot_lock = threading.Lock()
        self.val_only = val_only
        # Add new cache directory for output sequences
        self.output_sequences_dir = output_context_dir
        os.makedirs(self.output_sequences_dir, exist_ok=True)
        self.output_sequences_lock = threading.Lock()
        self.output_sequences_data = {}
        
        if os.path.exists(self.zeroshot_cache_file):
            print(f"load zeroshot answers from {self.zeroshot_cache_file}")
            self.zeroshot_answers = json.load(open(self.zeroshot_cache_file))
        else:
            os.makedirs(os.path.dirname(self.zeroshot_cache_file), exist_ok=True)
            self.zeroshot_answers = {}

    def _save_output_sequences(self, data_source):
        """Save output sequences to file for a specific data source"""
        if data_source not in self.output_sequences_data:
            return
            
        cache_file = os.path.join(self.output_sequences_dir, f"{data_source}_output_sequences.json")
        with self.output_sequences_lock:
            with open(cache_file, 'w') as f:
                json.dump(self.output_sequences_data[data_source], f, indent=2)

    def save_all_output_sequences(self):
        """Save all output sequences for all data sources"""
        with self.output_sequences_lock:
            for data_source in self.output_sequences_data:
                if data_source not in self.output_sequences_data:
                    continue
                    
                cache_file = os.path.join(self.output_sequences_dir, f"{data_source}_output_sequences.json")
                with open(cache_file, 'w') as f:
                    json.dump(self.output_sequences_data[data_source], f, indent=2)

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # Get scores and zeroshot info
            if not self.val_only:
                score, answer_zeroshot, answer_zeroshot_score = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth=ground_truth, 
                    zeroshot_answers=self.zeroshot_answers,
                    data_source=data_source,
                    use_utility_score=USE_UTILITY_SCORE,
                    use_generation_score=USE_GENERATION_SCORE
                )
                
            else:
                print(f"start output sequence")
                question, golden_answers, context_with_info, response_str = output_sequence(solution_str=sequences_str, ground_truth=ground_truth)
                print(f"output sequence end")
                score = 0
                
                # Store output sequence results
                with self.output_sequences_lock:
                    if data_source not in self.output_sequences_data:
                        self.output_sequences_data[data_source] = {}
                    
                    self.output_sequences_data[data_source][question] = {
                        'golden_answers': golden_answers,
                        'context_with_info': context_with_info,
                        'response_str': response_str
                    }
                    
                    # Periodically save to file (1% chance)
                    if random.random() < 0.01:
                        # Save to a temporary file first
                        temp_file = os.path.join(self.output_sequences_dir, f"{data_source}_output_sequences.json.tmp")
                        with open(temp_file, 'w') as f:
                            json.dump(self.output_sequences_data[data_source], f, indent=2)
                        # Atomic rename
                        final_file = os.path.join(self.output_sequences_dir, f"{data_source}_output_sequences.json")
                        os.rename(temp_file, final_file)
                        
                print(f"output sequence data end")

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, output_context_dir=config.output_context_dir)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, val_only=True, output_context_dir=config.output_context_dir)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            use_generation_score=USE_GENERATION_SCORE,
                            use_utility_score=USE_UTILITY_SCORE
                            )
    trainer.init_workers()
    trainer.fit()
    
    # Save all output sequences at the end of training
    reward_fn.save_all_output_sequences()
    val_reward_fn.save_all_output_sequences()


if __name__ == '__main__':
    main()
