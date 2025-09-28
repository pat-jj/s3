import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from verl import DataProto
import requests
import json

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: str = None
    topk: int = 3
    include_information: bool = False  # Whether to include search results in feedback
    generator_llm: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    output_context_dir: str = None
    
class LLMGenerationManager:
    """
    Search-C1: A search copilot that can be trained separately from the generator LLM.
    The generator LLM is treated as part of the environment.

    This implementation maintains output compatibility with Search-R1 for training,
    but conceptually separates the search copilot from the generator LLM.
    """
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,  # Worker group for the search copilot (to be trained)
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.timing_raw = {}
        # Load prompt templates for the generator LLM
        self.output_context_dir = config.output_context_dir
        # Initialize tensor helper for handling tensors
        from .tensor_helper import TensorHelper, TensorConfig
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        

    def _load_zeroshot_answers(self, filename):
        """Load zeroshot answers from file."""
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, IOError):
            print(f"Zeroshot answers file {filename} not found.")
            return {}

    def _load_prompt(self, filename):
        """Load prompt template from file."""
        try:
            with open(filename, 'r') as file:
                return file.read().strip()
        except (FileNotFoundError, IOError):
            # Return a default prompt if file not found
            raise ValueError(f"Prompt file {filename} not found.")

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str], List[str], List[bool]]:
        """Process responses to extract search queries."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Ensure responses end with </query> tag if it exists
        new_responses_str = []
        for resp in responses_str:
            if '</query>' in resp:
                resp = resp.split('</query>')[0] + '</query>'
                if '<query>' not in resp:
                    resp = '<query>\n' + resp
            new_responses_str.append(resp)
        responses_str = new_responses_str

        # Extract query information
        queries = []
        search_complete_flags = []
        
        for resp in responses_str:
            query_match = re.search(r'<query>(.*?)</query>', resp, re.DOTALL)
            search_complete_match = re.search(r'<search_complete>(.*?)</search_complete>', resp, re.DOTALL)
            
            # Check for search completion flag
            search_complete = False
            if search_complete_match:
                complete_text = search_complete_match.group(1).strip().lower()
                search_complete = complete_text == "true" or complete_text == "yes" or complete_text == "1" or complete_text == "y"
            
            # Extract query if present
            if query_match:
                query_text = query_match.group(1).strip()
                try:
                    import json
                    json_data = json.loads(query_text)
                    if 'query' in json_data:
                        queries.append(json_data['query'])
                    else:
                        queries.append("")
                except:
                    queries.append("")
            else:
                queries.append("")
                
            search_complete_flags.append(search_complete)
                    
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str, queries, search_complete_flags

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> DataProto:
        """
        Run the s3 loop with environment-based feedback and rewards.
        """
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        # Reset conversation histories for this batch
        self.conversation_histories = [""] * len(active_mask)
        
        # Extract the original questions from the initial inputs
        self.original_questions = [""] * len(active_mask)
        initial_inputs_str = self.tokenizer.batch_decode(
            initial_input_ids, 
            skip_special_tokens=True
        )
        
        # Extract questions from the initial inputs
        for i, input_text in enumerate(initial_inputs_str):
            question_matches = re.findall(r'<question>(.*?)</question>', input_text, re.DOTALL)
            if question_matches:
                # Use the last match of <question>...</question>
                self.original_questions[i] = question_matches[-1].strip()
            else:
                print(f"No <question>...</question> tags found in the initial input {input_text}")

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
                
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # Generate with active sequences
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # Process outputs 
            meta_info = gen_output.meta_info            
            responses_ids, responses_str, queries, search_complete_flags = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Execute search and get feedback from the environment
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            # Update active sequences
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            # Process observations (search results + feedback)
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # Final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # Process outputs - keeping exact compatibility with Search-R1
            meta_info = gen_output.meta_info            
            responses_ids, responses_str, queries, search_complete_flags = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Execute final predictions (without doing search)
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            # Update stats
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            next_obs_ids = self._process_next_obs(next_obs)

            # Update right side
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        
        # Store metadata for reward computation
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        return final_output

    def _example_level_pad(self, tensor, str_list, queries, end_flags, active_mask):
        """Pad tensors and lists back to full batch size - similar to original Search-R1."""
        if active_mask.all():
            return tensor, str_list, queries, end_flags
            
        full_size = active_mask.shape[0]
        active_indices = torch.where(active_mask)[0]
        
        # Pad tensor
        padded_tensor = torch.zeros(
            (full_size, tensor.shape[1]), 
            dtype=tensor.dtype, 
            device=tensor.device
        )
        padded_tensor[active_indices] = tensor
        
        # Pad string list
        padded_str = [""] * full_size
        for i, idx in enumerate(active_indices):
            padded_str[idx.item()] = str_list[i]
            
        # Pad queries
        padded_queries = [""] * full_size
        for i, idx in enumerate(active_indices):
            padded_queries[idx.item()] = queries[i]
            
        # Pad end flags
        padded_end_flags = [False] * full_size
        for i, idx in enumerate(active_indices):
            padded_end_flags[idx.item()] = end_flags[i]
            
        return padded_tensor, padded_str, padded_queries, padded_end_flags

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> Tuple[List[str], List[bool], List[int], List[int]]:
        """
        Execute predictions and generate external LLM feedback for Search-C1.
        """
        cur_actions, contents, search_complete_flags, important_doc_ids = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        # Track conversation history for each active example
        if not hasattr(self, 'conversation_histories'):
            self.conversation_histories = [""] * len(active_mask)
            
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search and search_queries:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            print("No search queries or no search is allowed")
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        search_result_counter = 0
        for i, (action, search_complete, active, doc_ids) in enumerate(zip(cur_actions, search_complete_flags, active_mask, important_doc_ids)):
            if not active and do_search:
                next_obs.append('')
                dones.append(True)
                valid_action.append(0)
                is_search.append(0)
            elif not active and not do_search:
                next_obs.append('')
                dones.append(True)
                valid_action.append(1)
                is_search.append(0)
            else:
                if action == 'search_complete' or not do_search:
                    next_obs.append('')
                    dones.append(True)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    if do_search:
                        search_result = search_results[search_result_counter].strip()
                        search_result_counter += 1

                        # Add important document IDs to conversation history
                        if doc_ids:
                            self.conversation_histories[i] += f"\nImportant documents: {doc_ids}\n"
                        
                        self.conversation_histories[i] += f"\nQuery: {contents[i]}\nSearch results: {search_result}"
                        
                        # Always include information tags
                        next_obs.append(f'\n\n<information>{search_result}</information>\n\n')
                        
                        dones.append(False)  # Always continue here, let the copilot decide in the next turn
                        valid_action.append(1)
                        is_search.append(1)
                    else:
                        next_obs.append('')
                        dones.append(True)
                        valid_action.append(1)
                        is_search.append(0)
                else:
                    # Invalid action
                    feedback = "\n\nThe information is not enough to answer the question. Let me dive deeper by generating a brand new search query between <query> and </query> tags in JSON format:\n<query>\n"
                    next_obs.append(feedback)
                    # next_obs.append('')
                    dones.append(False)
                    valid_action.append(0)
                    is_search.append(0)
            
        return next_obs, dones, valid_action, is_search
    
    
    
    def _check_feedback_for_stop(self, feedback: str) -> bool:
        """
        Check if the feedback indicates that we should stop searching.
        
        Args:
            feedback: The feedback from the generator LLM
            
        Returns:
            Boolean indicating whether to stop searching
        """
        # Look for explicit stop indicators in the feedback
        stop_patterns = [
            r'Stop Search:\s*Yes',
            r'stop searching',
            r'no need to search further',
            r'sufficient information',
            r'already have all the information needed',
            r'already have the answer',
            r'can answer the question',
            r'enough information to answer'
        ]
        
        for pattern in stop_patterns:
            if re.search(pattern, feedback, re.IGNORECASE):
                return True
                
        return False
    
        
    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str], List[bool], List[List[int]]]:
        """
        Process predictions to extract actions and content.
        Returns:
            Tuple of (actions, contents, search_complete_flags, important_doc_ids)
        """
        actions = []
        contents = []
        search_complete_flags = []
        important_doc_ids = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                # Extract search queries
                search_complete_match = re.search(r'<search_complete>(.*?)</search_complete>', prediction, re.DOTALL)
                query_match = re.search(r'<query>(.*?)</query>', prediction, re.DOTALL)
                important_info_match = re.search(r'<important_info>(.*?)</important_info>', prediction, re.DOTALL)
                
                # Parse important document IDs
                doc_ids = []
                if important_info_match:
                    try:
                        import json
                        doc_ids = json.loads(important_info_match.group(1).strip())
                        if not isinstance(doc_ids, list):
                            doc_ids = []
                    except:
                        doc_ids = []
                important_doc_ids.append(doc_ids)
                
                if query_match:
                    query_text = query_match.group(1).strip()
                    
                    # Check if the query is in JSON format
                    try:
                        # Try to parse as JSON
                        import json
                        json_data = json.loads(query_text)
                        try:
                            if 'query' in json_data:
                                try:
                                    content = json_data['query']
                                    if type(content) == list:
                                        content = content[0]
                                    elif type(content) == str:
                                        content = content
                                    else:
                                        print(f"Error in parsing query content from JSON: {query_text}")
                                        content = query_text
                                except:
                                    print(f"Error in parsing query: {query_text}")
                                    content = query_text
                                    
                                action = "search"
                            else:
                                content = query_text
                                action = "search"
                        except:
                            print(f"Error in json parsing: {json_data}")
                            content = query_text
                            action = "search"
                            
                    except json.JSONDecodeError:
                        # If not valid JSON, try to use the text directly
                        content = query_text
                        action = "search"
                        
                else:
                    content = ''
                    action = None
                    
                # Check for search completion flag
                search_complete = False
                if search_complete_match:
                    complete_text = search_complete_match.group(1).strip().lower()
                    search_complete = complete_text == "true" or complete_text == "yes" or complete_text == "1" or complete_text == "y"
                    if search_complete:
                        content = ""
                        action = "search_complete"
                
                actions.append(action)
                contents.append(content)
                search_complete_flags.append(search_complete)
            else:
                actions.append(None)
                contents.append('')
                search_complete_flags.append(False)
                important_doc_ids.append([])
            
        return actions, contents, search_complete_flags, important_doc_ids
    

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> DataProto:
        """Compose final output for the search copilot."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and info mask
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        try:
            results = self._batch_search(queries)['result']
        except Exception as e:
            print(f"Error in batch_search: {e}, queries: {queries}")
            return ["Error"] * len(queries)
            
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        """Call the search API."""
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        """Format retrieval results into a string."""
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            if "cube" in self.config.output_context_dir:
                content = doc_item['document']
                title = content['title']
                text = content['text']
            else:
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            
            # if "mirage" in self.config.output_context_dir:
            #     if "." in content:
            #         title = content.split(".")[0]
            #         text = content.split(".")[1]
            #     else:
            #         title = content.split("\n")[0]
            #         text = "\n".join(content.split("\n")[1:])
            # else:
            #     title = content.split("\n")[0]
            #     text = "\n".join(content.split("\n")[1:])
            # format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference