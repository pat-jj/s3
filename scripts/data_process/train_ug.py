# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


# def make_prefix(dp, template_type):
#     question = dp['question']

#     # NOTE: also need to change reward_score/countdown.py
#     if template_type == 'base':
#         """This works for any base model"""
#         prefix = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you lack some knowledge, you can call a search engine by <query> and it will return the top searched results between <information> and </information>. \
# You can search as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
#     else:
#         raise NotImplementedError
#     return prefix
# Note: The search query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately.

def make_prefix(dp, retriever):

    # input_str = """<|im_start|>system\nA conversation between User and Assistant. The User asks a question, and the Assistant solves it.<|im_end|>\n<|im_start|>user\n"""
    input_str = """You are a search copilot for the generation model. Based on a user's query, you will go through a loop of <think> -> <query> -> <information> -> <think> -> <important_info> -> <search_complete> -> <query> (if not complete) ..., to help the generation model to generate a better answer with more relevant information searched.
You should show your thinking process between <think> and </think>. You should show the search query between <query> and </query> in JSON format.
Based on the search query, we will return the top searched results between <information> and </information>. You need to first think (<think>) on the retrieved information and put the doc id (1, 2, 3) of the important documents between <important_info> and </important_info> (e.g., <important_info>[1, 2]</important_info>).
After reviewing the information, you must decide whether to continue searching with a new query or indicate that the search is complete. If you need more information, formulate a new search query OR use <search_complete>False</search_complete> to indicate you want to continue searching with a better query. If you have sufficient information, use <search_complete>True</search_complete> to indicate that you have gathered enough information for the generation model to produce an answer.
"""

    if retriever == "bm25":
        input_str += """Note: The search query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately."""

    input_str += """
For a question:
<question>
[user's question]
</question>

The loop is as follows:
<think>
[thinking process]
</think>
<query>
{
    "query": "[search query]"
} 
</query>
<information>
[top searched results]
</information>
<think>
[analyze the search results]
</think>
<important_info>
[doc ids]
</important_info>
<search_complete>
False
</search_complete>
<query>
{
    "query": "[search query]"
}
</query>
...... (several turns, max 4 turns in total)

<search_complete>
True
</search_complete>

Now, start the loop with the following question:
<question>
"""

    input_str +=  dp['question'] + """
</question>
"""
    return input_str



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--retriever', default="bm25")

    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        train_dataset = dataset['train']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                
                if "yes" in example['golden_answers'] or "no" in example['golden_answers'] or "true" in example['golden_answers'] or "false" in example['golden_answers'] or "Yes" in example['golden_answers'] or "No" in example['golden_answers'] or "True" in example['golden_answers'] or "False" in example['golden_answers']:
                    return None
                
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, args.retriever)
                solution = {
                    "question": example['question'],
                    "target": example['golden_answers']
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, f'train_{args.retriever}_ug.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
