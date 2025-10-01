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
import json
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


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
    parser.add_argument('--data_sources', default='hotpotqa')
    parser.add_argument('--retriever', default="e5")

    args = parser.parse_args()

    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        with open(f"/shared/eng/pj20/cube_data/{data_source}_with_index.json", "r") as f:
            test_dataset = json.load(f)

        # Process each item in the list of dictionaries
        processed_data = []
        for idx, example in enumerate(test_dataset):
            if 'answer' not in example:
                example['answer'] = example['gold_ans']   # for lveval dataset

            example['question'] = example['question'].strip()
            question = make_prefix(example, args.retriever)
            solution = {
                "question": example['question'],
                "target": example['answer'],
                "gt_docs": example['supporting_facts_index'] if 'supporting_facts_index' in example else []
            }

            data = {
                "question": example['question'],
                "answer": example['answer'],
                "supporting_facts_index": example['supporting_facts_index'] if 'supporting_facts_index' in example else [],
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
                    'split': 'test',
                    'index': idx,
                }
            }
            processed_data.append(data)
        
        all_dataset.append(processed_data)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Flatten the list of lists into a single list
    all_test_data = []
    for dataset in all_dataset:
        all_test_data.extend(dataset)
    
    # Convert to Dataset and save as parquet
    all_test_dataset = datasets.Dataset.from_list(all_test_data)
    all_test_dataset.to_parquet(os.path.join(local_dir, f'test_{args.retriever}_cube_{data_source}_pre.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
