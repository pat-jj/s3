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
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse

def make_prefix(dp):
    question = dp['question']
    prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    return prefix


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
            question = make_prefix(example)
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
    all_test_dataset.to_parquet(os.path.join(local_dir, f'test_{args.retriever}_cube_{data_source}_r1.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)