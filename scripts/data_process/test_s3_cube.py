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

def make_prefix(dp, retriever):
    input_str = """You are a search copilot for the generation model. Based on a user's query and initial searched results, you will first determine if the searched results are enough to produce an answer.
If the searched results are enough, you will use <search_complete>True</search_complete> to indicate that you have gathered enough information for the generation model to produce an answer.
If the searched results are not enough, you will go through a loop of <query> -> <information> -> <important_info> -> <search_complete> -> <query> (if not complete) ..., to help the generation model to generate a better answer with more relevant information searched.
You should show the search query between <query> and </query> in JSON format.
Based on the search query, we will return the top searched results between <information> and </information>. You need to put the doc ids of the important documents (up to 3 documents, within the current information window) between <important_info> and </important_info> (e.g., <important_info>[1, 4]</important_info>).
A search query MUST be followed by a <search_complete> tag if the search is not complete.
After reviewing the information, you must decide whether to continue searching with a new query or indicate that the search is complete. If you need more information, use <search_complete>False</search_complete> to indicate you want to continue searching with a better query. Otherwise, use <search_complete>True</search_complete> to terminate the search.
During the process, you can add reasoning process within <think></think> tag whenever you want. Note: Only the important information would be used for the generation model to produce an answer.
"""

    if retriever == "bm25":
        input_str += """Note: The search query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately."""

    input_str += """
For a question and initial searched results:
<question>
[user's question]
</question>
<information>
[initial searched results]
</information>

If the initial searched results are enough to produce an answer, you should output:
<search_complete>
True
</search_complete>

If the initial searched results are not enough to produce an answer, you should output:
<query>
{
    "query": "[search query]"
} 
</query>
<information>
[top searched results based on the above search query]
</information>
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
...... (can be several turns until <search_complete> is True)

<search_complete>
True
</search_complete>

Now, start the loop with the following question and initial searched results:
"""

    input_str += f"""
<question>
{dp['question']}
</question>
<information>
{dp['initial_searched_results'].strip()}
</information>
"""
    return input_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_sources', default='hotpotqa')
    parser.add_argument('--retriever', default="e5")
    parser.add_argument('--initial_searched_results_dir', default="data/cube/rag_e5_cube")
    args = parser.parse_args()

    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        with open(f"/shared/eng/pj20/cube_data/{data_source}_with_index.json", "r") as f:
            test_dataset = json.load(f)

        initial_searched_results = json.load(open(os.path.join(args.initial_searched_results_dir, f'{data_source}_output_sequences.json')))

        # Process each item in the list of dictionaries
        processed_data = []
        for idx, example in enumerate(test_dataset):
            if 'answer' not in example:
                example['answer'] = example['gold_ans']   # for lveval dataset

            example['question'] = example['question'].strip()
            example['initial_searched_results'] = initial_searched_results[example['question']]['context_with_info'].split("\nDoc 6")[0] + "\n"
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
                "initial_searched_results": example['initial_searched_results'],
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
    all_test_dataset.to_parquet(os.path.join(local_dir, f'test_{args.retriever}_cube_{data_source}.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)