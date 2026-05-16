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
Preprocess the MATH-lighteval dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=os.path.expanduser('~/PRM_filter/data/math500'))
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'DigitalLearningGmbH/MATH-lighteval'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split='test')


    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

        # for Qwen2.5-instruct
#     system_prompt = """You are a helpful Assistant that solves mathematical problems step-by-step. Your task is to provide a detailed solution process within specific tags.

# You MUST follow this exact format:
# 1. Your entire response must consist of a step-by-step reasoning process.
# 2. Each distinct logical step MUST be separated by a double newline. Do not use any other separators.
# 3. After all reasoning steps, you MUST provide the final answer on a new line.
# 4. The final answer MUST ONLY be the value enclosed within \\boxed{}. Do not add any text before or after the boxed answer on that line.

# Here is an example of the required format:

# User: Calculate 15 - (3 * 2).

# Assistant:
# First, calculate the expression inside the parentheses, which is 3 multiplied by 2.

# 3 * 2 equals 6.

# Next, subtract the result from the original number, which is 15 minus 6.

# 15 - 6 equals 9.

# \\boxed{9}"""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following

            answer = example.pop('solution')
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset = test_dataset.shuffle(seed=42)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'normal_train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'normal_test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)