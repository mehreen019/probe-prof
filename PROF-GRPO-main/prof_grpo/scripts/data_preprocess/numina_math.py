"""
Preprocess the Numia dataset to parquet format
"""

import os
import datasets
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from datasets import load_dataset

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=os.path.expanduser('~/PRM_filter/data/numina_math'))
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=10000000)
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-Math-1.5B')

    args = parser.parse_args()

    data_source = 'ScaleML-RLHF/numina_math'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    #dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    dataset = load_dataset(
        "parquet",
        data_files=os.path.expanduser("~/PRM_filter/data/numina_math/train_numina_raw.parquet")
    )
    # Shuffle the 'train' split
    dataset['train'] = dataset['train'].shuffle(seed=21)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    train_dataset = dataset['train']
    args.train_end = min(args.train_end, len(train_dataset))
    if args.train_end > 0:
        train_dataset = train_dataset.select(range(args.train_start, args.train_end))

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

    # You can then use this variable in your data processing function
    print(system_prompt)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following

            # We set the data_source as MATH so that we can use the reward model designed for MATH dataset
            
            reward_model = {
                "style": "rule",
                "ground_truth": example['reward_model']['ground_truth']
            }

            data = {
                "data_source": 'numina_math',
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
                "reward_model": reward_model,
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn
    
    def able_to_extract(example):
        if len(tokenizer.encode(example['problem'])) > 700:
            return False
        return True

    train_dataset = train_dataset.filter(able_to_extract)
    
    print(f"Train dataset size: {len(train_dataset)}")

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    print(train_dataset[0])
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'normal_train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
                                              