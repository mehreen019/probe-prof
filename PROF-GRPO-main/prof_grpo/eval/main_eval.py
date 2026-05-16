#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams
import json
import os
from func_timeout import func_timeout, FunctionTimedOut
try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default=os.path.expanduser("~/PRM_filter/eval/data"),
        metadata={"help": "the base directory containing dataset folders"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default=os.path.expanduser("~/PRM_filter/eval/output/model_name/global_step_i"),
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=8192,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.5,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="problem",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})
    prompt_type: Optional[str] = field(
        default="qwen25-math-cot",
        metadata={"help": "the type of the prompt"},
    )
    data_names: Optional[str] = field(
        default="math500",
        metadata={"help": "the name of the dataset"},
    )
    text_file_dir: Optional[str] = field(
        default="",
        metadata={"help": "the text file of the dataset"},
    )
    step: Optional[int] = field(
        default=0,
        metadata={"help": "the step of the model"},
    )
    model_name: Optional[str] = field(
        default="",
        metadata={"help": "the name of the model"},
    )


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
        return ret_score
    except TimeoutException:
        return timeout_score
    except Exception as e:
        print(f"Error during comparison: {e}")
        return 0.0


def gen_data(llm, tokenizer, data_name, script_args):
    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        top_p=1.0,
        max_tokens=4096,
        n=script_args.K,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    ds = load_dataset('json', data_files=f"{script_args.dataset_name_or_path}/{data_name}/test.jsonl", split="train")
    dataset_key = "problem" # default key
    if data_name in ["olympiadbench", "minerva_math"]:
        dataset_key = "question"

    if script_args.prompt_type == "qwen25-math-cot":
        ds = ds.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [#{"role":"system","content":f"You are a helpful Assistant that solves mathematical problems step-by-step. Your task is to provide a detailed solution process within specific tags. \n\nYou MUST follow this exact format: \n1. Your entire response must consist of a step-by-step reasoning process. \n2. Each distinct logical step MUST be separated by a double newline. Do not use any other separators. \n3. After all reasoning steps, you MUST provide the final answer on a new line. \n4. The final answer MUST ONLY be the value enclosed within \\boxed{{}}. Do not add any text before or after the boxed answer on that line. \n\nHere is an example of the required format: \n\nUser: Calculate 15 - (3 * 2). \n\nAssistant: \nFirst, calculate the expression inside the parentheses, which is 3 multiplied by 2. \n\n3 * 2 equals 6. \n\nNext, subtract the result from the original number, which is 15 minus 6. \n\n15 - 6 equals 9. \n\n\\boxed{{9}}"},
                        {"role":"user","content":x[dataset_key] + f' Let\'s think step by step and output the final answer within \\boxed{{}}.'}
                    ], 
                    tokenize=False, add_generation_prompt=True)
            }
        )
    elif script_args.prompt_type == "qwen25-step-cot":
        ds = ds.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [
                        {"role":"system","content":f"You are a helpful Assistant that solves mathematical problems step-by-step. Your task is to provide a detailed solution process within specific tags. \n\nYou MUST follow this exact format: \n1. Your entire response must consist of a step-by-step reasoning process. \n2. Each distinct logical step MUST be separated by a double newline. Do not use any other separators. \n3. After all reasoning steps, you MUST provide the final answer on a new line. \n4. The final answer MUST ONLY be the value enclosed within \\boxed{{}}. Do not add any text before or after the boxed answer on that line. \n\nHere is an example of the required format: \n\nUser: Calculate 15 - (3 * 2). \n\nAssistant: \nFirst, calculate the expression inside the parentheses, which is 3 multiplied by 2. \n\n3 * 2 equals 6. \n\nNext, subtract the result from the original number, which is 15 minus 6. \n\n15 - 6 equals 9. \n\n\\boxed{{9}}"},
                        {"role":"user","content":x[dataset_key]}
                    ], 
                    tokenize=False, add_generation_prompt=True)
            }
        )

    data_size = len(ds["prompt"])
    one_num_share = int(data_size / script_args.my_world_size)
    ds = ds.select(np.arange(script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share))

    print([script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share])
    print(ds, script_args.dataset_name_or_path)
    print(ds[0])
    
    prompts = ds["prompt"]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    completions = []
    used_prompts = []
    gathered_data = []
    for i, output in enumerate(outputs):
        tmp_data = {}
        for key in ds[0].keys():
            tmp_data[key] = ds[i][key]
        tmp_data["responses"] = [out.text for out in output.outputs]
        gathered_data.append(tmp_data)

    print("I collect ", len(gathered_data), "samples, each with ", script_args.K, " responses.")

    return gathered_data

def get_score(merged_json, script_args, data_name):
    all_scores = []
    all_vars = []
    #new_data = []

    for i in range(len(merged_json)):
        #merged_json[i]["scores"] = []
        tmp_scores = []
        #tmp_steps = []
        for j in range(len(merged_json[i]["responses"])):

            if data_name in ["math500", "aime24", "amc23", "minerva_math"]:
                gold_answer = merged_json[i]["answer"]
            elif data_name == "olympiadbench":
                gold_answer = merged_json[i]["final_answer"][0].strip("$")
            else:
                raise ValueError(f"Unsupported dataset: {data_name}")

            score = compute_score(merged_json[i]["responses"][j], gold_answer)
            tmp_scores.append(score)

            # 统计step数
            #resp = merged_json[i]["responses"][j]
            #step_count = len([seg for seg in resp.split("\n\n") if seg.strip()])
            #tmp_steps.append(step_count)
        #all_steps.append(tmp_steps)        
        all_scores.append(np.mean(tmp_scores))
        all_vars.append(np.var(tmp_scores))
        merged_json[i]["scores"] = tmp_scores
        #merged_json[i].update({"scores": tmp_scores})
        #new_data.append(merged_json[i])

    # merged_json = merged_json.add_column("scores", all_scores)
    #merged_json = merged_json.add_column("steps", all_steps)
    # data = merged_json.to_dict()
    records = merged_json #[dict(zip(data, t)) for t in zip(*data.values())]

    with open(f"{script_args.output_dir}/scored_{data_name}.jsonl", "w", encoding="utf8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write('\n')

    avg_score = np.mean(all_scores)
    avg_var = np.mean(all_vars)
    #avg_steps = np.mean(all_steps)
    print(f"Average score: {avg_score}")
    #print(f"Average steps: {avg_steps}")

    return avg_score, avg_var


def main(script_args):
    model_path = script_args.model_name_or_path
    print(f"Loading model ONCE: {model_path}")
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Using all of them for tensor parallelism.")
    
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=num_gpus,
        dtype="bfloat16",
        load_format="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data_list = script_args.data_names.split(",")
    results = []
    for data_name in data_list:
        print("=" * 50)
        print(f"Processing {data_name}...")
        gathered_data = gen_data(llm, tokenizer, data_name, script_args)

        # get score
        avg_score, avg_var = get_score(gathered_data, script_args, data_name)
        tmp = {"dataset":data_name,
               "score":avg_score,
               "var":avg_var}
        print(tmp)
        results.append(tmp)

    step_info = script_args.step
    final_result = {"step": step_info}
    for item in results:
        final_result[item['dataset']] = {"score":item['score'], "var":item['var']}

    file_path = f"{script_args.text_file_dir}/{script_args.model_name}/temp{script_args.temperature}_K{script_args.K}.jsonl"

    # Ensure the directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(file_path, "a", encoding="utf8") as f:
        json.dump(final_result, f, ensure_ascii=False)
        f.write('\n')



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if not os.path.exists(script_args.output_dir):
        os.makedirs(script_args.output_dir)

    main(script_args)