"""
evaluate.py - P3 Evaluation Suite for PROF-GRPO Replication
============================================================
Implements:
  - pass@k (k=1,4,8,32) with unbiased estimator (Chen et al. 2021)
  - average@k (Ye et al. protocol)
  - RAC: Reasoning-Answer Consistency (LLM-as-judge, Qwen2.5-7B-Instruct)
  - Answer extraction via math-verify / boxed fallback

Usage (from notebook):
    from eval.evaluate import (
        load_model_from_adapter, generate_responses,
        compute_pass_at_k, compute_rac_scores, aggregate_results
    )
"""

from __future__ import annotations

import json
import math
import re
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. MODEL LOADING
# ---------------------------------------------------------------------------

def load_merged_model(adapter_path: str, base_model_id: str = "unsloth/Qwen2.5-Math-1.5B-Instruct"):
    """
    Load a LoRA adapter checkpoint and merge into the base model.
    Returns (model, tokenizer) ready for generation.

    If adapter_path points to a directory with no optimizer.pt it is already
    a merged model and we load it directly.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    adapter_path = str(adapter_path)

    # Check whether this is a PEFT adapter or a fully merged model
    is_adapter = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))

    if is_adapter:
        try:
            peft_config = PeftConfig.from_pretrained(adapter_path)
            effective_base = peft_config.base_model_name_or_path or base_model_id
        except Exception:
            effective_base = base_model_id

        # Always load tokenizer from the base model — the adapter checkpoint's
        # tokenizer.json is incomplete (missing the full Qwen vocab/merges).
        print(f"[load] base model: {effective_base}")
        print(f"[load] adapter  : {adapter_path}")
        print(f"[load] loading tokenizer from base model...")
        tokenizer = AutoTokenizer.from_pretrained(effective_base, trust_remote_code=True)

        base = AutoModelForCausalLM.from_pretrained(
            effective_base,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
    else:
        print(f"[load] merged model: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    print(f"[load] VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. DATASET LOADING
# ---------------------------------------------------------------------------

def load_eval_dataset(name: str, max_samples: Optional[int] = None):
    """
    Load an evaluation dataset and return list of dicts with keys
    'problem' and 'answer'.

    Supported names: 'math500', 'olympiadbench', 'aime'
    """
    from datasets import load_dataset

    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        records = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]

    elif name == "olympiadbench":
        ds = load_dataset("OpenBMB/OlympiadBench", split="test_en", trust_remote_code=True)
        records = []
        for r in ds:
            ans = r.get("final_answer") or r.get("answer") or ""
            if isinstance(ans, list):
                ans = ans[0]
            ans = str(ans).strip().strip("$")
            records.append({"problem": r["question"], "answer": ans})

    elif name == "aime":
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        records = [{"problem": r["problem"], "answer": str(r["answer"])} for r in ds]

    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from math500, olympiadbench, aime")

    if max_samples:
        records = records[:max_samples]

    print(f"[dataset] {name}: {len(records)} problems loaded")
    return records


# ---------------------------------------------------------------------------
# 3. GENERATION
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def _build_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_responses(
    model,
    tokenizer,
    records: list[dict],
    n: int = 32,
    temperature: float = 0.8,
    max_new_tokens: int = 1024,
    batch_size: int = 4,
    save_path: Optional[str] = None,
) -> list[dict]:
    """
    Generate n responses per problem.

    Returns list of dicts:
        {problem, answer, responses: [str * n]}

    Saves incrementally to save_path (JSONL) so restarts are safe.
    """
    import time

    results = []
    seen_problems = set()

    # Resume from previous run
    if save_path and os.path.exists(save_path):
        with open(save_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    results.append(rec)
                    seen_problems.add(rec["problem"])
        print(f"[generate] Resuming — {len(results)} problems already done")

    remaining = [r for r in records if r["problem"] not in seen_problems]
    total = len(records)
    n_done = len(seen_problems)
    print(f"[generate] Generating {n} responses × {len(remaining)} problems (batch={batch_size})")

    writer = None
    if save_path:
        writer = open(save_path, "a", encoding="utf-8")

    session_start = time.time()

    try:
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i : i + batch_size]
            batch_start = time.time()
            prompts = [_build_prompt(tokenizer, r["problem"]) for r in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=n,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            for j, rec in enumerate(batch):
                seqs = outputs[j * n : (j + 1) * n]
                responses = [
                    tokenizer.decode(s[prompt_len:], skip_special_tokens=True)
                    for s in seqs
                ]
                correct = [check_answer(r, rec["answer"]) for r in responses]
                n_correct = sum(correct)
                avg_len = sum(len(r) for r in responses) / len(responses)

                entry = {
                    "problem": rec["problem"],
                    "answer": rec["answer"],
                    "responses": responses,
                    "correct": correct,
                }
                results.append(entry)
                if writer:
                    writer.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    writer.flush()
                n_done += 1

                print(
                    f"  [{n_done:>4}/{total}] correct={n_correct}/{n}"
                    f"  avg_len={avg_len:.0f}"
                    f"  ans={str(rec['answer'])[:12]:<12}"
                    f"  ({rec['problem'][:55].strip()}...)"
                )

            batch_elapsed = time.time() - batch_start
            total_elapsed = time.time() - session_start
            rate = n_done / max(total_elapsed, 1)
            eta = (total - n_done) / rate if rate > 0 else 0
            import torch as _torch
            vram = _torch.cuda.memory_allocated() / 1e9
            print(
                f"  --- batch {i//batch_size + 1}"
                f" | {batch_elapsed:.1f}s"
                f" | elapsed {total_elapsed/60:.1f}m"
                f" | ETA {eta/60:.1f}m"
                f" | {rate:.3f} prob/s"
                f" | VRAM {vram:.2f}GB\n"
            )

    finally:
        if writer:
            writer.close()

    print(f"[generate] Complete — {len(results)} problems total")
    return results


# ---------------------------------------------------------------------------
# 4. ANSWER VERIFICATION
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> Optional[str]:
    """Extract content of \\boxed{...} with balanced-brace handling."""
    m = re.search(r"\\boxed\{", text)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def check_answer(response: str, gold: str) -> bool:
    """
    Return True if response answers correctly.
    Uses math-verify (robust symbolic/LaTeX comparison) with boxed fallback.
    """
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
        from math_verify.errors import TimeoutException

        verify = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        gold_boxed = f"\\boxed{{{gold}}}"
        score, _ = verify([gold_boxed], [response])
        return score > 0.5
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: direct boxed string comparison
    pred = _extract_boxed(response)
    if pred is None:
        return False
    return pred.strip() == str(gold).strip()


# ---------------------------------------------------------------------------
# 5. PASS@K METRICS
# ---------------------------------------------------------------------------

def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator (Chen et al. 2021, HumanEval).
    n = total samples, c = correct samples, k = k in pass@k.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k(
    results: list[dict], ks: tuple[int, ...] = (1, 4, 8, 16, 32)
) -> dict:
    """
    Compute pass@k for each k, averaged over all problems.

    results: output of generate_responses (already scored, or will score now)
    Returns dict: {f'pass@{k}': float, ...}
    """
    # Score each response if not already scored
    for rec in results:
        if "correct" not in rec:
            rec["correct"] = [
                check_answer(r, rec["answer"]) for r in rec["responses"]
            ]

    metrics = {}
    for k in ks:
        scores = []
        for rec in results:
            n = len(rec["responses"])
            if k > n:
                continue
            c = sum(rec["correct"])
            scores.append(_pass_at_k_unbiased(n, c, k))
        if scores:
            metrics[f"pass@{k}"] = float(np.mean(scores))

    # average@k = mean of (correct / n) — matches Ye et al. average@16 protocol
    for k in ks:
        scores = []
        for rec in results:
            n = len(rec["responses"])
            if k > n:
                continue
            c = sum(rec["correct"][:k])
            scores.append(c / k)
        if scores:
            metrics[f"average@{k}"] = float(np.mean(scores))

    return metrics


# ---------------------------------------------------------------------------
# 6. RAC: REASONING-ANSWER CONSISTENCY (LLM-as-judge)
# ---------------------------------------------------------------------------

RAC_PROMPT = """You are evaluating whether a mathematical reasoning chain logically supports its stated final answer.

Problem: {problem}
Reasoning: {reasoning}
Final Answer: {answer}

Score from 0 to 1:
1.0 = reasoning steps directly and correctly lead to the final answer with no logical gaps
0.5 = reasoning is partially valid but contains gaps or minor errors that do not fully justify the answer
0.0 = reasoning is flawed, irrelevant, or does not support the final answer even if the answer is correct

Output only a number between 0 and 1."""


def _split_reasoning_answer(response: str):
    """Split a response into (reasoning, final_answer) where final_answer is the boxed content."""
    boxed = _extract_boxed(response)
    if boxed is None:
        return response.strip(), ""

    # Everything before the last \\boxed is the reasoning chain
    last_boxed_pos = response.rfind("\\boxed")
    reasoning = response[:last_boxed_pos].strip()
    return reasoning, boxed


def compute_rac_scores(
    results: list[dict],
    judge_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    correct_only: bool = True,
    max_responses_per_problem: int = 4,
    save_path: Optional[str] = None,
) -> list[dict]:
    """
    Score RAC for each (problem, response) using an LLM judge.

    correct_only: if True, only score responses that were marked correct.
    max_responses_per_problem: cap per problem to limit API cost.

    Saves results to save_path (JSONL) and returns augmented records.

    NOTE: Load this function in a separate kernel/session from generation
    to avoid having both the policy model and judge on the GPU simultaneously.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[RAC] Loading judge: {judge_model_id}")
    judge_tok = AutoTokenizer.from_pretrained(judge_model_id, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    print(f"[RAC] Judge VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Resume
    seen = set()
    if save_path and os.path.exists(save_path):
        with open(save_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    seen.add(r["problem"])
        print(f"[RAC] Resuming — {len(seen)} problems already scored")

    writer = None
    if save_path:
        writer = open(save_path, "a", encoding="utf-8")

    try:
        for rec in results:
            if rec["problem"] in seen:
                continue

            rac_scores = []
            candidates = list(enumerate(rec["responses"]))

            if correct_only and "correct" in rec:
                candidates = [(i, r) for i, r in candidates if rec["correct"][i]]

            candidates = candidates[:max_responses_per_problem]

            for idx, response in candidates:
                reasoning, pred_answer = _split_reasoning_answer(response)
                if not reasoning or not pred_answer:
                    continue

                prompt = RAC_PROMPT.format(
                    problem=rec["problem"],
                    reasoning=reasoning[:1500],  # truncate very long chains
                    answer=pred_answer,
                )
                messages = [{"role": "user", "content": prompt}]
                input_text = judge_tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = judge_tok(input_text, return_tensors="pt").to(judge_model.device)

                with torch.no_grad():
                    out = judge_model.generate(
                        **inputs,
                        max_new_tokens=8,
                        do_sample=False,
                        temperature=1.0,
                    )

                raw = judge_tok.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

                try:
                    score = float(re.search(r"[01](?:\.\d+)?", raw).group())
                    score = max(0.0, min(1.0, score))
                except Exception:
                    score = 0.5  # neutral fallback

                rac_scores.append({"response_idx": idx, "rac": score, "raw": raw})

            entry = {
                "problem": rec["problem"],
                "answer": rec["answer"],
                "rac_scores": rac_scores,
                "mean_rac": float(np.mean([s["rac"] for s in rac_scores])) if rac_scores else None,
            }
            if writer:
                writer.write(json.dumps(entry, ensure_ascii=False) + "\n")
                writer.flush()

            seen.add(rec["problem"])

    finally:
        if writer:
            writer.close()

    # Load all results back
    all_rac = []
    if save_path and os.path.exists(save_path):
        with open(save_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rac.append(json.loads(line))

    print(f"[RAC] Scored {len(all_rac)} problems")
    return all_rac


# ---------------------------------------------------------------------------
# 7. AGGREGATE & PRINT RESULTS
# ---------------------------------------------------------------------------

def aggregate_results(pass_k_metrics: dict, rac_records: list[dict], dataset_name: str = ""):
    """Print a clean results table."""
    label = f" [{dataset_name}]" if dataset_name else ""

    print(f"\n{'='*55}")
    print(f"  RESULTS{label}")
    print(f"{'='*55}")

    for key in ["average@1", "average@4", "average@8", "average@16", "average@32"]:
        if key in pass_k_metrics:
            print(f"  {key:<15} {pass_k_metrics[key]*100:.2f}%")

    print()
    for key in ["pass@1", "pass@4", "pass@8", "pass@16", "pass@32"]:
        if key in pass_k_metrics:
            print(f"  {key:<15} {pass_k_metrics[key]*100:.2f}%")

    if rac_records:
        valid = [r["mean_rac"] for r in rac_records if r.get("mean_rac") is not None]
        if valid:
            print(f"\n  RAC (mean)     {np.mean(valid):.3f}  (n={len(valid)} problems)")

    print(f"{'='*55}\n")

    return {**pass_k_metrics, "rac_mean": float(np.mean(valid)) if rac_records and valid else None}
