"""
Day 2: Baseline GRPO Full Training
===================================
Launches the full 30-40 hour Baseline GRPO training run.

Prerequisites:
- Day 1 smoke test MUST have passed
- All functions from Day 1 are available

Run this in the same Kaggle notebook after Day 1 completes.
"""

import torch
import time
from datetime import datetime
from datasets import load_dataset
import random
import json
import re

# ============================================================================
# NUMINAMATH-SPECIFIC ANSWER EXTRACTION (overrides Day 1 functions)
# ============================================================================

def extract_boxed(text):
    """
    Extract content from \boxed{...}, handling nested braces.
    NuminaMath solutions end with \boxed{answer}.
    """
    # Find \boxed{ and extract content with balanced braces
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None

    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i-1].strip()
    return None


def normalize_latex(expr):
    """
    Normalize LaTeX expression for comparison.
    Removes whitespace, normalizes common variations.
    """
    if expr is None:
        return None

    # Remove all whitespace
    expr = re.sub(r'\s+', '', expr)

    # Normalize common LaTeX variations
    expr = expr.replace('\\cdot', '*')
    expr = expr.replace('\\times', '*')
    expr = expr.replace('\\div', '/')
    expr = expr.replace('\\frac', 'frac')
    expr = expr.replace('\\dfrac', 'frac')
    expr = expr.replace('\\tfrac', 'frac')

    return expr.lower()


def extract_answer(text):
    """
    Extract answer from model output - NuminaMath compatible.
    Priority: \boxed{} > #### > "answer is" > last number
    """
    if text is None:
        return None

    # 1. Try \boxed{} first (NuminaMath format)
    boxed = extract_boxed(text)
    if boxed:
        return boxed

    # 2. GSM8K format: #### answer
    gsm_match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if gsm_match:
        return gsm_match.group(1).strip()

    # 3. "The answer is X" - capture full expression
    ans_match = re.search(r'[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)', text)
    if ans_match:
        return ans_match.group(1).strip()

    # 4. Fallback: last \boxed-like pattern or number
    # Look for any remaining mathematical expression
    return None


def extract_ground_truth_numina(example):
    """
    Extract ground truth from NuminaMath-CoT dataset.
    The 'solution' field contains step-by-step with \boxed{answer} at end.
    """
    if "solution" in example and example["solution"]:
        boxed = extract_boxed(str(example["solution"]))
        if boxed:
            return boxed

    # Fallback to 'answer' field if exists
    if "answer" in example and example["answer"]:
        return str(example["answer"]).strip()

    return None


def binary_outcome_reward(prompt, response, ground_truth):
    """
    Binary outcome reward for NuminaMath: +1 if correct, -1 if incorrect.
    Compares normalized LaTeX expressions.
    """
    if ground_truth is None:
        return -1.0

    # Extract answer from model response
    predicted = extract_answer(response)

    if predicted is None:
        return -1.0

    # Normalize both for comparison
    pred_norm = normalize_latex(predicted)
    gt_norm = normalize_latex(ground_truth)

    if pred_norm is None or gt_norm is None:
        return -1.0

    # String comparison (exact match after normalization)
    if pred_norm == gt_norm:
        return 1.0

    # Try numeric comparison as fallback
    try:
        pred_num = float(re.sub(r'[^\d.\-]', '', predicted))
        gt_num = float(re.sub(r'[^\d.\-]', '', ground_truth))
        if abs(pred_num - gt_num) < 1e-6:
            return 1.0
    except (ValueError, TypeError):
        pass

    return -1.0


print("=" * 60)
print("DAY 2: BASELINE GRPO FULL TRAINING")
print("=" * 60)
print(f"Started: {datetime.now()}")
print("=" * 60)

# ============================================================================
# SECTION 1: LOAD FULL TRAINING DATASET
# ============================================================================

print("\n" + "=" * 60)
print("Loading full training dataset...")
print("=" * 60)

# Load full NuminaMath/GSM8K (not just 10 samples)
try:
    train_dataset_full = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    print(f"✅ Loaded NuminaMath-CoT: {len(train_dataset_full)} examples")
except:
    print("⚠️  NuminaMath not available, using GSM8K")
    train_dataset_full = load_dataset("openai/gsm8k", "main", split="train")
    print(f"✅ Loaded GSM8K: {len(train_dataset_full)} examples")

# Per Ye et al.: 1024 prompts per iteration
# Sample randomly if dataset > 1024
random.seed(42)
if len(train_dataset_full) > 1024:
    train_indices = random.sample(range(len(train_dataset_full)), 1024)
    train_dataset_full = train_dataset_full.select(train_indices)

train_prompts_full = [format_prompt(ex) for ex in train_dataset_full]
print(f"✅ Full training set: {len(train_prompts_full)} prompts")

# ============================================================================
# SECTION 2: EXTRACT GROUND TRUTH ANSWERS
# ============================================================================

print("\n" + "=" * 60)
print("Extracting ground truth answers...")
print("=" * 60)

ground_truths = [extract_ground_truth_numina(ex) for ex in train_dataset_full]

# Check extraction quality
none_count = sum(1 for gt in ground_truths if gt is None)
print(f"   Extracted: {len(ground_truths) - none_count} valid, {none_count} failed")
print(f"✅ Extracted {len(ground_truths)} ground truth answers")
print(f"   Sample GT: {ground_truths[:3]}")

# Create lookup dict: prompt → ground_truth
prompt_to_gt = dict(zip(train_prompts_full, ground_truths))

# ============================================================================
# SECTION 3: CREATE REWARD FUNCTION WITH GROUND TRUTH LOOKUP
# ============================================================================

print("\n" + "=" * 60)
print("Creating reward function...")
print("=" * 60)

def reward_function_with_gt(prompts, completions, **kwargs):
    """Wrapper reward function for GRPO trainer"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        gt = prompt_to_gt.get(prompt)
        if gt is None:
            print(f"⚠️  Warning: No GT for prompt: {prompt[:50]}...")
            rewards.append(-1.0)
        else:
            reward = binary_outcome_reward(prompt, completion, gt)
            rewards.append(reward)
    return rewards

# Test extraction and reward
print("\n[DEBUG] Testing NuminaMath extraction:")
test_prompt = train_prompts_full[0]
test_gt = ground_truths[0]
print(f"   Ground truth: '{test_gt}'")

# Test with correct boxed answer
test_response_correct = f"Step 1: solve\\nStep 2: compute\\nThe answer is \\boxed{{{test_gt}}}"
test_response_wrong = "The answer is \\boxed{999}"

reward_correct = binary_outcome_reward(test_prompt, test_response_correct, test_gt)
reward_wrong = binary_outcome_reward(test_prompt, test_response_wrong, test_gt)

print(f"   Correct response reward: {reward_correct} (should be 1.0)")
print(f"   Wrong response reward: {reward_wrong} (should be -1.0)")

if reward_correct == 1.0 and reward_wrong == -1.0:
    print("✅ Reward function working correctly!")
else:
    print("❌ Reward function needs debugging")

# ============================================================================
# SECTION 4: CREATE FULL BASELINE GRPO TRAINER
# ============================================================================

print("\n" + "=" * 60)
print("Creating full Baseline GRPO trainer...")
print("=" * 60)

from trl import GRPOTrainer

# Use same config as smoke test, but full dataset
trainer_baseline = GRPOTrainer(
    model=model,
    args=grpo_config,  # Same config from Day 1
    processing_class=tokenizer,
    reward_funcs=[reward_function_with_gt],  # Must be a list
    train_dataset=[{"prompt": p} for p in train_prompts_full],  # List of dicts
)

# Save initial model (for recovery)
model.save_pretrained("./baseline_grpo_checkpoints/step_0")
tokenizer.save_pretrained("./baseline_grpo_checkpoints/step_0")

print(f"✅ Trainer created")
print(f"   Training on: {len(train_prompts_full)} prompts")
print(f"   Generations per prompt: {grpo_config.num_generations}")

# ============================================================================
# SECTION 5: START TRAINING WITH MONITORING
# ============================================================================

print("\n" + "=" * 60)
print("🚀 BASELINE GRPO TRAINING STARTED")
print("=" * 60)
print(f"   Time: {datetime.now()}")
print(f"   Dataset: {len(train_prompts_full)} prompts")
print(f"   Generations per prompt: {grpo_config.num_generations}")
print(f"   Expected duration: ~30-40 GPU hours")
print(f"   Checkpoints: every 200 steps → ./baseline_grpo_checkpoints/")
print("=" * 60)
print()
print("⚠️  MONITORING CHECKLIST (check every ~2 hours):")
print("   ✅ Training loss decreasing")
print("   ✅ VRAM stable (not creeping up)")
print("   ✅ Reward distribution: mix of +1.0 and -1.0")
print("   ✅ Checkpoints saving every 200 steps")
print("   ⚠️  If loss is NaN → stop and debug")
print("   ⚠️  If all rewards are -1 → verifier broken")
print("=" * 60)

start_time = time.time()

try:
    # This will run for many hours - monitor via Kaggle notebook logs
    trainer_baseline.train()

    # Training completed successfully
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ BASELINE GRPO TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"   Training time: {elapsed/3600:.2f} hours")
    print(f"   Final checkpoint: {trainer_baseline.state.global_step} steps")
    print(f"   Final loss: {trainer_baseline.state.log_history[-1]['loss']:.4f}")
    print(f"{'='*60}")

except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"⚠️  TRAINING INTERRUPTED BY USER")
    print(f"{'='*60}")
    print(f"   Training time so far: {elapsed/3600:.2f} hours")
    print(f"   Last checkpoint: {trainer_baseline.state.global_step}")
    print(f"   Can resume from checkpoint later")
    print(f"{'='*60}")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"❌ TRAINING FAILED")
    print(f"{'='*60}")
    print(f"   Error: {e}")
    print(f"   Training time before failure: {elapsed/3600:.2f} hours")
    print(f"   Last checkpoint: {trainer_baseline.state.global_step}")
    print(f"{'='*60}")
    raise

# Save training metadata
metadata_baseline = {
    "model": "Qwen2.5-Math-1.5B",
    "training_type": "Baseline GRPO",
    "dataset": "NuminaMath-CoT" if "NuminaMath" in str(type(train_dataset_full)) else "GSM8K",
    "num_prompts": len(train_prompts_full),
    "final_step": trainer_baseline.state.global_step,
    "training_time_hours": elapsed / 3600,
    "final_loss": float(trainer_baseline.state.log_history[-1]['loss']),
    "hyperparameters": {
        "learning_rate": grpo_config.learning_rate,
        "num_generations": grpo_config.num_generations,
        "batch_size": grpo_config.per_device_train_batch_size,
        "gradient_accumulation_steps": grpo_config.gradient_accumulation_steps,
    }
}

with open("./baseline_grpo_checkpoints/training_metadata.json", "w") as f:
    json.dump(metadata_baseline, f, indent=2)

print("\n✅ Training metadata saved to training_metadata.json")

# ============================================================================
# END OF DAY 2: BASELINE TRAINING
# ============================================================================

print(f"\n{'='*60}")
print(f"🎉 DAY 2 COMPLETE")
print(f"{'='*60}")
print(f"✅ Baseline GRPO training finished")
print(f"✅ Model checkpoints saved")
print(f"✅ Training metadata logged")
print(f"\n🚀 NEXT: Day 3 - Launch PROF-GRPO training in parallel")
print(f"{'='*60}")
