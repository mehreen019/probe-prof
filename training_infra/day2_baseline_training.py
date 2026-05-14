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

def extract_ground_truth(example):
    """Extract numerical ground truth from dataset"""
    if "answer" in example:
        return extract_answer(str(example["answer"]))
    elif "solution" in example:
        # Parse solution for final answer
        return extract_answer(str(example["solution"]))
    else:
        raise KeyError("No answer field in dataset")

ground_truths = [extract_ground_truth(ex) for ex in train_dataset_full]
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

def reward_function_with_gt(prompts, responses):
    """Wrapper reward function for GRPO trainer"""
    rewards = []
    for prompt, response in zip(prompts, responses):
        gt = prompt_to_gt.get(prompt)
        if gt is None:
            print(f"⚠️  Warning: No GT for prompt: {prompt[:50]}...")
            rewards.append(-1.0)
        else:
            reward = binary_outcome_reward(prompt, response, gt)
            rewards.append(reward)
    return rewards

# Test on first few
test_rewards = reward_function_with_gt(train_prompts_full[:3], ["The answer is X"]*3)
print(f"✅ Test rewards: {test_rewards}")

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
    config=grpo_config,  # Same config from Day 1
    tokenizer=tokenizer,
    reward_funcs=reward_function_with_gt,
    train_dataset=train_prompts_full,  # Full 1024 prompts
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
