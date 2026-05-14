"""
Day 3: PROF-GRPO Training with Filtering
=========================================
Launches PROF-GRPO training with PRM-based filtering (8 → 4 rollouts).

Prerequisites:
- Day 1 smoke test passed
- Day 2 baseline training launched (running in parallel)
- prof_filter.py module available

Run this in a SECOND Kaggle notebook on separate GPU.
"""

import torch
import time
from datetime import datetime
from trl import GRPOConfig, GRPOTrainer
import json

print("=" * 60)
print("DAY 3: PROF-GRPO TRAINING WITH FILTERING")
print("=" * 60)
print(f"Started: {datetime.now()}")
print("=" * 60)

# ============================================================================
# SECTION 1: IMPORT PRM PIPELINE
# ============================================================================

print("\n" + "=" * 60)
print("Importing PROF pipeline...")
print("=" * 60)

from prof_filter import PROFPipeline

# Initialize PROF pipeline (loads PRM in 4-bit)
prof_pipeline = PROFPipeline(prm_model_name="Qwen/Qwen2.5-Math-PRM-7B")
print(f"✅ PROF pipeline loaded")
print(f"   PRM VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================================
# SECTION 2: IMPLEMENT PROF REWARD WRAPPER
# ============================================================================

print("\n" + "=" * 60)
print("Implementing PROF reward wrapper...")
print("=" * 60)

class PROFRewardWrapper:
    """
    Wraps binary outcome reward + PROF filtering.

    TRL's GRPOTrainer calls reward_funcs after generation.
    We intercept here to:
    1. Compute outcome rewards
    2. Apply PROF filtering (8 → 4)
    3. Return rewards only for filtered samples
    """

    def __init__(self, ground_truth_map, prof_pipeline, binary_outcome_reward):
        """
        Args:
            ground_truth_map: Dict mapping prompts to ground truth answers
            prof_pipeline: PROFPipeline instance
            binary_outcome_reward: Function computing binary rewards
        """
        self.gt_map = ground_truth_map
        self.prof = prof_pipeline
        self.binary_outcome_reward = binary_outcome_reward
        self.filtering_stats = []

    def __call__(self, prompts, responses):
        """
        TRL calls this with (prompts, responses) after generation.

        Args:
            prompts: List of prompt strings
            responses: List of response strings (n=8 per prompt by default)

        Returns:
            rewards: List of rewards (only for PROF-kept samples)
        """
        # Group responses by prompt (assumes n=8 per prompt)
        n_per_prompt = 8  # Per Ye et al.
        assert len(responses) == len(prompts) * n_per_prompt, \
            f"Expected {len(prompts) * n_per_prompt} responses, got {len(responses)}"

        all_filtered_rewards = []

        for i, prompt in enumerate(prompts):
            # Get this prompt's n=8 rollouts
            start_idx = i * n_per_prompt
            rollouts = responses[start_idx:start_idx + n_per_prompt]

            # Compute outcome rewards
            gt = self.gt_map[prompt]
            outcome_rewards = [
                self.binary_outcome_reward(prompt, r, gt)
                for r in rollouts
            ]

            # Apply PROF filtering (8 → 4)
            filtered = self.prof.filter_rollouts(rollouts, outcome_rewards)

            # Extract rewards for kept samples
            filtered_rewards = filtered['rewards']
            all_filtered_rewards.extend(filtered_rewards)

            # Log stats
            self.filtering_stats.append(filtered['stats'])

        return all_filtered_rewards

print("✅ PROFRewardWrapper class defined")

# Create PROF-wrapped reward function
# Note: Assumes prompt_to_gt and binary_outcome_reward exist from Day 1/2
prof_reward_wrapper = PROFRewardWrapper(
    ground_truth_map=prompt_to_gt,
    prof_pipeline=prof_pipeline,
    binary_outcome_reward=binary_outcome_reward
)

print(f"✅ PROF reward wrapper instantiated")

# ============================================================================
# SECTION 3: CONFIGURE PROF-GRPO TRAINER
# ============================================================================

print("\n" + "=" * 60)
print("Configuring PROF-GRPO trainer...")
print("=" * 60)

# Same hyperparameters as baseline (only difference is filtering)
prof_grpo_config = GRPOConfig(
    output_dir="./prof_grpo_checkpoints",

    # Same hyperparameters as baseline
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-6,
    optim="adamw_torch",
    warmup_steps=10,

    # GRPO settings (same as baseline)
    num_generations=8,  # Generate n=8, but only m=4 used after filtering
    temperature=1.0,
    kl_coef=0.001,
    epsilon_low=0.2,
    epsilon_high=0.28,
    entropy_coef=0.001,

    # Generation
    max_new_tokens=4096,
    max_length=4096,

    # Checkpointing
    logging_steps=10,
    save_steps=200,
    save_total_limit=5,

    # Memory
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
)

print("✅ PROF-GRPO config loaded (same hyperparameters as baseline)")

# ============================================================================
# SECTION 4: TEST PROF FILTERING ON SMALL BATCH
# ============================================================================

print("\n" + "=" * 60)
print("Testing PROF filtering on small batch...")
print("=" * 60)

# Test the reward wrapper before full training
test_prompts = train_prompts_full[:10]

# Generate test responses (manually, to inspect filtering)
test_all_responses = []
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_return_sequences=8,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    test_all_responses.extend(responses)  # Flatten: 10 prompts × 8 = 80 responses

# Test reward wrapper (should return 10 × 4 = 40 rewards after filtering)
print("Running PROF filtering on test batch...")
test_rewards = prof_reward_wrapper(test_prompts, test_all_responses)

print(f"\n✅ Test filtering results:")
print(f"   Input: {len(test_all_responses)} responses (10 prompts × 8)")
print(f"   Output: {len(test_rewards)} rewards (expected: 10 × 4 = 40)")
print(f"   Sample stats: {prof_reward_wrapper.filtering_stats[-1]}")

assert len(test_rewards) == 40, f"Expected 40 rewards, got {len(test_rewards)}"
print("✅ PROF filtering test passed!")

# Clear filtering stats before real training
prof_reward_wrapper.filtering_stats = []

# ============================================================================
# SECTION 5: CREATE PROF-GRPO TRAINER
# ============================================================================

print("\n" + "=" * 60)
print("Creating PROF-GRPO trainer...")
print("=" * 60)

# Create PROF-GRPO trainer with filtering reward function
trainer_prof = GRPOTrainer(
    model=model,  # Same model architecture as baseline
    config=prof_grpo_config,  # Same hyperparameters
    tokenizer=tokenizer,
    reward_funcs=prof_reward_wrapper,  # <<< PROF-wrapped rewards
    train_dataset=train_prompts_full,
)

# Save initial model
model.save_pretrained("./prof_grpo_checkpoints/step_0")
tokenizer.save_pretrained("./prof_grpo_checkpoints/step_0")

print(f"✅ Trainer created")
print(f"   Training on: {len(train_prompts_full)} prompts")
print(f"   Filtering: 8 → 4 rollouts per prompt via PROF")

# ============================================================================
# SECTION 6: LAUNCH PROF-GRPO TRAINING
# ============================================================================

print("\n" + "=" * 60)
print("🚀 PROF-GRPO TRAINING STARTED")
print("=" * 60)
print(f"   Time: {datetime.now()}")
print(f"   Dataset: {len(train_prompts_full)} prompts")
print(f"   Filtering: PROF Algorithm 1 (PRM-based)")
print(f"   Rollouts: 8 generated → 4 kept per prompt")
print(f"   Expected duration: ~30-40 GPU hours")
print(f"   Checkpoints: every 200 steps → ./prof_grpo_checkpoints/")
print("=" * 60)
print()
print("⚠️  MONITORING CHECKLIST (check every ~2 hours):")
print("   ✅ Training loss decreasing")
print("   ✅ VRAM stable")
print("   ✅ Filtering stats: correct/incorrect balance ~equal")
print("   ⚠️  If loss is NaN → stop and debug")
print("=" * 60)

start_time = time.time()

try:
    # This will run for many hours
    trainer_prof.train()

    # Training completed successfully
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ PROF-GRPO TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"   Training time: {elapsed/3600:.2f} hours")
    print(f"   Final checkpoint: {trainer_prof.state.global_step} steps")
    print(f"   Final loss: {trainer_prof.state.log_history[-1]['loss']:.4f}")
    print(f"   Total filtering iterations: {len(prof_reward_wrapper.filtering_stats)}")
    print(f"{'='*60}")

except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"⚠️  TRAINING INTERRUPTED BY USER")
    print(f"{'='*60}")
    print(f"   Training time so far: {elapsed/3600:.2f} hours")
    print(f"   Last checkpoint: {trainer_prof.state.global_step}")
    print(f"   Can resume from checkpoint later")
    print(f"{'='*60}")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"❌ TRAINING FAILED")
    print(f"{'='*60}")
    print(f"   Error: {e}")
    print(f"   Training time before failure: {elapsed/3600:.2f} hours")
    print(f"   Last checkpoint: {trainer_prof.state.global_step}")
    print(f"{'='*60}")
    raise

# ============================================================================
# SECTION 7: SAVE TRAINING METADATA + FILTERING STATS
# ============================================================================

print("\n" + "=" * 60)
print("Saving training metadata and filtering stats...")
print("=" * 60)

# Compute filtering statistics
import numpy as np

filtering_stats = prof_reward_wrapper.filtering_stats
avg_correct_removed = np.mean([s['correct_removed'] for s in filtering_stats])
avg_incorrect_removed = np.mean([s['incorrect_removed'] for s in filtering_stats])
avg_correct_kept = np.mean([s['correct_kept'] for s in filtering_stats])
avg_incorrect_kept = np.mean([s['incorrect_kept'] for s in filtering_stats])

print(f"\n📊 FILTERING STATISTICS (averaged over {len(filtering_stats)} iterations):")
print(f"   Avg correct removed: {avg_correct_removed:.2f} / 8")
print(f"   Avg incorrect removed: {avg_incorrect_removed:.2f} / 8")
print(f"   Avg correct kept: {avg_correct_kept:.2f}")
print(f"   Avg incorrect kept: {avg_incorrect_kept:.2f}")

# Save metadata
metadata_prof = {
    "model": "Qwen2.5-Math-1.5B",
    "training_type": "PROF-GRPO",
    "dataset": "NuminaMath-CoT" if "NuminaMath" in str(type(train_dataset_full)) else "GSM8K",
    "num_prompts": len(train_prompts_full),
    "final_step": trainer_prof.state.global_step,
    "training_time_hours": elapsed / 3600,
    "final_loss": float(trainer_prof.state.log_history[-1]['loss']),
    "filtering_stats_summary": {
        "avg_correct_removed": float(avg_correct_removed),
        "avg_incorrect_removed": float(avg_incorrect_removed),
        "avg_correct_kept": float(avg_correct_kept),
        "avg_incorrect_kept": float(avg_incorrect_kept),
    },
    "hyperparameters": {
        "learning_rate": prof_grpo_config.learning_rate,
        "num_generations": prof_grpo_config.num_generations,
        "policy_update_size": 4,  # m=4 after filtering
        "lambda_reg": 10,
        "H_lambda": 30,
    }
}

with open("./prof_grpo_checkpoints/training_metadata.json", "w") as f:
    json.dump(metadata_prof, f, indent=2)

# Save detailed filtering stats as CSV
import pandas as pd
df_stats = pd.DataFrame(filtering_stats)
df_stats.to_csv("./prof_grpo_checkpoints/filtering_statistics.csv", index=False)

print(f"✅ Training metadata saved to training_metadata.json")
print(f"✅ Filtering stats saved to filtering_statistics.csv")

# ============================================================================
# END OF DAY 3: PROF-GRPO TRAINING
# ============================================================================

print(f"\n{'='*60}")
print(f"🎉 DAY 3 COMPLETE")
print(f"{'='*60}")
print(f"✅ PROF-GRPO training finished")
print(f"✅ Model checkpoints saved")
print(f"✅ Training metadata logged")
print(f"✅ Filtering statistics saved")
print(f"\n🚀 NEXT: Day 4 - Monitor both trainings, pull mid-checkpoints")
print(f"{'='*60}")
