# PROF-GRPO Implementation Plan (Updated with Official Repo Insights)

## Key Insights from Official Implementation

After analyzing the official Amazon PROF-GRPO repository, here are the critical differences from our current implementation:

### 1. Architecture Difference: PRM as Reward Model vs Filter

**Official Implementation:**
- Uses `Qwen2.5-Math-PRM-7B` as a **separate reward model worker** (`ProcessRewardModelWorker`)
- PRM computes **step-level scores** using `\n\n` as step delimiter (not `<extra_0>` tokens)
- PRM scores are used to compute a **consistency score** (`scores_ic`)
- Filtering happens **after** reward computation, not integrated into the reward function

**Our Current Implementation:**
- Uses PRM with `<extra_0>` token markers (Qwen model card format)
- Integrates filtering directly in the PROF algorithm

### 2. Step-Level Scoring Algorithm

The official implementation uses `compute_verify_step_level_score()`:

```python
def compute_verify_step_level_score(scores: torch.Tensor, final_rewards: torch.Tensor, gamma: float):
    """
    scores: 2D tensor from PRM, -99.0 indicates padding
    final_rewards: 1D tensor (0 or 1 for incorrect/correct)

    Returns:
    - scores_ic: consistency scores (higher = more consistent)
    - len_segs: number of valid steps per sample
    """
    valid_mask = (scores != -99.0)
    len_segs = valid_mask.sum(dim=1)
    score_uns_mean = scores[valid_mask].mean()
    score_processed = (scores - score_uns_mean) * 2
    final_rewards_scaled = (2 * final_rewards - 1).unsqueeze(1)
    scaled_values = score_processed * final_rewards_scaled
    scaled_values_masked = scaled_values * valid_mask
    scores_ic = torch.sum(scaled_values_masked, dim=1) / torch.clamp(len_segs, min=1)
    return scores_ic.cpu().numpy(), len_segs.cpu().numpy()
```

### 3. Filtering Algorithm

The official `filter_trajectories()` function:

```python
# Key parameters from config:
# - prof_filter = 4 (number to remove per group, keeps 8-4=4 from n=8)
# - len_step_reward_coef = 10.0 (penalty for steps < 2 or > 30)

# For each prompt (uid):
# 1. Compute prm_critic = prm_val - len_step_reward_coef * ((len_seg < 2) or (len_seg > 30))
# 2. Separate into pos_samples (reward=1) and neg_samples (reward=0)
# 3. Calculate delta = len(pos) - len(neg)
# 4. n_pos_to_remove = min(n_remove, (delta + n_remove) / 2)
# 5. Remove lowest prm_critic from positive samples
# 6. Remove randomly from negative samples (or lowest in "Both" variant)
```

### 4. Training Configuration (from run_prof_grpo.sh)

```bash
# Key hyperparameters:
n=8                          # rollouts per prompt
prof_filter=4                # trajectories to remove (keeps 4)
len_step_reward_coef=10.0    # step length penalty
gamma=0.99                   # PRM discount factor
clip_ratio_high=0.2          # PPO clip ratio
learning_rate=1e-6           # actor learning rate
train_batch_size=1024        # prompts per batch
max_prompt_length=1024
max_response_length=3072
```

---

## Updated Implementation Plan

### Phase 1: Adapt to TRL/Unsloth (Kaggle Compatible)

Since we're using TRL's `GRPOTrainer` instead of `verl`, we need to adapt:

1. **PRM Scoring**: Keep using `<extra_0>` markers (Qwen model card format) but compute step-level consistency scores similar to official
2. **Custom Reward Function**: Implement PROF filtering inside a custom reward wrapper
3. **Memory Optimization**: Keep policy model + PRM on same GPU with careful memory management

### Phase 2: Implementation Steps

#### Step 1: Update Step-Level Scoring (DONE - needs refinement)
- Current `make_step_rewards()` extracts per-step probabilities correctly
- Need to add `compute_verify_step_level_score()` style consistency calculation

#### Step 2: Update PROF Filtering Algorithm
Replace current filtering with official algorithm:

```python
def prof_filter_official(
    responses,           # List of response texts
    outcome_rewards,     # List of {0, 1} rewards
    step_scores,         # List of step reward lists from PRM
    n_remove=4,          # Number to remove (prof_filter param)
    len_step_reward_coef=10.0,
    H_lambda=30          # Max steps threshold
):
    """
    Official PROF filtering algorithm adapted for TRL.

    For each prompt group:
    1. Compute consistency score (scores_ic) for each response
    2. Apply step length penalty
    3. Separate into correct/incorrect groups
    4. Remove lowest-scoring correct, random incorrect
    5. Return filtered indices
    """
```

#### Step 3: Integrate with GRPOTrainer

Option A: **Post-hoc Filtering** (Recommended)
- Let GRPOTrainer generate all rollouts
- Filter responses BEFORE computing advantages
- Requires modifying GRPOTrainer or using callbacks

Option B: **Reward Masking**
- Give filtered-out responses reward of 0 (neutral)
- Less accurate but works with unmodified GRPOTrainer

#### Step 4: Training Loop

```python
# Day 3: PROF-GRPO Training
for step in training_steps:
    # 1. Generate n=8 rollouts per prompt
    rollouts = generate_rollouts(prompts, n=8)

    # 2. Compute outcome rewards (ORM)
    outcome_rewards = [binary_reward(r) for r in rollouts]

    # 3. Compute PRM step scores
    step_scores = [compute_prm_scores(r) for r in rollouts]

    # 4. Apply PROF filtering (n=8 -> m=4)
    kept_indices = prof_filter_official(
        rollouts, outcome_rewards, step_scores,
        n_remove=4, len_step_reward_coef=10.0
    )

    # 5. Update policy on filtered rollouts only
    grpo_update(rollouts[kept_indices], outcome_rewards[kept_indices])
```

---

## Day 3 Notebook Structure

```
SECTION 1: Load trained baseline model (or start fresh)
SECTION 2: Implement official PROF scoring
SECTION 3: Implement official PROF filtering
SECTION 4: Create PROF-GRPO reward wrapper
SECTION 5: Configure PROF-GRPO trainer
SECTION 6: Run PROF-GRPO training
SECTION 7: Compare baseline vs PROF-GRPO results
```

---

## Key Differences Summary

| Aspect | Our Current | Official | Action |
|--------|-------------|----------|--------|
| Step delimiter | `<extra_0>` tokens | `\n\n` in text | Keep `<extra_0>` (works with Qwen PRM) |
| Consistency score | Mean step reward | `(score - mean) * 2 * sign(outcome)` | **Update** |
| Filtering | Balance k+/k- removal | Remove from pos by score, neg randomly | **Update** |
| Step penalty | `lambda_reg=10` constant | `len_step_reward_coef * I(H<2 or H>30)` | **Update** |
| Target kept | `m=4` from `n=8` | `prof_filter=4` to remove | Same (4 kept) |

---

## Memory Budget (T4 16GB)

| Component | VRAM |
|-----------|------|
| Qwen2.5-Math-1.5B (fp16 + LoRA) | ~4 GB |
| Qwen2.5-Math-PRM-7B (bf16 inference) | ~14 GB |
| **Total** | ~18 GB (TOO HIGH) |

**Solution**: Load PRM only during scoring, offload during training:
```python
# Option 1: CPU offload PRM between scoring
# Option 2: Use 4-bit PRM (~4GB) - but may have NaN issues
# Option 3: Use smaller PRM (1.5B if available)
# Option 4: Score in batches with memory clearing
```

Recommended: **Option 4** - Score PRM in small batches with `torch.cuda.empty_cache()` between batches

---

## Next Steps

1. Update notebook Section 5 with official PROF algorithm
2. Add memory-efficient PRM scoring with batch processing
3. Implement PROF filtering wrapper for GRPOTrainer
4. Run comparative training: Baseline vs PROF-GRPO
5. Evaluate on Math500 test set
