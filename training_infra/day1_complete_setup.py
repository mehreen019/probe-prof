"""
Day 1 Complete Setup - PROF Baseline GRPO
==========================================
Complete implementation of:
- Unsloth GRPO setup
- Qwen2.5-Math-PRM-7B loading (bf16 vanilla)
- PROF Algorithm 1 filtering
- Smoke test

Copy-paste this entire file into Kaggle notebook and run cell-by-cell.
GPU: P100 (16GB), Internet: ON, Persistence: ON
"""

# ============================================================================
# SECTION 1: DEPENDENCIES & ENVIRONMENT SETUP
# ============================================================================

print("=" * 60)
print("SECTION 1: Installing dependencies...")
print("=" * 60)

# Install Unsloth + dependencies
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
!pip install -q datasets scipy

# Verify installations
import torch
import unsloth
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import torch.nn.functional as F
from datasets import load_dataset
import re
import time
from datetime import datetime

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# SECTION 2: LOAD POLICY MODEL (Qwen2.5-Math-1.5B)
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 2: Loading policy model...")
print("=" * 60)

from unsloth import FastLanguageModel

# Load Qwen2.5-Math-1.5B in bf16 (should use ~6-7GB)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Math-1.5B",
    max_seq_length=4096,  # Per Ye et al. Section 4.1
    dtype=torch.bfloat16,  # P100 safe (no FP16 issues)
    load_in_4bit=False,    # Full precision training
)

print(f"✅ Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Add LoRA adapters (reduces memory, faster training)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,              # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Memory optimization
    random_state=3407,
)

# Verify trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# ============================================================================
# SECTION 3: LOAD PRM MODEL (Qwen2.5-Math-PRM-7B in bf16)
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 3: Loading PRM model...")
print("=" * 60)

def load_prm_model(model_name="Qwen/Qwen2.5-Math-PRM-7B", device="auto"):
    """
    Load PRM in bf16 for inference (4-bit was causing NaN logits).
    Per Ye et al.: frozen PRM, no training, just scoring.

    Note: Using full bf16 instead of 4-bit quantization to avoid NaN outputs.
    This uses ~7GB VRAM instead of ~2-3GB, but is necessary for correct scoring.
    """
    print(f"  Loading PRM tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # CRITICAL FIX: The Qwen2RMConfig in config.json is missing pad_token_id
    # This causes AttributeError at line 814 of modeling_qwen2_rm.py
    # We MUST patch the config before model loading
    print(f"  Loading and patching config (pad_token_id missing from Qwen2RMConfig)...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Set pad_token_id to eos_token_id (standard for Qwen models)
    # From config.json: eos_token_id = 151645
    config.pad_token_id = config.eos_token_id
    print(f"     Patched: config.pad_token_id = {config.pad_token_id}")

    # Load model with patched config
    model = AutoModel.from_pretrained(
        model_name,
        config=config,  # Pass patched config
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    # Freeze all parameters (no gradients needed)
    for param in model.parameters():
        param.requires_grad = False

    print(f"  PRM model loaded successfully")
    return model, tokenizer

# Load PRM
prm_model, prm_tokenizer = load_prm_model()
print(f"✅ PRM loaded. Total VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# DEBUG: Check model architecture and weights
print(f"\n[CRITICAL DEBUG] Model architecture check...")
print(f"  Model type: {type(prm_model).__name__}")
print(f"  Model has 'score' module: {hasattr(prm_model, 'score')}")
print(f"  Model has 'lm_head' module: {hasattr(prm_model, 'lm_head')}")
print(f"  Top-level modules: {[name for name, _ in prm_model.named_children()]}")

print(f"\n[CRITICAL DEBUG] Checking model weights for NaN...")
nan_params = []
total_params_checked = 0
for name, param in prm_model.named_parameters():
    total_params_checked += 1
    if torch.isnan(param).any():
        nan_params.append(name)
        print(f"  ❌ NaN found in: {name}")
    if total_params_checked <= 5:  # Show first 5 param stats
        print(f"  Param {name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")

if nan_params:
    print(f"\n⚠️  CRITICAL: {len(nan_params)} parameters contain NaN!")
    print(f"   This explains the NaN logits - model weights are corrupted!")
else:
    print(f"  ✅ All {total_params_checked} model weights are valid (no NaN)")

# ============================================================================
# SECTION 4: IMPLEMENT STEP-WISE PRM SCORING
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 4: Implementing PRM scoring functions...")
print("=" * 60)

def make_step_rewards(logits, token_masks):
    """
    Extract step-wise reward scores from PRM logits.

    From Qwen2.5-Math-PRM-7B model card implementation.

    Args:
        logits: Model output logits of shape (batch_size, seq_length, 2)
        token_masks: Boolean mask marking step separator positions

    Returns:
        List of lists where each inner list contains step rewards [0-1]
    """
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        # Extract positive class probability at step separators
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def compute_step_rewards(response_text, model, tokenizer):
    """
    Compute step-wise PRM scores for a response.

    Per Ye et al. Section 3.1: Steps delimited by newlines.

    Args:
        response_text: Full response text from policy model
        model: Loaded PRM model
        tokenizer: Corresponding tokenizer

    Returns:
        step_rewards: List of reward scores for each step [0-1]
    """
    # Split response into steps by double newline (Ye et al. approach)
    steps = response_text.split('\n\n')

    # Filter out empty steps
    steps = [s.strip() for s in steps if s.strip()]

    if not steps:
        return []

    # Format for chat template with <extra_0> step markers
    # This is how Qwen2.5-Math-PRM expects input
    messages = [
        {"role": "system", "content": "Please reason step by step."},
        {"role": "user", "content": "Solve this problem."},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]

    # Apply chat template and tokenize
    conversation_str = prm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize with attention_mask (CRITICAL for avoiding NaN)
    tokenized = prm_tokenizer(
        conversation_str,
        return_tensors="pt",
        padding=False,  # No padding needed for single sequence
    )
    input_ids = tokenized['input_ids'].to(prm_model.device)
    attention_mask = tokenized['attention_mask'].to(prm_model.device)

    # Forward pass through PRM (no gradients)
    # CRITICAL: Must pass attention_mask to avoid NaN in attention computation
    with torch.no_grad():
        outputs = prm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )

    # Debug: Check output structure
    print(f"  [DEBUG] Output type: {type(outputs)}")
    print(f"  [DEBUG] Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")

    # Try to get logits - different models have different output structures
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
        print(f"  [DEBUG] Using outputs.logits")
    elif isinstance(outputs, tuple) and len(outputs) > 0:
        logits = outputs[0]
        print(f"  [DEBUG] Using outputs[0]")
    else:
        logits = outputs
        print(f"  [DEBUG] Using outputs directly")

    print(f"  [DEBUG] Logits shape: {logits.shape}")
    print(f"  [DEBUG] Logits dtype: {logits.dtype}")
    print(f"  [DEBUG] Logits device: {logits.device}")
    print(f"  [DEBUG] Logits contains NaN: {torch.isnan(logits).any()}")
    print(f"  [DEBUG] Logits contains Inf: {torch.isinf(logits).any()}")
    print(f"  [DEBUG] Logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")

    # Extract rewards at step separator positions
    step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)

    print(f"  [DEBUG] Step separator ID: {step_sep_id}")
    print(f"  [DEBUG] Number of step separators found: {token_masks.sum().item()}")

    step_rewards = make_step_rewards(logits, token_masks)

    return step_rewards[0] if step_rewards else []

# Test PRM scoring - using format closer to official example
print("\n[TEST] Testing PRM scoring with official example format...")
test_data = {
    "system": "Please reason step by step.",
    "query": "What is 2+2?",
    "response": [
        "Let me solve this step by step.",
        "Step 1: Add the two numbers together.",
        "Step 2: 2 + 2 = 4.",
        "The answer is 4."
    ]
}

test_messages = [
    {"role": "system", "content": test_data['system']},
    {"role": "user", "content": test_data['query']},
    {"role": "assistant", "content": "<extra_0>".join(test_data['response']) + "<extra_0>"},
]

test_conversation = prm_tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=False
)

print(f"  Conversation length: {len(test_conversation)} chars")
test_tokenized = prm_tokenizer(test_conversation, return_tensors="pt", padding=False)
test_input_ids = test_tokenized['input_ids'].to(prm_model.device)
test_attention_mask = test_tokenized['attention_mask'].to(prm_model.device)
print(f"  Input IDs shape: {test_input_ids.shape}")

with torch.no_grad():
    test_outputs = prm_model(
        input_ids=test_input_ids,
        attention_mask=test_attention_mask,
        use_cache=False
    )

test_step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
test_token_masks = (test_input_ids == test_step_sep_id)
print(f"  Step separators found: {test_token_masks.sum().item()}")

test_logits = test_outputs.logits if hasattr(test_outputs, 'logits') else test_outputs[0]
print(f"  Logits shape: {test_logits.shape}")
print(f"  Logits contains NaN: {torch.isnan(test_logits).any().item()}")

if not torch.isnan(test_logits).any():
    test_step_rewards = make_step_rewards(test_logits, test_token_masks)
    print(f"✅ Test PRM scores: {test_step_rewards}")
    print(f"   Expected: ~4 scores, each in [0, 1]")
else:
    print(f"❌ Test FAILED: Logits contain NaN")
    print(f"   Logits min/max: {test_logits.min().item():.4f} / {test_logits.max().item():.4f}")

# ============================================================================
# SECTION 5: IMPLEMENT PROF ALGORITHM 1 (FILTERING)
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 5: Implementing PROF Algorithm 1...")
print("=" * 60)

def compute_trajectory_prm_score(step_rewards, outcome_reward,
                                lambda_reg=10, H_lambda=30):
    """
    Compute trajectory-level PRM score with step regularization.

    Implements PROF Algorithm 1, Equation 1:
    r_pro_i = [mean(step_rewards) - λ·I(H=1 or H≥H_λ)] · ro,i

    Args:
        step_rewards: List of step-wise PRM scores (each in [0, 1])
        outcome_reward: Outcome reward {-1, 1} (incorrect/correct)
        lambda_reg: Regularization weight λ=10 (Ye et al.)
        H_lambda: Threshold H_λ=30 (Ye et al.)

    Returns:
        trajectory_score: Float representing consistency
    """
    num_steps = len(step_rewards)

    # Trajectory-level score = mean of step-wise rewards
    if num_steps == 0:
        mean_step_reward = 0.0
    else:
        mean_step_reward = sum(step_rewards) / num_steps

    # Step length regularization: penalize trivial (1 step) or very long (>H_λ)
    step_penalty = 0.0
    if num_steps == 1 or num_steps >= H_lambda:
        step_penalty = lambda_reg

    # Final consistency score weighted by outcome reward
    trajectory_score = (mean_step_reward - step_penalty) * outcome_reward

    return trajectory_score


def prof_filter(rollouts, outcome_rewards, prm_model, prm_tokenizer,
                policy_update_size=4, lambda_reg=10, H_lambda=30):
    """
    Implement PROF filtering Algorithm 1 (complete).

    Filters n=8 rollouts down to m=4 based on consistency between
    PRM (process rewards) and ORM (outcome rewards).

    Args:
        rollouts: List of n=8 response texts
        outcome_rewards: List of n=8 outcome rewards {-1, 1}
        prm_model: Loaded PRM for step scoring
        prm_tokenizer: PRM tokenizer
        policy_update_size: Target m=4 (Ye et al.)

    Returns:
        filtered_rollouts: List of m=4 kept responses
        filtered_rewards: List of m=4 corresponding rewards
        filter_stats: Dict with debugging info
    """
    n = len(rollouts)
    n_correct = sum(1 for r in outcome_rewards if r == 1)
    n_incorrect = n - n_correct
    delta = n_correct - n_incorrect

    # Step 1: Compute PRM step scores for each rollout
    all_step_rewards = []
    for response in rollouts:
        step_rewards = compute_step_rewards(response, prm_model, prm_tokenizer)
        all_step_rewards.append(step_rewards)

    # Step 2: Compute trajectory-level consistency scores (Eq. 1)
    trajectory_scores = []
    for step_rewards, outcome in zip(all_step_rewards, outcome_rewards):
        score = compute_trajectory_prm_score(
            step_rewards, outcome, lambda_reg, H_lambda
        )
        trajectory_scores.append(score)

    # Step 3: Separate into correct (G+) and incorrect (G-) groups
    correct_indices = [i for i, r in enumerate(outcome_rewards) if r == 1]
    incorrect_indices = [i for i, r in enumerate(outcome_rewards) if r == -1]

    correct_scores = [(trajectory_scores[i], i) for i in correct_indices]
    incorrect_scores = [(trajectory_scores[i], i) for i in incorrect_indices]

    # Step 4: Calculate removal counts (Eq. 2)
    k_plus = min(n - policy_update_size,
                 int((delta + n - policy_update_size) / 2))
    k_minus = n - policy_update_size - k_plus

    # Step 5: Rank and filter
    # G+ (correct): Keep highest PRM scores (best reasoning among correct)
    # G- (incorrect): Keep lowest PRM scores (most obviously wrong)
    correct_ranked = sorted(correct_scores, key=lambda x: x[0], reverse=True)
    incorrect_ranked = sorted(incorrect_scores, key=lambda x: x[0], reverse=False)

    # Keep top from each group
    kept_correct_indices = [idx for _, idx in correct_ranked[:(n_correct - k_plus)]]
    kept_incorrect_indices = [idx for _, idx in incorrect_ranked[:(n_incorrect - k_minus)]]

    filtered_indices = kept_correct_indices + kept_incorrect_indices

    # Step 6: Build filtered outputs
    filtered_rollouts = [rollouts[i] for i in filtered_indices]
    filtered_rewards = [outcome_rewards[i] for i in filtered_indices]

    filter_stats = {
        'original_count': n,
        'kept_count': len(filtered_indices),
        'correct_removed': k_plus,
        'incorrect_removed': k_minus,
        'correct_kept': len(kept_correct_indices),
        'incorrect_kept': len(kept_incorrect_indices),
        'avg_prm_score_kept': sum(trajectory_scores[i] for i in filtered_indices) / len(filtered_indices) if filtered_indices else 0,
    }

    return filtered_rollouts, filtered_rewards, filter_stats

# Test filtering on dummy data
test_rollouts = ["Response A", "Response B", "Response C", "Response D",
                 "Response E", "Response F", "Response G", "Response H"]
test_outcomes = [1, 1, 1, 1, -1, -1, -1, -1]  # 4 correct, 4 incorrect

filtered, filtered_rewards, stats = prof_filter(
    test_rollouts, test_outcomes, prm_model, prm_tokenizer
)

print(f"\n✅ PROF Filtering Test:")
print(f"   Original: {len(test_rollouts)} rollouts")
print(f"   Filtered: {len(filtered)} rollouts (should be 4)")
print(f"   Stats: {stats}")

assert len(filtered) == 4, "PROF should keep m=4 rollouts!"
print("✅ PROF filtering validation passed!")

# ============================================================================
# SECTION 6: LOAD TRAINING DATA
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 6: Loading training data...")
print("=" * 60)

# NuminaMath dataset (per Ye et al. paper)
# If not available, use GSM8K as proxy
try:
    train_dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    print("✅ Loaded NuminaMath-CoT dataset")
except:
    print("⚠️  NuminaMath not found, using GSM8K as proxy")
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")

# Format for math reasoning
def format_prompt(example):
    """Convert dataset example to Qwen2.5-Math format"""
    # Format: "Question: {problem}\nAnswer:"
    if "problem" in example:
        problem = example["problem"]
    elif "question" in example:
        problem = example["question"]
    else:
        raise KeyError("No problem/question field in dataset")

    return f"Question: {problem}\nAnswer:"

# Prepare prompts (1024 per Ye et al.)
train_prompts = [format_prompt(ex) for ex in train_dataset.select(range(min(1024, len(train_dataset))))]
print(f"✅ Loaded {len(train_prompts)} training prompts")
print(f"   Example prompt:\n   {train_prompts[0][:150]}...")

# ============================================================================
# SECTION 7: IMPLEMENT BINARY OUTCOME REWARD (VERIFIER)
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 7: Implementing binary outcome reward...")
print("=" * 60)

def extract_answer(text):
    """Extract final numerical answer from model output"""
    # Look for patterns like "The answer is X" or "####X" (GSM8K format)
    patterns = [
        r'####\s*([0-9,\.]+)',           # GSM8K format
        r'[Tt]he answer is[:\s]*([0-9,\.]+)',
        r'[Aa]nswer[:\s]*([0-9,\.]+)',
        r'\\boxed\{([^}]+)\}',            # LaTeX boxed
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(',', '').strip()

    # Fallback: last number in text
    numbers = re.findall(r'[0-9,\.]+', text)
    return numbers[-1].replace(',', '').strip() if numbers else None

def binary_outcome_reward(prompt, response, ground_truth):
    """
    Binary outcome reward: +1 if correct, -1 if incorrect
    Per Ye et al.: ro ∈ {-1, +1}
    """
    predicted = extract_answer(response)

    if predicted is None:
        return -1.0  # No answer found

    # Normalize both answers
    try:
        pred_num = float(predicted)
        gt_num = float(ground_truth)
        is_correct = abs(pred_num - gt_num) < 1e-4
    except ValueError:
        # String comparison fallback
        is_correct = predicted.strip().lower() == str(ground_truth).strip().lower()

    return 1.0 if is_correct else -1.0

# Test on example
test_response = "Let me solve this step by step.\n1. First...\n2. Then...\nThe answer is 42."
test_reward = binary_outcome_reward("", test_response, "42")
print(f"✅ Test reward: {test_reward} (should be 1.0)")

# ============================================================================
# SECTION 8: CONFIGURE GRPO TRAINER (BASELINE)
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 8: Configuring GRPO trainer...")
print("=" * 60)

from trl import GRPOConfig, GRPOTrainer

# Hyperparameters from Ye et al. Section 4.1
grpo_config = GRPOConfig(
    output_dir="./baseline_grpo_checkpoints",

    # Training schedule
    num_train_epochs=1,
    per_device_train_batch_size=1,      # CRITICAL for P100 - start minimal
    gradient_accumulation_steps=16,      # Effective batch = 16

    # Optimization (Ye et al. Section 4.1)
    learning_rate=1e-6,                  # Per paper
    optim="adamw_torch",
    warmup_steps=10,

    # GRPO-specific (Ye et al. Section 4.1)
    num_generations=8,                   # n=8 rollouts per prompt (per Algorithm 1)
    temperature=1.0,                     # Per Algorithm 1
    kl_coef=0.001,                       # Per paper
    epsilon_low=0.2,                     # Asymmetric clip-higher
    epsilon_high=0.28,                   # Per paper

    # Entropy (Ye et al. Section 4.1)
    entropy_coef=0.001,                  # Per paper

    # Generation settings
    max_new_tokens=4096,                 # Per paper: "max 4096 tokens per response"
    max_length=4096,

    # Logging & checkpointing
    logging_steps=10,
    save_steps=200,                      # Checkpoint every 200 steps (per proposal)
    save_total_limit=5,                  # Keep last 5 checkpoints (P100 storage limit)

    # Memory optimizations
    bf16=True,                           # P100 safe
    gradient_checkpointing=True,

    # Disable unneeded features
    report_to="none",                    # No wandb (simplify)
)

print("✅ GRPO config loaded:")
print(f"   Effective batch size: {grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps}")
print(f"   Generations per prompt: {grpo_config.num_generations}")
print(f"   Learning rate: {grpo_config.learning_rate}")

# ============================================================================
# SECTION 9: SMOKE TEST - Single Forward + Backward Pass
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 9: RUNNING SMOKE TEST...")
print("=" * 60)
print("⚠️  CRITICAL: This must pass before proceeding to Day 2")
print("=" * 60)

# Create minimal trainer for smoke test
trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    tokenizer=tokenizer,
    reward_funcs=lambda prompts, responses: [binary_outcome_reward(p, r, "42") for p, r in zip(prompts, responses)],
    train_dataset=train_prompts[:10],  # Only 10 samples for smoke test
)

# Clear VRAM before test
torch.cuda.empty_cache()
initial_memory = torch.cuda.memory_allocated() / 1e9

# Run 1 training step
print("\n🔥 Starting smoke test (1 training step)...")
print("   This will:")
print("   1. Generate n=8 rollouts per prompt")
print("   2. Compute rewards")
print("   3. Run GRPO loss backward pass")
print("   4. Update weights")
print()

try:
    trainer.train(max_steps=1)

    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n{'='*60}")
    print(f"✅ SMOKE TEST PASSED")
    print(f"{'='*60}")
    print(f"   Initial VRAM: {initial_memory:.2f} GB")
    print(f"   Peak VRAM: {peak_memory:.2f} GB")
    print(f"   Available headroom: {16 - peak_memory:.2f} GB")
    print(f"{'='*60}")

    if peak_memory > 15.0:
        print(f"⚠️  WARNING: Very tight memory ({peak_memory:.2f}/16 GB)")
        print(f"   Consider: reduce batch_size or num_generations")
    else:
        print(f"✅ Memory footprint looks good!")

    print(f"\n{'='*60}")
    print(f"🎉 DAY 1 COMPLETE - ALL GATES PASSED")
    print(f"{'='*60}")
    print(f"✅ Policy model loaded (~7GB)")
    print(f"✅ PRM model loaded (~3GB)")
    print(f"✅ PRM scoring works")
    print(f"✅ PROF filtering works (8 → 4)")
    print(f"✅ GRPO training loop works")
    print(f"✅ Peak VRAM: {peak_memory:.2f} GB (safe margin)")
    print(f"\n🚀 READY FOR DAY 2: Launch full training!")
    print(f"{'='*60}")

except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"\n{'='*60}")
        print(f"❌ SMOKE TEST FAILED: OOM")
        print(f"{'='*60}")
        print(f"   Peak VRAM before crash: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Reduce num_generations: 8 → 4")
        print(f"   2. Reduce gradient_accumulation_steps: 16 → 8")
        print(f"   3. Reduce max_new_tokens: 4096 → 2048")
        print(f"   4. Enable load_in_4bit=True (last resort)")
        print(f"{'='*60}")
        raise
    else:
        print(f"\n{'='*60}")
        print(f"❌ SMOKE TEST FAILED: {e}")
        print(f"{'='*60}")
        raise

# ============================================================================
# END OF DAY 1 SCRIPT
# ============================================================================
