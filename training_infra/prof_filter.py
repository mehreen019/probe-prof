"""
prof_filter.py - PROF Algorithm 1 Implementation
================================================
Complete, self-contained PRM filtering pipeline.

Usage:
    from prof_filter import PROFPipeline

    prof = PROFPipeline()
    filtered = prof.filter_rollouts(rollouts, outcome_rewards)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig


class PROFPipeline:
    """Complete PROF filtering pipeline - PROF Algorithm 1 implementation."""

    def __init__(self, prm_model_name="Qwen/Qwen2.5-Math-PRM-7B"):
        """
        Load frozen PRM in bf16 for inference-only scoring.

        Args:
            prm_model_name: HuggingFace model name for PRM

        Note: Using full bf16 instead of 4-bit to avoid NaN outputs.
        """
        print(f"Loading PRM: {prm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(prm_model_name, trust_remote_code=True)

        # Load config and patch pad_token_id if missing
        config = AutoConfig.from_pretrained(prm_model_name, trust_remote_code=True)
        if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # Use SDPA attention to avoid potential NaN issues
        config._attn_implementation = "sdpa"

        self.model = AutoModel.from_pretrained(
            prm_model_name,
            config=config,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Full bf16 instead of 4-bit
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()

        # Freeze all parameters (no gradients needed)
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"✅ PRM loaded successfully")

    def compute_step_rewards(self, response_text):
        """
        Compute PRM scores for each step in response.

        Steps are delimited by double newlines (Ye et al. Section 3.1).

        Args:
            response_text: Full response text from policy model

        Returns:
            List of step-wise reward scores [0-1]
        """
        # Split response into steps by double newline
        steps = [s.strip() for s in response_text.split('\n\n') if s.strip()]

        if not steps:
            return []

        # Format for chat template with <extra_0> step markers
        # This is the required format for Qwen2.5-Math-PRM
        messages = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user", "content": "Solve this problem."},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]

        conversation_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize with attention_mask (CRITICAL for avoiding NaN)
        tokenized = self.tokenizer(conversation_str, return_tensors="pt", padding=False)
        input_ids = tokenized['input_ids'].to(self.model.device)
        attention_mask = tokenized['attention_mask'].to(self.model.device)

        # Forward pass through PRM (no gradients)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )

        # Get logits - handle different output structures
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            logits = outputs[0]
        else:
            logits = outputs

        # Extract rewards at step separator positions
        step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)

        # Check for NaN before softmax
        if torch.isnan(logits).any():
            print(f"  [WARNING] NaN detected in PRM logits, returning zeros")
            num_steps = token_masks.sum().item()
            return [0.5] * num_steps

        # Softmax to get probabilities
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)

        # Extract positive class probabilities
        sample = probabilities[0]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]

        return positive_probs.cpu().tolist()

    def filter_rollouts(self, rollouts, outcome_rewards,
                       policy_update_size=4, lambda_reg=10, H_lambda=30):
        """
        Apply PROF Algorithm 1 filtering.

        Filters n=8 rollouts down to m=4 based on consistency between
        PRM (process rewards) and ORM (outcome rewards).

        Args:
            rollouts: List of n=8 response texts
            outcome_rewards: List of n=8 outcome rewards {-1, 1}
            policy_update_size: Target m=4 (default from Ye et al.)
            lambda_reg: Regularization weight λ=10 (default from Ye et al.)
            H_lambda: Step threshold H_λ=30 (default from Ye et al.)

        Returns:
            dict with keys:
                'rollouts': List of m=4 kept responses
                'rewards': List of m=4 corresponding rewards
                'indices': Indices of kept rollouts
                'stats': Dict with debugging info
        """
        n = len(rollouts)
        n_correct = sum(1 for r in outcome_rewards if r == 1)
        n_incorrect = n - n_correct
        delta = n_correct - n_incorrect

        # Step 1: Compute trajectory-level consistency scores (Equation 1)
        trajectory_scores = []
        for response, outcome in zip(rollouts, outcome_rewards):
            step_rewards = self.compute_step_rewards(response)
            num_steps = len(step_rewards)

            if num_steps == 0:
                mean_step_reward = 0.0
            else:
                mean_step_reward = sum(step_rewards) / num_steps

            # Step length regularization: penalize trivial (1 step) or very long (>H_λ)
            step_penalty = 0.0
            if num_steps == 1 or num_steps >= H_lambda:
                step_penalty = lambda_reg

            # r_pro_i = [mean(step_rewards) - λ·I(H=1 or H≥H_λ)] · ro,i
            trajectory_score = (mean_step_reward - step_penalty) * outcome
            trajectory_scores.append(trajectory_score)

        # Step 2: Separate into correct (G+) and incorrect (G-) groups
        correct_indices = [i for i, r in enumerate(outcome_rewards) if r == 1]
        incorrect_indices = [i for i, r in enumerate(outcome_rewards) if r == -1]

        correct_scores = [(trajectory_scores[i], i) for i in correct_indices]
        incorrect_scores = [(trajectory_scores[i], i) for i in incorrect_indices]

        # Step 3: Calculate removal counts (Equation 2)
        k_plus = min(n - policy_update_size, int((delta + n - policy_update_size) / 2))
        k_minus = n - policy_update_size - k_plus

        # Step 4: Rank and filter
        # G+ (correct): Keep highest PRM scores (best reasoning among correct)
        # G- (incorrect): Keep lowest PRM scores (most obviously wrong)
        correct_ranked = sorted(correct_scores, key=lambda x: x[0], reverse=True)
        incorrect_ranked = sorted(incorrect_scores, key=lambda x: x[0], reverse=False)

        # Keep top from each group
        kept_correct = [idx for _, idx in correct_ranked[:(n_correct - k_plus)]]
        kept_incorrect = [idx for _, idx in incorrect_ranked[:(n_incorrect - k_minus)]]

        filtered_indices = kept_correct + kept_incorrect

        return {
            'rollouts': [rollouts[i] for i in filtered_indices],
            'rewards': [outcome_rewards[i] for i in filtered_indices],
            'indices': filtered_indices,
            'stats': {
                'correct_removed': k_plus,
                'incorrect_removed': k_minus,
                'correct_kept': len(kept_correct),
                'incorrect_kept': len(kept_incorrect),
            }
        }


# Standalone test function
def test_prof_pipeline():
    """Test the PROF pipeline on dummy data."""
    print("Testing PROF pipeline...")

    prof = PROFPipeline()

    test_rollouts = ["Response A", "Response B", "Response C", "Response D",
                     "Response E", "Response F", "Response G", "Response H"]
    test_outcomes = [1, 1, 1, 1, -1, -1, -1, -1]  # 4 correct, 4 incorrect

    result = prof.filter_rollouts(test_rollouts, test_outcomes)

    print(f"✅ Original: {len(test_rollouts)} rollouts")
    print(f"✅ Filtered: {len(result['rollouts'])} rollouts (should be 4)")
    print(f"✅ Stats: {result['stats']}")

    assert len(result['rollouts']) == 4, "PROF should keep m=4 rollouts!"
    print("✅ PROF pipeline test passed!")


if __name__ == "__main__":
    test_prof_pipeline()
