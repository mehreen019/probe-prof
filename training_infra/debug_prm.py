"""
Debug script for Qwen2.5-Math-PRM-7B NaN issue
===============================================
Minimal test to diagnose why PRM outputs NaN logits.
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

print("=" * 60)
print("PRM DEBUG SCRIPT")
print("=" * 60)

# Step 1: Load model and tokenizer
print("\n1. Loading PRM model...")
model_name = "Qwen/Qwen2.5-Math-PRM-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"   ✅ Tokenizer loaded")

# Load config
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(f"   ✅ Config loaded: {type(config).__name__}")
print(f"   Config attributes: {dir(config)[:10]}...")

# Patch pad_token_id
if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
    config.pad_token_id = tokenizer.eos_token_id
    print(f"   Patched pad_token_id = {config.pad_token_id}")

# Load model
model = AutoModel.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()
print(f"   ✅ Model loaded: {type(model).__name__}")
print(f"   Model device: {next(model.parameters()).device}")
print(f"   Model dtype: {next(model.parameters()).dtype}")

# Step 2: Test with simple input
print("\n2. Testing with simple input...")
test_text = "Step 1: Do something.<extra_0>Step 2: Do more.<extra_0>"

# Method 1: Direct encoding
print("\n   Method 1: Direct encoding")
input_ids = tokenizer.encode(test_text, return_tensors="pt").to(model.device)
print(f"   Input shape: {input_ids.shape}")
print(f"   Input IDs: {input_ids[0].tolist()[:20]}...")

with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=False)

print(f"   Output type: {type(outputs)}")
print(f"   Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")

# Try different ways to access logits
if hasattr(outputs, 'logits'):
    logits = outputs.logits
    print(f"   ✅ Found outputs.logits")
elif isinstance(outputs, tuple):
    logits = outputs[0]
    print(f"   ✅ Using outputs[0] (tuple indexing)")
else:
    logits = outputs
    print(f"   ✅ Using outputs directly")

print(f"   Logits shape: {logits.shape}")
print(f"   Logits dtype: {logits.dtype}")
print(f"   Logits device: {logits.device}")
print(f"   Logits contains NaN: {torch.isnan(logits).any().item()}")
print(f"   Logits contains Inf: {torch.isinf(logits).any().item()}")

if not torch.isnan(logits).any():
    print(f"   Logits min: {logits.min().item():.4f}")
    print(f"   Logits max: {logits.max().item():.4f}")
    print(f"   Logits mean: {logits.mean().item():.4f}")
    print(f"   Sample logits[0, 0]: {logits[0, 0].tolist()}")
else:
    print(f"   ⚠️ Logits contain NaN!")
    # Check which positions have NaN
    nan_mask = torch.isnan(logits)
    print(f"   NaN positions: {nan_mask.sum().item()} / {logits.numel()}")
    print(f"   First few positions with NaN: {torch.where(nan_mask[0])[0].tolist()[:10]}")

# Step 3: Test with chat template
print("\n3. Testing with chat template...")
messages = [
    {"role": "system", "content": "Please reason step by step."},
    {"role": "user", "content": "Solve: What is 2+2?"},
    {"role": "assistant", "content": "Step 1: Add the numbers.<extra_0>Step 2: The answer is 4.<extra_0>"},
]

conversation_str = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
print(f"   Conversation length: {len(conversation_str)} chars")
print(f"   First 200 chars:\n{conversation_str[:200]}")

input_ids = tokenizer.encode(conversation_str, return_tensors="pt").to(model.device)
print(f"   Input shape: {input_ids.shape}")

# Find step separators
step_sep_id = tokenizer.encode("<extra_0>")[0]
step_sep_positions = torch.where(input_ids[0] == step_sep_id)[0].tolist()
print(f"   Step separator ID: {step_sep_id}")
print(f"   Step separator positions: {step_sep_positions}")

with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=False)

if hasattr(outputs, 'logits'):
    logits = outputs.logits
elif isinstance(outputs, tuple):
    logits = outputs[0]
else:
    logits = outputs

print(f"   Logits shape: {logits.shape}")
print(f"   Logits contains NaN: {torch.isnan(logits).any().item()}")

if not torch.isnan(logits).any():
    print(f"   ✅ SUCCESS: No NaN in logits!")
    print(f"   Logits at step separators:")
    for pos in step_sep_positions:
        step_logits = logits[0, pos].tolist()
        print(f"      Position {pos}: {step_logits}")
else:
    print(f"   ❌ FAILURE: NaN detected in logits")

# Step 4: Check model's forward method signature
print("\n4. Model details...")
print(f"   Model class: {type(model).__name__}")
print(f"   Model forward signature: {model.forward.__doc__}")
print(f"   Model config type: {type(model.config).__name__}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
