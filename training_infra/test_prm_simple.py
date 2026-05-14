"""
Simple PRM test - minimal code to diagnose NaN issue
====================================================
"""

import torch
from transformers import AutoModel, AutoTokenizer

print("=" * 60)
print("SIMPLE PRM TEST")
print("=" * 60)

# Test 1: Load model with minimal settings
print("\n1. Loading model with minimal settings...")
model_name = "Qwen/Qwen2.5-Math-PRM-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"   ✅ Tokenizer loaded")
print(f"   Tokenizer class: {type(tokenizer).__name__}")

# Test 2: Load model WITHOUT config patching
print("\n2. Loading model directly (no config patching)...")
try:
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    print(f"   ✅ Model loaded: {type(model).__name__}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    # Try with float32
    print("\n2b. Trying with float32...")
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).eval()
    print(f"   ✅ Model loaded in float32: {type(model).__name__}")

# Test 3: Simple forward pass
print("\n3. Testing simple forward pass...")
test_text = "Hello world"
input_ids = tokenizer.encode(test_text, return_tensors="pt").to(model.device)
print(f"   Input shape: {input_ids.shape}")

with torch.no_grad():
    outputs = model(input_ids=input_ids)

logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
print(f"   Output shape: {logits.shape}")
print(f"   Contains NaN: {torch.isnan(logits).any().item()}")

if not torch.isnan(logits).any():
    print(f"   ✅ SUCCESS! Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}")
    print(f"   Sample logits[0, 0]: {logits[0, 0].tolist()}")
else:
    print(f"   ❌ FAILURE: NaN detected")
    print(f"   Trying to find where NaN starts...")

    # Check each layer
    print("\n   Checking model internals...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"      ❌ NaN in parameter: {name}")
        else:
            print(f"      ✅ OK: {name} (shape: {param.shape})")
            if len(list(model.named_parameters())) > 20:  # Only show first few
                break

# Test 4: Try with the official format
print("\n4. Testing with step marker format...")
messages = [
    {"role": "system", "content": "Please reason step by step."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "Step 1: Add 2 and 2<extra_0>Step 2: The result is 4<extra_0>"},
]

conversation = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
print(f"   Conversation length: {len(conversation)} chars")
print(f"   First 150 chars: {conversation[:150]}")

input_ids = tokenizer.encode(conversation, return_tensors="pt").to(model.device)
print(f"   Input shape: {input_ids.shape}")

# Check for step separator
step_sep_id = tokenizer.encode("<extra_0>")[0]
sep_positions = torch.where(input_ids[0] == step_sep_id)[0].tolist()
print(f"   Step separator ID: {step_sep_id}")
print(f"   Found at positions: {sep_positions}")

with torch.no_grad():
    outputs = model(input_ids=input_ids)

logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
print(f"   Output shape: {logits.shape}")
print(f"   Contains NaN: {torch.isnan(logits).any().item()}")

if not torch.isnan(logits).any():
    print(f"   ✅ SUCCESS with step format!")
    print(f"   Logits at first step separator (pos {sep_positions[0] if sep_positions else 'N/A'}):")
    if sep_positions:
        print(f"      {logits[0, sep_positions[0]].tolist()}")
else:
    print(f"   ❌ Still NaN with step format")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

# Conclusion
print("\n📋 DIAGNOSIS:")
if not torch.isnan(logits).any():
    print("   ✅ Model works! The issue was elsewhere.")
    print("   Check: Input formatting, chat template, or tokenization")
else:
    print("   ❌ Model produces NaN in forward pass")
    print("   Possible causes:")
    print("   1. Model weights are corrupted on HuggingFace")
    print("   2. Incompatible transformers version")
    print("   3. CUDA/PyTorch compatibility issue")
    print("   4. Need different precision (try float32 instead of bf16)")
    print("\n   💡 SOLUTION: Try loading in float32 or use a different PRM model")
