from datasets import load_dataset
import os

# Load the dataset (specify split if needed)
ds = load_dataset("Chenlu123/numia_prompt_ppo", split="train")

# Make sure output directory exists
os.makedirs(os.path.expanduser("~/PRM/data"), exist_ok=True)

# Save to Parquet
ds.to_parquet(os.path.expanduser("~/PRM_filter/data/numina_math/train_numina_raw.parquet"))