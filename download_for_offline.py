#!/usr/bin/env python3
"""
Download models and datasets for offline Kaggle usage.

Run this script locally (with internet) before uploading to Kaggle.

Usage:
    python download_for_offline.py

This will create:
    ./offline_assets/
        models/
            qwen-math-1.5b/
            qwen-prm-7b/
        datasets/
            numina-math-cot-5k/
        wheels/
            *.whl files for offline pip install
"""

import os
import sys
import subprocess

def download_wheels(output_dir):
    """Download pip wheels for offline installation."""
    wheels_dir = os.path.join(output_dir, "wheels")
    os.makedirs(wheels_dir, exist_ok=True)

    # Packages needed that might not be on Kaggle or need specific versions
    packages = [
        "trl<0.9.0",
        "unsloth",
        "peft",
        "bitsandbytes",
    ]

    print("\n[0/4] Downloading pip wheels for offline install...")
    for pkg in packages:
        print(f"  Downloading: {pkg}")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "download",
                pkg,
                "-d", wheels_dir,
                "--no-deps"  # Don't download dependencies (Kaggle has most)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"    Warning: Failed to download {pkg}: {e}")

    # Also download with dependencies for safety
    print("  Downloading with dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "download",
            "trl<0.9.0", "unsloth",
            "-d", wheels_dir,
        ], check=True, capture_output=True)
    except:
        pass

    print(f"  Wheels saved to: {wheels_dir}")
    return wheels_dir


def main():
    print("=" * 60)
    print("PROF-GRPO Offline Asset Downloader")
    print("=" * 60)

    # Check dependencies
    try:
        from huggingface_hub import snapshot_download
        from datasets import load_dataset
    except ImportError:
        print("\nMissing dependencies. Install with:")
        print("  pip install huggingface_hub datasets")
        sys.exit(1)

    # Create output directory
    output_dir = "./offline_assets"
    os.makedirs(output_dir, exist_ok=True)

    # Download pip wheels first
    #download_wheels(output_dir)

    # =========================================================
    # 1. Download Policy Model
    # =========================================================
    print("\n[1/3] Downloading Qwen2.5-Math-1.5B-Instruct...")
    policy_path = os.path.join(output_dir, "models", "qwen-math-1.5b")

    try:
        snapshot_download(
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            local_dir=policy_path,
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
            resume_download=True
        )
        print(f"  Saved to: {policy_path}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # =========================================================
    # 2. Download PRM Model
    # =========================================================
    print("\n[2/3] Downloading Qwen2.5-Math-PRM-7B...")
    print("  (This is ~14GB, may take a while)")
    prm_path = os.path.join(output_dir, "models", "qwen-prm-7b")

    try:
        snapshot_download(
            "Qwen/Qwen2.5-Math-PRM-7B",
            local_dir=prm_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"  Saved to: {prm_path}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # =========================================================
    # 3. Download Dataset
    # =========================================================
    print("\n[3/3] Downloading NuminaMath-CoT dataset...")
    dataset_path = os.path.join(output_dir, "datasets", "numina-math-cot-5k")

    try:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train[:5000]")
        ds.save_to_disk(dataset_path)
        print(f"  Saved {len(ds)} samples to: {dataset_path}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nAll assets saved to: {os.path.abspath(output_dir)}/")
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR KAGGLE:")
    print("=" * 60)
    print("""
1. ZIP everything:
   zip -r offline_assets.zip ./offline_assets

2. Upload to Kaggle as a Dataset:
   - Go to kaggle.com/datasets
   - Click "New Dataset"
   - Upload offline_assets.zip
   - Name it something like "prof-grpo-offline-assets"

3. Add to your notebook:
   - Open your Kaggle notebook
   - Click "Add Input" on the right sidebar
   - Search for your uploaded dataset
   - Add it

4. In your notebook Cell 0, set:
   OFFLINE_MODE = True
   OFFLINE_POLICY_MODEL_PATH = "/kaggle/input/prof-grpo-offline-assets/models/qwen-math-1.5b"
   OFFLINE_PRM_MODEL_PATH = "/kaggle/input/prof-grpo-offline-assets/models/qwen-prm-7b"
   OFFLINE_DATASET_PATH = "/kaggle/input/prof-grpo-offline-assets/datasets/numina-math-cot-5k"

5. In Cell 1 (dependencies), ADD this at the top:
   # Install from offline wheels
   !pip install /kaggle/input/prof-grpo-offline-assets/wheels/*.whl --no-index --find-links /kaggle/input/prof-grpo-offline-assets/wheels/

6. Turn OFF internet in Kaggle notebook settings, then run!
""")


if __name__ == "__main__":
    main()
