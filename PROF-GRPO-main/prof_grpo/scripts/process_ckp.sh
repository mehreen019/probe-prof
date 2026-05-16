#!/bin/bash
set -ex

# Define base paths for model checkpoints and merged output
MODEL_BASE_DIR="/opt/dlami/nvme/${USER}_ckpoints/prm_filter/Qwen2.5-Math-1.5B-dapocliph-numina_math-n8-std0."
MERGE_BASE_DIR="/opt/dlami/nvme/${USER}_ckpoints/prm_filter/Merge_Qwen2.5-Math-1.5B-dapocliph-numina_math-n8-std0."
MERGER_SCRIPT="scripts/model_merger.py"

# Iterate over step numbers (e.g., 10, 20, ..., 70)
for STEP in $(seq 10 10 70); do
    SRC_DIR="${MODEL_BASE_DIR}/global_step_${STEP}/actor"
    TGT_DIR="${MERGE_BASE_DIR}/global_step_${STEP}/actor"
    REPO_NAME="Qwen2.5-Math-1.5B-dapocliph-numina_math-n8-std0.-step${STEP}"
    HF_REPO="mytestprm/${REPO_NAME}"

    # Merge the FSDP checkpoint into a HuggingFace-compatible format
    python ${MERGER_SCRIPT} merge \
        --backend fsdp \
        --local_dir "${SRC_DIR}" \
        --target_dir "${TGT_DIR}"

    # Create the target repo on Hugging Face (set to private)
    huggingface-cli repo create "${REPO_NAME}" --type model --private -y

    # Ensure Git LFS is enabled for large model files
    git lfs install

    # Upload the merged model to the Hugging Face repo
    transformers-cli repo upload "${TGT_DIR}" \
        --repo-id "${HF_REPO}" \
        --commit-message "Upload merged model at step ${STEP}"
done
