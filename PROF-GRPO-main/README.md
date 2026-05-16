<div align="center">

# Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training
[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2509.03403v1) 
</div>

## Table of Contents
- [Introduction](#introduction)


## Introduction
### Current Dilemma: Process vs. Outcome

Current Reinforcement learning with verifiable rewards (RLVR) faces a challenge:

- Outcome Reward Models (ORMs): They are "results-oriented," caring only whether the final answer is correct. This is reliable but also "blind"—they cannot distinguish between a logically perfect solution and one with flawed reasoning steps but happens to be correct.

- Process Reward Models (PRMs): They are "process-oriented" and can evaluate each step of the reasoning. This provides finer-grained guidance, but PRMs themselves are often inaccurate and susceptible to "reward hacking" by the model.

Simply blending the two often introduces unstable training signals and can even lead to worse performance.

### Our Solution: PROF - PRocess cOnsistency Filter

We propose **PRocess cOnsistency Filter (PROF)**, an effective data process curation method that the strengths of process rewards (PRMs) and outcome rewards (ORMs). The core idea of PROF is not to use the PRM's score directly for training, but rather to use it as a "consistency filter" to select the highest-quality training data. Specifically, PROF filters the correct and incorrect samples separately and remove the most deceptive "noise":

- Correct: remove the responses with the lowest PRM values averaged over steps
- Incorrect: remove  the responses with the highest PRM values averaged over steps

The number to remove in correct and incorrect group is to balance the correct-incorrect ratio.
<p align="center">
  <img src="fig/PROF_alg.jpeg" width="72%" />
</p>

### Main Results

<p align="center">
  <img src="fig/Qwen2.5-Math-1.5B_data_temp1.0_comparison_main.png" width="45%" />
  <img src="fig/7B_model_comparison_temp1.0_main.png" width="45%" />
</p>

<p align="center">
  
| Model | Algorithm | Math500 | Minerva Math | Olympiad Bench | AIME24 | AMC23 | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-Math-1.5B-base | Base | 39.9 | 11.4 | 19.1 | 3.5 | 23.6 | 19.5 |
| | GRPO | 70.3 | 29.1 | 33.0 | 9.0 | 44.5 | 37.2 |
| | Blend | 67.6 | 27.8 | 31.1 | 7.7 | 42.5 | 35.3 |
| | **PROF-GRPO** | **73.2** | **30.0** | **36.1** | **9.6** | **49.1** | **39.6** |
| Qwen2.5-Math-7B-base | Base | 42.0 | 12.8 | 19.2 | 12.9 | 30.0 | 23.4 |
| | GRPO | 81.6 | 37.2 | 45.5 | **20.6** | 64.4 | 49.9 |
| | Blend | 81.7 | 36.7 | 45.0 | 15.2 | 58.0 | 47.3 |
| | **PROF-GRPO** | **83.1** | **39.0** | **47.8** | 17.5 | **70.9** | **51.7** |

_**Table 1:** Performance of different algorithms across five benchmarks including Math500, Minerva Math, Olympiad Bench, AMC2023 and AIME2024. We denote Blend-PRM-GRPO by Blend for short. We tune all the algorithms to their best performance and refer the readers to Appendix for the detailed parameter setup. The reported accuracy is average@16 under temperature 1.0._
</div>

### Main Takeways:

1. **PROF-GRPO v.s. Vanilla GRPO:** Compared to vanilla GRPO, PROF-GRPO consistently improves final accuracy by around $2\\%$. More improtantly, PROF-GRPO can effectively differentiate and filter out the flawed responses with correct final answer and reduce misleading gradients, while ORM cannot distinguish those samples. An example of flawed responses filtered out by PROF is presented below.
<p align="center">
  <img src="fig/Example_flaw_process.png" width="90%" />
</p>

2. **PROF-GRPO v.s. Blend:** Blend denotes naively mixing PRM with ORM in the objective and is a common way to apply PRM into RL training. As shown in the figure and table above, Blend is outperformed by PROF-GRPO by over $4\\%$ and suffers from severe reward hacking due to the mixing gradients of PRM and ORM.
3. **Improved Reasoning Quality:** Our method not only increases the accuracy of the final answer but also strengthens the quality and reliability of the model's intermediate reasoning steps by segmenting CoT into more **simple and easy-to-verify** steps.
<p align="center">
  <img src="fig/mc_prm_bar.png" width="38%" />
  <img src="fig/step_number_bar.png" width="20%" />
  <img src="fig/prm_bar.png" width="20%" />
  <img src="fig/claude_pref_bar.png" width="20%" />
</p>

4. **Filter separately:** By comparing the performance with w/o Separation, which ranks and filters correct and incorrect samples together, we find that separate then into two subgroups and balance the correct/incorrect ratio is essential. Additionally, filtering only correct samples with consistency and incorrect samples randomly (Filter-Correct) performs comparably to filtering both correct and incorrect samples with consistency (Filter-Both), while "Both" converges more efficiently. This implies that the quality of correct responses is the crucial part of online training. Moreover, through experiments on more rollout numbers and LLaMA-3.2-3B-instruct, we find that for robustness, especially when the PRM is untrustworthy or out-of-distribution, Filter-correct is the safer option as it constrains the PRM's influence.

<p align="center">
  <img src="fig/Qwen2.5-Math-1.5B_data_temp1.0_comparison_+-.png" width="45%" />
  <img src="fig/7B_model_comparison_temp1.0_+-.png" width="45%" />
  <img src="fig/nroll_plot.png" width="45%" />
</p>

<div align="center">

<p align="center">
  
| Algorithm | Math500 | Minerva Math | Olympiad Bench | AIME24 | AMC23 | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Base | 30.0 | 8.8 | 6.1 | 2.3 | 10.6 | 11.6 |
| GRPO | 50.5 | 18.8 | 17.9 | 5.0 | 25.6 | 23.6 |
| Blend-PRM-GRPO | 37.2 | 13.1 | 9.9 | 1.0 | 17.2 | 15.7 |
| PROF-GRPO (Both) | 50.4 | 19.1 | 18.7 | 3.5 | 27.8 | 23.9 |
| **PROF-GRPO (Correct)** | **52.4** | **19.5** | **19.8** | **6.7** | **28.6** | **25.4** |
| PROF-GRPO (Incorrect) | 49.0 | 18.0 | 17.3 | 5.4 | 23.9 | 22.7 |

_**Table 2:** The test accuracy of different methods initialized from LLaMA-3.2-3B-instruct that is average@16 under temperature 1.0 and further averaged across all the five benchmarks._

</div>

## Quick Start

### 1) Environment Setup

```bash
# 1. Create and activate a clean python environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2. Install project package
pip install -e .

# 3. Optional extras (pick based on your runtime backend)
# vLLM rollout
pip install -e ".[vllm]"
# math evaluation utilities
pip install -e ".[math]"
```

If you use GPU training, make sure your CUDA/PyTorch stack matches your machine.

### 2) Data Preprocessing

```bash
# Build Math500 parquet files
python scripts/data_preprocess/math_dataset.py \
  --local_dir "$HOME/PRM_filter/data/math500"

# Build Numina-Math parquet files
python scripts/data_preprocess/numina_math.py \
  --local_dir "$HOME/PRM_filter/data/numina_math"
```

After preprocessing, confirm these files exist:

- `$HOME/PRM_filter/data/math500/normal_train.parquet`
- `$HOME/PRM_filter/data/math500/normal_test.parquet`
- `$HOME/PRM_filter/data/numina_math/normal_train.parquet`

### 3) Run Training Scripts (`scripts/`)

All scripts are under `scripts/`:

- `run_grpo.sh`: GRPO baseline
- `run_prof_grpo.sh`: PROF-GRPO (both correct/incorrect filtering)
- `run_prof_grpo_correct.sh`: PROF-GRPO (correct-side filtering focus)
- `run_blend_grpo.sh`: Blend-PRM-GRPO baseline
- `process_ckp.sh`: checkpoint merge/upload helper

Run examples:

```bash
# GRPO baseline
bash scripts/run_grpo.sh

# PROF-GRPO (both)
bash scripts/run_prof_grpo.sh

# PROF-GRPO (correct)
bash scripts/run_prof_grpo_correct.sh

# Blend baseline
bash scripts/run_blend_grpo.sh
```

Training logs are written to `logs/<project_name>/<experiment_name>.log`.

### 4) Evaluation and Plotting

```bash
# Evaluate a model checkpoint (adjust paths/model args as needed)
python eval/main_eval.py \
  --model_name_or_path "Qwen/Qwen2.5-Math-7B" \
  --dataset_name_or_path "$HOME/PRM_filter/eval/data" \
  --output_dir "$HOME/PRM_filter/eval/output/demo_model/global_step_0" \
  --data_names "math500,minerva_math,olympiadbench"

# Plot evaluation curve
python eval/plot.py
```

### 5) Minimal Testing Checklist

Before long training runs, do a quick smoke test:

```bash
# package import + cli parse checks
python -m verl.trainer.main_ppo --help
python eval/main_eval.py --help

# preprocess one dataset first
python scripts/data_preprocess/math_dataset.py --local_dir "$HOME/PRM_filter/data/math500"
```

Then run one training script with a smaller local config (fewer GPUs, smaller batch, fewer epochs) and verify:

- log file is created under `logs/`
- first evaluation step completes
- checkpoints are saved to configured `trainer.default_local_dir`

This confirms your environment, data paths, and training pipeline are wired correctly before full-scale runs.
