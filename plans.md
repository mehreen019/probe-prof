# AI4MATH @ ICML 2026 — WORKSHOP PAPER PROPOSAL (REVISED)
## Does PROF Fix What It Claims to Fix? A Diagnostic Replication of Process Consistency Filtering via Pass@k and Reasoning–Answer Consistency
### Evaluating entropy collapse and shortcut exploitation in PROF-filtered GRPO-trained math LLMs via pass@k scaling and CoT-faithfulness metrics

**Submission deadline:** May 25, 2026 | **Track:** NL Mathematical Reasoning | **Length:** 4–6 pages

---

## 1. Problem Statement

Reinforcement Learning with Verifiable Rewards (RLVR), trained via GRPO, has become the dominant paradigm for improving mathematical reasoning in LLMs. The standard setup provides a binary outcome reward: +1 if the final answer matches the ground truth, 0 otherwise. This sparse reward creates two well-documented failure modes:

**Shortcut exploitation** — models learn to arrive at correct answers through flawed, hallucinatory, or superficial reasoning chains. The reward is satisfied even when intermediate logic is wrong. Alam & Rastogi (arXiv:2510.27044, NeurIPS 2025 Math-AI Workshop) document exactly this phenomenon: RLVR improves evaluation metrics primarily by reinforcing superficial heuristics rather than acquiring new reasoning strategies.

**Entropy collapse** — RLVR aggressively narrows the policy distribution toward rewarded trajectories. While pass@1 improves, pass@k at large k plateaus or regresses, meaning the model loses diversity of reasoning paths. This is the central finding of Yue et al. (arXiv:2504.13837, AI4Math 2025 Best Paper / NeurIPS 2025 Best Paper Runner-Up), who show that base models eventually overtake RLVR-trained models at large k, and has since been independently confirmed by Chen et al. (arXiv:2508.10751) and Sullivan & Koller (arXiv:2509.21154).

**The recent PROF paper** (Ye et al., arXiv:2509.03403, Sep 2025) was explicitly motivated by these failure modes. PROF uses a frozen PRM offline to filter training trajectories before passing a clean subset to standard GRPO. Ye et al. demonstrate that PROF-GRPO consistently improves average@16 accuracy over baseline GRPO and over naive PRM-blending approaches.

**The gap this paper fills.** Ye et al. evaluate PROF using accuracy metrics only (average@16), and their code has not been released. But the failure modes that motivated PROF are not directly observable from average@16 accuracy alone:

- PROF's claim to suppress entropy collapse rests entirely on **training-time actor entropy curves**, never on test-time generation diversity or pass@k scaling (the Yue et al. framework is absent from PROF's bibliography).
- A model can improve accuracy while still learning shortcuts (Alam & Rastogi, 2025), and average@16 gains do not reveal whether chain-of-thought reasoning is logically valid.
- No prior work has independently replicated PROF-GRPO or evaluated it using the diversity and reasoning-quality diagnostics the community has developed since its publication.

We ask the question Ye et al. did not: **does PROF's accuracy improvement survive the pass@k and CoT-faithfulness diagnostics that define the current standard for evaluating RLVR training?**

---

## 2. Technical Approach

### 2.1 What We Do

We provide an independent replication of PROF-GRPO from Ye et al. on Qwen2.5-Math-1.5B — implementing their algorithm from the paper description since no code has been released — and add two evaluation axes their paper does not include:

**Pass@k scaling curves** (k = 1, 4, 8, 32) — the direct observable for entropy collapse established by Yue et al. (2025). If PROF preserves generative diversity, pass@k should scale meaningfully with k. If PROF helps accuracy but not diversity, pass@k curves will be indistinguishable from baseline GRPO despite the accuracy gap. PROF does not report any pass@k results; the only diversity evidence in the paper is a training-time actor entropy plot.

**Reasoning–Answer Consistency (RAC)** — a CoT-faithfulness evaluation measuring whether a model's chain-of-thought steps logically support its stated final answer, implemented as an LLM-as-judge score following the spirit of Wen et al.'s CoT-Pass@K (arXiv:2506.14245) and Huang et al.'s CACR (arXiv:2510.10104). Unlike pass@k, which measures *whether* diverse solutions exist, RAC directly probes *whether* the reasoning leading to correct answers is logically valid — targeting shortcut exploitation specifically. We frame RAC not as a new metric concept but as the application of established CoT-faithfulness evaluation to the specific question of whether PROF's filtering removes shortcut responses from the training distribution or merely decorrelates them from the reward signal.

### 2.2 PROF Implementation

Ye et al. provide complete pseudocode in Algorithm 1. Since no code has been released (the GitHub repository github.com/Chenluye99/PROF is public but reads "Codes and scripts will be released soon!"), we implement the following pipeline from scratch:

**Step 1 — Rollout generation.** For each prompt x, sample n=8 responses from the current policy at temperature 1.0. Each response receives a binary outcome reward ro ∈ {−1, +1} from the verifier.

**Step 2 — PRM scoring.** Load Qwen2.5-Math-PRM-7B in 4-bit quantization (inference only, no gradient) and score each rollout step-by-step. Steps are delimited by newlines, following Ye et al. Section 3.1. The trajectory-wise consistency score is:

```
r_pro_i = [mean(r_1, ..., r_H) − λ · I(H=1 or H≥Hλ)] · ro_i
```

where λ=10 and Hλ=30 are Ye et al.'s regularization hyperparameters.

**Step 3 — Balanced filtering.** Rollouts are divided into correct group G+ (ro=+1) and incorrect group G− (ro=−1). The number to remove from each group is computed via Ye et al.'s equation 2 to maximize outcome-reward variance in the kept set. G+ is ranked by r_pro descending; G− is ranked by r_pro ascending. The kept m=4 trajectories are passed to GRPO.

**Step 4 — Standard GRPO update.** The reward function and training objective are identical to baseline GRPO. The only variable is which trajectories survive to the update step.

**Replication checkpoint.** Ye et al. report ~37.2% average accuracy for baseline GRPO and ~39.6% for PROF-GRPO on Qwen2.5-Math-1.5B-base (Table 2, average across five benchmarks; note these are average@16 numbers, not pass@1). We use these as replication targets. Ye et al. also report PROF underperforming vanilla GRPO on AIME2024 (17.5% vs 20.6% for the 7B model), which we treat as an instability signal to monitor during replication.

**Hyperparameters (from Ye et al. Section 4.1):** AdamW optimizer, lr 1×10⁻⁶, asymmetric clip-higher with εlow=0.2, εhigh=0.28, 1024 prompts per iteration, KL coefficient 0.001, entropy coefficient 0.001, max 4096 tokens per response.

### 2.3 RAC Metric Definition and Prior Art Framing

We define Reasoning–Answer Consistency (RAC) as a scalar in [0, 1] measuring whether a model's chain-of-thought steps logically support its stated final answer, implemented as an LLM-as-judge score using frozen Qwen2.5-7B-Instruct.

**Relationship to prior work.** The core idea — judging whether CoT entails the final answer — appears in at least four prior works we explicitly build on:

| Prior work | Metric | Key difference from RAC |
|---|---|---|
| Wen et al. (2506.14245) | CoT-Pass@K | Correctness of CoT is binary (correct/incorrect), aggregated over k samples | 
| Huang et al. ACRE (2510.10104) | CACR | Tests consistency by shuffling answer options; multimodal QA setting |
| RACE (2506.04832) | Joint reasoning–answer eval | Hallucination detection framing; not specific to RLVR training comparison |
| PROF Appendix A.2 | Claude-3.7-Sonnet judge | Applied only to filtered-out responses during method validation, not to trained model outputs |

RAC's contribution is not the concept but the **application**: applying a continuous LLM-judge score to trained model outputs at evaluation time, across PROF-filtered and unfiltered GRPO checkpoints, to quantify whether PROF's filtering translates into higher CoT–answer faithfulness in the deployed model — a question none of the above papers addresses.

**RAC prompt structure:**

```
You are evaluating whether a mathematical reasoning chain
logically supports its stated final answer.

Problem: {problem}
Reasoning: {chain_of_thought}
Final Answer: {answer}

Score from 0 to 1:
1.0 = reasoning steps directly and correctly lead to the
      final answer with no logical gaps
0.5 = reasoning is partially valid but contains gaps or
      minor errors that do not fully justify the answer
0.0 = reasoning is flawed, irrelevant, or does not support
      the final answer even if the answer is correct

Output only a number between 0 and 1.
```

We validate this prompt on 50 held-out problems with ground-truth reasoning quality labels before scoring the full evaluation set, specifically checking that RAC scores are lower for shortcut responses (correct answer, identifiably flawed reasoning) than for genuinely correct responses.

### 2.4 Experimental Conditions

| Condition | Training data | Reward signal | Purpose |
|---|---|---|---|
| Baseline GRPO | All rollouts (unfiltered) | Binary outcome | Replicates standard RLVR; establishes entropy-collapse baseline |
| PROF-GRPO | PROF-filtered rollouts (Ye et al. Algorithm 1) | Binary outcome | Replicates Ye et al.; subject of diagnostic evaluation |

The reward function and training objective are identical across conditions. The only variable is the composition of the training dataset.

### 2.5 What Each Outcome Tells Us

**If PROF improves both accuracy and pass@k / RAC:** this provides the first empirical validation — using the Yue et al. framework that PROF's authors did not apply — that offline PRM filtering addresses the failure modes motivating its design, not just surface accuracy. This strengthens the case for PROF as a paradigm.

**If PROF improves accuracy but not pass@k / RAC (dissociation):** this is the more important finding. It would mean PROF shifts the distribution toward correct answers without preserving reasoning diversity or eliminating shortcut exploitation — improving the metric its authors measured while the underlying failure modes persist. This is a clear, actionable finding with direct implications for what offline filtering alone cannot fix.

Either outcome is publishable. The experimental design is falsifiable, the metrics are grounded in prior literature, and the question being asked is one the community needs answered before PROF becomes a standard component of RLVR pipelines.

---

## 3. Core Claims

**Primary falsifiable claim:** Ye et al. demonstrate PROF-GRPO improves average@16 accuracy over baseline GRPO. We test whether these gains are accompanied by (a) improved pass@k scaling (Yue et al. framework), indicating preserved generative diversity, and (b) higher Reasoning–Answer Consistency (following Wen et al. / Huang et al.), indicating reduced shortcut exploitation.

**Secondary objectives:**
1. Provide the first independent replication of Ye et al.'s accuracy results without access to their code.
2. Establish whether the pass@k entropy collapse documented by Yue et al. (2025) persists under PROF filtering — PROF's own entropy analysis is training-time only and does not address test-time generation diversity.
3. Quantify RAC in PROF-filtered vs. unfiltered GRPO checkpoints, providing a continuous CoT-faithfulness signal where PROF's own evaluation used only binary correct/incorrect labels.
4. Stress-test the specific PRM (Qwen2.5-Math-PRM-7B) that PROF's filtering depends on, contextualizing results against Tiwari et al.'s finding (arXiv:2603.06621) that 43% of RL reward gains on this PRM are attributable to stylistic shortcuts.

---

## 4. Related Work and Positioning

This paper sits at the intersection of four recent lines of work.

**PROF (Ye et al., arXiv:2509.03403, Sep 2025)** is our direct subject of study. We replicate their method from the paper description and extend their evaluation. We are not proposing a competing method; we are providing diagnostic analysis their paper does not contain, and an independent replication given no code has been released. The specific gap: PROF evaluates at average@16 only, does not report pass@k for any k > 1, does not cite or engage with the Yue et al. entropy-collapse framework, and does not report RAC-style CoT-faithfulness on trained model outputs.

**Entropy collapse in RLVR** is the methodological backbone of our pass@k evaluation. Yue et al. (arXiv:2504.13837, AI4Math 2025 Best Paper) establish the pass@k crossover framework. Wen et al. (arXiv:2506.14245) extend this with CoT-Pass@K, which additionally requires chain-of-thought correctness. Sullivan & Koller (arXiv:2509.21154) prove GRPO implicitly induces a PRM, providing a theoretical lens for interpreting PROF's redundancy with the explicit PRM. Our contribution is applying this diagnostic suite to PROF specifically.

**CoT-faithfulness and reasoning–answer consistency** has been operationalized by Wen et al. (CoT-Pass@K, 2506.14245), Huang et al. (CACR/ACRE, 2510.10104), and RACE (2506.04832). Our RAC is an application of this established approach to the PROF evaluation question, not a novel metric concept.

**PRM robustness** is examined by Tiwari et al. (arXiv:2603.06621, PRM-BiasBench), who study Qwen2.5-Math-PRM-7B — the exact PRM PROF uses — and document that adversarial sequences push PRM rewards above 0.9 on logically flawed reasoning. PROF's offline use of the PRM avoids live reward hacking but does not guarantee immunity to stylistic biases in the PRM's learned scoring function; this is the motivating risk for our RAC evaluation.

**Shortcut exploitation** in RLVR is documented by Alam & Rastogi (arXiv:2510.27044, NeurIPS 2025 Math-AI Workshop). RAC is our operationalization of their finding for continuous math reasoning, following the LLM-judge approach rather than their programmatic case-study design.

---

## 5. Team Responsibilities

| | Person | Role | Key Deliverables |
|---|---|---|---|
| **P1** | Training & RL infrastructure | Set up GRPO training loop (verl); implement PROF Algorithm 1 integration; train Baseline GRPO and PROF-GRPO on Qwen2.5-Math-1.5B; manage Kaggle P100 quota and checkpointing | Two model checkpoints; replication accuracy verified against Ye et al. Table 2 |
| **P2** | PRM pipeline & filtering | Deploy frozen Qwen2.5-Math-PRM-7B for step-level scoring (4-bit, inference only); implement trajectory scoring and balanced filtering from Algorithm 1; run smoke test to verify correct/incorrect balance; document filtering statistics | PROF-filtered dataset; filtering stats; smoke test results |
| **P3** | Evaluation & analysis | Validate RAC prompt on 50 held-out examples; implement pass@1, pass@k (k=4,8,32), and CoT-Pass@K evaluation suite; run RAC scoring; evaluate on MATH-500, OlympiadBench, AIME; produce pass@k scaling curves and all figures | All quantitative results + figures |

---

## 6. Experiment Setup

### 6.1 Models

| Role | Model | Purpose |
|---|---|---|
| Policy | Qwen2.5-Math-1.5B-base | Training target — matches Ye et al.'s smaller model |
| Frozen PRM | Qwen2.5-Math-PRM-7B (4-bit) | Step-level scorer for PROF filtering — inference only |
| RAC / CoT judge | Qwen2.5-7B-Instruct (frozen) | LLM-as-judge for Reasoning–Answer Consistency scoring |

### 6.2 Data

Following Ye et al., we use NuminaMath as the training prompt set. For evaluation: MATH-500 and OlympiadBench (in-distribution, matching Ye et al.'s Table 2) and AIME as an OOD probe.

### 6.3 Key Metrics

| Metric | What it measures | Replication target / reference |
|---|---|---|
| Average@16 | Surface accuracy; matches Ye et al.'s evaluation protocol | ~37.2% (GRPO), ~39.6% (PROF-GRPO) per Ye et al. Table 2 |
| Pass@k (k=4,8,32) | Test-time generation diversity / entropy collapse | Yue et al. 2025 framework; no PROF-specific prior target |
| CoT-Pass@K | Joint CoT-correctness + answer-correctness diversity | Wen et al. 2506.14245; no PROF-specific prior target |
| RAC | Continuous LLM-judge CoT–answer faithfulness | Following Wen / Huang et al.; no PROF-specific prior target |
| OOD accuracy (AIME) | Generalization; stress-tests Ye et al.'s instability on AIME | Note: Ye et al. report PROF underperforms GRPO at 7B on AIME |

### 6.4 Compute Budget

Primary training on Kaggle P100 (16GB) — Qwen2.5-Math-1.5B in bf16, ~6GB VRAM with headroom. PRM scoring on T4 (16GB) with 4-bit Qwen2.5-Math-PRM-7B, inference only. RAC judging via Kaggle T4, frozen Qwen2.5-7B-Instruct. Total estimated Kaggle quota: ~40–50 GPU-hours across both accounts.

---

## 7. Timeline (13 days, May 12–25)

| Days | Task | Owner | Deliverable |
|---|---|---|---|
| **May 12–13** | Environment setup: implement GRPO training loop; implement PRM inference pipeline; implement Algorithm 1 from Ye et al.; smoke-test filtering on 100 problems (verify correct/incorrect balance) | P1 + P2 | Working pipeline; filtering verified |
| **May 14–15** | Validate RAC/CoT-faithfulness prompt on 50 held-out examples; begin Baseline GRPO training run on full dataset | P1 + P3 | RAC prompt finalized; baseline training started |
| **May 16–17** | PROF-GRPO training run in parallel; monitor both runs for stability and AIME-style instability | P1 + P2 | Two model checkpoints |
| **May 18–19** | Run average@16, pass@k (k=4,8,32), CoT-Pass@K on both checkpoints across MATH-500, OlympiadBench; run RAC scoring | P3 | Core quantitative results |
| **May 20–21** | AIME OOD evaluation; qualitative shortcut examples; plot pass@k scaling curves; contextualise against Tiwari et al. PRM-BiasBench findings if PRM anomalies observed | P3 | All figures and tables |
| **May 22–23** | Write paper: intro, related work, method, experiments, analysis | All | Full draft |
| **May 24–25** | Revise, proofread, finalize figures, submit on OpenReview before AOE deadline | All | Submitted paper |

**Critical path:** The smoke test on Days 12–13 is non-negotiable — it catches Algorithm 1 bugs before compute is wasted. The RAC prompt validation on Days 14–15 is independent of training and must complete before evaluation begins. P3's CoT-judge work and P1's training run in parallel on Days 14–17.

**Contingencies:**
- If PROF releases code before submission, use it and state this explicitly — the diagnostic evaluation remains the contribution regardless.
- If a v2 of arXiv:2509.03403 appears reporting pass@k, the independent replication remains valuable but the positioning shifts toward RAC + PRM-robustness analysis.
- If AIME instability makes the 1.5B model's OOD results uninterpretable, drop AIME from the main results and report it in an appendix.

---

## 8. Expected Outcomes

**If PROF improves both accuracy and pass@k / RAC:** first empirical validation using the Yue et al. framework that offline PRM filtering addresses its stated failure modes, not just surface accuracy. Narrative: PROF's training-time entropy improvement translates to test-time diversity — here is the evidence that was missing.

**If PROF improves accuracy but shows no pass@k or RAC benefit (dissociation):** the more novel finding. PROF shifts the distribution toward correct answers while leaving entropy collapse and shortcut exploitation intact — improving the metric its authors measured while the underlying failure modes persist. This is a clear, actionable finding.

**If results are mixed or noisy:** the methodological contribution stands — an independent replication of Ye et al.'s accuracy results, the first application of the Yue pass@k framework to PROF, and a CoT-faithfulness evaluation using Wen et al.'s approach. This is a legitimate 4-page workshop contribution.

**Minimum viable paper:** replication of Ye et al.'s average@16 results, pass@k curves showing the presence or absence of entropy collapse under PROF, and RAC scores on generated traces — with honest framing of RAC as an application of established CoT-faithful

this is our plan, but it's 14th may and we still haven't started work. can you split everything between 3 people so that we can be done with experiments in the next 5 days at most?



Looking at your timeline, you're already 2 days behind and need to compress 13 days into 5 (May 14–18) to leave May 19–25 for writing. Here's the restructured plan with ruthless parallelism.

---

## Revised 5-Day Execution Plan (May 14–18)

The core insight: P1, P2, and P3 must work **fully in parallel from day one**. Nothing waits for anything else except the smoke test gate on Day 1.

---

### Day 1 — Today, May 14: Setup + Smoke Test Gate

Everything happens today. This is your most critical day.

**P1 — GRPO training loop**
- Install verl, verify Qwen2.5-Math-1.5B loads in bf16 on P100 (~6GB)
- Implement standard GRPO training loop with binary outcome reward
- Verify a single training step runs without OOM
- *Deliverable by EOD:* one forward + backward pass confirmed working

**P2 — PRM + Algorithm 1**
- Load Qwen2.5-Math-PRM-7B in 4-bit on T4, confirm inference runs
- Implement the trajectory consistency score formula and balanced filtering from Algorithm 1
- Run smoke test on 100 problems: verify correct/incorrect group balance looks sane
- *Deliverable by EOD:* filtering pipeline confirmed; smoke test passed

**P3 — RAC prompt validation**
- Pull 50 held-out problems with known reasoning quality (manually label or use existing MATH-500 annotated errors)
- Run RAC judge (frozen Qwen2.5-7B-Instruct) on all 50
- Confirm RAC scores are lower for shortcut responses than genuine ones — if not, revise prompt today
- *Deliverable by EOD:* RAC prompt finalized and validated

**Gate:** P1 and P2 must sync at end of Day 1. If smoke test fails, debug tonight. Training does **not** start until filtering is confirmed correct.

---

### Day 2 — May 15: Both Training Runs Start Simultaneously

**P1 — Launch Baseline GRPO**
- Start full training run on NuminaMath, Qwen2.5-Math-1.5B
- Set up checkpointing every 200 steps
- Monitor for instability; log training curves
- *This run takes ~30–40 GPU-hours, so it must start today*

**P2 — Launch PROF-GRPO in parallel**
- Plug P1's training loop into P2's filtering pipeline
- Start PROF-GRPO training run on second Kaggle account simultaneously
- Monitor filtering statistics each iteration (how many trajectories survive, correct/incorrect ratio)
- *Deliverable:* both runs live and stable by end of Day 2

**P3 — Build evaluation suite**
- Implement pass@1, pass@k (k=4, 8, 32), CoT-Pass@K evaluation harness
- Test harness on a toy checkpoint or base model to confirm it runs
- Prepare MATH-500, OlympiadBench, AIME prompt sets in the right format
- *Deliverable:* evaluation harness ready to plug in a checkpoint the moment training finishes

> **Scope cut to make this feasible:** Drop CoT-Pass@K from the main results if it adds implementation complexity — pass@k and RAC already cover both failure modes. CoT-Pass@K can go in a footnote or appendix.

---

### Day 3 — May 16: Training Continues + Early Evaluation

**P1 — Monitor + mid-run checkpoint**
- Pull a mid-training checkpoint (~50% of steps) from both runs
- Flag any signs of the AIME instability Ye et al. report at 7B — less likely at 1.5B but worth watching
- If either run is diverging, diagnose today while there's still time to restart

**P2 — Intermediate filtering stats**
- Compute and log: average trajectories filtered per batch, PRM score distributions for G+ vs G−, proportion of 1-step responses hitting the λ penalty
- These become Table 1 or Figure 1 in the paper — important to collect during training, not after

**P3 — Run evaluation on mid-run checkpoints**
- Run pass@k evaluation on mid-training checkpoints for both models
- This is a sanity check, not final results — but it tells you early whether the dissociation signal is present
- Begin drafting figure templates (pass@k curves, RAC bar chart) with placeholder data

---

### Day 4 — May 17: Training Ends, Full Evaluation Runs

**P1 — Confirm final checkpoints**
- Both training runs should complete today (or early May 18 at latest)
- Verify average@16 is in the right ballpark: ~37% baseline, ~39.5% PROF per Ye et al.
- If numbers are badly off (e.g., >3% gap from replication target), flag immediately — this becomes a finding to report, not a failure to hide

**P2 — Final PRM scoring pass**
- Run Qwen2.5-Math-PRM-7B over the final evaluation generations from both models
- Compute PRM score distributions on evaluation outputs — this contextualizes Tiwari et al.'s stylistic shortcut finding if anomalies appear

**P3 — Full evaluation suite on final checkpoints**
- Run pass@k (k=1, 4, 8, 32) on MATH-500 and OlympiadBench for both models
- Run RAC scoring on all generated traces
- Run AIME for OOD probe (if results are interpretable; if not, appendix)
- *Deliverable by EOD:* all numbers in a spreadsheet; pass@k curves plotted

---

### Day 5 — May 18: Results Locked, Writing Begins

**All three — joint analysis session (synchronous, first thing)**
- Identify which outcome scenario you're in: both improve / dissociation / mixed
- Decide framing within 2 hours — this determines the paper's narrative arc
- Lock all figures and tables (no more rerunning experiments after this point)

**P1 — Write Sections 1 and 2** (Problem Statement + Technical Approach)
- You know the method best; draft these while P3 finalizes figures

**P2 — Write Section 4 and the Appendix** (Related Work + filtering statistics table)
- Filtering stats table from Day 3 goes here

**P3 — Write Section 3 and finalize all figures** (Experiments + Results)
- Pass@k scaling curves, RAC bar chart, accuracy comparison table

**Target by end of May 18:** complete rough draft exists. May 19–25 is revision, proofreading, and submission — which is a comfortable buffer.

---

## What to Cut if You're Running Behind

In rough priority order, drop these to protect the core contribution:

1. **AIME OOD evaluation** — interesting but not load-bearing; move to appendix if tight
2. **CoT-Pass@K** — RAC already covers shortcut exploitation; CoT-Pass@K is redundant for a 4-page workshop paper
3. **Tiwari et al. PRM-BiasBench contextualization** — mention in related work prose, don't run additional analysis
4. **Mid-run checkpoint analysis** — useful but Day 3 evaluation is optional if training is running clean

**Non-negotiable core:** average@16 replication + pass@k curves + RAC scores. Everything else is supporting material.

---

## Single-Line Summary Per Person

**P1** owns the training infrastructure and is the integration point — nothing runs without you, so your Day 1 deliverable is the hardest gate.

**P2** owns the PRM pipeline and filtering — your smoke test on Day 1 is the second gate, and your filtering stats during training are the only inside-the-box view of what PROF is actually doing.

**P3** owns evaluation and figures — you can work almost entirely independently once the RAC prompt is validated, and your harness should be ready before the checkpoints land.