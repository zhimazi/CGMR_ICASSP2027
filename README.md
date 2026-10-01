# CGMR: Comparator-Guided Minimum-Information Repair

Reference implementation for **“When Recognition Fails Before Competitive Hypotheses Disappear: Rethinking Catastrophic Forgetting in Continual Multilingual ASR”** (ICASSP 2027 submission).

CGMR is built around a process view of forgetting: under finite-beam decoding, old-language recognition can regress before competitive hypotheses disappear. That intermediate, **ranking-accessible** state creates a repair window in which the model still contains useful alternatives but ranks them incorrectly.

## Abstract

Catastrophic forgetting in continual multilingual ASR is usually measured through degradation of the final 1-best prediction. CGMR separates two events that endpoint metrics conflate: **recognition regression** and **loss of comparator-competitive hypotheses from the beam**. When regression occurs first, the adapted recognizer enters a ranking-accessible state in which useful old-language candidates remain available but are misranked.

**Comparator-Guided Minimum-Information Repair (CGMR)** repairs this state without replacing the adapted model's candidate support. The pre-adaptation comparator supplies a reference error boundary, while candidates and their anchor scores come from the adapted recognizer itself. CGMR then finds the KL-nearest posterior that satisfies the comparator-defined language-level edit budget and amortizes the projected targets back into a single recognizer. After repair, the comparator and candidate lists are discarded: inference uses ordinary beam search with no test-time reranker.

Experiments in the accompanying paper study this process on CL-MASR and FLEURS and evaluate the repair across heterogeneous multilingual ASR backbones.

## Highlights

- **Process diagnosis, not endpoint-only forgetting.** The code explicitly separates recognition regression from competitive-candidate loss through N/A/D states and observed-grid first passage.
- **A repair window with intact support.** CGMR targets state **A**, where 1-best recognition has regressed but a comparator-competitive alternative is still accessible.
- **Comparator as a boundary, not a reranker.** The comparator contributes the risk budget; candidates and their native ordering come from the adapted model.
- **Minimum-information revision.** The target posterior is the KL-nearest feasible projection under a language-level edit constraint.
- **No tuned repair temperature.** The language multiplier is determined by the comparator budget through monotone one-dimensional bisection.
- **Model-level repair.** Projected targets are amortized into the recognizer, so deployment requires neither comparator access nor test-time reranking.

## Method in Code

The repository is organized around the paper's conceptual steps rather than around experiment-specific run folders.

| Paper concept | Implementation |
|---|---|
| N / A / D forgetting states | `cgmr/states.py::classify_state` |
| First observed regression and candidate loss | `cgmr/states.py::first_passage` |
| Anchor posterior and Gibbs tilt | `cgmr/projection.py::gibbs_tilt_from_log_scores` |
| Language-level comparator budget | `cgmr/projection.py::project_language_cohort` |
| Monotone solution for the shared multiplier | `cgmr/projection.py::solve_language_lambda` |
| Amortization on the fixed candidate support | `cgmr/objective.py::restricted_target_cross_entropy` |
| Memory-efficient exact score gradient | `cgmr/objective.py::restricted_score_gradient` |

The central projection is

```text
q_i*(h) ∝ p_i(h) exp(-lambda_l d_i(h))
```

where the shared `lambda_l` is chosen so that the language-level expected edit risk meets the comparator-defined budget.

## Minimal Example

A model-free example demonstrates the two central operations—state diagnosis and language-level projection:

```bash
python examples/minimal_cgmr.py
```

The core projection code is NumPy-only and can be inspected or unit-tested without loading an ASR backbone.

## Optional ASR Reference Path

`scripts/run_er_cgmr_micro.py` provides a compact Whisper-based integration example. It routes state diagnosis, posterior projection, and amortization through the reusable `cgmr/` modules.

The accompanying Common Voice configuration is an executable validation example rather than a paper-scale reproduction package.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch and torchaudio should be installed separately for the target CUDA environment.

Run the method-level tests with:

```bash
pytest -q
```

## Repository Layout

```text
CGMR_ICASSP2027/
├── cgmr/
│   ├── states.py        # N/A/D diagnosis and first passage
│   ├── projection.py    # comparator-budgeted I-projection
│   └── objective.py     # amortization objective
├── examples/
│   └── minimal_cgmr.py  # model-free method demo
├── scripts/
│   ├── prepare_cv15_subset.py
│   └── run_er_cgmr_micro.py
├── configs/
│   └── micro_cv15_seed2027.json
├── tests/
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Release Scope

This repository releases the **method implementation and reference code**. Trained model weights, LoRA adapter weights, paper-scale checkpoints, intermediate beam caches, and training logs are not distributed.

## Citation

If you use this code, please cite the accompanying manuscript:

> Peihong Zhang, Mingzhuo Zhou, Shuyu Li, and Shengchen Li.  
> **When Recognition Fails Before Competitive Hypotheses Disappear: Rethinking Catastrophic Forgetting in Continual Multilingual ASR.**

Formal proceedings metadata will be added after publication.
