# CGMR: Comparator-Guided Minimum-Information Repair

Reference implementation for **“When Recognition Fails Before Competitive Hypotheses Disappear: Rethinking Catastrophic Forgetting in Continual Multilingual ASR”** (ICASSP 2027 submission).

## Abstract

**CGMR studies catastrophic forgetting in continual multilingual ASR through the evolution of the model’s hypothesis space, rather than only through endpoint error degradation.**

Under finite-beam decoding, recognition can fail before competitive old-language hypotheses disappear: the adapted model may still contain a comparator-competitive candidate in its beam, but rank a higher-error hypothesis above it. We call this the **ranking-accessible** regime.

**Comparator-Guided Minimum-Information Repair (CGMR)** targets this regime by projecting the adapted candidate posterior under a comparator-defined edit budget while minimizing its KL deviation from the adapted model. The projected targets are then amortized back into the recognizer, so inference uses ordinary beam search with **no comparator and no test-time reranking**.

Experiments on CL-MASR and FLEURS show that ranking-first forgetting is common and that ranking-accessible failures are substantially more repairable than failures after competitive-candidate loss.

## Highlights

- **A process view of forgetting.** CGMR separates recognition regression from competitive-hypothesis loss instead of treating all endpoint degradation as the same failure.
- **Ranking-accessible failures.** Old-language recognition can degrade while comparator-competitive hypotheses remain inside the adapted model’s beam.
- **Minimum-information repair.** CGMR performs a language-level KL projection under a comparator-defined edit budget rather than directly minimizing expected edit risk.
- **Model-level amortization.** The repaired posterior is amortized back into the recognizer; inference requires neither the comparator nor a test-time reranker.
- **State-conditioned repairability.** Ranking-accessible failures are markedly more recoverable than failures after competitive candidates have disappeared.
- **Backbone transfer.** The same repair principle is evaluated with Whisper-small, OWSM v3.1-small, and SeamlessM4T-v2-Large.

## Method at a Glance

For an old-language utterance, the pre-adaptation comparator defines the reference error boundary `b_i`. The adapted recognizer is then diagnosed from its own finite beam:

- **N — comparator-competitive:** the adapted 1-best is no worse than the comparator.
- **A — ranking-accessible:** the adapted 1-best has regressed, but a comparator-competitive alternative is still present in the beam.
- **D — candidate loss:** no comparator-competitive candidate remains accessible.

CGMR targets **state A**: the failure has occurred at the ranking level, while useful support is still available.

```text
pre-adaptation comparator C
        │
        └── baseline error boundary b_i

adapted recognizer P
        │
        ├── beam candidates S_i
        ├── N / A / D diagnosis
        └── retain state-A samples
                    │
                    ▼
            anchor posterior p_i
                    │
                    ▼
      comparator-budgeted KL projection
                    │
                    ▼
             repaired target q_i*
                    │
                    ▼
          amortize back into model
                    │
                    ▼
           ordinary beam decoding
```

The comparator is used as a **boundary, not a reranker**: candidates and anchor scores come from the adapted recognizer, while the comparator supplies the edit budget used to construct the repair target.

## Core Projection

For each old-language cohort, CGMR solves the minimum-information projection

```text
q_i*(h) ∝ p_i(h) exp(-lambda_l d_i(h))
```

subject to the comparator-defined language-level edit budget. The shared `lambda_l` is determined by monotone one-dimensional bisection; it is not a tuned repair temperature.

After projection, the repaired old-language targets are fitted jointly with current-language supervision, producing a single recognizer for ordinary inference.

## Method in Code

The repository follows the paper’s conceptual structure rather than experiment-specific run folders.

| Paper concept | Implementation |
|---|---|
| N / A / D forgetting states | `cgmr/states.py::classify_state` |
| First observed regression and candidate loss | `cgmr/states.py::first_passage` |
| Anchor posterior and Gibbs tilt | `cgmr/projection.py::gibbs_tilt_from_log_scores` |
| Language-level comparator budget | `cgmr/projection.py::project_language_cohort` |
| Monotone solution for the shared multiplier | `cgmr/projection.py::solve_language_lambda` |
| Amortization on the fixed candidate support | `cgmr/objective.py::restricted_target_cross_entropy` |
| Exact score-gradient form | `cgmr/objective.py::restricted_score_gradient` |

## Key Results

| Observation | Result |
|---|---:|
| Ranking-first onset — CL-MASR | **82.86%** |
| Ranking-first onset — FLEURS | **86.73%** |
| State-A vs. state-D complete-repair gap with CGMR | **+21.96 pp** |

See the paper for the complete retention–adaptation comparisons, ablations, temporal analyses, and cross-backbone results.

## Minimal Example

A model-free example demonstrates state diagnosis and the language-level projection without loading an ASR backbone:

```bash
python examples/minimal_cgmr.py
```

The core projection code is NumPy-only, making the central method easy to inspect and unit-test independently of a speech model.

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
│   └── micro_cv15_example.json
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
