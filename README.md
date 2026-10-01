# CGMR: Comparator-Guided Minimum-Information Repair

> Code and supplementary material for **“When Recognition Fails Before Competitive Hypotheses Disappear: Rethinking Catastrophic Forgetting in Continual Multilingual ASR”** (ICASSP 2027 submission).

CGMR revisits catastrophic forgetting in continual multilingual ASR from a **process** perspective: recognition can regress before competitive hypotheses disappear from a finite beam. This ranking-accessible stage exposes a repair window in which useful old-language hypotheses still exist but are misranked.

## Abstract

Catastrophic forgetting in continual multilingual automatic speech recognition (ASR) is commonly assessed through degradation in previously learned languages, which is often interpreted as loss of acquired language capability. Under finite-beam decoding, however, recognition can regress while competitive hypotheses remain accessible but are misranked. By tracking old-language utterances throughout adaptation, we show that forgetting typically enters a ranking-accessible stage before competitive-hypothesis loss. This temporal separation reveals a repair window in which useful alternatives remain available despite degraded 1-best recognition. Building on this observation, we propose **Comparator-Guided Minimum-Information Repair (CGMR)**, which uses the pre-adaptation model as a reference to minimally revise the posterior over still-accessible candidates and fits the repaired targets back into a single recognizer. Experiments across benchmarks and multilingual ASR backbones show that CGMR improves old-language recognition while preserving current-language performance, with ranking-accessible failures being substantially more repairable than failures after candidate loss. Inference requires neither comparator access nor test-time reranking.

## Highlights

- **Forgetting is often ranking-first.** Ranking-first onset accounts for **82.86%** on CL-MASR and **86.73%** on FLEURS.
- **Replay changes progression more than onset.** Under ER, ranking-first onset remains **81.46%**, while the standardized post-onset progression contrast reverses sign.
- **CGMR repairs ranking-accessible failures.** It performs a minimum-information posterior revision under a comparator-defined edit budget.
- **The effect transfers across backbones.** CGMR improves old-language recognition with Whisper-small, OWSM v3.1-small, and SeamlessM4T-v2-Large.
- **State A is more repairable than state D.** The complete-repair-rate gap reaches **+21.96 pp** under CGMR.
- **No test-time comparator or reranker is required.** The repaired targets are amortized into a single recognizer.

## Method Overview

For an old-language utterance, let the pre-adaptation comparator define baseline error `b_i`. Under the adapted model:

- **N**: 1-best recognition remains comparator-competitive.
- **A**: 1-best recognition regresses, but a comparator-competitive candidate remains in the beam.
- **D**: no comparator-competitive candidate remains in the beam.

CGMR acts on state-A samples. For each old-language cohort, it solves the minimum-KL projection

```text
q_i*(h) ∝ p_i(h) exp(-lambda_l d_i(h))
```

subject to the comparator-defined aggregate edit budget. The language-specific `lambda_l` is obtained by one-dimensional bisection. The projected old-language targets are then trained jointly with current-language CE, producing one repaired recognizer for ordinary beam-search inference.

## Experimental Setup

- **Benchmarks:** CL-MASR and FLEURS
- **Primary backbone:** Whisper-small
- **Cross-backbone evaluation:** OWSM v3.1-small and SeamlessM4T-v2-Large
- **Continual adaptation:** LoRA
- **Default decoding:** beam size `K=8`, zero length penalty
- **Paper-scale reporting:** arithmetic mean ± sample standard deviation across five protocol seeds `{2027, 2028, 2029, 2030, 2031}`
- **CGMR sampling:** 1/2 language-balanced old samples + 1/2 current-language samples
- **Inference:** repaired recognizer only; no comparator and no test-time reranking

The manuscript-scale protocol is machine-readable in [configs/paper_protocol.json](configs/paper_protocol.json). Implementation-level settings and data budgets are collected in [supplementary/hyperparameters.md](supplementary/hyperparameters.md), and the artifact/reproducibility boundary is documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Main Results

### Temporal diagnosis

| Setting | Censored (%) | Ranking-first (%) | Ψstd (pp) |
|---|---:|---:|---:|
| CL-MASR | 28.8 | **82.86** [79.72, 85.94] | **+16.66** [6.77, 26.77] |
| CL-MASR + ER | 44.4 | **81.46** [77.70, 84.99] | **−32.52** [−44.68, −21.40] |
| FLEURS | 35.7 | **86.73** [83.66, 89.55] | **+13.21** [4.18, 22.33] |

### CGMR across datasets and backbones

| Backbone | Dataset | Method | Old gain ↑ | Current Δ ↓ |
|---|---|---|---:|---:|
| Whisper-small | CL-MASR | CGMR | **+1.97 ± 0.18** | **−1.05 ± 0.22** |
| Whisper-small | CL-MASR | ER+CGMR | **+3.74 ± 0.22** | **−0.97 ± 0.11** |
| Whisper-small | FLEURS | CGMR | **+2.16 ± 0.15** | **−1.17 ± 0.08** |
| OWSM v3.1-small | FLEURS | CGMR | **+1.89 ± 0.17** | **−1.09 ± 0.10** |
| SeamlessM4T-v2-Large | FLEURS | CGMR | **+1.77 ± 0.13** | **−1.06 ± 0.07** |

Full comparison and ablation tables are provided as CSV files under [supplementary/](supplementary/).

## Supplementary Material

The repository includes additional diagnostics that are useful for interpreting the main claims but are too detailed for the five-page paper:

- **checkpoint-resolution robustness** of ranking-first onset;
- **beam-size / search-budget control** at `K ∈ {4, 8, 16}`;
- **parameterization control** contrasting LoRA-style adaptation with dense decoder adaptation;
- full machine-readable versions of Tables 1–3 and the Fig. 4 repairability contrast;
- implementation-level hyperparameters and data-budget details.

See [supplementary/README.md](supplementary/README.md).

## Reproducibility Contract

The repository uses two deliberately separate artifact tiers:

1. **Manuscript-scale protocol and reported aggregates.** The five-seed contract, model/training settings, data budgets, and aggregation semantics are recorded in `configs/paper_protocol.json`. The helper `scripts/aggregate_seed_results.py` refuses to aggregate incomplete or duplicated seed sets.
2. **Public executable validation.** The Common Voice 15.0 micro path is a compact end-to-end implementation check. It exercises the CGMR pipeline but is not presented as a generator of the manuscript tables.

Large external assets such as dataset audio, upstream model weights, paper-scale checkpoints, feature caches, and machine-specific manifests are not redistributed. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for details.

## Reference Implementation

The public code is intentionally compact:

```text
cgmr/
└── projection.py                 # N/A/D states + comparator-budgeted projection

scripts/
├── prepare_cv15_subset.py        # compact public validation panel
├── run_er_cgmr_micro.py          # end-to-end ER → CGMR reference path
└── aggregate_seed_results.py     # strict five-seed reporting utility

configs/
├── micro_cv15_seed2027.json      # executable public reference configuration
└── paper_protocol.json           # manuscript-scale protocol and reporting contract
```

The compact Common Voice validation is provided to make the core algorithm inspectable and executable; the paper-scale numerical results are reported separately in `supplementary/`.

## Repository Layout

```text
CGMR_ICASSP2027/
├── README.md
├── REPRODUCIBILITY.md
├── cgmr/
├── scripts/
├── configs/
├── supplementary/
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Citation

If you use this code, please cite the accompanying manuscript:

> Peihong Zhang, Mingzhuo Zhou, Shuyu Li, and Shengchen Li.  
> **When Recognition Fails Before Competitive Hypotheses Disappear: Rethinking Catastrophic Forgetting in Continual Multilingual ASR.**

Formal proceedings metadata will be added after publication.
