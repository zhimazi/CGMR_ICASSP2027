# Supplementary Material

This directory collects numerical results and implementation details that complement the five-page ICASSP manuscript.

The main paper focuses on the central diagnosis and CGMR repair mechanism. The material below exposes additional robustness checks and protocol details for readers who want to inspect the evidence more closely.

## 1. Paper tables in machine-readable form

- `table1_temporal_diagnosis.csv` — temporal diagnosis of forgetting.
- `table2_retention_adaptation.csv` — retention–adaptation results, including cross-backbone evaluation.
- `table3_ablation.csv` — CGMR design ablations on CL-MASR with Whisper-small.
- `fig4_repairability_contrasts.csv` — state-A versus state-D complete-repair-rate contrasts.

These files mirror the numerical values reported in the submitted manuscript.

## 2. Temporal robustness beyond the main table

The archived trajectory analysis includes several controls that support the interpretation that ranking-first onset is not a checkpoint-grid artifact:

| Analysis | Result |
|---|---:|
| Native-top state assignment | 82.81% ranking-first [79.47, 86.01] |
| Finer 0–64 checkpoint grid | 82.90% ranking-first |
| Excluding onsets spanning the two longest checkpoint intervals | 83.76% ranking-first |
| Comparator exact token path still in adapted beam at ranking-first onset | 57.13% |
| Comparator normalized transcript still in adapted beam at ranking-first onset | 69.51% [64.88, 74.02] |
| Ranking share of first-event positive edit mass | 79.20% [74.55, 83.95] |
| Ranking-first rows still accessible at the next checkpoint | 88.47% |
| Ranking-first rows that later enter D | 30.16% |
| Median observed delay before later D transition | 201 updates |

The corresponding machine-readable file is `extra_temporal_robustness.csv`.

## 3. Beam-size / search-budget control

At CL-MASR seed 2027, update 490:

| Beam K | Beam-oracle mER ↓ | Competitive-candidate survival ↑ | Ordinary 1-best mER ↓ |
|---:|---:|---:|---:|
| 4 | 21.71 | 32.67% | 26.50 |
| 8 | 19.56 | 46.19% | 26.44 |
| 16 | 17.60 | 58.29% | 27.85 |

Increasing the search budget exposes substantially better surviving candidates, while the model's native 1-best decision does not improve correspondingly. This separates candidate accessibility from candidate ranking.

See `extra_beam_size_control.csv`.

## 4. Parameterization control

A dense-decoder control was run on the same adaptation stream with 153.6M trainable decoder parameters. By update 16, 76.42% of rows had already entered support loss. Its final first-passage counts were 79 ranking-first, 603 support-first, and 22 censored; only 11.63% of observed onsets were ranking-first, and ranking contributed 5.53% of first-event edit mass.

This control shows that the temporal ordering is not assumed to be universal: the dominant failure path depends on the geometry of adaptation.

See `extra_parameterization_control.csv`.

## 5. Hyperparameters

Paper-declared settings and implementation-level reference settings are summarized in `hyperparameters.md`.

### Scope note

The additional controls above come from the archived diagnostic runs used during method development and analysis. Their scope is stated explicitly in each table (for example, seed 2027 for the beam-size control). They are included as supplementary evidence rather than as replacements for the five-seed headline results in the submitted paper.
