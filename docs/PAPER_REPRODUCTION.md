# Paper reproduction map

This document maps the submitted paper's main empirical claims to public repository artifacts and makes the current reproduction boundary explicit.

## Source of truth

The **submitted 5-page ICASSP 2027 PDF** is the source of truth for the paper's reported values. The files under `paper_results/` are direct transcriptions of that PDF.

The executable Common Voice 15.0 micro validation in this repository is a separate implementation check. It is not the source of the paper's Tables 1–3.

## Claim-to-artifact map

| Paper item | What it supports | Public artifact | Current status |
|---|---|---|---|
| Table 1 | ranking-first onset and post-onset progression on CL-MASR / CL-MASR+ER / FLEURS | `paper_results/table1_temporal_diagnosis.csv` | reported values archived; exact paper-scale trajectory runner/config still to be released |
| Eq. (2) | N/A/D state definition | `cgmr/projection.py::classify_state` | implemented and unit-tested |
| Eq. (5)–(7) | comparator-budgeted minimum-KL projection and bisection | `cgmr/projection.py` | implemented and unit-tested |
| Eq. (8) | amortization into a single recognizer | `scripts/run_er_cgmr_micro.py` | executable reference implementation; paper-scale run config not yet released |
| Table 2 | retention–adaptation results and cross-backbone transfer | `paper_results/table2_retention_adaptation.csv` | reported values archived; exact paper-scale training/evaluation lineage still to be released |
| Table 3 | design ablations | `paper_results/table3_ablation.csv` | reported values archived; exact ablation scripts/configs still to be released |
| Fig. 4 | state-A failures are more repairable than state-D failures | `paper_results/fig4_repairability_contrasts.csv` | annotated paper contrasts archived; source held-out row outputs still to be released |

## What an exact reproduction release still needs

A complete paper-scale release should include the following, tied to immutable commits and artifact hashes:

1. Final CL-MASR and FLEURS data preparation scripts and manifest hashes.
2. Exact five seed IDs.
3. Exact continual-adaptation configs for each backbone.
4. Exact temporal checkpoint grid and state-trajectory analysis script.
5. Exact CGMR, ER+CGMR, N-best MWER, and baseline configs.
6. Exact Table 3 ablation configs.
7. Per-seed raw metrics from which the reported mean ± standard deviation values are computed.
8. Held-out repairability rows underlying Fig. 4.
9. Model/checkpoint revisions or hashes sufficient to trace every reported result.
10. Environment capture for the paper-scale runs.

The specific hyperparameter fields that are missing from the 5-page paper are listed in `docs/HYPERPARAMETERS.md`.

## Integrity rule

Do not back-fill missing paper-scale settings from the micro validation or from earlier manuscript drafts. If a setting cannot be traced to the final experiment lineage that produced the submitted numbers, keep it explicitly unresolved rather than guessing.
