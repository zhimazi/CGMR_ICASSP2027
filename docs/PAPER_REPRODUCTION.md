# Paper-to-repository map

This document links the main manuscript claims to the public code and result artifacts in this repository.

## Mapping

| Manuscript item | Role | Repository artifact |
|---|---|---|
| N/A/D state definition | process-level forgetting states | `cgmr/projection.py::classify_state` |
| Comparator-budgeted minimum-KL projection | core CGMR objective | `cgmr/projection.py` |
| Language-level bisection for lambda | projection solver | `cgmr/projection.py` |
| Amortization into a single recognizer | model-level repair | `scripts/run_er_cgmr_micro.py` |
| Table 1 | temporal diagnosis | `paper_results/table1_temporal_diagnosis.csv` |
| Table 2 | retention–adaptation results | `paper_results/table2_retention_adaptation.csv` |
| Table 3 | CGMR design ablations | `paper_results/table3_ablation.csv` |
| Fig. 4 | state-conditioned repairability | `paper_results/fig4_repairability_contrasts.csv` |
| Experimental settings | protocol / hyperparameters | `docs/HYPERPARAMETERS.md` |
| Executable implementation check | compact end-to-end path | `configs/micro_cv15_seed2027.json` + `scripts/run_er_cgmr_micro.py` |

## Result artifacts

The CSV files under `paper_results/` are machine-readable copies of the numerical values reported in the submitted manuscript. They provide a stable reference for downstream plotting, checking, or comparison.

## Executable reference path

The Common Voice 15.0 example exercises the main implementation stages:

```text
comparator
  → continual adaptation
  → beam candidate extraction
  → state-A selection
  → comparator-budgeted projection
  → CGMR amortization
  → endpoint evaluation
```

This example is deliberately compact so that the implementation path can be inspected and run without redistributing the full paper-scale training assets.

## Paper-scale protocol

The manuscript-level experimental constraints and archived implementation settings are consolidated in `docs/HYPERPARAMETERS.md`. This includes beam settings, LoRA adaptation, update counts, learning rates, construction-pool sizes, temporal checkpoint grids, and other protocol details for which archived evidence is available.

## Reproduction scope

The repository releases the method implementation, experimental protocol, manuscript-level result tables, and provenance for the executable validation. Large external assets—dataset audio, base-model weights, paper-scale checkpoints, feature caches, and machine-specific manifests—are not bundled.

As with many large ASR experiments, exact numerical reruns can additionally depend on dataset snapshots, model revisions, and software/hardware environment. The released artifacts are organized so that the method, evaluation protocol, and reported results can be inspected independently.
