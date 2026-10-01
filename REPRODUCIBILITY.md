# Reproducibility and artifact scope

This repository separates **manuscript-scale experimental reporting** from the **compact executable public validation**.

That distinction is intentional: the paper-scale experiments use larger datasets, multiple backbones, five reporting seeds, temporal checkpoint sweeps, and held-out repairability cohorts, while the public micro validation is designed to make the core CGMR path easy to inspect and execute.

## Artifact tiers

### 1. Manuscript-scale protocol

The machine-readable protocol is in `configs/paper_protocol.json`. It fixes:

- the five-seed reporting contract;
- aggregation semantics (arithmetic mean and sample standard deviation, `ddof=1`);
- primary LoRA settings and trainable-parameter count;
- adaptation and repair learning rates;
- beam-search settings;
- CL-MASR checkpoint and data budgets;
- FLEURS checkpoint and data budgets;
- CGMR sampling, projection, and comparator-access rules.

The seed identifiers used by the manuscript protocol are:

`2027, 2028, 2029, 2030, 2031`.

### 2. Public executable validation

`configs/micro_cv15_seed2027.json` and `scripts/run_er_cgmr_micro.py` provide a compact end-to-end validation path on a public Common Voice panel.

This path exercises:

`comparator -> continual adaptation -> beam extraction -> state-A diagnosis -> comparator-budgeted projection -> repair -> endpoint evaluation`.

It is an implementation validation, not a claim that the compact run regenerates the manuscript tables.

## Five-seed reporting contract

For every experimental condition reported as `mean ± std`:

1. exactly five unique seed identifiers must be present;
2. the point estimate is the arithmetic mean across seeds;
3. the dispersion term is the sample standard deviation (`ddof=1`);
4. aggregation is performed after each seed produces its complete condition-level metric;
5. incomplete seed sets are not silently averaged.

The helper `scripts/aggregate_seed_results.py` enforces this contract and fails on missing, duplicated, or unexpected seeds.

Example schema for Table-2-style aggregation:

```csv
seed,section,backbone,dataset,method,old_gain,current_delta
2027,main,Whisper-small,CL-MASR,CGMR,1.83,-1.10
...
```

The public repository intentionally does not synthesize missing per-seed values from aggregate means and standard deviations.

## Deterministic reference behavior

The executable reference runner fixes Python, NumPy, and PyTorch seeds, disables TF32, disables flash and memory-efficient SDP kernels, enables deterministic algorithms where available, and uses deterministic beam search.

Exact numerical identity across machines is still not guaranteed because ASR pipelines can depend on:

- CUDA, cuDNN, and kernel versions;
- GPU architecture;
- dataset snapshot and audio decoder behavior;
- model/tokenizer revision;
- floating-point reduction order.

The target is **protocol reproducibility and claim auditability**, not bitwise identity.

## Data and model assets

Large or externally licensed assets are not redistributed in this repository. In particular, the repository does not bundle:

- dataset audio;
- upstream pretrained model weights;
- paper-scale intermediate checkpoints;
- feature caches;
- machine-specific absolute-path manifests.

The repository instead records data budgets, filtering rules, model identifiers, method code, and machine-readable result tables.

## Evaluation isolation

Construction, development, temporal-diagnosis, and final evaluation cohorts are treated as distinct roles. Repairability comparisons are evaluated on a disjoint held-out cohort.

Comparator information is used for target construction only. The final repaired recognizer is evaluated with ordinary decoding and does not require comparator access or test-time reranking.

## Result provenance

Files under `supplementary/` are the manuscript-level result artifacts and diagnostic summaries. The executable micro validation is a separate implementation check.

For readers auditing a result, the intended chain is:

`reported claim -> supplementary table -> paper protocol -> core implementation -> public executable sanity check`.

This makes the method and reporting assumptions inspectable without requiring redistribution of every large training artifact.
