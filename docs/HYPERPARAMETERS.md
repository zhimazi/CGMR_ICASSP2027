# Hyperparameters and experimental protocol

This file separates settings that are **explicitly stated in the submitted ICASSP 2027 paper** from settings that are required for exact reproduction but are **not stated in the 5-page paper**.

The submitted PDF is the source of truth for the paper-level claims. The bundled micro-validation config (`configs/micro_cv15_seed2027.json`) is an independent executable sanity check and must **not** be treated as the paper-scale configuration. Likewise, values from earlier manuscript drafts or exploratory runs should not be promoted here unless they are verified against the exact final experiment lineage that produced the submitted tables.

## Explicitly stated in the submitted paper

| Item | Paper-specified setting |
|---|---|
| Primary backbone | Whisper-small |
| Cross-backbone evaluation | OWSM v3.1-small; SeamlessM4T-v2-Large |
| Continual adaptation parameterization | LoRA is used |
| CL-MASR sequence | Base10, then Frisian, then Interlingua; final stage evaluated on 11 old languages |
| FLEURS sequence | old: English, German, Spanish, French; current: Swahili |
| Default decoding beam | beam size K = 8 |
| Default length penalty | 0 |
| Reporting | mean ± standard deviation across five seeds |
| CL-MASR metrics | old languages: mER; current language: WER |
| FLEURS metrics | WER for both old and current languages |
| Repair construction data | equally sized labeled construction pool per old language |
| Standalone repair controls | share the same post-adaptation checkpoint P, construction pool, current-language data, trainable parameters, and update budget |
| N-best MWER control | uses the same state-A samples and beam candidates as CGMR |
| ER data budget | same old-language storage budget during adaptation |
| Repairability evaluation | disjoint held-out cohort |
| Comparator/reference access | restricted to target construction |
| CGMR old/current sampling mixture | 1/2 language-balanced old samples + 1/2 current-language samples |
| Inference | repaired recognizer only; ordinary beam search; no comparator and no test-time reranking |
| Projection strength | language-specific lambda is solved by one-dimensional bisection; it is not tuned as a temperature |

## Required for exact reproduction but not stated in the submitted 5-page PDF

The following values should be released from the **verified final paper runs** before this repository claims exact end-to-end reproduction:

- Exact seed IDs used for the five reported runs.
- Optimizer(s).
- Adaptation learning rate and any scheduler / warmup / weight decay.
- Repair learning rate and any scheduler / warmup / weight decay.
- Batch size, effective batch size, and gradient accumulation.
- Number of continual-adaptation updates / epochs.
- Number of repair updates / epochs.
- LoRA rank, alpha/scaling, dropout, and exact target module names for each backbone.
- Exact trainable parameter counts.
- Maximum generation length / maximum new tokens.
- Exact checkpoint grid used for temporal first-passage analysis.
- Construction-pool sizes, replay-buffer sizes, replay/current sampling ratios, and held-out cohort sizes.
- Exact dataset releases, manifests, filtering rules, split hashes, and preprocessing scripts.
- Text normalization, tokenization, locale/language prompting, and EOS scoring details as implemented.
- Base model identifiers and immutable revisions / hashes.
- Mixed-precision settings and numerical-determinism settings.
- Software versions (Python, PyTorch, Transformers, PEFT, CUDA, etc.).
- Hardware used for the reported paper runs.
- Baseline-specific settings for LwF, ER, SVT, WFC, UGP, N-best MWER, and all Table 3 ablations.
- Checkpoint and result-artifact hashes for the numbers reported in the paper.

## Current repository status

The repository currently contains:

- a directly executable **micro validation** with its own fully specified config;
- the core CGMR state definition and posterior projection implementation;
- unit tests for the core projection;
- direct transcriptions of the submitted paper's Tables 1–3 and Fig. 4 contrasts under `paper_results/`.

It does **not** yet contain a verified paper-scale config carrying every missing value listed above. Until those exact final-run settings are recovered, the repository should continue to describe itself as a core implementation plus paper-result archive, rather than as a complete end-to-end reproduction package.

## Template for the verified paper config

When the final experiment lineage is recovered, add immutable configs such as:

```yaml
experiment:
  seed_ids: []
  dataset_release:
  manifest_hash:
  backbone:
  backbone_revision:

adaptation:
  optimizer:
  learning_rate:
  scheduler:
  warmup:
  weight_decay:
  batch_size:
  gradient_accumulation:
  updates:

lora:
  rank:
  alpha:
  dropout:
  target_modules: []
  trainable_parameters:

decoding:
  beam_size: 8
  length_penalty: 0
  max_generation_length:

repair:
  optimizer:
  learning_rate:
  batch_size:
  gradient_accumulation:
  updates:
  old_current_sampling: [0.5, 0.5]
  construction_rows_per_old_language:

temporal_analysis:
  checkpoint_grid: []
  cohort_rows_per_language:

environment:
  python:
  pytorch:
  transformers:
  peft:
  cuda:
  gpu:
```

Do not fill unknown fields by inference. Only commit values traceable to the final runs that produced the submitted paper results.
