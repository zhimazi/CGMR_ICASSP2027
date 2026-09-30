# Experimental protocol and hyperparameters

This document collects the experimental settings associated with the submitted manuscript and the archived experiment assets accompanying the project.

The first table contains settings explicitly stated in the submitted paper. The second table records implementation-level values found in archived experiment material. Where an archived value is not independently confirmed by the submitted five-seed results, it is labeled accordingly rather than inferred.

## Manuscript-level protocol

| Component | Setting |
|---|---|
| Primary backbone | Whisper-small |
| Cross-backbone evaluation | OWSM v3.1-small; SeamlessM4T-v2-Large |
| Continual adaptation | LoRA |
| CL-MASR sequence | Base10 → Frisian → Interlingua; final stage evaluates 11 old languages |
| FLEURS sequence | old: English, German, Spanish, French; current: Swahili |
| Default beam size | 8 |
| Default length penalty | 0 |
| Reporting | mean ± standard deviation across five seeds |
| CL-MASR metrics | old languages: mER; current language: WER |
| FLEURS metrics | WER for old and current languages |
| Repair construction | equally sized labeled construction pool per old language |
| Standalone repair controls | same post-adaptation checkpoint P, construction pool, current-language data, trainable parameters, and update budget |
| N-best MWER control | same state-A samples and beam candidates as CGMR |
| ER storage | same old-language storage budget during adaptation |
| Repairability evaluation | disjoint held-out cohort |
| Comparator/reference use | target construction only |
| CGMR sampling | 1/2 language-balanced old samples + 1/2 current-language samples |
| Projection strength | language-specific lambda solved by one-dimensional bisection |
| Inference | repaired recognizer only; no comparator and no test-time reranking |

## Archived implementation settings

These values are directly recoverable from the supplied historical manuscript/config/result assets and are useful for reconstructing the paper-scale protocol.

| Component | Archived setting | Evidence status |
|---|---:|---|
| LoRA rank | 8 | archived manuscript |
| LoRA insertion scope | 120 decoder projection modules | archived manuscript |
| Trainable parameters | 1,916,928 | archived manuscript |
| Adaptation optimizer | AdamW | archived manuscript |
| Adaptation learning rate | 1e-4 | archived manuscript |
| Adaptation batch size | 8 | archived manuscript |
| CL-MASR adaptation updates | 490 | archived manuscript + checkpoint naming |
| CL-MASR checkpoint grid | 0, 16, 32, 48, 64, 256, 457, 490 | archived manuscript |
| Maximum generation length | 128 | archived result metadata |
| Repair learning rate | 1.25e-5 | archived manuscript + checkpoint/result naming |
| Repair updates | 64 | archived manuscript |
| CL-MASR temporal rows | 128 per old language | archived manuscript |
| CL-MASR rescoring rows | 64 per old language | archived manuscript |
| CL-MASR construction rows | 48 per old language | archived manuscript |
| CL-MASR dev rows | 32 per old language | archived manuscript |
| CL-MASR final-test rows | 128 per old language | archived manuscript / endpoint shape |
| Current-language construction/dev/test | 59 / 128 / 128 | archived manuscript |
| Expanded current test | 584 total | archived manuscript |
| FLEURS comparator training | 1,536 balanced old-language updates | archived manuscript |
| FLEURS current-language adaptation | 512 Swahili updates | archived manuscript |
| FLEURS temporal grid | 16, 32, 64, 128, 256, 512 | archived manuscript |
| FLEURS repair pool | 128 rows per old language | archived manuscript |
| FLEURS current repair pool | 512 Swahili rows | archived manuscript |
| FLEURS dev panel | 128 rows per locale | archived manuscript |
| FLEURS final evaluation | official test split after 30 s Whisper-window filtering | archived manuscript |

## Interpretation of archived settings

The archived paper-scale material includes an earlier three-seed lineage (2027–2029), whereas the submitted manuscript reports five-seed aggregates. For that reason, settings in the archived table are provided as implementation-level protocol evidence; they should not be interpreted as proof that every final five-seed run used an identical hidden environment or artifact revision.

The settings that are directly stated in the manuscript—such as backbone family, LoRA adaptation, beam size 8, zero length penalty, five-seed reporting, matched repair controls, and 1/2 old + 1/2 current CGMR sampling—are the strongest paper-level constraints.

## Compact executable example

The runnable Common Voice 15.0 validation uses its own fully specified config at:

`configs/micro_cv15_seed2027.json`

Its main settings are:

```text
seed: 2027
backbone: Whisper-small
LoRA rank / alpha / dropout: 8 / 16 / 0
beam size: 8
max new tokens: 64
comparator steps: 128
ER steps: 64
CGMR steps: 16
adaptation LR: 1e-4
repair LR: 1.25e-5
effective batch size: 8
```

This compact configuration is intended for implementation validation rather than paper-scale benchmarking.

## Provenance fields not fixed by the released artifacts

For exact reruns, the following details may still depend on the original final-run environment and data snapshot:

- exact five seed IDs;
- immutable base-model revisions/hashes;
- final dataset manifest hashes and filtering snapshots;
- exact LoRA target-module names / alpha / dropout for every paper-scale backbone;
- scheduler, warmup, weight decay, and mixed-precision details when not recorded in archived metadata;
- full baseline-specific hyperparameters for all comparison methods;
- final per-seed checkpoint hashes and environment captures.

These fields are not guessed in the repository. The available settings above are recorded at the strongest evidence level supported by the archived assets.
