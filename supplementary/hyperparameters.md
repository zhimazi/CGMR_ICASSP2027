# Hyperparameters and Protocol Details

## Paper-declared settings

| Component | Setting |
|---|---|
| Primary backbone | Whisper-small |
| Cross-backbone evaluation | OWSM v3.1-small; SeamlessM4T-v2-Large |
| Continual adaptation | LoRA |
| CL-MASR sequence | Base10 → Frisian → Interlingua; final stage evaluates 11 old languages |
| FLEURS sequence | old: English, German, Spanish, French; current: Swahili |
| Default beam size | 8 |
| Default length penalty | 0 |
| Reporting | mean ± sample standard deviation across five seeds |
| Manuscript seed IDs | 2027, 2028, 2029, 2030, 2031 |
| Seed aggregation | arithmetic mean; sample std with ddof=1 |
| CL-MASR metrics | old languages: mER; current language: WER |
| FLEURS metrics | WER for old and current languages |
| Repair construction | equally sized labeled construction pool per old language |
| N-best MWER control | same state-A samples and beam candidates as CGMR |
| ER storage | same old-language storage budget during adaptation |
| Repairability | disjoint held-out cohort |
| Comparator/reference use | target construction only |
| CGMR sampling | 1/2 language-balanced old + 1/2 current language |
| Projection strength | language-specific lambda solved by one-dimensional bisection |
| Inference | repaired recognizer only; no comparator and no test-time reranking |

The machine-readable version of this protocol is `../configs/paper_protocol.json`. Five-seed aggregation can be checked with `../scripts/aggregate_seed_results.py`.

## Implementation-level reference settings from archived paper-scale runs

| Component | Setting |
|---|---:|
| LoRA rank | 8 |
| LoRA insertion scope | 120 decoder projection modules |
| Trainable parameters | 1,916,928 |
| Adaptation optimizer | AdamW |
| Adaptation learning rate | 1e-4 |
| Adaptation batch size | 8 |
| CL-MASR adaptation updates | 490 |
| CL-MASR checkpoint grid | 0, 16, 32, 48, 64, 256, 457, 490 |
| Maximum generation length | 128 |
| Repair learning rate | 1.25e-5 |
| Repair updates | 64 |
| CL-MASR temporal rows | 128 per old language |
| CL-MASR rescoring rows | 64 per old language |
| CL-MASR construction rows | 48 per old language |
| CL-MASR dev rows | 32 per old language |
| CL-MASR final-test rows | 128 per old language |
| Current-language construction / dev / test | 59 / 128 / 128 |
| Expanded current test | 584 total |
| FLEURS comparator training | 1,536 balanced old-language updates |
| FLEURS current-language adaptation | 512 Swahili updates |
| FLEURS temporal grid | 16, 32, 64, 128, 256, 512 |
| FLEURS repair pool | 128 rows per old language |
| FLEURS current repair pool | 512 Swahili rows |
| FLEURS dev panel | 128 rows per locale |
| FLEURS final evaluation | official test split after 30 s Whisper-window filtering |

## Public reference defaults

The compact executable runner fixes additional engineering defaults so that the implementation path is deterministic and inspectable.

| Component | Setting |
|---|---:|
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| AdamW betas | (0.9, 0.999) |
| AdamW epsilon | 1e-8 |
| AdamW weight decay | 0.01 |
| Scheduler | constant |
| Warmup | 0 |
| Gradient clipping | 5.0 |
| Mixed precision | fp16 autocast |
| TF32 | disabled |
| Flash / memory-efficient SDP | disabled |
| Deterministic algorithms | enabled where supported |

These defaults describe the released reference implementation. They are recorded separately from manuscript-declared settings so that implementation choices are not confused with paper-level claims.

The compact public runner uses its own explicit configuration in `../configs/micro_cv15_seed2027.json`.

### Reproduction note

The manuscript-scale tables report five-seed aggregates. The public micro validation is a separate executable check and does not stand in for the paper-scale seed runs.

Some implementation-level values above are reconstructed from archived paper-scale configuration and result traces rather than a single frozen final-run manifest. They document the experimental protocol and should not be interpreted as a bitwise-reproduction guarantee across dataset/model revisions or hardware/software environments.

See `../REPRODUCIBILITY.md` for the artifact boundary, determinism policy, and five-seed aggregation contract.
