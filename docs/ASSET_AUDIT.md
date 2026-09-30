# Historical asset audit

This document records a static audit of the research archives supplied during repository reconstruction. Its purpose is to distinguish **final-paper facts**, **historical run evidence**, and **unresolved reproduction details**.

## Audited archives

The audit covered three supplied archives:

- `ICASSP2027_continue_work(1)(6).zip`
- `seed2027_er_cgmr_validation_20260917.zip`
- `seed2027_pilot_deliverables(1).zip`

No training or decoding was executed for this audit. Files were inspected statically (manuscript sources, JSON result metadata, launch scripts, logs, and configs).

## Critical finding

The large historical paper/results archive corresponds to an **earlier three-seed lineage** with seed IDs `2027, 2028, 2029`. The submitted 5-page PDF reports **five-seed mean ± standard deviation** results and contains different headline values.

Therefore, the older archive is useful for recovering likely protocol details, but it **cannot be treated as the verified final five-seed experiment lineage** unless the missing final-run configs/logs establish continuity.

This is why the repository does not silently promote the historical settings below into a `configs/paper/*.yaml` file.

## Evidence classes

- **FINAL-PDF** — explicitly stated in the submitted 5-page PDF.
- **HISTORICAL-RESULT** — directly present in historical run/result metadata.
- **HISTORICAL-MANUSCRIPT** — declared in an earlier manuscript source, but no complete executable config is present.
- **MICRO-ONLY** — verified only for the independent Common Voice 15.0 micro validation.
- **MISSING** — not present in the supplied archives.

## Recovered historical CL-MASR protocol

| Item | Recovered value | Evidence | Final-paper status |
|---|---|---|---|
| Historical seeds | 2027, 2028, 2029 | HISTORICAL-RESULT | **not sufficient**: submitted PDF reports five seeds |
| Backbone | Whisper-small | HISTORICAL-MANUSCRIPT + FINAL-PDF | consistent |
| Adaptation type | LoRA | HISTORICAL-MANUSCRIPT + FINAL-PDF | consistent |
| LoRA rank | 8 | HISTORICAL-MANUSCRIPT | not independently verified for final five-seed runs |
| LoRA insertion scope | 120 decoder projections | HISTORICAL-MANUSCRIPT | exact module names missing |
| Trainable parameters | 1,916,928 | HISTORICAL-MANUSCRIPT | final-run count log missing |
| Current-language adaptation | 490 CE updates | HISTORICAL-MANUSCRIPT; final P paths contain `step0490` | plausible historical lineage, not final-five-seed verified |
| Batch size | 8 | HISTORICAL-MANUSCRIPT | final config missing |
| Optimizer | AdamW | HISTORICAL-MANUSCRIPT | final config missing |
| Adaptation LR | 1e-4 | HISTORICAL-MANUSCRIPT | final config missing |
| Historical checkpoint grid | 0, 16, 32, 48, 64, 256, 457, 490 | HISTORICAL-MANUSCRIPT | no compatible no-replay LoRA per-checkpoint cache in supplied archive |
| Beam | 8 | HISTORICAL-RESULT + FINAL-PDF | confirmed for the historical result files; default K=8 in final PDF |
| Historical max generation length | 128 | HISTORICAL-RESULT + HISTORICAL-MANUSCRIPT | final PDF does not state this |
| Length penalty | 0 | FINAL-PDF | confirmed |
| Normalization | Whisper normalization | HISTORICAL-MANUSCRIPT | exact final implementation/version missing |
| Language prompt | oracle locale | HISTORICAL-MANUSCRIPT | exact final implementation missing |
| Sequence scoring | teacher-forced continuation score including EOS | HISTORICAL-MANUSCRIPT | method is consistent with final paper definition; executable final implementation missing |
| Repair updates | 64 | HISTORICAL-MANUSCRIPT | final config missing |
| Repair LR | 1.25e-5 | HISTORICAL-MANUSCRIPT; several historical checkpoint/result names contain `lr1p25e5` | likely historical setting, not final-five-seed verified |
| Historical old-language temporal rows | 128 per old language | HISTORICAL-MANUSCRIPT | actual final manifest missing |
| Historical rescoring/beam rows | 64 per old language | HISTORICAL-MANUSCRIPT | actual final manifest missing |
| Historical construction rows | 48 per old language | HISTORICAL-MANUSCRIPT | actual final manifest missing |
| Historical dev rows | 32 per old language | HISTORICAL-MANUSCRIPT | actual final manifest missing |
| Historical final-test rows | 128 per old language | HISTORICAL-MANUSCRIPT; historical endpoint rows are consistent with 128/language | final manifest/hash missing |
| Historical current construction/dev/test | 59 / 128 / 128 | HISTORICAL-MANUSCRIPT | final manifest/hash missing |
| Historical expanded current test | 584 total | HISTORICAL-MANUSCRIPT | per-row expanded-current manifest not supplied |

Historical P/CGMR/MWER result metadata also records a common CL-MASR split label `second_official`, beam size 8, maximum generation length 128, and a P checkpoint path ending in `continuation_step0490_lora.pt`.

## Recovered historical FLEURS protocol

| Item | Recovered value | Evidence | Final-paper status |
|---|---|---|---|
| Old languages | English, German, Spanish, French | HISTORICAL-MANUSCRIPT + FINAL-PDF | consistent |
| Current language | Swahili | HISTORICAL-MANUSCRIPT + FINAL-PDF | consistent |
| Comparator training | 1,536 balanced old-language updates | HISTORICAL-MANUSCRIPT | final config missing |
| P adaptation | 512 Swahili updates | HISTORICAL-MANUSCRIPT | final config missing |
| Historical temporal grid | 16, 32, 64, 128, 256, 512 | HISTORICAL-MANUSCRIPT | final trajectory assets missing |
| Repair old-language pool | 128 rows per old language | HISTORICAL-MANUSCRIPT | final manifest missing |
| Repair current-language pool | 512 Swahili rows | HISTORICAL-MANUSCRIPT | final manifest missing |
| Dev panel | 128 rows per locale | HISTORICAL-MANUSCRIPT | final manifest missing |
| Final evaluation | complete official test split after 30 s Whisper-window filter | HISTORICAL-MANUSCRIPT | exact filtered IDs/hash missing |
| Launcher | `tmp/server_scripts/run_fleurs_seed.sh` in the archive | HISTORICAL-RESULT | references a missing project tree and a missing phase-2 script |

The historical FLEURS launcher points to a project root `/home/zph/continual_masr_icassp27`, a local Whisper-small model directory, `data/FLEURS/manifests`, and a missing `scripts/run_fleurs_seed_phase2.sh`. The launcher therefore documents intent but is not a self-contained reproduction entry point.

## What the supplied archives do not contain

The supplied archives do **not** contain a complete final paper-scale training project. In particular, they lack at least:

- the verified final five seed IDs and per-seed raw outputs corresponding to the submitted tables;
- complete final training configs / CLI arguments;
- the full `src/` training/evaluation project referenced by historical launchers;
- exact LoRA target module names, alpha/scaling, and dropout for the historical paper-scale runs;
- actual CL-MASR and FLEURS manifests with immutable hashes;
- the final no-replay LoRA temporal checkpoint rows/caches;
- FLEURS phase-2 executable script and per-utterance final endpoint outputs;
- OWSM v3.1-small and SeamlessM4T-v2-Large scripts/checkpoints/configs;
- exact baseline configs for LwF, ER, SVT, WFC, UGP, N-best MWER, and Table 3 ablations;
- checkpoint hashes for the submitted five-seed results.

## Assets that must not be substituted

Two supplied sources are valid for their own purposes but must not be used to fill final-paper gaps:

1. The **dense fine-tuning temporal control** has a detailed optimizer record, but it is a dense-decoder control rather than the LoRA trajectory studied as the main protocol.
2. The **Common Voice 15.0 micro validation** has a complete executable config, but it is an independent sanity check and does not reproduce the submitted CL-MASR/FLEURS tables.

## Practical recovery target

To turn the repository into a true exact-reproduction package, the most useful next files to recover from the original experiment machine are, in priority order:

1. final paper run configs or `args.json` / command logs;
2. a directory listing of the final five seed run folders;
3. CL-MASR and FLEURS manifest files and their hashes;
4. per-seed summaries used to build the submitted Tables 1–3;
5. temporal per-checkpoint state files for the no-replay LoRA runs;
6. FLEURS phase-2 script/config;
7. cross-backbone configs and result summaries;
8. environment lockfile / package freeze and model revision identifiers.

Until those are recovered, the public repository should remain explicit that it contains the core method, a micro executable check, submitted-paper numeric transcriptions, and a historical-protocol audit—not a complete exact reproduction.
