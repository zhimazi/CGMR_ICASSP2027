# CGMR: Comparator-Guided Minimum-Information Repair

Research code accompanying **“When Recognition Fails Before Competitive Hypotheses Disappear: Rethinking Catastrophic Forgetting in Continual Multilingual ASR.”**

CGMR studies catastrophic forgetting in continual multilingual ASR from a process perspective. Under finite-beam decoding, an old-language utterance can become worse at 1-best recognition while a baseline-competitive hypothesis is still present in the beam. CGMR uses this **ranking-accessible** regime as a repair window: it minimally revises the posterior over accessible candidates using a pre-adaptation comparator, then amortizes the repaired targets back into a single recognizer. Inference uses ordinary beam search and does not require the comparator or test-time reranking.

> **Release status.** This repository now provides (i) the core executable CGMR implementation and micro sanity check, and (ii) machine-readable transcriptions of the submitted paper's main numerical results. It is **not yet an exact end-to-end reproduction of the paper-scale CL-MASR/FLEURS experiments** because the complete verified final-run configs, manifests, and raw per-seed lineage have not all been recovered.

## Submitted-paper artifacts

The submitted PDF is treated as the source of truth for reported paper numbers.

- `paper_results/table1_temporal_diagnosis.csv` — submitted Table 1.
- `paper_results/table2_retention_adaptation.csv` — submitted Table 2.
- `paper_results/table3_ablation.csv` — submitted Table 3.
- `paper_results/fig4_repairability_contrasts.csv` — Fig. 4 state-A vs state-D annotated contrasts.
- `docs/HYPERPARAMETERS.md` — what the 5-page paper specifies, what it omits, and what must be released for exact reproduction.
- `docs/PAPER_REPRODUCTION.md` — claim-to-code/result map and current reproduction status.

**Important:** the Common Voice 15.0 micro experiment under `examples/` is an implementation sanity check, not a substitute for the paper-scale CL-MASR/FLEURS results.

## Method at a glance

For an old-language utterance, let the pre-adaptation comparator define a baseline edit count `b_i`. Let the adapted model produce a beam candidate set. We distinguish three states:

- **N**: the adapted 1-best is comparator-competitive.
- **A**: the adapted 1-best is worse than the comparator, but a comparator-competitive candidate is still accessible in the beam.
- **D**: no comparator-competitive candidate remains in the beam.

CGMR repairs only state-A samples. For each old-language cohort, it finds the minimum-KL posterior revision satisfying a comparator-defined aggregate edit budget. The projected candidate posterior has the form

```text
q_i*(h) ∝ p_i(h) exp(-lambda_l * d_i(h)),
```

where `lambda_l` is solved by one-dimensional bisection for each language. The projected old-language targets are then trained jointly with teacher-forced current-language CE, yielding one repaired recognizer.

## What is implemented in the minimal release

The current executable pipeline includes:

- Whisper-small with LoRA adaptation.
- Custom locale-token handling for `fy-NL` and `ia` following the CL-MASR-style setup used in the validation code.
- Deterministic data-panel construction from Common Voice 15.0.
- Experience replay (ER) adaptation.
- Beam-8 candidate extraction.
- Exact teacher-forced sequence log-probabilities including EOS.
- State-A sample selection using comparator error, adapted 1-best error, and beam oracle error.
- Language-level comparator edit budgets.
- Minimum-KL / Gibbs-tilt posterior projection.
- Per-language bisection for `lambda_l`.
- CGMR amortization with old/current loss balancing.
- Endpoint WER/CER evaluation and paired bootstrap summaries.

The minimal pipeline is intentionally small enough to serve as an executable implementation check. It should not be used as the source of the paper's main numerical tables.

## Repository layout

The minimum public repository should look like this:

```text
CGMR_ICASSP2027/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── CITATION.cff
├── configs/
│   └── micro_cv15_seed2027.json
├── scripts/
│   ├── prepare_cv15_subset.py
│   └── run_er_cgmr_micro.py
├── cgmr/
│   ├── __init__.py
│   └── projection.py
├── tests/
│   ├── test_state.py
│   └── test_projection.py
├── examples/
│   ├── seed2027_final_summary.json
│   └── seed2027_provenance.json
├── paper_results/
│   ├── README.md
│   ├── table1_temporal_diagnosis.csv
│   ├── table2_retention_adaptation.csv
│   ├── table3_ablation.csv
│   └── fig4_repairability_contrasts.csv
└── docs/
    ├── REPRODUCIBILITY.md
    ├── HYPERPARAMETERS.md
    └── PAPER_REPRODUCTION.md
```

Do **not** commit local audio, model weights, feature caches, absolute-path manifests, temporary LaTeX files, or large experiment directories.

## Installation

The validation environment used Python 3.12 with CUDA. A tested software stack was:

```text
torch 2.8.0
torchaudio 2.8.0
transformers 4.56.2
peft 0.17.1
accelerate 1.10.1
jiwer 4.0.0
av 15.1.0
pandas 2.3.2
scipy 1.16.2
safetensors 0.6.2
sentencepiece 0.2.1
```

Create an environment, install a PyTorch build compatible with your CUDA version, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

# Example for CUDA 12.8. Adjust for your system.
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

A CUDA GPU is required by the current end-to-end reference script. The projection unit tests are CPU-only.

Run the lightweight tests from the repository root:

```bash
pytest -q
```

## Data preparation

The minimal example uses four Common Voice 15.0 locales:

```text
old languages: fy-NL, tr, ja
current language: ia
```

Arrange each locale as follows:

```text
data/cv15/
├── fy-NL/
│   ├── train.tsv
│   ├── test.tsv
│   └── extracted/**/*.mp3
├── tr/
├── ja/
└── ia/
```

The repository does not redistribute Common Voice audio. Please obtain the dataset under its original terms.

Generate the deterministic micro manifest:

```bash
python scripts/prepare_cv15_subset.py \
  --config configs/micro_cv15_seed2027.json \
  --data-root data/cv15 \
  --output experiment/manifests/micro_panel_seed2027.jsonl
```

**Important:** generated manifests may contain machine-specific audio paths. Keep them out of Git, or regenerate them locally.

## Run the minimal ER → CGMR experiment

You can use a local Whisper-small directory or a Hugging Face model identifier supported by `from_pretrained`.

```bash
python scripts/run_er_cgmr_micro.py \
  --config configs/micro_cv15_seed2027.json \
  --manifest experiment/manifests/micro_panel_seed2027.jsonl \
  --model-path openai/whisper-small \
  --output-dir experiment/output
```

The script performs:

```text
build/load Whisper-small + LoRA
        ↓
train comparator C on old-language samples
        ↓
evaluate C and cache comparator repair errors
        ↓
experience replay adaptation
        ↓
beam-8 decoding on the old-language repair pool
        ↓
select state-A samples
        ↓
solve language-level CGMR projection by bisection
        ↓
amortize projected targets + current-language CE
        ↓
evaluate ER and ER→CGMR endpoints
```

## Main output files

The run directory contains, among others:

```text
experiment/output/
├── C_endpoint.json
├── ER_endpoint.json
├── ER_to_CGMR_endpoint.json
├── ER_to_CGMR_targets.json
├── final_summary.json
├── paired_uncertainty.json
├── predictions_seed2027.csv
├── training_history.json
├── run.log
└── adapters/
```

`ER_to_CGMR_targets.json` is useful for inspecting the actual state-A cohort, beam candidates, language-level `lambda`, comparator budget, and projected candidate posterior.

## Minimal validation configuration

The bundled sanity-check configuration uses approximately the following setup:

```text
seed: 2027
backbone: Whisper-small
LoRA rank / alpha / dropout: 8 / 16 / 0
beam size: 8
max new tokens: 64
old languages: fy-NL, tr, ja
current language: ia
comparator steps: 128
ER steps: 64
CGMR steps: 16
adaptation LR: 1e-4
repair LR: 1.25e-5
```

This configuration is deliberately smaller than the manuscript experiments.

## Relation to the paper experiments

The paper evaluates the forgetting process and CGMR on CL-MASR and FLEURS, primarily with Whisper-small, and also reports cross-backbone experiments with OWSM v3.1-small and SeamlessM4T-v2-Large. The manuscript's default beam size is 8 with zero length penalty, and reported results are mean ± standard deviation across five seeds.

The exact values printed in the submitted paper are archived under `paper_results/`. See `docs/HYPERPARAMETERS.md` for the distinction between settings stated in the paper and settings that must still be recovered from the final paper-scale run lineage.

The current minimal public implementation should therefore be interpreted as **method code plus an executable sanity check**, not as a claim that every paper table is reproducible from this release.

A full reproduction release should eventually add:

```text
configs/clmasr_whisper_small.yaml
configs/fleurs_whisper_small.yaml
configs/fleurs_owsm_small.yaml
configs/fleurs_seamlessm4t.yaml
scripts/diagnose_forgetting.py
scripts/run_cgmr.py
scripts/evaluate.py
scripts/reproduce_table1.sh
scripts/reproduce_table2.sh
scripts/reproduce_table3.sh
```

Only add these entries to the README as runnable commands after the corresponding code, manifests, checkpoints/config lineage, and expected outputs have been verified.

## Reproducibility notes

The reference implementation uses deterministic seeds and performs a small repeated-decoding consistency check before training. The exact numerical result can still depend on GPU, CUDA, PyTorch, Transformers, dataset release, and generated manifest.

For a scientific release, record at least:

- Git commit.
- Python/package versions.
- GPU/CUDA versions.
- Dataset release and manifest hash.
- Model identifier/revision.
- Random seed.
- Beam size and generation limit.
- Tokenization and normalization rules.
- LoRA target modules and trainable parameter count.
- Checkpoint hashes for results reported in the paper.

## Known limitations of the minimal release

- It is a single-seed micro experiment, not the paper's multi-seed benchmark reproduction.
- It uses a public Common Voice 15.0 subset rather than the manuscript's original CL-MASR experiment cohort.
- It contains ER → CGMR validation, but does not yet expose the complete paper-scale temporal `tau_rec` / `tau_loss` analysis pipeline.
- It does not yet provide verified FLEURS, OWSM, or SeamlessM4T training/evaluation entry points.
- It should not be used to regenerate the paper's main tables until the original experiment lineage is restored and checked.

## Data and model licenses

This repository contains code only unless explicitly stated otherwise. Common Voice, Whisper, OWSM, SeamlessM4T, and other third-party assets remain subject to their original licenses and terms. No third-party dataset or base-model weights should be redistributed through this repository without checking those terms.

## Citation

If you use this code, please cite the accompanying manuscript:

> Peihong Zhang, Mingzhuo Zhou, Shuyu Li, and Shengchen Li.  
> **When Recognition Fails Before Competitive Hypotheses Disappear: Rethinking Catastrophic Forgetting in Continual Multilingual ASR.**

Formal conference/BibTeX metadata should be added here once the bibliographic record is finalized.

## License

The bundled `LICENSE` currently states **all rights reserved**; it does not grant an open-source license. Replace it with the license chosen by the repository owners before advertising the project as open source. Third-party datasets and models remain subject to their original terms.

## Contact

For implementation or reproducibility questions, please open a GitHub issue.
