# CGMR: Comparator-Guided Minimum-Information Repair

Research code accompanying **“When Recognition Fails Before Competitive Hypotheses Disappear: Rethinking Catastrophic Forgetting in Continual Multilingual ASR.”**

CGMR studies catastrophic forgetting in continual multilingual ASR from a process perspective. Under finite-beam decoding, recognition can regress while a comparator-competitive hypothesis is still accessible in the beam. CGMR uses this ranking-accessible regime as a repair window: it minimally revises the posterior over accessible candidates using a pre-adaptation comparator, then amortizes the repaired targets back into a single recognizer. Inference uses ordinary beam search and does not require the comparator or test-time reranking.

## What this repository provides

This release contains four complementary pieces:

- **Core method implementation** for N/A/D state assignment, comparator-budgeted posterior projection, language-level bisection, and CGMR amortization.
- **Experimental protocol and hyperparameters** documented in `docs/HYPERPARAMETERS.md`.
- **Machine-readable paper results** for Tables 1–3 and Fig. 4 under `paper_results/`.
- **Executable validation example** on a compact Common Voice 15.0 setup for checking the implementation path end to end.

The repository is intended as a reference implementation and experimental record for the manuscript. Large training artifacts such as dataset audio, model checkpoints, feature caches, and machine-specific manifests are not redistributed.

## Method at a glance

For an old-language utterance, let the pre-adaptation comparator define a baseline edit count `b_i`. Let the adapted model produce a finite beam candidate set. We distinguish three states:

- **N**: the adapted 1-best remains comparator-competitive.
- **A**: the adapted 1-best is worse than the comparator, but a comparator-competitive candidate remains in the beam.
- **D**: no comparator-competitive candidate remains in the beam.

CGMR repairs state-A samples. For each old-language cohort, it finds the minimum-KL posterior revision satisfying a comparator-defined aggregate edit budget:

```text
q_i*(h) ∝ p_i(h) exp(-lambda_l * d_i(h))
```

The language-specific `lambda_l` is solved by one-dimensional bisection. Projected old-language targets are then combined with current-language teacher-forced CE, producing a single repaired recognizer.

## Paper results

The values reported in the submitted manuscript are available in machine-readable form:

```text
paper_results/
├── table1_temporal_diagnosis.csv
├── table2_retention_adaptation.csv
├── table3_ablation.csv
└── fig4_repairability_contrasts.csv
```

See `docs/PAPER_REPRODUCTION.md` for the mapping between manuscript claims, code, and released result artifacts.

## Repository layout

```text
CGMR_ICASSP2027/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── cgmr/
│   ├── __init__.py
│   └── projection.py
├── configs/
│   └── micro_cv15_seed2027.json
├── scripts/
│   ├── prepare_cv15_subset.py
│   └── run_er_cgmr_micro.py
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
    ├── HYPERPARAMETERS.md
    ├── PAPER_REPRODUCTION.md
    └── REPRODUCIBILITY.md
```

## Installation

The executable validation was checked with Python 3.12 and the following software stack:

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

Install a PyTorch build compatible with your CUDA environment, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

# Example for CUDA 12.8
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Run the CPU-only core tests from the repository root:

```bash
pytest -q
```

## Executable validation example

The compact example uses Common Voice 15.0 with:

```text
old languages: fy-NL, tr, ja
current language: ia
backbone: Whisper-small
seed: 2027
```

Expected data layout:

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

Generate the validation manifest:

```bash
python scripts/prepare_cv15_subset.py \
  --config configs/micro_cv15_seed2027.json \
  --data-root data/cv15 \
  --output experiment/manifests/micro_panel_seed2027.jsonl
```

Run the ER → CGMR pipeline:

```bash
python scripts/run_er_cgmr_micro.py \
  --config configs/micro_cv15_seed2027.json \
  --manifest experiment/manifests/micro_panel_seed2027.jsonl \
  --model-path openai/whisper-small \
  --output-dir experiment/output
```

The validation path covers comparator training, adaptation, beam candidate extraction, state-A selection, language-level projection, CGMR amortization, and endpoint evaluation.

## Paper-scale experimental protocol

The manuscript evaluates CGMR primarily with Whisper-small and additionally with OWSM v3.1-small and SeamlessM4T-v2-Large. The released protocol records the settings stated in the manuscript together with implementation-level settings available from archived experiment assets.

Key paper-level settings include:

```text
continual adaptation: LoRA
default beam size: 8
default length penalty: 0
reporting: mean ± standard deviation across five seeds
CL-MASR old metric: mER
CL-MASR current metric: WER
FLEURS metric: WER
CGMR old/current sampling: 1/2 + 1/2
inference: single repaired recognizer; no comparator or reranking
```

See `docs/HYPERPARAMETERS.md` for the full protocol table, including archived implementation settings such as optimizer, learning rates, update counts, LoRA configuration, and construction-pool sizes where supported by the available experiment records.

## Reproducibility scope

This repository releases the method code, evaluation protocol, reported paper metrics, and a runnable implementation check. Exact numerical reproduction of large multilingual ASR experiments can depend on dataset revisions, model revisions, manifests, checkpoints, and software/hardware environment. The repository therefore records available provenance separately from the compact executable example.

Third-party datasets and base-model weights remain subject to their original licenses and are not redistributed here.

## Citation

If you use this code, please cite the accompanying manuscript:

> Peihong Zhang, Mingzhuo Zhou, Shuyu Li, and Shengchen Li.  
> **When Recognition Fails Before Competitive Hypotheses Disappear: Rethinking Catastrophic Forgetting in Continual Multilingual ASR.**

Formal proceedings metadata can be added after publication.

## License

The current `LICENSE` permits inspection and research reproducibility but does not grant a permissive open-source license. Third-party datasets and models remain subject to their original terms.

## Contact

For implementation or reproducibility questions, please open a GitHub issue.
