# Reproducibility notes for the minimal public release

This repository package is an executable **method sanity check**, not a complete
paper-scale reproduction release.

## Source of the bundled implementation

The two executable scripts and the JSON config were extracted from the validated
seed-2027 ER -> CGMR micro-run deliverable supplied by the authors. The example
summary and provenance files are copied from the same deliverable.

The standalone `cgmr/projection.py` module and its unit tests were factored out
for readability and to make the paper's N/A/D state definition and language-level
posterior projection testable without a GPU. The end-to-end micro script keeps its
validated integrated implementation unchanged.

## Validated micro-run environment

The bundled provenance records the following environment for the source run:

- Python 3.12.14
- PyTorch 2.8.0+cu128
- Transformers 4.56.2
- PEFT 0.17.1
- NVIDIA GeForce RTX 4060 (8 GiB class)
- Seed 2027

See `examples/seed2027_provenance.json` for the complete source record.

## Important scope boundary

The supplied config describes an independent Common Voice 15.0 plausibility
check with old languages `fy-NL`, `tr`, `ja` and current language `ia`.
It is not the paper's full CL-MASR/FLEURS benchmark reproduction and should not
be cited as regenerating the manuscript tables.

A paper-scale release still needs the verified temporal forgetting pipeline
(`tau_rec`, `tau_loss`, ranking-first statistics), the exact CL-MASR and FLEURS
experiment lineage, multi-seed orchestration, and verified OWSM / SeamlessM4T
entry points.
