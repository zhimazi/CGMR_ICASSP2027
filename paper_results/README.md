# Submitted-paper numerical results

This directory contains **direct transcriptions of the numerical results in the submitted ICASSP 2027 PDF**.

These CSV files are intended to make the paper's reported numbers machine-readable and easy to audit. They are **not** presented as freshly regenerated outputs from the current minimal code release.

- `table1_temporal_diagnosis.csv`: Table 1, temporal diagnosis of forgetting.
- `table2_retention_adaptation.csv`: Table 2, retention–adaptation results.
- `table3_ablation.csv`: Table 3, CGMR ablations on CL-MASR with Whisper-small.
- `fig4_repairability_contrasts.csv`: the state-A versus state-D complete-repair-rate differences annotated in Fig. 4.

The independent micro validation under `examples/` is a separate implementation sanity check and should not be used as a substitute for these paper-scale results.

For the status of executable reproduction and missing final-run settings, see:

- `docs/PAPER_REPRODUCTION.md`
- `docs/HYPERPARAMETERS.md`
