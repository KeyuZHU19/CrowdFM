# AAAI Experiment Plan for CbR

This file is the live execution plan. Update status, commands, seeds, failures, and result paths as experiments are run.

## Paper claims

| ID | Claim | Required evidence |
|---|---|---|
| C1 | The conditional audit controls false rejection under the fitted null. | Nominal-vs-empirical rejection curves and p-value histograms. |
| C2 | CbR detects structured noise-model misspecification. | Power/AUROC curves over severity, density, and coalition size. |
| C3 | Residual routing improves selective aggregation risk. | Risk-coverage curves, AURC, and oracle-router gap. |
| C4 | Cross-fitting and item conditioning are necessary. | Leakage and global-prior ablations. |
| C5 | The detector remains useful on realistic sparse assignment graphs. | Real-mask semi-synthetic experiments. |

## Phase 0 — Infrastructure

- [x] Cross-fit split implementation.
- [x] Posterior-expected confusion estimator.
- [x] Item-conditioned predicted disagreement.
- [x] Signed standardized residual and two-sided spectral statistic.
- [x] Fixed-nuisance conditional Monte Carlo test.
- [x] CrowdFM evaluation entrypoint.
- [ ] Repeated cross-fitting and result aggregation.
- [ ] YAML experiment configuration and result schema.
- [ ] GPU/CPU profiling.

## Phase 1 — Null calibration

Use eight representative synthetic configurations spanning:

- workers: 20, 50, 100;
- items: 200, 500, 1000;
- classes: 2, 5, 10;
- labels per item: 3, 5, 10;
- balanced and imbalanced class priors.

Run 300 independent worlds per configuration. Report empirical rejection at alpha in `{0.01, 0.05, 0.10, 0.20}`. Use `B=199` during development and `B=999` for final calibration figures.

- [ ] Implement in-prior synthetic generator.
- [ ] Run smoke calibration, 20 worlds/configuration.
- [ ] Run final calibration, 300 worlds/configuration.
- [ ] Produce calibration plot and p-value histogram.

## Phase 2 — OOD power

Four primary alternatives, five severity levels, 100 worlds per point:

1. Class-conditioned confusion outside the pretraining range.
2. Conditional worker coalition dependence.
3. Worker temporal/block drift.
4. Item-type-dependent worker expertise.

Report AUROC, AUPRC, TPR at 5% FPR, and power at alpha=0.05. Include an observationally equivalent coalition as a documented detectability-limit negative control.

- [ ] Implement OOD injectors.
- [ ] Run severity sweeps.
- [ ] Run density and coalition-size sweeps.
- [ ] Produce power curves.

## Phase 3 — Real-mask semi-synthetic

Preserve the worker-item masks of six representative CrowdFM datasets. Initial targets:

- LabelMe
- RTE
- Trec
- Dog
- Bird
- ZC_all

For each mask, run four OOD families, three severity levels, and 50 worlds. Verify exact available dataset directory names before launching.

- [ ] Implement mask extraction and relabeling.
- [ ] Run 6 x 4 x 3 x 50 worlds.
- [ ] Compare item-conditioned CbR against global-prior residuals.

## Phase 4 — Real benchmarks

Run all CrowdFM datasets with five cross-fit seeds. Main baselines:

- Majority vote
- Dawid-Skene
- GLAD
- MACE
- DGN
- CrowdFM
- CrowdFM + CbR selective routing

Report all datasets in the appendix and eight representative datasets in the main paper. Real data have no definitive OOD oracle; report aggregation accuracy, audit p-value, rejection frequency, runtime, and case studies without claiming real-data OOD AUROC.

- [ ] Audit all available datasets with the official checkpoint.
- [ ] Integrate classical baselines.
- [ ] Implement pre-specified fallback routing.
- [ ] Produce real-benchmark table.

## Phase 5 — Selective risk and ablations

Required ablations:

- no cross-fitting;
- global class prior instead of item posterior;
- largest positive eigenvalue only;
- Frobenius norm;
- maximum entry statistic;
- global synthetic threshold;
- CrowdFM entropy/margin confidence.

Report risk-coverage curves, AURC, accuracy at fixed coverage, and router regret relative to an oracle router.

- [ ] Implement routing metrics.
- [ ] Run core ablations.
- [ ] Run bootstrap count sensitivity `B in {99,199,499,999}`.

## Seeds and reproducibility

Training seeds: `42, 43, 44`.
Cross-fit/evaluation seeds: `42, 43, 44, 45, 46`.
Every output JSON must record the Git commit, full configuration, dataset, seed, runtime, and result path.

## Compute allocation

Use independent jobs rather than distributed training:

- 4090-1/2/3: model or head seeds;
- 4090-4: real-data evaluation and development;
- V100S-1: null calibration;
- V100S-2: OOD power;
- V100S-3: real-mask semi-synthetic.

One 24 GB GPU is sufficient for an individual job. Residual construction and bootstrap can run on CPU or GPU; benchmark both before the final sweep.
