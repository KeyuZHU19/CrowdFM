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
- [x] Local unit-test smoke validation.
- [x] Official-checkpoint baseline evaluation over all 16 available dataset directories.
- [x] One-seed real-data CbR smoke audit with `B=99`.
- [x] YAML calibration configuration and incremental JSON result schema.
- [ ] Add worker nuisance-support and posterior-quality diagnostics.
- [ ] Repeated cross-fitting and result aggregation.
- [ ] GPU/CPU profiling.

## Phase 1 — Null calibration

Use eight representative synthetic configurations spanning:

- workers: 20, 50, 100;
- items: 200, 500, 1000;
- classes: 2, 5, 10;
- labels per item: 3, 5, 10;
- balanced and imbalanced class priors.

Run 300 independent worlds per configuration. Report empirical rejection at alpha in `{0.01, 0.05, 0.10, 0.20}`. Use `B=199` during development and `B=999` for final calibration figures.

### Calibration decomposition

Before running the full fitted pipeline, separate statistical calibration from nuisance-estimation error using four variants on the same synthetic worlds and the same cross-fit split:

1. **Oracle posterior + oracle confusion**: `q_k` is one-hot ground truth and `P_i` is the generating confusion matrix. This isolates the residual statistic and Monte Carlo test.
2. **CrowdFM posterior + oracle confusion**: isolates posterior error.
3. **Oracle posterior + estimated confusion**: isolates confusion-estimation error.
4. **CrowdFM posterior + estimated confusion**: evaluates the complete deployment pipeline.

Do not interpret real-data rejection rates until variant 1 is calibrated and the gap between variants 1--4 is understood.

First run the quick integration check:

```bash
python run_cbr_calibration.py config=config/cbr_calibration_quick.yaml
```

Then run the development calibration sweep:

```bash
python run_cbr_calibration.py config=config/cbr_calibration.yaml
```

The quick configuration runs two small settings, two worlds per setting, all four variants, and `B=19`. The development configuration runs four settings, 20 worlds per setting, all four variants, and `B=99`. Results are written incrementally to `log/cbr_calibration_quick.json` and `log/cbr_calibration_smoke.json`.

- [x] Implement fixed-degree Dawid--Skene synthetic null generator with known truth and worker confusion matrices.
- [x] Implement the four-way oracle/fitted calibration decomposition.
- [x] Add quick/development configurations, incremental runner, summary statistics, and synthetic unit tests.
- [ ] Run quick integration calibration.
- [ ] Run smoke calibration, 20 worlds/configuration.
- [ ] Inspect rejection rates and posterior/confusion errors for all four variants.
- [ ] Run final calibration, 300 worlds/configuration.
- [ ] Produce calibration plot and p-value histogram.

Decision rule after the smoke run:

- Variant 1 inflated: debug residual/bootstrap implementation before any model changes.
- Variant 2 inflated relative to 1: improve or recalibrate the CrowdFM item posterior/context split.
- Variant 3 inflated relative to 1: replace the provisional expected-count confusion estimator with hierarchical shrinkage or a learned head.
- Only variant 4 inflated: study interaction between posterior and confusion errors and repeated cross-fitting.

## Phase 2 — OOD power

Four primary alternatives, five severity levels, 100 worlds per point:

1. Class-conditioned confusion outside the pretraining range.
2. Conditional worker coalition dependence.
3. Worker temporal/block drift.
4. Item-type-dependent worker expertise.

Report AUROC, AUPRC, TPR at 5% FPR, and power at alpha=0.05. Include an observationally equivalent coalition as a documented detectability-limit negative control.

- [ ] Implement OOD injectors.
- [ ] Run conditional-coalition dependence first as the initial power sanity check.
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

- [ ] Audit all available datasets with the official checkpoint over five cross-fit seeds.
- [ ] Report rejection frequency across splits rather than treating one split as ground truth.
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
- CrowdFM entropy/margin confidence;
- hierarchical/global-confusion shrinkage versus a uniform Dirichlet prior;
- worker support thresholds for nuisance confusion estimation.

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

## Execution log

### 2026-07-16 PT — Initial local validation

- Command: `pytest -q`
- Result before the seed-parser regression test was added: `5 passed in 1.57s`.
- Command: `python evaluate.py checkpoint_path=checkpoint.pt output_path=log/crowdfm_baseline.json`
- Evaluated all 16 available dataset directories successfully.
- Mean task accuracy: `0.8229134873`.
- Mean reported per-dataset runtime: `0.0849569976` seconds.
- Baseline output: `log/crowdfm_baseline.json`.
- `evaluate_cbr.py` exposed a CLI parsing issue because dlwheel preserved `seeds=[42]` as a string. The entrypoint now normalizes scalar, sequence, JSON-string, and comma-separated seed specifications.

### 2026-07-16 PT — One-seed real-data CbR smoke audit

- Command: `python evaluate_cbr.py checkpoint_path=checkpoint.pt seeds=42 cbr.num_bootstrap=99 cbr.alpha=0.05 output_path=log/cbr_smoke.json`.
- All 16 datasets completed and had nonzero audit edges and supported worker pairs.
- Rejected 14/16 datasets at `alpha=0.05` (`87.5%`). Only RTE (`p=0.23`) and SP (`p=0.90`) were not rejected.
- Twelve datasets attained the minimum possible p-value `1/(B+1)=0.01`; Bird had `p=0.04` and PosSent had `p=0.02`.
- This is a diagnostic of the current fitted predictive null, not evidence that 14 real datasets are truly OOD. The provisional null combines CrowdFM posteriors with a stationary conditionally independent per-worker confusion model estimated by posterior expected counts; the high rejection rate indicates that this null and/or its nuisance estimates are too restrictive for most real datasets.
- Raw spectral statistics are not directly comparable across datasets; instance-conditional bootstrap p-values are the relevant quantities.

### 2026-07-16 PT — Four-way calibration implementation

- Added `src/cfm/audit/synthetic.py` with deterministic fixed-degree Dawid--Skene worlds and known confusion matrices.
- Added `src/cfm/audit/calibration.py`, `run_cbr_calibration.py`, `config/cbr_calibration_quick.yaml`, and `config/cbr_calibration.yaml`.
- All four variants reuse the same world, cross-fit split, and original CrowdFM node features.
- Results are checkpointed after every completed world and include posterior accuracy, confusion MAE, p-values, rejection decisions, runtime, and Git commit.
- Added two synthetic tests; isolated local execution of the new tests returned `2 passed`.
- Next local validation: pull the branch, run the full suite (expected `8 passed`), then launch the quick calibration before the 20-world sweep.
