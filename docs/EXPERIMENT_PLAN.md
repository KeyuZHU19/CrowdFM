# Experiment Plan: Residual-Audited Crowd Foundation Models

This is the live plan for the revised masked-annotation predictive audit.  The confusion-based experiments are historical diagnostics and must not be continued as the primary method.

## Paper claims

| ID | Claim | Required evidence |
|---|---|---|
| C1 | A crowd foundation model can learn a transferable conditional distribution for held-out worker responses. | Masked-annotation NLL, Brier score, and calibration on unseen worlds and real matrices. |
| C2 | Conditional Monte Carlo controls false rejection when the learned response law is calibrated. | Nominal-vs-empirical rejection curves and p-value histograms. |
| C3 | Marginal and spectral residuals detect complementary deployment shifts. | Power curves by shift family and severity; component ablations. |
| C4 | Cross-fitting is necessary and worker residual structure localizes coherent failures. | Leakage ablation and coalition-localization AUC. |
| C5 | Audit-based defer improves selective aggregation rather than merely detecting synthetic OOD labels. | Risk--coverage, AURC, accepted-set accuracy, and oracle-router gap. |

## Phase 0 — Implementation and local validation

- [x] Add `PredictiveCFM` with a K-invariant masked-annotation response head.
- [x] Add leakage-free context/audit edge splitting.
- [x] Add marginal categorical log-score residual.
- [x] Add signed worker-pair residual covariance and two-sided operator norm.
- [x] Add exact conditional Monte Carlo component tests and Bonferroni joint p-value.
- [x] Add masked-response plus task-truth training objective.
- [x] Add training and evaluation entrypoints.
- [x] Preserve the old confusion implementation as a reproducible negative control.
- [ ] Pull the branch and run the complete unit-test suite.
- [ ] Run a one-batch training smoke test and verify that response-head gradients and checkpoint markers are correct.
- [ ] Add checkpoint-resume support to `train_predictive.py` if long runs require it.

Commands:

```bash
git pull --ff-only origin agent/cbr-audit-core
pytest -q
python train_predictive.py config=config/predictive_train.yaml epochs=2 batch_size=2 output_dir=log/predictive_smoke
```

## Phase 1 — Response model validation

The audit is meaningful only if `r_ik` is a useful held-out response predictor.  Before any OOD experiment, compare:

1. option-frequency baseline;
2. worker empirical-frequency baseline;
3. Dawid--Skene posterior predictive;
4. 3PL/GLAD-style low-dimensional predictor;
5. frozen CrowdFM backbone plus trained response head;
6. jointly fine-tuned PredictiveCFM.

Metrics:

- annotation NLL;
- Brier score;
- top-label accuracy;
- expected calibration error;
- reliability diagrams stratified by `K`, worker support, and task degree;
- worker cold-start and low-support performance.

Training regimes:

- synthetic-only masked-response training;
- synthetic task-truth plus masked-response multi-task training;
- synthetic pretraining plus self-supervised real-matrix response training;
- frozen backbone versus joint fine-tuning.

Decision rule: do not interpret audit p-values until the response predictor outperforms unconditional and worker-frequency baselines and has acceptable in-prior calibration.

## Phase 2 — Null calibration

Use unseen worlds sampled from the same declared pretraining distribution.  Evaluate at `alpha in {0.01, 0.05, 0.10}` over strata of:

- `M in {20, 50, 100}`;
- `N in {200, 500, 1000}`;
- `K in {2, 5, 10, 20}`;
- labels per task;
- class imbalance;
- worker support and task difficulty.

Report:

- marginal rejection;
- dependence rejection;
- Bonferroni joint rejection;
- p-value histograms;
- calibration error and confidence intervals.

Use `B=199` for development and `B=999` for final figures.  A badly calibrated learned response law is a model failure; Monte Carlo is not expected to repair it.

Required controls:

- oracle response probabilities from the generator;
- learned response probabilities;
- labels exposed to the model as a deliberate leakage control;
- shuffled query-worker identities;
- untrained response head, which must fail or be refused by the pipeline.

## Phase 3 — Structured deployment shift

Hold out entire mechanisms, not merely parameter values, from response-head training:

1. class-conditioned error directions;
2. stronger task-difficulty dependence;
3. worker--worker coalition dependence;
4. temporal worker drift;
5. non-random worker assignment;
6. Sybil workers and targeted label attacks;
7. class or option semantics absent from training.

For each family, run severity, density, worker-support, and coalition-size sweeps.  Report:

- AUROC/AUPRC;
- power at `alpha=0.05`;
- marginal versus dependence component power;
- leading-eigenvector worker localization;
- annotation-prediction degradation;
- aggregation-accuracy degradation.

Include an observationally equivalent alternative as a negative control.  The paper must state that such a process is information-theoretically invisible to the checked response law.

## Phase 4 — Real-mask semi-synthetic evaluation

Preserve real worker--task masks and inject controlled responses.  Candidate datasets include LabelMe, RTE, Trec, Dog, Bird, and ZC_all after verifying exact directory names.

This phase separates shift detection from unrealistic dense synthetic overlap.  Evaluate every method under the same mask and known injected mechanism.

- [ ] Implement mask extraction.
- [ ] Implement response sampling from in-prior and held-out mechanisms.
- [ ] Run at least six masks, four mechanisms, and multiple severities.
- [ ] Report power and worker localization with mask-specific confidence intervals.

## Phase 5 — Real benchmarks

Train the response head without using deployment gold task labels.  On real datasets report:

- CrowdFM aggregation accuracy where gold truth is available for evaluation only;
- masked-annotation NLL/Brier;
- marginal, dependence, and joint p-values;
- rejection stability across cross-fit seeds;
- runtime and bootstrap cost;
- qualitative residual-eigenvector case studies.

Do not label a real matrix “OOD” solely because it is rejected.  The statistically correct interpretation is incompatibility with the learned response law.

## Phase 6 — Selective aggregation

Compare the audit gate against:

- CrowdFM entropy and margin;
- split-view or worker-subsampling instability;
- latent embedding OOD scores;
- response NLL without the dependence statistic;
- dependence statistic without marginal calibration;
- majority-vote and classical-model diagnostics.

Fallbacks must be pre-specified and evaluated rather than called safe by assumption.  Report:

- coverage;
- accepted-set risk/accuracy;
- AURC;
- accuracy at fixed coverage;
- fallback penalty on false rejections;
- gain on detected failure families;
- oracle-router gap.

## Required ablations

- no cross-fitting;
- response head trained with and without task-truth loss;
- frozen versus fine-tuned backbone;
- binary disagreement residual versus full categorical residual;
- marginal-only, dependence-only, and joint test;
- Frobenius norm, maximum entry, and one-sided eigenvalue instead of the two-sided operator norm;
- worker-context support threshold;
- audit/context fraction;
- Monte Carlo count `B in {99,199,499,999}`.

## Reproducibility

Training seeds: `42, 43, 44`.

Evaluation/cross-fit seeds: `42, 43, 44, 45, 46`.

Every output must record:

- Git commit;
- full model/training/audit configuration;
- checkpoint path;
- dataset/world seed;
- split seed;
- runtime;
- response metrics;
- component and joint p-values.

## Historical execution log

### Official CrowdFM baseline

- Evaluated all 16 available dataset directories.
- Mean task accuracy: `0.8229134873`.
- Mean reported per-dataset runtime: `0.0849569976` seconds.

### Confusion-based smoke audit

- Rejected 14/16 real datasets at `alpha=0.05`.
- This was not evidence that 14 datasets were truly OOD; it showed that the manufactured stationary confusion null was too restrictive or poorly estimated.

### Confusion calibration decomposition

- Oracle response law produced approximately nominal rejection.
- Estimated response law produced `70--74%` rejection in the first full sweep.
- Hierarchical shrinkage reduced but did not solve inflation: estimated-confusion rejection remained `28.75%` with CrowdFM `q` and `51.25%` with oracle `q` aggregated over 80 worlds.
- Support sweeps reduced disagreement MAE but did not establish a clean audit because nuisance estimation and audit power grew together.

These results motivate the current direct masked-annotation response model.  See `CALIBRATION_RESULTS.md` for details.
