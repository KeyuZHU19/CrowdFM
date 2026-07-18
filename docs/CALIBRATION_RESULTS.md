# CbR Calibration History and Method Pivot

This file records why the first confusion-based implementation was retired.  Raw outputs remain under `log/` and are not committed.

## 1. Legacy hypothesis

The initial implementation used a frozen CrowdFM item posterior `q_k`, estimated a worker-specific class-conditional confusion matrix `P_i(a|c)` from nuisance tasks, and audited held-out pairwise disagreements.  This implicitly assumed that a post-hoc deployment estimator could supply the response law missing from the discriminative CrowdFM checkpoint.

## 2. Fixed plug-in confusion sweep

At `alpha=0.05`:

| Configuration | Oracle q + oracle P | CrowdFM q + oracle P | Oracle q + estimated P | CrowdFM q + estimated P |
|---|---:|---:|---:|---:|
| small_binary | 0.00 | 0.00 | 0.00 | 0.00 |
| medium_multiclass | 0.15 | 0.10 | 1.00 | 1.00 |
| large_multiclass | 0.00 | 0.00 | 1.00 | 1.00 |
| sparse_imbalanced | 0.10 | 0.10 | 0.80 | 0.95 |

Aggregated over 80 worlds:

- oracle posterior + oracle confusion: `5/80 = 6.25%`;
- CrowdFM posterior + oracle confusion: `4/80 = 5.00%`;
- oracle posterior + estimated confusion: `56/80 = 70.00%`;
- CrowdFM posterior + estimated confusion: `59/80 = 73.75%`.

The residual/Monte Carlo core was approximately calibrated when the generating response law was supplied.  The failure was the deployment confusion bridge.

## 3. Uncertainty corrections that failed

The following changes did not repair multiclass calibration:

1. parametric refit bootstrap;
2. Dirichlet posterior-predictive bootstrap;
3. symmetric prior-strength sweep over `0.1`, `0.01`, and `0.001`;
4. leave-one-worker-out global hierarchical shrinkage with strength `10`.

The full hierarchical sweep remained anti-conservative:

| Configuration | Oracle q + oracle P | CrowdFM q + oracle P | Oracle q + estimated P | CrowdFM q + estimated P |
|---|---:|---:|---:|---:|
| small_binary | 0.00 | 0.00 | 0.20 | 0.05 |
| medium_multiclass | 0.15 | 0.10 | 0.55 | 0.25 |
| large_multiclass | 0.00 | 0.00 | 0.75 | 0.65 |
| sparse_imbalanced | 0.10 | 0.10 | 0.55 | 0.20 |

Aggregated estimated-confusion rejection was `41/80 = 51.25%` with oracle `q` and `23/80 = 28.75%` with CrowdFM `q`.  In the ten-class setting, entrywise confusion MAE was only about `0.023`, yet rejection remained `65--75%`; small coherent response-law bias is amplified by the spectral statistic.

## 4. Support sweep

A later sweep increased total tasks to raise mean worker/class nuisance support.  The estimated disagreement became more accurate, but audit sample size increased at the same time.

| Setting | Mean support | Disagreement MAE | Estimated-P rejection |
|---|---:|---:|---:|
| K=5 | 5 | 0.0780 | 0.60 |
| K=5 | 20 | 0.0571 | 0.40 |
| K=5 | 50 | 0.0415 | 0.40 |
| K=10 | 5 | 0.0867 | 0.60 |
| K=10 | 20 | 0.0606 | 0.80 |

This experiment showed that the estimator error decreases, but it did not validate the method: nuisance estimation and audit power were scaled together.  More importantly, even a consistent full-confusion estimator would remain an external model not produced by CrowdFM.

## 5. Conceptual diagnosis

The official CrowdFM checkpoint predicts task truth but does not expose

```math
Pr(A_{ik}=a | context, i, k).
```

The legacy implementation attempted to manufacture this quantity as

```math
\sum_c q_k(c)\widehat P_i(a|c).
```

That changes the object being audited.  A rejection may indicate CrowdFM failure, confusion-family misspecification, estimator error, task-dependent worker behavior, or worker dependence.  Consequently it is not a clean audit of the foundation model.

The conclusion is not “find a better prior.”  It is:

> Do not infer a large post-hoc nuisance model and call it the predictive law of a frozen discriminative checkpoint.

## 6. New primary method

The primary implementation now trains the model to predict held-out annotations directly:

```math
r_{ik}(a)=Pr_\theta(A_{ik}=a | G_C,i,k).
```

The deployment audit evaluates:

1. marginal held-out log-score calibration;
2. structured cross-worker residual dependence;
3. exact conditional Monte Carlo p-values for the fixed response rows;
4. Bonferroni joint rejection and defer routing.

No confusion matrix is estimated at deployment.  See `docs/IDEA.md` and the `predictive_*` modules.

## 7. Status of legacy files

The following files are retained only for reproducibility of the negative result:

- `src/cfm/audit/disagreement.py`;
- `src/cfm/audit/bootstrap.py`;
- `src/cfm/audit/pipeline.py`;
- `src/cfm/audit/calibration.py`;
- `run_cbr_calibration.py`;
- legacy calibration YAML files.

They should not be used for new OOD-power experiments.

## 8. Next experiment

The next load-bearing experiment is not another confusion-prior sweep.  It is to train `PredictiveCFM` with masked-annotation loss and establish:

1. held-out response NLL/Brier calibration on unseen in-prior worlds;
2. nominal joint audit rejection under those worlds;
3. power on entire response mechanisms held out from pretraining;
4. improvement in risk--coverage after defer routing.
