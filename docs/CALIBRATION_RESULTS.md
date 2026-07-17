# CbR Calibration Results

This file records calibration results and the implementation decisions taken from them. Raw experiment outputs remain under `log/` and are not committed.

## Fixed plug-in bootstrap sweep

The first 20-world sweep held the estimated confusion matrix fixed inside every bootstrap replicate. At `alpha=0.05`, the observed rejection rates were:

| Configuration | Oracle q + oracle P | CrowdFM q + oracle P | Oracle q + estimated P | CrowdFM q + estimated P |
|---|---:|---:|---:|---:|
| small_binary | 0.00 | 0.00 | 0.00 | 0.00 |
| medium_multiclass | 0.15 | 0.10 | 1.00 | 1.00 |
| large_multiclass | 0.00 | 0.00 | 1.00 | 1.00 |
| sparse_imbalanced | 0.10 | 0.10 | 0.80 | 0.95 |

Aggregated over the 80 worlds:

- oracle posterior + oracle confusion: `5/80 = 6.25%`;
- CrowdFM posterior + oracle confusion: `4/80 = 5.00%`;
- oracle posterior + estimated confusion: `56/80 = 70.00%`;
- CrowdFM posterior + estimated confusion: `59/80 = 73.75%`.

The residual statistic and Monte Carlo test are therefore approximately calibrated when the generating confusion matrices are supplied. CrowdFM posterior error is not the dominant failure. The problem is sparse multiclass confusion uncertainty.

## Confusion-refit bootstrap

A parametric refit bootstrap was tested next: simulate nuisance and audit labels from the plug-in estimate, re-estimate confusion from simulated nuisance labels, then recompute the audit statistic. The two quick multiclass worlds still rejected in both estimated-confusion variants. Refitting around a biased or weakly identified plug-in matrix did not reproduce the discrepancy between the unknown generating matrix and its nuisance estimate.

## Prior-strength sweep

The quick experiment was repeated with `prior_strength` in `{0.1, 0.01, 0.001}` and `B=99`.

- Multiclass `oracle_q_estimated_p` remained rejected in both worlds for every prior.
- Reducing the prior caused the raw spectral statistic to grow sharply, reaching about `189` at `prior_strength=0.001`.
- Confusion MAE remained around `0.10`; it did not improve as the prior approached zero.

This rules out excessive shrinkage toward the uniform matrix as the sole explanation. With sparse worker-by-class support, reducing regularization instead makes low-count rows unstable.

## Current implementation decision

The default fitted audit now uses a posterior-predictive confusion calibration:

1. compute Dirichlet posterior parameters from nuisance expected counts;
2. draw a confusion matrix `P*` from that posterior in each replicate;
3. simulate held-out audit labels using `P*`;
4. compute both the observed and replicated residual statistics under the same `P*` draw;
5. form a paired posterior-predictive p-value.

This directly integrates confusion uncertainty rather than pretending the plug-in matrix is known or refitting around the same uncertain point estimate. Oracle-confusion variants continue to use fixed-confusion Monte Carlo. CrowdFM remains frozen; no training is introduced.

## Next run

```bash
git pull --ff-only origin agent/cbr-audit-core
pytest -q
python run_cbr_calibration.py config=config/cbr_calibration_quick.yaml
```

The expected test count is `10 passed`. The new quick output is written to `log/cbr_calibration_quick_posterior.json`. Only after the multiclass estimated-confusion variants stop rejecting systematically should the 20-world configuration be run:

```bash
python run_cbr_calibration.py config=config/cbr_calibration.yaml
```

Its output is `log/cbr_calibration_smoke_posterior.json`.
