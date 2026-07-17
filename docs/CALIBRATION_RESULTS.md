# CbR Calibration Results

This file records calibration results and the implementation decision taken from them. Raw experiment outputs remain under `log/` and are not committed.

## Fixed-nuisance bootstrap sweep

Command:

```bash
python run_cbr_calibration.py config=config/cbr_calibration.yaml
```

The first 20-world sweep used a fixed plug-in confusion matrix inside every bootstrap replicate. At `alpha=0.05`, the observed rejection rates were:

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

## Interpretation

The residual statistic and Monte Carlo test are approximately calibrated when the generating confusion matrices are supplied. Replacing the oracle item posterior by the CrowdFM context-only posterior does not materially inflate the aggregate rejection rate. The failure is concentrated in the plug-in confusion estimate, especially when the number of classes is large or nuisance support per worker/class row is sparse.

The fixed-nuisance bootstrap simulates audit labels from `P_hat` while treating `P_hat` as known. It therefore omits uncertainty from estimating the worker confusion matrices. Small estimation errors become a structured residual mean shift after aggregation across many worker pairs.

## Implementation decision

The default fitted audit now uses a parametric confusion-refit bootstrap:

1. simulate nuisance and audit labels from the fitted predictive model;
2. re-estimate each worker confusion matrix from the simulated nuisance labels;
3. recompute the audit residual and spectral statistic using the replicate estimate.

Oracle-confusion variants continue to use the fixed-confusion Monte Carlo test. The task posterior remains fixed during this first correction because the oracle-confusion experiments show that CrowdFM posterior error is not the dominant calibration failure.

## Next run

```bash
git pull --ff-only origin agent/cbr-audit-core
pytest -q
python run_cbr_calibration.py config=config/cbr_calibration_quick.yaml
```

The updated quick output is written to `log/cbr_calibration_quick_refit.json`. If the estimated-confusion variants no longer reject systematically, run:

```bash
python run_cbr_calibration.py config=config/cbr_calibration.yaml
```

The updated 20-world output is written to `log/cbr_calibration_smoke_refit.json`.
