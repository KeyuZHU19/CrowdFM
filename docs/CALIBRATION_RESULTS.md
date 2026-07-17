# CbR Calibration Results

This file records calibration results and implementation decisions. Raw experiment outputs remain under `log/` and are not committed.

## Fixed plug-in confusion sweep

The first 20-world sweep used a fixed estimated confusion matrix inside each bootstrap replicate. At `alpha=0.05`:

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

The residual statistic and Monte Carlo core are approximately calibrated when the generating confusion matrices are supplied. CrowdFM posterior error is not the dominant failure. The failure is concentrated in estimating a full worker-specific multiclass confusion matrix from sparse nuisance labels.

## Uncertainty corrections that were insufficient

Two uncertainty treatments were tested on the quick binary and five-class settings:

1. parametric refit bootstrap, which re-simulates nuisance labels and re-estimates confusion matrices per replicate;
2. paired Dirichlet posterior-predictive bootstrap.

Both removed no systematic rejection in the quick five-class setting: both estimated-confusion variants rejected `2/2` worlds. Sweeping the symmetric Dirichlet prior strength over `0.1`, `0.01`, and `0.001` also failed. Smaller prior strength caused the spectral statistic to become unstable because worker/class rows had only a few effective observations.

## Current diagnosis

For the quick five-class configuration, approximately half of `150 x 5 = 750` annotations are nuisance annotations. Spread over 30 workers and five truth classes, each worker/class confusion row has only about `2.5` effective labels on average. Estimating five probabilities independently for every such row is therefore not identifiable. Bootstrap calibration cannot repair a systematically underidentified nuisance model.

## Current implementation decision

The default estimator now uses hierarchical global confusion shrinkage:

1. pool nuisance expected counts across all workers for each truth class;
2. construct a leave-one-worker-out global class-conditional confusion row;
3. use that row as the Dirichlet prior mean for the held-out worker;
4. combine it with the worker's own sparse counts using prior strength `10`;
5. retain posterior-predictive calibration for the remaining uncertainty.

This remains training-free. The official CrowdFM checkpoint is frozen throughout.

## Next run

```bash
git pull --ff-only origin agent/cbr-audit-core
pytest -q
python run_cbr_calibration.py config=config/cbr_calibration_quick.yaml
```

Expected test count: `12 passed`.

The quick result is written to:

```text
log/cbr_calibration_quick_hierarchical.json
```

Do not run the 20-world sweep until the two quick multiclass estimated-confusion p-values are no longer both at the rejection boundary.
