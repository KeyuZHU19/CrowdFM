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

## Hierarchical global shrinkage

The estimator was then changed to a leave-one-worker-out global class-conditional prior with strength `10`. The two-world quick check looked promising, but the full 20-world sweep remained substantially anti-conservative:

| Configuration | Oracle q + oracle P | CrowdFM q + oracle P | Oracle q + estimated P | CrowdFM q + estimated P |
|---|---:|---:|---:|---:|
| small_binary | 0.00 | 0.00 | 0.20 | 0.05 |
| medium_multiclass | 0.15 | 0.10 | 0.55 | 0.25 |
| large_multiclass | 0.00 | 0.00 | 0.75 | 0.65 |
| sparse_imbalanced | 0.10 | 0.10 | 0.55 | 0.20 |

Aggregated over 80 worlds:

- oracle posterior + oracle confusion: `5/80 = 6.25%`;
- CrowdFM posterior + oracle confusion: `4/80 = 5.00%`;
- oracle posterior + estimated confusion: `41/80 = 51.25%`;
- CrowdFM posterior + estimated confusion: `23/80 = 28.75%`.

Hierarchical shrinkage therefore improves the fitted pipeline but does not solve null calibration. In the ten-class setting, the mean entrywise confusion MAE is only about `0.023`, yet rejection remains `65--75%`. Entrywise MAE is consequently not an adequate diagnostic for the pairwise disagreement residual.

## Current diagnosis

The synthetic generator draws the off-diagonal distribution independently for every worker and truth class. The hierarchical estimator assumes those rows share a pooled class-conditional structure. With about five nuisance labels per worker/class row, the independently generated rows cannot be recovered and the shared prior is structurally biased. This is an identifiability and model-family issue, not a bootstrap implementation issue.

The softer CrowdFM posterior yields lower fitted rejection than the oracle one-hot posterior because it spreads effective counts across classes and acts as additional regularization. This does not imply that the learned posterior is more correct than the oracle posterior.

## Current implementation decision

The code now records two diagnostics in every calibration world:

1. mean, median, and minimum effective nuisance support per worker/class row;
2. audit-pair disagreement MAE between the fitted and generating confusion models.

A dedicated support sweep isolates whether the full Dawid--Skene estimator approaches calibration as worker/class support increases. It uses only the oracle-posterior variants so posterior error cannot obscure the result.

## Next run

```bash
git pull --ff-only origin agent/cbr-audit-core
pytest -q
python run_cbr_calibration.py config=config/cbr_calibration_support.yaml
```

Output:

```text
log/cbr_calibration_support.json
```

Do not start OOD power experiments yet. If rejection decreases toward nominal as support increases, retain sparse independent-row Dawid--Skene as an underidentification stress test and use either a structured hierarchical synthetic null or a lower-dimensional confusion family for the realistic sparse calibration experiment.
