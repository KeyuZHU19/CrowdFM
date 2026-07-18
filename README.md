# CrowdFM

Official implementation for the ICLR 2026 paper:
**Towards a Foundation Model for Crowdsourced Label Aggregation**.

![CrowdFM Overview](assets/CrowdFM.jpg)

## Installation

```bash
uv sync
source .venv/bin/activate
```

For development tests:

```bash
uv sync --extra dev
pytest -q
```

## Original CrowdFM usage

### Evaluate

By default, evaluation loads `checkpoint.pt`.

```bash
python evaluate.py
```

Optional:

```bash
python evaluate.py checkpoint_path=<path>
```

Results are written to `log/perform.json` by default.

### Train

```bash
python train.py --backup
```

Optional:

```bash
python train.py --backup --resume
python train.py --backup name=<experiment_name>
```

Training logs and checkpoints are saved under `log/<experiment_name>/` by default.

## Residual-audited PredictiveCFM

The research branch extends CrowdFM with a masked-annotation response head.  The head predicts

```math
Pr(A_{ik}=a\mid\text{context graph},i,k)
```

for annotations hidden from the model.  A deployment audit then tests marginal response calibration and structured cross-worker residual dependence with a conditional Monte Carlo test.

This is a compatibility audit of the held-out response law.  It is not a proof that aggregated task labels are correct.

### Train the response head

Start from the official CrowdFM checkpoint and train masked-annotation prediction:

```bash
python train_predictive.py config=config/predictive_train.yaml
```

The default configuration freezes the original backbone and trains the new head.  Joint fine-tuning is enabled with:

```bash
python train_predictive.py \
  config=config/predictive_train.yaml \
  freeze_backbone=false
```

A trained predictive checkpoint records a `response_head_trained` marker.  The audit refuses an untrained head.

### Run the deployment audit

```bash
python evaluate_predictive_audit.py \
  config=config/predictive_audit.yaml \
  checkpoint_path=log/predictive_cfm/10000.pt
```

Each dataset result includes:

- marginal response p-value;
- worker-dependence p-value;
- Bonferroni joint p-value;
- reject/defer decision;
- marginal and spectral statistics;
- audit/context support counts.

### Main implementation

- `src/cfm/model/PredictiveCFM.py`: masked-annotation response head;
- `src/cfm/audit/predictive_split.py`: leakage-free context/audit split;
- `src/cfm/audit/predictive.py`: categorical and spectral residuals;
- `src/cfm/audit/predictive_bootstrap.py`: conditional Monte Carlo test;
- `src/cfm/audit/predictive_pipeline.py`: deployment pipeline;
- `src/cfm/audit/predictive_training.py`: training objective.

## Historical confusion-based CbR

The earlier branch code combined CrowdFM task posteriors with a deployment-fitted worker confusion matrix.  Full calibration experiments showed that this external nuisance bridge remained strongly anti-conservative in sparse multiclass settings.  The implementation and runners are retained only to reproduce that negative result; they are not the recommended audit.

## Documentation

- `docs/IDEA.md`: formal research problem, method, guarantees, and limitations;
- `docs/CBR_SPEC.md`: implementation contract and formulas;
- `docs/CALIBRATION_RESULTS.md`: confusion-estimator failure history and method pivot;
- `docs/EXPERIMENT_PLAN.md`: revised experimental plan.

## Citation

```bibtex
@inproceedings{liu2026crowdfm,
    title={Towards a Foundation Model for Crowdsourced Label Aggregation},
    author={Liu, Hao and Liu, Jiacheng and Tang, Feilong and Chen, Long and Yu, Jiadi and Zhu, Yanmin and Dong, Qiwen and Yu, Yichuan and Hou, Xiaofeng},
    booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
    year={2026},
}
```
