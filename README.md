# CrowdFM

Official implementation for the ICLR 2026 paper:
**Towards a Foundation Model for Crowdsourced Label Aggregation**.

![CrowdFM Overview](assets/CrowdFM.jpg)

## Installation

```bash
uv sync
source .venv/bin/activate
```

For development and CbR tests:

```bash
uv sync --extra dev
pytest -q
```

## Usage

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

### CbR real-data smoke audit

```bash
python evaluate_cbr.py \
  checkpoint_path=checkpoint.pt \
  seeds=42 \
  cbr.num_bootstrap=99 \
  output_path=log/cbr_smoke.json
```

### CbR four-way null calibration

The calibration experiment runs the same synthetic worlds and cross-fit splits under four variants:

1. oracle item posterior and oracle worker confusion;
2. CrowdFM posterior and oracle confusion;
3. oracle posterior and estimated confusion;
4. CrowdFM posterior and estimated confusion.

Run the development sweep with:

```bash
python run_cbr_calibration.py config=config/cbr_calibration.yaml
```

The default configuration runs four synthetic settings, 20 worlds per setting, and 99 Monte Carlo replicates. Results are written incrementally to `log/cbr_calibration_smoke.json`, so interrupted runs retain completed worlds.

### Train

```bash
python train.py --backup
```

Optional:

```bash
python train.py --backup --resume
```

```bash
python train.py --backup name=<experiment_name>
```

Training logs and checkpoints are saved under `log/<experiment_name>/` by default.

## CbR documentation

- `docs/CBR_IDEA.md`: research motivation and contribution framing.
- `docs/CBR_SPEC.md`: statistical definitions and implementation constraints.
- `docs/EXPERIMENT_PLAN.md`: live AAAI experiment plan and execution log.

## Citation

```bibtex
@inproceedings{liu2026crowdfm,
    title={Towards a Foundation Model for Crowdsourced Label Aggregation},
    author={Liu, Hao and Liu, Jiacheng and Tang, Feilong and Chen, Long and Yu, Jiadi and Zhu, Yanmin and Dong, Qiwen and Yu, Yichuan and Hou, Xiaofeng},
    booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
    year={2026},
}
```
