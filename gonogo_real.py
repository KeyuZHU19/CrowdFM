"""Go/No-Go 3-way comparison on real datasets (zero-shot, no adaptation).

X = official frozen CrowdFM (backbone from checkpoint.pt), Y = no-mechanism zero-shot,
Z = full CrowdSI zero-shot. One seed's full/nomech models per run; average over seeds
in post. Gold labels only for scoring.
"""
from __future__ import annotations
import json
from pathlib import Path
import dlwheel, numpy as np, torch
from cfm.data import load_data
from cfm.model.CrowdSIFM import CrowdSIFM


def load_full(mc, ckpt, device, disable):
    cfg = dict(mc); cfg["disable_mechanism"] = disable
    m = CrowdSIFM(**cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=True); m.eval(); return m


@torch.no_grad()
def main():
    cfg = dlwheel.setup(); device = cfg.device; mc = cfg.model.to_dict()
    official = CrowdSIFM(**dict(mc)).to(device)
    official.load_crowdfm_checkpoint(torch.load(cfg.official_checkpoint, map_location=device, weights_only=False), strict=True); official.eval()
    full = load_full(mc, cfg.full_checkpoint, device, False)
    nomech = load_full(mc, cfg.nomech_checkpoint, device, True)

    names = load_data.get_dataset_list(cfg)
    rows = {}
    for name in names:
        try:
            data = load_data.run(cfg, selected_dataset=name)[name]
            y = data.task_y.to(device).long(); v = y >= 0
            if not bool(v.any()):
                continue
            X = float((official(data, sample_mechanism=False)["hat_task_option_base"].argmax(-1)[v] == y[v]).float().mean())
            Y = float((nomech(data, sample_mechanism=False)["hat_task_option"].argmax(-1)[v] == y[v]).float().mean())
            Z = float((full(data, sample_mechanism=False)["hat_task_option"].argmax(-1)[v] == y[v]).float().mean())
            rows[name] = {"X_official": X, "Y_nomech": Y, "Z_full": Z, "Z_minus_X": Z-X, "Y_minus_X": Y-X}
            print(f"[{name}] X={X:.4f} Y={Y:.4f} Z={Z:.4f} Z-X={Z-X:+.4f}", flush=True)
        except Exception as e:
            print(f"[{name}] SKIP: {type(e).__name__}: {e}", flush=True)
    dsX = np.mean([r["Z_minus_X"] for r in rows.values()])
    dsYX = np.mean([r["Y_minus_X"] for r in rows.values()])
    print(f"\nMEAN Z-X (full vs official) = {dsX:+.4f}   MEAN Y-X (nomech vs official) = {dsYX:+.4f}")
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps({"rows": rows, "mean_Z_minus_X": float(dsX), "mean_Y_minus_X": float(dsYX)}, indent=2))


if __name__ == "__main__":
    main()
