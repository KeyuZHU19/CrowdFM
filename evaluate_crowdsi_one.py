"""Evaluate CrowdSI-FM on a SINGLE real dataset (cfg.dataset). One job per dataset."""
from __future__ import annotations

import json
from pathlib import Path

import dlwheel
import torch

from cfm.data import load_data
from cfm.model.CrowdSIFM import CrowdSIFM
from cfm.si.pipeline import CrowdSIPipelineConfig, run_crowdsi


def main() -> None:
    cfg = dlwheel.setup()
    model = CrowdSIFM(**cfg.model.to_dict()).to(cfg.device)
    ck = torch.load(cfg.checkpoint_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    model.eval()
    pipe = CrowdSIPipelineConfig(**dict(cfg.crowdsi.to_dict()))

    name = cfg.dataset
    datasets = load_data.run(cfg, selected_dataset=name)
    data = datasets[name]
    data.to(cfg.device)
    result = run_crowdsi(model, data, config=pipe, seed=int(cfg.get("base_seed", 42000)))
    rec = {
        "dataset": name,
        "num_worker": int(data.num_worker), "num_task": int(data.num_task),
        "num_option": int(data.num_option), "num_edges": int(data.triple.shape[1]),
        "e_value": result["e_value"], "log_e_value": result["log_e_value"],
        "e_value_threshold": result["e_value_threshold"],
        "adaptation_supported": result["adaptation_supported"],
        "used_adaptation": result["used_adaptation"],
    }
    if "accuracy" in result:
        rec.update({
            "adapted_accuracy": result["accuracy"],
            "zero_shot_accuracy": result["zero_shot_accuracy"],
            "crowdfm_accuracy": result["crowdfm_accuracy"],
            "adaptation_delta": result["accuracy"] - result["zero_shot_accuracy"],
            "arch_delta": result["zero_shot_accuracy"] - result["crowdfm_accuracy"],
        })
    out = Path(cfg.get("output_path", f"log/real_{name}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    acc = (f" adapted={rec.get('adapted_accuracy'):.4f} zero={rec.get('zero_shot_accuracy'):.4f} "
           f"crowdfm={rec.get('crowdfm_accuracy'):.4f}" if "adapted_accuracy" in rec else "")
    print(f"[{name}] M={rec['num_worker']} N={rec['num_task']} K={rec['num_option']} "
          f"E={rec['e_value']:.4g} adapt={rec['used_adaptation']}{acc}")


if __name__ == "__main__":
    main()
