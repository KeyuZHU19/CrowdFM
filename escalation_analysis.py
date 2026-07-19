"""Escalation-gate analysis: which gold-free dataset-level signal predicts where expert
anchoring pays off? Correlate candidate triggers with the REALIZED CrowdGuard gain.

Candidate triggers (no gold):
  cf_entropy   : mean CrowdFM predictive entropy (uncertainty -> room to fix)
  cf_maxconf   : mean max-posterior (inverse of above)
  disagree     : mean worker-vote entropy per task (raw annotation disagreement)
  evalue       : task-disjoint e-value (response-process shift) [from prior CrowdSI run if available]
Target: realized VOI anchor gain at B=20 (from runs/crowdguard/real.json).
Reports Spearman/Pearson correlation of each trigger with the realized gain.
"""
from __future__ import annotations
import json, glob, math
from pathlib import Path
import dlwheel, numpy as np, torch
from cfm.data import load_data
from cfm.model.ResidualCFM import ResidualMechanismCFM


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    def rank(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


@torch.no_grad()
def main():
    cfg = dlwheel.setup(); device = cfg.device
    m = ResidualMechanismCFM(**dict(cfg.model.to_dict())).to(device)
    sd = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=True); m.eval()
    real = json.load(open(cfg.gains_json))["rows"]  # realized gains per dataset
    # optional prior e-values
    evals = {}
    for f in glob.glob(cfg.get("evalue_glob", "runs/p1_full_s42/real_chunks/*.json")):
        try:
            d = json.load(open(f)); evals[d["dataset"]] = float(d.get("e_value", float("nan")))
        except Exception:
            pass
    B = int(cfg.get("gain_budget", 20))
    names = load_data.get_dataset_list(cfg)
    recs = []
    for name in names:
        if name not in real or real[name]["crowdfm"] < 0.3:  # skip loading-artifact datasets
            continue
        data = load_data.run(cfg, selected_dataset=name)[name]
        y = data.task_y.cpu().numpy(); N, K = data.num_task, data.num_option
        a = data.triple[1].cpu().numpy(); t = data.triple[2].cpu().numpy()
        q = torch.softmax(m(data, sample_mechanism=False)["hat_task_option_base"], -1).cpu().numpy()
        ent = float((-(q * np.log(q + 1e-12)).sum(1)).mean())
        maxc = float(q.max(1).mean())
        # per-task vote entropy
        ve = []
        for k in range(N):
            vv = np.bincount(a[t == k], minlength=K).astype(float)
            if vv.sum() > 0:
                p = vv / vv.sum(); ve.append(-(p * np.log(p + 1e-12)).sum())
        disagree = float(np.mean(ve)) if ve else 0.0
        gain = real[name].get(f"voi_{B}", real[name]["crowdfm"]) - real[name]["crowdfm"]
        recs.append({"name": name, "crowdfm": real[name]["crowdfm"], "gain": gain,
                     "cf_entropy": ent, "cf_maxconf": maxc, "disagree": disagree,
                     "evalue": evals.get(name, float("nan")),
                     "log_evalue": math.log(max(evals.get(name, 1e-12), 1e-12))})
        print(f"[{name:9s}] CrowdFM={real[name]['crowdfm']:.3f} gain@B{B}={gain:+.3f} "
              f"ent={ent:.3f} maxconf={maxc:.3f} disagree={disagree:.3f} E={evals.get(name,float('nan')):.3g}", flush=True)
    g = [r["gain"] for r in recs]
    print(f"\n=== correlation with realized VOI gain@B{B} (n={len(recs)}) ===")
    for sig in ["cf_entropy", "cf_maxconf", "disagree", "log_evalue", "crowdfm"]:
        xs = [r[sig] for r in recs]
        ok = [(x, gg) for x, gg in zip(xs, g) if not (isinstance(x, float) and math.isnan(x))]
        if len(ok) > 2:
            xx, gg = zip(*ok)
            print(f"  {sig:12s} spearman={spearman(xx, gg):+.3f}  pearson={np.corrcoef(xx, gg)[0,1]:+.3f}  (n={len(ok)})")
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps({"records": recs}, indent=2))


if __name__ == "__main__":
    main()
