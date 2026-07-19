"""Go/No-Go clean 3-way comparison on an identical synthetic world bank.

  X = official frozen CrowdFM   : backbone loaded from checkpoint.pt, NO CrowdSI
      training; prediction = hat_task_option_base (backbone truth head).
  Y = capacity-matched no-mech  : full CrowdSI architecture with disable_mechanism,
      trained identically; prediction = zero-shot hat_task_option (C(Z)=0).
  Z = full CrowdSI              : prediction = zero-shot hat_task_option at mu_0.

All three are evaluated on the SAME worlds (same base_seed) so deltas are paired and
the official baseline is never touched by CrowdSI training.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
import dlwheel, numpy as np, torch
from cfm.data.crowdsi_simulator import CrowdSISimulator
from cfm.model.CrowdSIFM import CrowdSIFM


def seed_all(s):
    random.seed(s); np.random.seed(s % (2**32)); torch.manual_seed(s)


def load_full(model_cfg, ckpt, device, disable):
    cfg = dict(model_cfg); cfg["disable_mechanism"] = disable
    m = CrowdSIFM(**cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=True); m.eval(); return m


def load_official(model_cfg, ckpt, device):
    m = CrowdSIFM(**dict(model_cfg)).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_crowdfm_checkpoint(ck, strict=True); m.eval(); return m


def mv_pred(data):
    a = data.triple[1].cpu().numpy(); t = data.triple[2].cpu().numpy()
    p = np.zeros(data.num_task, dtype=np.int64)
    for k in range(data.num_task):
        aa = a[t == k]
        if len(aa): p[k] = np.bincount(aa, minlength=data.num_option).argmax()
    return torch.tensor(p)


@torch.no_grad()
def main():
    cfg = dlwheel.setup(); device = torch.device(cfg.device)
    mc = cfg.model.to_dict()
    official = load_official(mc, cfg.official_checkpoint, device)
    full = load_full(mc, cfg.full_checkpoint, device, disable=False)
    nomech = load_full(mc, cfg.nomech_checkpoint, device, disable=True)

    sim_kwargs = dict(cfg.simulator.to_dict())
    fam = cfg.get("families", None)
    if fam: sim_kwargs["mechanism_families"] = [f.strip() for f in str(fam).split(",")]
    sim = CrowdSISimulator(**sim_kwargs)
    N = int(cfg.get("num_worlds", 200)); base = int(cfg.get("base_seed", 90000))
    recs = []
    for i in range(N):
        seed_all(base + i)
        data = sim.generate(); data.to(device)
        y = data.task_y.to(device).long(); v = y >= 0
        ox = official(data, sample_mechanism=False)["hat_task_option_base"].argmax(-1)
        yz = nomech(data, sample_mechanism=False)["hat_task_option"].argmax(-1)
        zz = full(data, sample_mechanism=False)["hat_task_option"].argmax(-1)
        recs.append({
            "X_official": float((ox[v] == y[v]).float().mean()),
            "Y_nomech": float((yz[v] == y[v]).float().mean()),
            "Z_full": float((zz[v] == y[v]).float().mean()),
            "mv": float((mv_pred(data).to(device)[v] == y[v]).float().mean()),
        })
        if (i + 1) % 50 == 0:
            def m(k): return np.mean([r[k] for r in recs])
            print(f"  {i+1}/{N} X={m('X_official'):.4f} Y={m('Y_nomech'):.4f} Z={m('Z_full'):.4f}")
    def agg(k):
        vv = np.array([r[k] for r in recs]);
        return {"mean": float(vv.mean()), "ci95": float(1.96*vv.std(ddof=1)/math.sqrt(len(vv)))}
    def paired(a, b):
        d = np.array([r[a]-r[b] for r in recs])
        return {"mean": float(d.mean()), "ci95": float(1.96*d.std(ddof=1)/math.sqrt(len(d)))}
    out = {"n": len(recs), "families": sim_kwargs["mechanism_families"],
           "X_official": agg("X_official"), "Y_nomech": agg("Y_nomech"), "Z_full": agg("Z_full"),
           "mv": agg("mv"),
           "Z_minus_X": paired("Z_full","X_official"), "Y_minus_X": paired("Y_nomech","X_official"),
           "Z_minus_Y": paired("Z_full","Y_nomech"), "records": recs}
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps(out, indent=2))
    print(f"\n[{sim_kwargs['mechanism_families']}] n={len(recs)}")
    print(f"  X official-frozen = {out['X_official']['mean']:.4f}")
    print(f"  Y no-mechanism    = {out['Y_nomech']['mean']:.4f}   (Y-X={out['Y_minus_X']['mean']:+.4f} +/-{out['Y_minus_X']['ci95']:.4f})")
    print(f"  Z full CrowdSI    = {out['Z_full']['mean']:.4f}   (Z-X={out['Z_minus_X']['mean']:+.4f} +/-{out['Z_minus_X']['ci95']:.4f})")
    print(f"  Z-Y (mechanism)   = {out['Z_minus_Y']['mean']:+.4f} +/-{out['Z_minus_Y']['ci95']:.4f}")


if __name__ == "__main__":
    main()
