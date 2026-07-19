"""6-arm clean comparison for the residual/meta method on a frozen CrowdFM.

  A1 official   : frozen CrowdFM truth head (X)
  A2 no-mech    : residual head on frozen backbone, no Z (capacity control)
  A3 zeroshot   : mechanism residual, zero-shot (Z=mu0)
  A4 direct     : mechanism residual + direct response-likelihood latent adaptation
  A5 meta       : meta model, inner response-adapt Z_resp -> learned g_psi -> Z_agg
  A6 oracle     : mechanism residual, latent tuned on GOLD truth (upper bound)

All arms scored on the SAME worlds. Go: A3/A5 - A1 >= +1pp (held-out), A5 > A4 and A5 > A2.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
import dlwheel, numpy as np, torch
import torch.nn.functional as F
from cfm.data.crowdsi_simulator import CrowdSISimulator
from cfm.model.ResidualCFM import ResidualMechanismCFM
from cfm.audit.pipeline import masked_data
from cfm.audit.predictive_split import make_annotation_audit_split
from cfm.si.likelihood import joint_annotation_log_likelihood


def seed_all(s): random.seed(s); np.random.seed(s % 2**32); torch.manual_seed(s)

def load(mc, ckpt, device, use_mech):
    cfg = dict(mc); cfg["use_mechanism"] = use_mech
    m = ResidualMechanismCFM(**cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=True); m.eval(); return m

def acc(logits, y, v): return float((logits.argmax(-1)[v] == y[v]).float().mean())

def split_audit(data, seed):
    sp = make_annotation_audit_split(data.triple, data.num_task, num_worker=data.num_worker,
        audit_task_fraction=1.0, context_fraction=0.5, min_context_workers=1,
        min_audit_workers=2, min_worker_context_edges=1, seed=seed)
    ctx = masked_data(data, sp.context_edge_mask)
    idx = sp.audit_edge_indices
    return ctx, data.triple[0, idx].long(), data.triple[2, idx].long(), data.triple[1, idx].long()

def resp_nll(model, ctx, qw, qt, qa, z):
    o = model(ctx, query_workers=qw, query_tasks=qt, mechanism_latent=z, sample_mechanism=False)
    return -joint_annotation_log_likelihood(o["hat_task_option"], o["hat_annotation_given_truth"], qa, qt, reduction="mean")

def adapt_latent_response(model, ctx, qw, qt, qa, z0, steps, lr):
    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); l = resp_nll(model, ctx, qw, qt, qa, z); l.backward(); opt.step()
    return z.detach()

def adapt_latent_oracle(model, data, z0, y, v, steps, lr):
    z = z0.detach().clone().requires_grad_(True); opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); o = model(data, mechanism_latent=z, sample_mechanism=False)
        F.cross_entropy(o["hat_task_option"][v], y[v]).backward(); opt.step()
    return z.detach()


@torch.no_grad()
def _fwd(model, data, z=None, zagg=None):
    return model(data, mechanism_latent=z, agg_latent=zagg, sample_mechanism=False)


def main():
    cfg = dlwheel.setup(); device = cfg.device; mc = cfg.model.to_dict()
    mech = load(mc, cfg.mech_ckpt, device, True)
    nomech = load(mc, cfg.nomech_ckpt, device, False)
    meta = load(mc, cfg.meta_ckpt, device, True)
    steps = int(cfg.get("adapt_steps", 40)); lr = float(cfg.get("adapt_lr", 0.05))
    inner_lr = float(cfg.get("inner_lr", 0.1)); oracle_steps = int(cfg.get("oracle_steps", 60))
    sim_kw = dict(cfg.simulator.to_dict())
    fam = cfg.get("families", None)
    if fam: sim_kw["mechanism_families"] = [f.strip() for f in str(fam).split(",")]
    sim = CrowdSISimulator(**sim_kw)
    N = int(cfg.get("num_worlds", 150)); base = int(cfg.get("base_seed", 90000))
    recs = []
    for i in range(N):
        seed_all(base + i)
        data = sim.generate(); data.to(device)
        y = data.task_y.to(device).long(); v = y >= 0
        if not bool(v.any()): continue
        # A1 official (frozen head) — same from any model
        A1 = acc(_fwd(mech, data)["hat_task_option_base"], y, v)
        # A2 no-mech zero-shot, A3 mech zero-shot
        A2 = acc(_fwd(nomech, data)["hat_task_option"], y, v)
        A3 = acc(_fwd(mech, data)["hat_task_option"], y, v)
        # context/audit for adaptation
        ctx, qw, qt, qa = split_audit(data, base + i)
        z0 = mech.encode_mechanism(ctx)[1].detach()
        # A4 direct response adaptation on mech model
        z4 = adapt_latent_response(mech, ctx, qw, qt, qa, z0, steps, lr)
        A4 = acc(_fwd(mech, data, z=z4)["hat_task_option"], y, v)
        # A5 meta: inner response step + learned g_psi
        zm0 = meta.encode_mechanism(ctx)[1].detach()
        zc = zm0.clone().requires_grad_(True)
        l = resp_nll(meta, ctx, qw, qt, qa, zc); g = torch.autograd.grad(l, zc)[0]
        zprime = (zc - inner_lr * g).detach()
        with torch.no_grad():
            base_ctx = meta.encode_mechanism(ctx)[0]["z_t"].mean(0)
            dagg = meta.agg_update(torch.cat([zprime, zm0, base_ctx]))
            zagg = zprime + dagg
        A5 = acc(_fwd(meta, data, z=zm0, zagg=zagg)["hat_task_option"], y, v)
        # A6 oracle (mech latent tuned on gold)
        z6 = adapt_latent_oracle(mech, data, z0, y, v, oracle_steps, 0.05)
        A6 = acc(_fwd(mech, data, z=z6)["hat_task_option"], y, v)
        recs.append(dict(A1=A1, A2=A2, A3=A3, A4=A4, A5=A5, A6=A6, K=int(data.num_option)))
        if (i+1) % 30 == 0:
            m = lambda k: np.mean([r[k] for r in recs])
            print(f"  {i+1}/{N} A1={m('A1'):.4f} A2={m('A2'):.4f} A3={m('A3'):.4f} A4={m('A4'):.4f} A5={m('A5'):.4f} A6={m('A6'):.4f}", flush=True)
    def ag(k):
        vv = np.array([r[k] for r in recs]); return dict(mean=float(vv.mean()), ci95=float(1.96*vv.std(ddof=1)/math.sqrt(len(vv))))
    def pd(a, b):
        d = np.array([r[a]-r[b] for r in recs]); return dict(mean=float(d.mean()), ci95=float(1.96*d.std(ddof=1)/math.sqrt(len(d))))
    out = dict(n=len(recs), families=sim_kw["mechanism_families"],
               arms={k: ag(k) for k in ["A1","A2","A3","A4","A5","A6"]},
               deltas={"A3-A1(zeroshot-mech)": pd("A3","A1"), "A2-A1(nomech)": pd("A2","A1"),
                       "A4-A3(direct-adapt)": pd("A4","A3"), "A5-A1(meta)": pd("A5","A1"),
                       "A5-A4(meta-vs-direct)": pd("A5","A4"), "A5-A2(meta-vs-nomech)": pd("A5","A2"),
                       "A6-A1(oracle)": pd("A6","A1")}, records=recs)
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps(out, indent=2))
    print(f"\n[{sim_kw['mechanism_families']}] n={len(recs)}")
    for k in ["A1","A2","A3","A4","A5","A6"]: print(f"  {k} = {out['arms'][k]['mean']:.4f}")
    for k, d in out["deltas"].items(): print(f"  {k:24s} {d['mean']*100:+.2f}pp +/- {d['ci95']*100:.2f}")


if __name__ == "__main__":
    main()
