"""CrowdGuard on REAL crowd datasets: does active anchoring + reliability correction on a
frozen CrowdFM recover accuracy at small expert budget on real crowds?

Per dataset: CrowdFM baseline (frozen hat_task_option_base); then anchor-conditioned
correction (starts=CrowdFM) with selection in {random, entropy, VOI(ours)} at budgets.
Metric: effective accuracy (anchored counted correct) vs budget, and AURC. Gold labels
used only as the purchased anchors + for scoring.
"""
from __future__ import annotations
import json
from pathlib import Path
import dlwheel, numpy as np, torch
from cfm.data import load_data
from cfm.model.ResidualCFM import ResidualMechanismCFM


def correct(q_cf, w, a, t, anchors, y, M, N, K, lam=1.0):
    corr = np.ones(M); tot = np.full(M, 2.0)
    aset = set(int(x) for x in anchors)
    if aset:
        mask = np.array([int(x) in aset for x in t])
        np.add.at(tot, w[mask], 1.0)
        np.add.at(corr, w[mask], (a[mask] == y[t[mask]]).astype(float))
    rel = np.clip(corr / tot, 1e-2, 1 - 1e-2)
    wgt = np.log(rel / (1 - rel))
    logp = np.log(q_cf + 1e-12).copy()
    np.add.at(logp, (t, a), lam * wgt[w])
    return logp.argmax(1)


def eff_acc(pred, y, anchors):
    aset = set(int(x) for x in anchors)
    return float(np.mean([(k in aset) or (pred[k] == y[k]) for k in range(len(y))]))


def voi_adaptive(q, w, a, t, y, M, N, K, B):
    sc = np.ones(M); st = np.full(M, 2.0); chosen = []; aset = set(); E = len(w)
    cf_lab = q.argmax(1)
    dev_num = np.zeros(M); dev_den = np.zeros(M) + 1e-6
    np.add.at(dev_den, w, 1.0); np.add.at(dev_num, w, (a != cf_lab[t]).astype(float))
    dev = dev_num / dev_den
    for _ in range(B):
        rel = np.clip(sc / st, 1e-2, 1 - 1e-2); wgt = np.log(rel / (1 - rel))
        logp = np.log(q + 1e-12).copy(); np.add.at(logp, (t, a), wgt[w])
        logp -= logp.max(1, keepdims=True); qc = np.exp(logp); qc /= qc.sum(1, keepdims=True)
        risk = 1.0 - qc.max(1)
        anch_edge = np.array([int(tk) in aset for tk in t]) if aset else np.zeros(E, bool)
        infl = np.zeros(M); np.add.at(infl, w[~anch_edge], risk[t[~anch_edge]])
        rel_var = rel * (1 - rel) / (st + 1.0)
        wval = (rel_var + 0.5 * dev) * infl
        tval = np.zeros(N); np.add.at(tval, t, wval[w]); tval *= (0.5 + risk)
        if aset: tval[list(aset)] = -1.0
        k = int(np.argmax(tval))
        if tval[k] <= 0 and len(chosen) >= 1:
            # fall back to any unanchored
            rem = [x for x in range(N) if x not in aset]
            if not rem: break
            k = rem[0]
        chosen.append(k); aset.add(k)
        em = (t == k)
        for wi, ai in zip(w[em], a[em]): st[wi] += 1; sc[wi] += (ai == y[k])
    return chosen


@torch.no_grad()
def main():
    cfg = dlwheel.setup(); device = cfg.device
    m = ResidualMechanismCFM(**dict(cfg.model.to_dict())).to(device)
    sd = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=True); m.eval()
    budgets = [int(b) for b in cfg.get("budgets", [0, 10, 20, 50])]
    names = load_data.get_dataset_list(cfg)
    rows = {}
    for name in names:
        try:
            data = load_data.run(cfg, selected_dataset=name)[name]
            y = data.task_y.cpu().numpy()
            M, N, K = data.num_worker, data.num_task, data.num_option
            w = data.triple[0].cpu().numpy(); a = data.triple[1].cpu().numpy(); t = data.triple[2].cpu().numpy()
            q = torch.softmax(m(data, sample_mechanism=False)["hat_task_option_base"], -1).cpu().numpy()
            cf = eff_acc(q.argmax(1), y, [])
            cf_ent = -(q * np.log(q + 1e-12)).sum(1); ent_order = np.argsort(-cf_ent)
            rng = np.random.default_rng(12345)
            r = {"crowdfm": cf, "N": int(N), "M": int(M), "K": int(K)}
            for B in budgets:
                if B == 0: continue
                Bc = min(B, N - 1)
                rnd = list(rng.choice(N, size=Bc, replace=False))
                ent = list(ent_order[:Bc]); vo = voi_adaptive(q, w, a, t, y, M, N, K, Bc)
                r[f"random_{B}"] = eff_acc(correct(q, w, a, t, rnd, y, M, N, K), y, rnd)
                r[f"entropy_{B}"] = eff_acc(correct(q, w, a, t, ent, y, M, N, K), y, ent)
                r[f"voi_{B}"] = eff_acc(correct(q, w, a, t, vo, y, M, N, K), y, vo)
            rows[name] = r
            gains = " ".join(f"B{B}:{r.get(f'voi_{B}',cf)-cf:+.3f}" for B in budgets if B > 0)
            print(f"[{name:9s}] N={N:5d} M={M:4d} K={K:2d} CrowdFM={cf:.3f}  VOI gain {gains}", flush=True)
        except Exception as e:
            print(f"[{name}] SKIP {type(e).__name__}: {e}", flush=True)
    # summary means over datasets
    def mg(sel, B):
        vals=[rows[n][f"{sel}_{B}"]-rows[n]["crowdfm"] for n in rows if f"{sel}_{B}" in rows[n]]
        return float(np.mean(vals)) if vals else float("nan")
    summ = {f"{sel}_{B}": mg(sel, B) for sel in ["random","entropy","voi"] for B in budgets if B>0}
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps({"rows": rows, "mean_gain": summ}, indent=2))
    print("\n=== MEAN GAIN over datasets (method - CrowdFM) ===")
    for B in budgets:
        if B==0: continue
        print(f"  B={B:3d}: random={mg('random',B)*100:+.2f}pp  entropy={mg('entropy',B)*100:+.2f}pp  VOI={mg('voi',B)*100:+.2f}pp")


if __name__ == "__main__":
    main()
