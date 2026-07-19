"""Baseline comparison: does the frozen CrowdFM prior add value over classic anchor-using
aggregators?  All methods use the SAME VOI-selected anchors + same budget.

  crowdfm      frozen CrowdFM, no anchors                         [FM baseline]
  mv_rw        reliability-reweighted majority vote (anchors, NO CrowdFM)
  ds_anchor    semi-supervised Dawid-Skene (anchors clamped, NO CrowdFM)
  crowdguard   CrowdFM + reliability-log-odds correction (ours, uses CrowdFM + anchors)
Reports effective accuracy vs budget per method and the CrowdGuard advantage over the best
non-CrowdFM baseline (isolates the value of the frozen crowd FM).
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path
import dlwheel, numpy as np, torch
from cfm.data.crowdsi_simulator import CrowdSISimulator
from cfm.model.ResidualCFM import ResidualMechanismCFM


def seed_all(s): random.seed(s); np.random.seed(s % 2**32); torch.manual_seed(s)


def ds_em(w, a, t, M, N, K, n_iter=30, clamp=None, init_T=None):
    E = len(w); ar = np.arange(K)
    wi = np.repeat(w[:, None], K, 1); ci = np.repeat(ar[None, :], E, 0); ai = np.repeat(a[:, None], K, 1)
    T = init_T.copy() if init_T is not None else None
    if T is None:
        T = np.full((N, K), 1e-6); np.add.at(T, (t, a), 1.0); T /= T.sum(1, keepdims=True)
    ck = None
    if clamp:
        ck = np.array(list(clamp.keys())); cg = np.array(list(clamp.values())); T[ck] = 0; T[ck, cg] = 1.0
    for _ in range(n_iter):
        pi = T.sum(0) + 1e-6; pi /= pi.sum()
        cm = np.full((M, K, K), 1e-6); np.add.at(cm, (wi, ci, ai), T[t]); cm /= cm.sum(2, keepdims=True)
        logcm = np.log(cm[w][np.arange(E)[:, None], ar[None, :], a[:, None]] + 1e-12)
        logT = np.log(pi)[None, :].repeat(N, 0); np.add.at(logT, t, logcm)
        logT -= logT.max(1, keepdims=True); T = np.exp(logT); T /= T.sum(1, keepdims=True)
        if ck is not None: T[ck] = 0; T[ck, cg] = 1.0
    return T


def rel_weights(w, a, t, anchors, y, M):
    corr = np.ones(M); tot = np.full(M, 2.0); aset = set(int(x) for x in anchors)
    if aset:
        mk = np.array([int(x) in aset for x in t]); np.add.at(tot, w[mk], 1.0); np.add.at(corr, w[mk], (a[mk] == y[t[mk]]).astype(float))
    rel = np.clip(corr / tot, 1e-2, 1 - 1e-2); return np.log(rel / (1 - rel))


def mv_rw(w, a, t, anchors, y, M, N, K):
    wgt = np.maximum(rel_weights(w, a, t, anchors, y, M), 0) + 1e-3
    V = np.zeros((N, K)); np.add.at(V, (t, a), wgt[w]); return V.argmax(1)


def ds_anchor(w, a, t, anchors, y, M, N, K):
    return ds_em(w, a, t, M, N, K, clamp={int(k): int(y[k]) for k in anchors}).argmax(1)


def crowdguard(q, w, a, t, anchors, y, M, N, K):
    wgt = rel_weights(w, a, t, anchors, y, M)
    logp = np.log(q + 1e-12).copy(); np.add.at(logp, (t, a), wgt[w]); return logp.argmax(1)


def crowdguard_adaptive(q, w, a, t, anchors, y, M, N, K):
    """Anchor-calibrated FM trust: measure CrowdFM accuracy on the anchors and down-weight
    its prior when it is corrupted (fixes the extreme-attack failure where the FM itself is
    fooled). lam = normalized above-chance accuracy of CrowdFM on anchors, in [0,1]."""
    cf = q.argmax(1); aset = list(int(x) for x in anchors)
    acc = np.mean([cf[k] == y[k] for k in aset]) if aset else 0.7
    acc = float(np.clip(acc, 1e-2, 1 - 1e-2))
    lam = float(np.clip(np.log(acc / (1 - acc)), 0.0, 3.0))  # FM log-odds reliability on anchors
    wgt = rel_weights(w, a, t, anchors, y, M)
    logp = lam * np.log(q + 1e-12).copy(); np.add.at(logp, (t, a), wgt[w])
    return logp.argmax(1)


def eff_acc(pred, y, anchors):
    aset = set(int(x) for x in anchors); return float(np.mean([(k in aset) or (pred[k] == y[k]) for k in range(len(y))]))


def voi(q, w, a, t, y, M, N, K, B):
    sc = np.ones(M); st = np.full(M, 2.0); chosen = []; aset = set(); E = len(w)
    cf = q.argmax(1); dn = np.zeros(M); dd = np.zeros(M) + 1e-6
    np.add.at(dd, w, 1.0); np.add.at(dn, w, (a != cf[t]).astype(float)); dev = dn / dd
    for _ in range(B):
        rel = np.clip(sc / st, 1e-2, 1 - 1e-2); wg = np.log(rel / (1 - rel))
        lp = np.log(q + 1e-12).copy(); np.add.at(lp, (t, a), wg[w]); lp -= lp.max(1, keepdims=True)
        qc = np.exp(lp); qc /= qc.sum(1, keepdims=True); risk = 1 - qc.max(1)
        ae = np.array([int(tk) in aset for tk in t]) if aset else np.zeros(E, bool)
        infl = np.zeros(M); np.add.at(infl, w[~ae], risk[t[~ae]])
        rv = rel * (1 - rel) / (st + 1); wval = (rv + 0.5 * dev) * infl
        tv = np.zeros(N); np.add.at(tv, t, wval[w]); tv *= (0.5 + risk)
        if aset: tv[list(aset)] = -1
        k = int(np.argmax(tv)); chosen.append(k); aset.add(k)
        em = (t == k)
        for wi_, ai_ in zip(w[em], a[em]): st[wi_] += 1; sc[wi_] += (ai_ == y[k])
    return chosen


@torch.no_grad()
def main():
    cfg = dlwheel.setup(); device = cfg.device
    m = ResidualMechanismCFM(**dict(cfg.model.to_dict())).to(device)
    sd = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=True); m.eval()
    sk = dict(cfg.simulator.to_dict()); fam = cfg.get("families", None)
    if fam: sk["mechanism_families"] = [f.strip() for f in str(fam).split(",")]
    sim = CrowdSISimulator(**sk); Nw = int(cfg.get("num_worlds", 60)); base = int(cfg.get("base_seed", 66000))
    budgets = [int(b) for b in cfg.get("budgets", [0, 10, 20, 50])]
    acc = defaultdict(list)
    for i in range(Nw):
        seed_all(base + i); data = sim.generate(); data.to(device)
        y = data.task_y.cpu().numpy(); M, N, K = data.num_worker, data.num_task, data.num_option
        w = data.triple[0].cpu().numpy(); a = data.triple[1].cpu().numpy(); t = data.triple[2].cpu().numpy()
        q = torch.softmax(m(data, sample_mechanism=False)["hat_task_option_base"], -1).cpu().numpy()
        acc["crowdfm|0"].append(eff_acc(q.argmax(1), y, []))
        for B in budgets:
            if B == 0: continue
            Bc = min(B, N - 1); anch = voi(q, w, a, t, y, M, N, K, Bc)
            acc[f"mv_rw|{B}"].append(eff_acc(mv_rw(w, a, t, anch, y, M, N, K), y, anch))
            acc[f"ds_anchor|{B}"].append(eff_acc(ds_anchor(w, a, t, anch, y, M, N, K), y, anch))
            acc[f"crowdguard|{B}"].append(eff_acc(crowdguard(q, w, a, t, anch, y, M, N, K), y, anch))
            acc[f"cg_adaptive|{B}"].append(eff_acc(crowdguard_adaptive(q, w, a, t, anch, y, M, N, K), y, anch))
        if (i + 1) % 20 == 0: print(f"  {i+1}/{Nw}", flush=True)
    def ms(k):
        v = acc.get(k, []); n = len(v)
        return {"mean": float(np.mean(v)) if n else float("nan"),
                "ci95": float(1.96 * np.std(v) / max(n, 1) ** 0.5) if n > 1 else 0.0}
    keys = ["crowdfm|0"] + [f"{s}|{B}" for B in budgets if B > 0 for s in ["mv_rw", "ds_anchor", "crowdguard", "cg_adaptive"]]
    summ = {k: ms(k) for k in keys}
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps(summ, indent=2))
    cf = summ["crowdfm|0"]["mean"]
    print(f"\n[{sk['mechanism_families']}] CrowdFM={cf:.4f} (n={Nw})")
    print("  B    mv_rw    ds_anchor  CrowdGuard | CG-best_noFM")
    for B in budgets:
        if B == 0: continue
        mv = summ[f"mv_rw|{B}"]["mean"]; ds = summ[f"ds_anchor|{B}"]["mean"]; cg = summ[f"crowdguard|{B}"]["mean"]
        print(f"  {B:3d}  {mv:.4f}   {ds:.4f}    {cg:.4f}    | {(cg-max(mv,ds))*100:+.2f}pp")


if __name__ == "__main__":
    main()
