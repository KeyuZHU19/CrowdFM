"""Correction-strength test (fixed VOI selection): how close can the anchor-conditioned
correction get to the oracle-worker ceiling?

  crowdfm        frozen CrowdFM                                       [baseline]
  scalar         log q_CrowdFM + reliability-log-odds vote (current)  [lower bound]
  cfds           CrowdFM-prior semi-supervised Dawid-Skene: init T from CrowdFM, add
                 CrowdFM as a pseudo-annotator, estimate full per-worker K×K confusions,
                 anchors clamped to gold                              [candidate STRONG]
  oracle_worker  worker confusions from ALL gold                     [ceiling]
Anchors selected by adaptive VOI (fixed) so we isolate the correction.
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
    wi_idx = np.repeat(w[:, None], K, 1); ci_idx = np.repeat(ar[None, :], E, 0); ai_idx = np.repeat(a[:, None], K, 1)
    if init_T is not None:
        T = init_T.copy()
    else:
        T = np.full((N, K), 1e-6); np.add.at(T, (t, a), 1.0); T /= T.sum(1, keepdims=True)
    ck = None
    if clamp:
        ck = np.array(list(clamp.keys())); cg = np.array(list(clamp.values())); T[ck] = 0; T[ck, cg] = 1.0
    for _ in range(n_iter):
        pi = T.sum(0) + 1e-6; pi /= pi.sum()
        cm = np.full((M, K, K), 1e-6); np.add.at(cm, (wi_idx, ci_idx, ai_idx), T[t]); cm /= cm.sum(2, keepdims=True)
        logcm = np.log(cm[w][np.arange(E)[:, None], ar[None, :], a[:, None]] + 1e-12)
        logT = np.log(pi)[None, :].repeat(N, 0); np.add.at(logT, t, logcm)
        logT -= logT.max(1, keepdims=True); T = np.exp(logT); T /= T.sum(1, keepdims=True)
        if ck is not None: T[ck] = 0; T[ck, cg] = 1.0
    return T


def scalar_correct(q, w, a, t, anchors, y, M, N, K):
    corr = np.ones(M); tot = np.full(M, 2.0); aset = set(int(x) for x in anchors)
    if aset:
        mk = np.array([int(x) in aset for x in t]); np.add.at(tot, w[mk], 1.0); np.add.at(corr, w[mk], (a[mk] == y[t[mk]]).astype(float))
    rel = np.clip(corr / tot, 1e-2, 1 - 1e-2); wgt = np.log(rel / (1 - rel))
    logp = np.log(q + 1e-12).copy(); np.add.at(logp, (t, a), wgt[w]); return logp.argmax(1)


def cfds_correct(q, w, a, t, anchors, y, M, N, K):
    cf = q.argmax(1)
    w2 = np.concatenate([w, np.full(N, M)]); a2 = np.concatenate([a, cf]); t2 = np.concatenate([t, np.arange(N)])
    clamp = {int(k): int(y[k]) for k in anchors}
    T = ds_em(w2, a2, t2, M + 1, N, K, clamp=clamp, init_T=q.copy())
    return T.argmax(1)


def oracle_worker(w, a, t, y, M, N, K):
    E = len(w); ar = np.arange(K); cm = np.full((M, K, K), 1e-6)
    np.add.at(cm, (w, y[t], a), 1.0); cm /= cm.sum(2, keepdims=True)
    pi = np.bincount(y, minlength=K).astype(float) + 1e-6; pi /= pi.sum()
    logcm = np.log(cm[w][np.arange(E)[:, None], ar[None, :], a[:, None]] + 1e-12)
    logT = np.log(pi)[None, :].repeat(N, 0); np.add.at(logT, t, logcm); return logT.argmax(1)


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
        for wi, ai in zip(w[em], a[em]): st[wi] += 1; sc[wi] += (ai == y[k])
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
        acc["oracle_worker|0"].append(eff_acc(oracle_worker(w, a, t, y, M, N, K), y, []))
        for B in budgets:
            if B == 0: continue
            Bc = min(B, N - 1); anch = voi(q, w, a, t, y, M, N, K, Bc)
            acc[f"scalar|{B}"].append(eff_acc(scalar_correct(q, w, a, t, anch, y, M, N, K), y, anch))
            acc[f"cfds|{B}"].append(eff_acc(cfds_correct(q, w, a, t, anch, y, M, N, K), y, anch))
        if (i + 1) % 20 == 0: print(f"  {i+1}/{Nw}", flush=True)
    summ = {k: float(np.mean(v)) for k, v in acc.items()}
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps(summ, indent=2))
    cf = summ["crowdfm|0"]; ow = summ["oracle_worker|0"]
    print(f"\n[{sk['mechanism_families']}] CrowdFM={cf:.4f}  oracle_worker={ow:.4f} (ceiling +{(ow-cf)*100:.1f}pp)")
    for B in budgets:
        if B == 0: continue
        print(f"  B={B:3d}: scalar={summ.get(f'scalar|{B}',cf):.4f} ({(summ.get(f'scalar|{B}',cf)-cf)*100:+.2f})  "
              f"cfds={summ.get(f'cfds|{B}',cf):.4f} ({(summ.get(f'cfds|{B}',cf)-cf)*100:+.2f})  "
              f"[cfds captures {(summ.get(f'cfds|{B}',cf)-cf)/max(ow-cf,1e-6)*100:.0f}% of ceiling]")


if __name__ == "__main__":
    main()
