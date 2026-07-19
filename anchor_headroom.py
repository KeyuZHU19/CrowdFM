"""CrowdGuard Anchor Headroom Test.

New information source: a few TRUSTED expert anchors (gold Y on B tasks) -- unlike every
prior experiment, this injects truth information not derivable from the annotation graph
G. Question: do B anchors create real aggregation headroom on the REMAINING tasks over
frozen CrowdFM?  We DO NOT build the full method yet; we measure ceilings first.

Metric: effective accuracy over all N tasks, counting the B anchored tasks as correct
(we bought them), as a function of budget B. Baseline CrowdFM at B=0.

Methods:
  crowdfm         frozen CrowdFM (hat_task_option_base)          [baseline, B=0]
  mv, ds          majority vote / unsupervised Dawid-Skene       [baselines]
  oracle_worker   per-dataset worker-confusion model from ALL gold [CEILING: best worker model]
  anchor_ds       semi-supervised DS, anchors clamped to gold    [random / entropy selection]
  reweight        anchor-estimated worker reliability weighted vote
Go: at B/N <= 5%, some method >= CrowdFM + 2pp AND oracle_worker shows clear headroom.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
import dlwheel, numpy as np, torch
from cfm.data.crowdsi_simulator import CrowdSISimulator
from cfm.model.ResidualCFM import ResidualMechanismCFM


def seed_all(s): random.seed(s); np.random.seed(s % 2**32); torch.manual_seed(s)


def ds_em(w, a, t, M, N, K, n_iter=30, clamp=None):
    """Vectorised Dawid-Skene EM. clamp: dict task->gold (semi-supervised). Returns T [N,K]."""
    E = len(w); ar = np.arange(K)
    wi_idx = np.repeat(w[:, None], K, 1); ci_idx = np.repeat(ar[None, :], E, 0); ai_idx = np.repeat(a[:, None], K, 1)
    T = np.full((N, K), 1e-6)
    np.add.at(T, (t, a), 1.0); T /= T.sum(1, keepdims=True)
    ck = None
    if clamp:
        ck = np.array(list(clamp.keys())); cg = np.array(list(clamp.values()))
        T[ck] = 0; T[ck, cg] = 1.0
    for _ in range(n_iter):
        pi = T.sum(0) + 1e-6; pi /= pi.sum()
        cm = np.full((M, K, K), 1e-6)
        np.add.at(cm, (wi_idx, ci_idx, ai_idx), T[t])       # cm[w,:,a] += T[t]
        cm /= cm.sum(2, keepdims=True)
        logcm = np.log(cm[w][np.arange(E)[:, None], ar[None, :], a[:, None]] + 1e-12)  # [E,K]
        logT = np.log(pi)[None, :].repeat(N, 0)
        np.add.at(logT, t, logcm)
        logT -= logT.max(1, keepdims=True)
        T = np.exp(logT); T /= T.sum(1, keepdims=True)
        if ck is not None:
            T[ck] = 0; T[ck, cg] = 1.0
    return T


def oracle_worker(w, a, t, y, M, N, K):
    """Worker confusion from ALL gold, then aggregate (upper bound worker model)."""
    E = len(w); ar = np.arange(K)
    cm = np.full((M, K, K), 1e-6)
    np.add.at(cm, (w, y[t], a), 1.0); cm /= cm.sum(2, keepdims=True)
    pi = np.bincount(y, minlength=K).astype(float) + 1e-6; pi /= pi.sum()
    logcm = np.log(cm[w][np.arange(E)[:, None], ar[None, :], a[:, None]] + 1e-12)
    logT = np.log(pi)[None, :].repeat(N, 0)
    np.add.at(logT, t, logcm)
    return logT.argmax(1)


def reweight_vote(w, a, t, anchors, y, M, N, K):
    """Worker reliability from anchors -> weighted vote on all tasks."""
    corr = np.full(M, 1.0); tot = np.full(M, 2.0)  # Beta(1,1)-ish prior => 0.5
    aset = set(anchors)
    for wi, ai, ti in zip(w, a, t):
        if ti in aset:
            tot[wi] += 1; corr[wi] += (ai == y[ti])
    rel = corr / tot  # in (0,1)
    wgt = np.log(np.clip(rel, 1e-3, 1-1e-3) / np.clip(1-rel, 1e-3, 1-1e-3))
    wgt = np.maximum(wgt, 0)  # ignore anti-correlated unless strong
    V = np.zeros((N, K))
    for wi, ai, ti in zip(w, a, t): V[ti, ai] += wgt[wi] + 1e-3
    return V.argmax(1)


def cf_anchor_correct(q_cf, w, a, t, anchors, y, M, N, K, lam=1.0):
    """CrowdGuard-style correction that STARTS from CrowdFM (B=0 => CrowdFM).
    log-posterior_k = log q_cf[k] + lam * sum_i weight_i * onehot(vote_ik),
    weight_i = log-odds of worker i's anchor-estimated reliability (can be negative,
    so systematically-wrong/coalition workers get their votes SUBTRACTED)."""
    corr = np.full(M, 1.0); tot = np.full(M, 2.0)  # Beta(1,1) => rel 0.5 => weight 0 at B=0
    aset = set(int(x) for x in anchors)
    for wi, ai, ti in zip(w, a, t):
        if int(ti) in aset:
            tot[wi] += 1; corr[wi] += (ai == y[ti])
    rel = np.clip(corr / tot, 1e-2, 1 - 1e-2)
    wgt = np.log(rel / (1 - rel))  # 0 at prior, +/- with evidence
    logp = np.log(q_cf + 1e-12).copy()
    add = np.zeros((N, K))
    np.add.at(add, (t, a), wgt[w])
    logp += lam * add
    return logp.argmax(1)


def eff_acc(pred, y, anchors):
    """Accuracy over all tasks; anchored tasks count as correct (bought)."""
    N = len(y); aset = set(anchors)
    c = sum(1 for k in range(N) if (k in aset) or (pred[k] == y[k]))
    return c / N


@torch.no_grad()
def main():
    cfg = dlwheel.setup(); device = cfg.device
    m = ResidualMechanismCFM(**dict(cfg.model.to_dict())).to(device)
    sd = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=True); m.eval()
    sim_kw = dict(cfg.simulator.to_dict())
    fam = cfg.get("families", None)
    if fam: sim_kw["mechanism_families"] = [f.strip() for f in str(fam).split(",")]
    sim = CrowdSISimulator(**sim_kw)
    N_worlds = int(cfg.get("num_worlds", 120)); base = int(cfg.get("base_seed", 66000))
    budgets = [int(b) for b in cfg.get("budgets", [0, 5, 10, 20, 50])]

    from collections import defaultdict
    acc = defaultdict(list)
    for i in range(N_worlds):
        seed_all(base + i)
        data = sim.generate(); data.to(device)
        y = data.task_y.cpu().numpy()
        M, N, K = data.num_worker, data.num_task, data.num_option
        w = data.triple[0].cpu().numpy(); a = data.triple[1].cpu().numpy(); t = data.triple[2].cpu().numpy()
        out = m(data, sample_mechanism=False)
        q = torch.softmax(out["hat_task_option_base"], -1).cpu().numpy()
        cf_pred = q.argmax(1); cf_ent = -(q * np.log(q + 1e-12)).sum(1)
        # baselines (B=0)
        acc["crowdfm|0"].append(eff_acc(cf_pred, y, []))
        acc["mv|0"].append(eff_acc(ds_em(w, a, t, M, N, K, n_iter=0), y, []) if False else eff_acc(
            np.array([np.bincount(a[t == k], minlength=K).argmax() if (t == k).any() else 0 for k in range(N)]), y, []))
        acc["ds|0"].append(eff_acc(ds_em(w, a, t, M, N, K).argmax(1), y, []))
        acc["oracle_worker|0"].append(eff_acc(oracle_worker(w, a, t, y, M, N, K), y, []))
        rng = np.random.default_rng(base + i)
        ent_order = np.argsort(-cf_ent)
        for B in budgets:
            if B == 0: continue
            B = min(B, N - 1)
            rnd = list(rng.choice(N, size=B, replace=False))
            ent = list(ent_order[:B])
            clamp_r = {k: int(y[k]) for k in rnd}; clamp_e = {k: int(y[k]) for k in ent}
            acc[f"anchor_ds_rand|{B}"].append(eff_acc(ds_em(w, a, t, M, N, K, clamp=clamp_r).argmax(1), y, rnd))
            acc[f"anchor_ds_ent|{B}"].append(eff_acc(ds_em(w, a, t, M, N, K, clamp=clamp_e).argmax(1), y, ent))
            acc[f"reweight_rand|{B}"].append(eff_acc(reweight_vote(w, a, t, rnd, y, M, N, K), y, rnd))
            # CrowdGuard-style: correction on top of CrowdFM (B=0 => CrowdFM)
            acc[f"cf_correct_rand|{B}"].append(eff_acc(cf_anchor_correct(q, w, a, t, rnd, y, M, N, K), y, rnd))
            acc[f"cf_correct_ent|{B}"].append(eff_acc(cf_anchor_correct(q, w, a, t, ent, y, M, N, K), y, ent))
            # crowdfm + fill anchors (trivial: just count anchors correct, keep CrowdFM elsewhere)
            acc[f"crowdfm_anchored|{B}"].append(eff_acc(cf_pred, y, rnd))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{N_worlds} crowdfm={np.mean(acc['crowdfm|0']):.3f} oracle_worker={np.mean(acc['oracle_worker|0']):.3f}", flush=True)

    summ = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in acc.items()}
    res = {"families": sim_kw["mechanism_families"], "n_worlds": N_worlds, "budgets": budgets, "acc": summ}
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps(res, indent=2))
    cf = summ["crowdfm|0"]["mean"]; ow = summ["oracle_worker|0"]["mean"]
    print(f"\n[{sim_kw['mechanism_families']}] worlds={N_worlds}")
    print(f"  CrowdFM={cf:.4f}  MV={summ['mv|0']['mean']:.4f}  DS={summ['ds|0']['mean']:.4f}  "
          f"ORACLE_WORKER={ow:.4f}  (ceiling headroom = {(ow-cf)*100:+.2f}pp)")
    print("  budget-accuracy (effective, anchored=correct):")
    for B in budgets:
        if B == 0: continue
        def g(k): return summ.get(f"{k}|{B}", {}).get("mean", float("nan"))
        print(f"  B={B:3d}: cf_correct_rand={g('cf_correct_rand'):.4f} cf_correct_ent={g('cf_correct_ent'):.4f}  "
              f"anchor_ds_ent={g('anchor_ds_ent'):.4f} reweight={g('reweight_rand'):.4f}  "
              f"[cf_correct(clean) rand {g('cf_correct_rand')-cf:+.3f} / ent {g('cf_correct_ent')-cf:+.3f}]")


if __name__ == "__main__":
    main()
