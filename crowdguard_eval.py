"""CrowdGuard core experiment: ACTIVE anchor selection for expert-efficient correction.

Paper Q: when a crowd FM fails at deployment, use few expert anchors to actively identify
annotator reliability and correct the remaining tasks at minimal expert cost.

Fixed correction (starts = CrowdFM at B=0): log q_CrowdFM + Σ_i wgt_i·onehot(vote_ik),
wgt_i = anchor-estimated worker reliability log-odds. We compare SELECTION policies for
the anchor tasks under the same correction + same budget:
  random  : uniform
  entropy : highest CrowdFM predictive entropy (uncertainty sampling)
  voi     : worker-influence + coverage greedy (identify reliability of workers who
            influence many still-uncertain tasks; diminishing returns per worker) -- OURS
  oracle  : greedy on true remaining accuracy (upper bound on any selection policy)
Metric: effective accuracy (anchored counted correct) vs budget, + AURC (area under the
budget-error curve, lower better). Averaged over worlds x seeds.
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path
import dlwheel, numpy as np, torch
from cfm.data.crowdsi_simulator import CrowdSISimulator
from cfm.model.ResidualCFM import ResidualMechanismCFM


def seed_all(s): random.seed(s); np.random.seed(s % 2**32); torch.manual_seed(s)


def correct(q_cf, w, a, t, anchors, y, M, N, K, lam=1.0):
    """Anchor-conditioned correction on frozen CrowdFM (B=0 -> CrowdFM)."""
    corr = np.ones(M); tot = np.full(M, 2.0)
    aset = set(int(x) for x in anchors)
    if aset:
        mask = np.array([int(x) in aset for x in t])
        for wi, ai, ti in zip(w[mask], a[mask], t[mask]):
            tot[wi] += 1; corr[wi] += (ai == y[ti])
    rel = np.clip(corr / tot, 1e-2, 1 - 1e-2)
    wgt = np.log(rel / (1 - rel))
    logp = np.log(q_cf + 1e-12).copy()
    np.add.at(logp, (t, a), lam * wgt[w])
    return logp.argmax(1)


def eff_acc(pred, y, anchors):
    aset = set(int(x) for x in anchors)
    return float(np.mean([(k in aset) or (pred[k] == y[k]) for k in range(len(y))]))


def select_voi(w, t, N, B, cf_ent, task_workers):
    """Greedy: pick tasks whose workers are influential (high non-anchor degree) and not
    yet pinned by anchors (diminishing returns). Reliability-identification objective."""
    deg = np.bincount(w, minlength=N if False else w.max() + 1)
    Mw = w.max() + 1
    deg = np.bincount(w, minlength=Mw).astype(float)
    seen = np.zeros(Mw)  # anchor answers per worker so far
    chosen = []; avail = set(range(N))
    for _ in range(B):
        best_k, best_s = -1, -1e18
        for k in avail:
            ws = task_workers[k]
            if len(ws) == 0: continue
            s = np.sum(deg[ws] / (1.0 + seen[ws])) * (1.0 + 0.3 * cf_ent[k])
            if s > best_s: best_s, best_k = s, k
        if best_k < 0: break
        chosen.append(best_k); avail.discard(best_k)
        for i in task_workers[best_k]: seen[i] += 1
    return chosen


def select_voi_adaptive(q, w, a, t, y, M, N, K, B, task_workers):
    """ADAPTIVE VOI (ours): after each anchor, update worker reliability beliefs, then pick
    the task that resolves the most reliability-UNCERTAIN, INFLUENTIAL workers on currently
    RISKY tasks. Uses revealed anchor labels y_k to update (legitimate: we bought them)."""
    sc = np.ones(M); st = np.full(M, 2.0)  # Beta(1,1) correct/total per worker
    chosen = []; aset = set()
    E = len(w)
    # consensus-deviation prior: workers who disagree with CrowdFM's argmax are suspect
    cf_lab = q.argmax(1)
    dev_num = np.zeros(M); dev_den = np.zeros(M) + 1e-6
    np.add.at(dev_den, w, 1.0)
    np.add.at(dev_num, w, (a != cf_lab[t]).astype(float))
    dev = dev_num / dev_den  # disagreement rate with CrowdFM consensus
    for _ in range(B):
        rel = sc / st
        relc = np.clip(rel, 1e-2, 1 - 1e-2)
        wgt = np.log(relc / (1 - relc))
        logp = np.log(q + 1e-12).copy()
        np.add.at(logp, (t, a), wgt[w])
        logp -= logp.max(1, keepdims=True)
        qc = np.exp(logp); qc /= qc.sum(1, keepdims=True)
        risk = 1.0 - qc.max(1)  # [N]
        anchored_edge = np.array([int(tk) in aset for tk in t]) if aset else np.zeros(E, bool)
        infl = np.zeros(M)
        np.add.at(infl, w[~anchored_edge], risk[t[~anchored_edge]])  # worker influence on risky non-anchored tasks
        rel_var = relc * (1 - relc) / (st + 1.0)  # Beta variance ~ uncertainty
        # value of resolving worker i: uncertain OR consensus-deviating (suspect), and influential
        wval = (rel_var + 0.5 * dev) * infl
        # task value: worker-resolution value, boosted by the task's own risk (double benefit:
        # anchoring a likely-wrong task fixes it for free AND informs correction)
        tval = np.full(N, -1.0)
        for k in range(N):
            if k in aset: continue
            ws = task_workers[k]
            tval[k] = (wval[ws].sum() * (0.5 + risk[k])) if len(ws) else -1.0
        k = int(np.argmax(tval))
        if tval[k] < 0: break
        chosen.append(k); aset.add(k)
        em = (t == k)
        for wi, ai in zip(w[em], a[em]):
            st[wi] += 1; sc[wi] += (ai == y[k])
    return chosen


def select_oracle(q, w, a, t, y, M, N, K, B, task_workers):
    """Greedy on true resulting accuracy (upper bound). Expensive; used at modest B/worlds."""
    chosen = []; avail = list(range(N))
    for _ in range(B):
        best_k, best_acc = -1, -1.0
        for k in avail:
            cand = chosen + [k]
            acc = eff_acc(correct(q, w, a, t, cand, y, M, N, K), y, cand)
            if acc > best_acc: best_acc, best_k = acc, k
        if best_k < 0: break
        chosen.append(best_k); avail.remove(best_k)
    return chosen


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
    Nw = int(cfg.get("num_worlds", 80)); base = int(cfg.get("base_seed", 66000))
    budgets = [int(b) for b in cfg.get("budgets", [0, 5, 10, 20, 50])]
    do_oracle = bool(cfg.get("do_oracle", True)); oracle_maxB = int(cfg.get("oracle_maxB", 20))

    acc = defaultdict(list)
    for i in range(Nw):
        seed_all(base + i)
        data = sim.generate(); data.to(device)
        y = data.task_y.cpu().numpy()
        M, N, K = data.num_worker, data.num_task, data.num_option
        w = data.triple[0].cpu().numpy(); a = data.triple[1].cpu().numpy(); t = data.triple[2].cpu().numpy()
        q = torch.softmax(m(data, sample_mechanism=False)["hat_task_option_base"], -1).cpu().numpy()
        cf_ent = -(q * np.log(q + 1e-12)).sum(1)
        cf_pred = q.argmax(1)
        acc["crowdfm|0"].append(eff_acc(cf_pred, y, []))
        task_workers = [w[t == k] for k in range(N)]
        rng = np.random.default_rng(base + i)
        ent_order = np.argsort(-cf_ent)
        for B in budgets:
            if B == 0: continue
            Bc = min(B, N - 1)
            sel = {
                "random": list(rng.choice(N, size=Bc, replace=False)),
                "entropy": list(ent_order[:Bc]),
                "voi_static": select_voi(w, t, N, Bc, cf_ent, task_workers),
                "voi": select_voi_adaptive(q, w, a, t, y, M, N, K, Bc, task_workers),
            }
            if do_oracle and Bc <= oracle_maxB:
                sel["oracle"] = select_oracle(q, w, a, t, y, M, N, K, Bc, task_workers)
            for name, anch in sel.items():
                acc[f"{name}|{B}"].append(eff_acc(correct(q, w, a, t, anch, y, M, N, K), y, anch))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{Nw} crowdfm={np.mean(acc['crowdfm|0']):.3f}", flush=True)

    summ = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)} for k, v in acc.items()}
    cf = summ["crowdfm|0"]["mean"]
    def aurc(strategy):  # area under error-vs-budget (normalized), lower=better
        errs = [1 - cf] + [1 - summ.get(f"{strategy}|{B}", {"mean": cf})["mean"] for B in budgets if B > 0]
        return float(np.mean(errs))
    res = {"families": sim_kw["mechanism_families"], "n_worlds": Nw, "budgets": budgets,
           "crowdfm": cf, "acc": summ, "aurc": {s: aurc(s) for s in ["random", "entropy", "voi", "oracle"]}}
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps(res, indent=2))
    print(f"\n[{sim_kw['mechanism_families']}] worlds={Nw}  CrowdFM={cf:.4f}")
    print("  B    random   entropy   voi(OURS)  oracle")
    for B in budgets:
        if B == 0: continue
        def g(s): return summ.get(f"{s}|{B}", {}).get("mean", float("nan"))
        print(f"  {B:3d}  {g('random'):.4f}  {g('entropy'):.4f}   {g('voi'):.4f}    {g('oracle'):.4f}"
              f"   [voi-rand {(g('voi')-g('random'))*100:+.2f} | voi-ent {(g('voi')-g('entropy'))*100:+.2f}]")
    print("  AURC(lower=better): " + "  ".join(f"{s}={res['aurc'][s]:.4f}" for s in ["random", "entropy", "voi", "oracle"]))


if __name__ == "__main__":
    main()
