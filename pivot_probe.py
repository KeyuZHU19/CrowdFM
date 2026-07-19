"""PIVOT probe: can response-predictive evidence tell us WHEN CrowdFM is wrong?

New research question (CrowdSI accuracy line is falsified/stopped). We DON'T try to beat
CrowdFM accuracy. We ask whether per-task response-predictive signals identify the tasks
CrowdFM gets wrong -- enabling selective / risk-aware aggregation (abstention), which the
verified positives (emission head predicts annotations; e-value fires under mismatch)
could actually support.

Per task we score error-detection signals and compare their risk-coverage against
CrowdFM's own confidence (the strong standard baseline):
  conf      : 1 - max_c q_k(c)               (CrowdFM uncertainty; BASELINE)
  vote_ent  : entropy of the empirical votes  (simple baseline)
  resp_nll  : per-task mean marginal response NLL under the model (NEW response signal)
  margin    : 1 - (top1 - top2 posterior)     (baseline)
Metrics: AUROC(signal, is_error) and AURC (area under selective-risk vs coverage).
Reported on a hard regime (coalition/adversarial) where CrowdFM has real error to catch.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
import dlwheel, numpy as np, torch
from cfm.data.crowdsi_simulator import CrowdSISimulator
from cfm.model.ResidualCFM import ResidualMechanismCFM


def seed_all(s): random.seed(s); np.random.seed(s % 2**32); torch.manual_seed(s)

def auroc(sig, lab):
    sig = np.asarray(sig, float); lab = np.asarray(lab, int)
    p = int(lab.sum()); n = len(lab) - p
    if p == 0 or n == 0: return float("nan")
    order = np.argsort(sig, kind="mergesort"); ranks = np.empty(len(sig)); ranks[order] = np.arange(1, len(sig)+1)
    # average ties
    s = sig[order]; i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and s[j+1] == s[i]: j += 1
        if j > i:
            a = (ranks[order[i]]+ranks[order[j]])/2
            for k in range(i, j+1): ranks[order[k]] = a
        i = j+1
    return float((ranks[lab==1].sum() - p*(p+1)/2) / (p*n))

def aurc(sig, err):
    """Area under risk-coverage: abstain highest-signal first. Lower is better."""
    sig = np.asarray(sig, float); err = np.asarray(err, float)
    order = np.argsort(sig)  # keep low-signal (confident) first
    err = err[order]; cum = np.cumsum(err) / np.arange(1, len(err)+1)
    return float(cum.mean())


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
    N = int(cfg.get("num_worlds", 150)); base = int(cfg.get("base_seed", 55000))

    rows = {"err": [], "conf": [], "vote_ent": [], "resp_nll": [], "margin": []}
    accs = []
    for i in range(N):
        seed_all(base + i)
        data = sim.generate(); data.to(device)
        y = data.task_y.to(device).long()
        out = m(data, sample_mechanism=False)
        q = torch.softmax(out["hat_task_option_base"], -1)  # OFFICIAL CrowdFM posterior
        pred = q.argmax(-1)
        correct = (pred == y).float()
        accs.append(float(correct.mean()))
        conf = q.max(-1).values
        top2 = q.topk(2, dim=-1).values
        margin = 1 - (top2[:, 0] - top2[:, 1])
        # per-task response signals: marginal predictive NLL of each task's own annotations
        w = data.triple[0].long(); a = data.triple[1].long(); t = data.triple[2].long()
        allw = w; allt = t
        emis = m(data, query_workers=allw, query_tasks=allt, sample_mechanism=False)
        log_em = torch.log_softmax(emis["hat_annotation_given_truth"], -1)  # [E,K,K]
        qk_edge = q[allt]  # [E,K]
        pe = log_em.exp()  # [E,K(truth),K(report)]
        marg = torch.einsum("ec,ecr->er", qk_edge, pe)
        marg = (marg / marg.sum(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
        edge_nll = -torch.log(marg.gather(1, a[:, None]).squeeze(1))  # [E]
        K = data.num_option
        # aggregate per task
        for k in range(data.num_task):
            mask = (t == k)
            if not bool(mask.any()): continue
            votes = a[mask]
            vc = torch.bincount(votes, minlength=K).float(); vp = vc / vc.sum()
            vent = float(-(vp * (vp + 1e-12).log()).sum())
            rows["err"].append(int(correct[k].item() == 0.0))
            rows["conf"].append(float(1 - conf[k].item()))
            rows["margin"].append(float(margin[k].item()))
            rows["vote_ent"].append(vent)
            rows["resp_nll"].append(float(edge_nll[mask].mean().item()))
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{N} acc={np.mean(accs):.3f} err_rate={np.mean(rows['err']):.3f}", flush=True)

    err = rows["err"]
    res = {"n_worlds": len(accs), "n_tasks": len(err), "families": sim_kw["mechanism_families"],
           "crowdfm_acc": float(np.mean(accs)), "err_rate": float(np.mean(err))}
    for sig in ["conf", "margin", "vote_ent", "resp_nll"]:
        res[sig] = {"auroc_err": auroc(rows[sig], err), "aurc": aurc(rows[sig], err)}
    # combined conf+resp via simple standardized sum (no fit) and normalized product
    z = lambda x: (np.array(x) - np.mean(x)) / (np.std(x) + 1e-9)
    combo = z(rows["conf"]) + z(rows["resp_nll"])
    res["conf+resp"] = {"auroc_err": auroc(combo, err), "aurc": aurc(combo, err)}
    # LEARNED combination (2-fold CV logistic): does resp add ORTHOGONAL value over conf?
    try:
        from sklearn.linear_model import LogisticRegression
        Xa = np.array([z(rows["conf"]), z(rows["margin"]), z(rows["vote_ent"]), z(rows["resp_nll"])]).T
        Xc = np.array([z(rows["conf"]), z(rows["margin"]), z(rows["vote_ent"])]).T  # no resp
        ya = np.array(err)
        half = len(ya) // 2
        def cv_auroc(X):
            aucs = []
            for tr, te in [(slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))]:
                lr = LogisticRegression(max_iter=500).fit(X[tr], ya[tr])
                aucs.append(auroc(lr.predict_proba(X[te])[:, 1], ya[te]))
            return float(np.mean(aucs))
        res["learned_all"] = {"auroc_err": cv_auroc(Xa)}
        res["learned_no_resp"] = {"auroc_err": cv_auroc(Xc)}
    except Exception as e:
        res["learned_all"] = {"auroc_err": float("nan"), "err": str(e)}
        res["learned_no_resp"] = {"auroc_err": float("nan")}
    res["random_aurc"] = float(np.mean(err))  # abstain-at-random baseline risk
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).write_text(json.dumps(res, indent=2))
    print(f"\n[{sim_kw['mechanism_families']}] worlds={res['n_worlds']} tasks={res['n_tasks']} "
          f"CrowdFM acc={res['crowdfm_acc']:.3f} err={res['err_rate']:.3f}")
    print("  signal        AUROC(err)   AURC(lower=better)")
    for sig in ["conf", "margin", "vote_ent", "resp_nll", "conf+resp"]:
        print(f"  {sig:12s}  {res[sig]['auroc_err']:.4f}       {res[sig]['aurc']:.4f}")
    print(f"  random AURC = {res['random_aurc']:.4f}")


if __name__ == "__main__":
    main()
