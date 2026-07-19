"""Phase-1 in-prior predictive validation harness for CrowdSI-FM.

Generates held-out synthetic worlds from a (possibly family-restricted) simulator,
loads a trained CrowdSIFM checkpoint, and reports the paper's gate metrics:

  * CrowdFM accuracy (frozen backbone truth head)
  * CrowdSI zero-shot accuracy (mechanism-conditioned truth head at mu_0)
  * Majority-vote accuracy (aggregation baseline)
  * Held-out response predictive NLL (model marginal) vs option-freq / worker-freq
  * Assignment AUROC/AUPRC vs a density-only baseline
  * Mechanism posterior diagnostics (primitive-weight entropy, cross-view KL)

Usage:
  python eval_synth.py config=config/eval_synth.yaml checkpoint_path=... [families=irt,class_bias]
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import dlwheel
import numpy as np
import torch

from cfm.audit.pipeline import masked_data
from cfm.audit.predictive_split import make_annotation_audit_split
from cfm.data.crowdsi_simulator import CrowdSISimulator
from cfm.model.CrowdSIFM import CrowdSIFM
from cfm.si.likelihood import joint_annotation_log_likelihood


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def auroc_direct(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _assign_avg_ranks(scores, ranks)
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _assign_avg_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def auprc_direct(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # AP = sum over thresholds of (recall_i - recall_{i-1}) * precision_i
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(((recall - recall_prev) * precision).sum())


def majority_vote_accuracy(data) -> float:
    workers = data.triple[0].cpu().numpy()
    answers = data.triple[1].cpu().numpy()
    tasks = data.triple[2].cpu().numpy()
    y = data.task_y.cpu().numpy()
    correct = 0
    total = 0
    for k in range(data.num_task):
        a = answers[tasks == k]
        if len(a) == 0:
            continue
        pred = np.bincount(a, minlength=data.num_option).argmax()
        correct += int(pred == y[k])
        total += 1
    return correct / max(1, total)


@torch.no_grad()
def evaluate_world(model, data, device, seed):
    """Return per-world metric dict."""
    data.to(device)
    out = model(data, sample_mechanism=False)
    y = data.task_y.to(device).long()
    valid = y >= 0
    crowdfm_pred = out["hat_task_option_base"].argmax(-1)
    zshot_pred = out["hat_task_option"].argmax(-1)
    crowdfm_acc = (crowdfm_pred[valid] == y[valid]).float().mean().item()
    zshot_acc = (zshot_pred[valid] == y[valid]).float().mean().item()
    mv_acc = majority_vote_accuracy(data)

    # primitive weights entropy (collapse diagnostic)
    w = out["mechanism_weights"].cpu().numpy()
    prim_entropy = float(-(w * np.log(w + 1e-12)).sum())
    mech_logvar_mean = float(out["mechanism_log_variance"].mean().item())

    # cross-view posterior consistency: two independent masked context views
    split1 = make_annotation_audit_split(
        data.triple, data.num_task, num_worker=data.num_worker,
        audit_task_fraction=1.0, context_fraction=0.5,
        min_context_workers=1, min_audit_workers=2, min_worker_context_edges=0, seed=seed)
    split2 = make_annotation_audit_split(
        data.triple, data.num_task, num_worker=data.num_worker,
        audit_task_fraction=1.0, context_fraction=0.5,
        min_context_workers=1, min_audit_workers=2, min_worker_context_edges=0, seed=seed + 7)
    c1 = masked_data(data, split1.context_edge_mask)
    c2 = masked_data(data, split2.context_edge_mask)
    o1 = model(c1, sample_mechanism=False)
    o2 = model(c2, sample_mechanism=False)
    view_l2 = float((o1["mechanism_mean"] - o2["mechanism_mean"]).pow(2).sum().sqrt().item())

    # response predictive NLL on held-out audit edges, using context o1
    audit_idx = split1.audit_edge_indices
    qw = data.triple[0, audit_idx].long()
    qt = data.triple[2, audit_idx].long()
    qa = data.triple[1, audit_idx].long()
    ctx_out = model(c1, query_workers=qw, query_tasks=qt, sample_mechanism=False)
    # joint per-task NLL (model's proper object), mean per task
    joint_ll = joint_annotation_log_likelihood(
        ctx_out["hat_task_option"], ctx_out["hat_annotation_given_truth"],
        qa, qt, reduction="mean").item()
    model_joint_nll = -joint_ll

    # marginal per-edge predictive for baseline comparison
    log_task = torch.log_softmax(ctx_out["hat_task_option"], -1)  # [N,K]
    log_em = torch.log_softmax(ctx_out["hat_annotation_given_truth"], -1)  # [Q,K,K]
    qt_dev = qt.to(device)
    # marginal P(A_e = a) = sum_c q_k(c) P_e(a|c)
    qk = log_task[qt_dev].exp()  # [Q,K]
    pe = log_em.exp()  # [Q,K,K] truth,report
    marg = torch.einsum("qc,qcr->qr", qk, pe)  # [Q,K]
    marg = marg / marg.sum(-1, keepdim=True).clamp_min(1e-12)
    edge_nll_model = -torch.log(marg.gather(1, qa.to(device)[:, None]).squeeze(1).clamp_min(1e-12)).mean().item()

    # option-frequency baseline (from context answers)
    ctx_answers = data.triple[1, torch.nonzero(split1.context_edge_mask).flatten()].cpu().numpy()
    freq = np.bincount(ctx_answers, minlength=data.num_option).astype(np.float64) + 0.5
    freq = freq / freq.sum()
    qa_np = qa.cpu().numpy()
    edge_nll_optfreq = float(-np.log(freq[qa_np] + 1e-12).mean())

    # worker-frequency baseline: per-worker answer distribution from context
    ctx_idx = torch.nonzero(split1.context_edge_mask).flatten().cpu().numpy()
    ctx_w = data.triple[0].cpu().numpy()[ctx_idx]
    ctx_a = data.triple[1].cpu().numpy()[ctx_idx]
    worker_counts = np.full((data.num_worker, data.num_option), 0.5)
    for wi, ai in zip(ctx_w, ctx_a):
        worker_counts[wi, ai] += 1
    worker_probs = worker_counts / worker_counts.sum(1, keepdims=True)
    qw_np = qw.cpu().numpy()
    edge_nll_workerfreq = float(-np.log(worker_probs[qw_np, qa_np] + 1e-12).mean())

    # assignment AUROC: positives = observed edges, negatives = sampled unobserved pairs
    pos_w = data.triple[0].long()
    pos_t = data.triple[2].long()
    n_pos = pos_w.numel()
    observed = torch.zeros(data.num_worker * data.num_task, dtype=torch.bool)
    observed[pos_w.cpu() * data.num_task + pos_t.cpu()] = True
    cand = torch.nonzero(~observed).flatten()
    g = torch.Generator().manual_seed(seed + 3)
    sel = cand[torch.randperm(cand.numel(), generator=g)[:n_pos]]
    neg_w = torch.div(sel, data.num_task, rounding_mode="floor")
    neg_t = sel.remainder(data.num_task)
    all_w = torch.cat([pos_w.cpu(), neg_w]).to(device)
    all_t = torch.cat([pos_t.cpu(), neg_t]).to(device)
    aout = model(data, query_workers=all_w, query_tasks=all_t, sample_mechanism=False)
    assign_scores = aout["hat_assignment_logit"].cpu().numpy()
    assign_labels = np.concatenate([np.ones(n_pos), np.zeros(len(neg_w))])
    a_auroc = auroc_direct(assign_scores, assign_labels)
    a_auprc = auprc_direct(assign_scores, assign_labels)
    # density baseline: task degree as score (popular tasks more likely assigned) -> trivial
    task_deg = np.bincount(pos_t.cpu().numpy(), minlength=data.num_task).astype(np.float64)
    dens_scores = np.concatenate([task_deg[pos_t.cpu().numpy()], task_deg[neg_t.cpu().numpy()]])
    dens_auroc = auroc_direct(dens_scores, assign_labels)

    return {
        "family": getattr(data, "mechanism_family_name", "unknown"),
        "num_worker": int(data.num_worker), "num_task": int(data.num_task),
        "num_option": int(data.num_option), "num_edges": int(data.triple.shape[1]),
        "crowdfm_acc": crowdfm_acc, "zeroshot_acc": zshot_acc, "mv_acc": mv_acc,
        "zeroshot_minus_crowdfm": zshot_acc - crowdfm_acc,
        "model_joint_nll": model_joint_nll,
        "edge_nll_model": edge_nll_model,
        "edge_nll_optfreq": edge_nll_optfreq,
        "edge_nll_workerfreq": edge_nll_workerfreq,
        "assign_auroc": a_auroc, "assign_auprc": a_auprc, "assign_dens_auroc": dens_auroc,
        "prim_entropy": prim_entropy, "mech_logvar_mean": mech_logvar_mean,
        "view_l2": view_l2,
    }


def _agg(records, key):
    vals = np.array([r[key] for r in records if not (isinstance(r[key], float) and math.isnan(r[key]))], dtype=np.float64)
    if len(vals) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "ci95": float("nan"), "n": 0}
    return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
            "ci95": float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0),
            "n": len(vals)}


def main():
    cfg = dlwheel.setup()
    device = cfg.device
    model = CrowdSIFM(**cfg.model.to_dict()).to(device)
    ck = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    model.load_state_dict(sd, strict=True)
    model.eval()

    fam = cfg.get("families", None)
    sim_kwargs = dict(cfg.simulator.to_dict())
    if fam:
        sim_kwargs["mechanism_families"] = [f.strip() for f in str(fam).split(",")]
    simulator = CrowdSISimulator(**sim_kwargs)

    num_worlds = int(cfg.get("num_worlds", 200))
    base_seed = int(cfg.get("base_seed", 42000))
    records = []
    for i in range(num_worlds):
        _seed_all(base_seed + i)
        data = simulator.generate()
        try:
            rec = evaluate_world(model, data, device, base_seed + i)
        except Exception as e:
            print(f"world {i} failed: {e}")
            continue
        records.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{num_worlds} zshot={_agg(records,'zeroshot_acc')['mean']:.4f} "
                  f"crowdfm={_agg(records,'crowdfm_acc')['mean']:.4f}")

    keys = ["crowdfm_acc", "zeroshot_acc", "mv_acc", "zeroshot_minus_crowdfm",
            "model_joint_nll", "edge_nll_model", "edge_nll_optfreq", "edge_nll_workerfreq",
            "assign_auroc", "assign_auprc", "assign_dens_auroc",
            "prim_entropy", "mech_logvar_mean", "view_l2"]
    summary = {k: _agg(records, k) for k in keys}
    families = sim_kwargs["mechanism_families"]
    by_family = {}
    for f in families:
        fr = [r for r in records if r["family"] == f]
        if fr:
            by_family[f] = {k: _agg(fr, k) for k in ["crowdfm_acc", "zeroshot_acc", "mv_acc", "assign_auroc"]}

    result = {"checkpoint": cfg.checkpoint_path, "families": families,
              "num_worlds": len(records), "base_seed": base_seed,
              "summary": summary, "by_family": by_family, "records": records}
    out = Path(cfg.get("output_path", "log/eval_synth.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("\n=== SUMMARY ===")
    for k in keys:
        s = summary[k]
        print(f"  {k:24s} {s['mean']:+.4f} +/- {s['ci95']:.4f} (n={s['n']})")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
