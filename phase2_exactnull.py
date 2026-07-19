"""Phase-2 exact-null e-value calibration for CrowdSI-FM (claim C4).

Under the fixed plug-in null H0: Z = mu_0, the cross-fitted e-value must satisfy
E_0[E] <= 1 and hence Pr_0(E >= 1/alpha) <= alpha.

We construct data for which H0 is *exactly true* by:
  1. drawing a world structure (mask + dimensions) from the simulator;
  2. computing mu_0 from the context graph (the context answers are kept fixed);
  3. regenerating the audit answers (fold A + fold B) from the model's own
     predictive law at mu_0 -- one shared latent truth per task, then per-edge
     emission -- which is precisely the null generating process;
  4. running the standard crossfit_e_value with the *same* split seed.

If the empirical Pr(E >= 1/alpha) is at or below alpha, the finite-sample gate
is calibrated and the theorem's implementation is correct.
"""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import dlwheel
import numpy as np
import torch

from cfm.audit.pipeline import masked_data
from cfm.data.crowdsi_simulator import CrowdSISimulator
from cfm.model.CrowdSIFM import CrowdSIFM
from cfm.si.adaptation import AdaptationConfig, crossfit_e_value
from cfm.si.split import make_task_crossfit_split


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


@torch.no_grad()
def regenerate_audit_answers_from_null(model, data, split_seed, device, gen_seed,
                                       ctx_cfg):
    """Replace audit answers with samples from the model at mu_0. Returns new data."""
    split = make_task_crossfit_split(
        data.triple, data.num_task, num_worker=data.num_worker,
        audit_task_fraction=ctx_cfg["audit_task_fraction"],
        context_fraction=ctx_cfg["context_fraction"],
        min_context_workers=ctx_cfg["min_context_workers"],
        min_audit_workers=ctx_cfg["min_audit_workers"],
        min_worker_context_edges=ctx_cfg["min_worker_context_edges"],
        seed=split_seed)
    context_data = masked_data(data, split.context_edge_mask)
    audit_mask = split.audit_edge_mask
    audit_idx = torch.nonzero(audit_mask).flatten()
    qw = data.triple[0, audit_idx].long()
    qt = data.triple[2, audit_idx].long()

    out = model(context_data, query_workers=qw, query_tasks=qt,
                mechanism_latent=None, sample_mechanism=False)
    mu0 = out["mechanism_mean"]
    # recompute at explicit mu0 to be safe (identical, but explicit)
    out = model(context_data, query_workers=qw, query_tasks=qt,
                mechanism_latent=mu0, sample_mechanism=False)
    task_logits = out["hat_task_option"]          # [N,K]
    emission_logits = out["hat_annotation_given_truth"]  # [Q,K,K] truth,report
    q_truth = torch.softmax(task_logits, dim=-1)
    p_emit = torch.softmax(emission_logits, dim=-1)

    dev = q_truth.device
    g = torch.Generator(device=dev.type)
    g.manual_seed(gen_seed)

    # one shared truth per audit task, sampled from q_theta(Y_k | G_C, mu_0)
    audit_tasks = torch.unique(qt)                      # [T]
    truth_per_task = torch.multinomial(
        q_truth[audit_tasks], 1, generator=g).squeeze(1)  # [T]
    # map each audit edge's task -> its sampled truth (vectorised)
    max_task = int(audit_tasks.max().item()) + 1
    task_to_truth = torch.full((max_task,), -1, dtype=torch.long, device=dev)
    task_to_truth[audit_tasks] = truth_per_task
    edge_truth = task_to_truth[qt.to(dev)]             # [Q]
    # emission row per edge for its truth, then sample the reported option
    Q = qw.numel()
    probs = p_emit[torch.arange(Q, device=dev), edge_truth]  # [Q,K]
    sampled = torch.multinomial(probs, 1, generator=g).squeeze(1)  # [Q]
    new_answers = data.triple[1].clone()
    new_answers[audit_idx] = sampled.to(new_answers.device)

    new_data = copy.deepcopy(data)
    new_triple = data.triple.clone()
    new_triple[1] = new_answers
    new_data.triple = new_triple
    new_data.setup()
    new_data.to(device)
    return new_data, split


def main():
    cfg = dlwheel.setup()
    device = torch.device(cfg.device)
    model = CrowdSIFM(**cfg.model.to_dict()).to(device)
    ck = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    model.eval()

    sim_kwargs = dict(cfg.simulator.to_dict())
    fam = cfg.get("families", None)
    if fam:
        sim_kwargs["mechanism_families"] = [f.strip() for f in str(fam).split(",")]
    simulator = CrowdSISimulator(**sim_kwargs)

    ctx_cfg = dict(
        audit_task_fraction=float(cfg.get("audit_task_fraction", 1.0)),
        context_fraction=float(cfg.get("context_fraction", 0.5)),
        min_context_workers=int(cfg.get("min_context_workers", 1)),
        min_audit_workers=int(cfg.get("min_audit_workers", 2)),
        min_worker_context_edges=int(cfg.get("min_worker_context_edges", 1)),
    )
    adapt_cfg = AdaptationConfig(
        steps=int(cfg.get("adaptation_steps", 100)),
        learning_rate=float(cfg.get("adaptation_learning_rate", 0.05)),
        likelihood_temperature=float(cfg.get("adaptation_likelihood_temperature", 1.0)),
        kl_weight=float(cfg.get("adaptation_kl_weight", 1.0)),
        num_elbo_samples=int(cfg.get("adaptation_num_elbo_samples", 4)),
    )
    num_predictive_samples = int(cfg.get("num_predictive_samples", 32))

    num_worlds = int(cfg.get("num_worlds", 1000))
    base_seed = int(cfg.get("base_seed", 70000))
    log_e_values = []
    records = []
    for i in range(num_worlds):
        _seed_all(base_seed + i)
        try:
            data = simulator.generate()
            data.to(device)
            split_seed = base_seed + i
            null_data, _ = regenerate_audit_answers_from_null(
                model, data, split_seed, device, gen_seed=base_seed + 100000 + i,
                ctx_cfg=ctx_cfg)
            res = crossfit_e_value(
                model, null_data, adaptation_config=adapt_cfg,
                alpha=0.05, num_predictive_samples=num_predictive_samples,
                audit_task_fraction=ctx_cfg["audit_task_fraction"],
                context_fraction=ctx_cfg["context_fraction"],
                min_context_workers=ctx_cfg["min_context_workers"],
                min_audit_workers=ctx_cfg["min_audit_workers"],
                min_worker_context_edges=ctx_cfg["min_worker_context_edges"],
                seed=split_seed)
            log_e_values.append(res.log_e_value)
            records.append({
                "log_e": res.log_e_value, "e": res.e_value,
                "log_e_ab": res.log_e_value_a_to_b, "log_e_ba": res.log_e_value_b_to_a,
                "num_task": int(null_data.num_task), "num_option": int(null_data.num_option),
            })
        except Exception as e:
            records.append({"error": str(e)})
        if (i + 1) % 50 == 0:
            le = np.array(log_e_values)
            e = np.exp(le)
            print(f"  {i+1}/{num_worlds} n={len(le)} E[E]={e.mean():.3f} "
                  f"Pr(E>=20)={(e>=20).mean():.3f} Pr(E>=10)={(e>=10).mean():.3f}")

    le = np.array(log_e_values)
    e = np.exp(np.clip(le, None, 700))
    n = len(le)
    def rate(alpha):
        thr = 1.0 / alpha
        r = float((e >= thr).mean())
        se = math.sqrt(max(r * (1 - r), 1e-12) / n)
        return {"alpha": alpha, "threshold": thr, "rate": r, "ci95": 1.96 * se,
                "nominal_ok": r <= alpha + 1.96 * se}
    summary = {
        "checkpoint": cfg.checkpoint_path, "families": sim_kwargs["mechanism_families"],
        "n_worlds": n, "mean_E": float(e.mean()), "median_E": float(np.median(e)),
        "mean_logE": float(le.mean()),
        "calibration": {f"alpha_{a}": rate(a) for a in (0.01, 0.05, 0.10)},
    }
    out = Path(cfg.get("output_path", "log/phase2_exactnull.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print("\n=== EXACT-NULL CALIBRATION ===")
    print(f"n={n}  E[E]={e.mean():.3f} (should be <=1)  median E={np.median(e):.3f}")
    for a in (0.01, 0.05, 0.10):
        r = rate(a)
        flag = "OK" if r["nominal_ok"] else "ANTI-CONSERVATIVE"
        print(f"  alpha={a:.2f}  Pr(E>={r['threshold']:.0f})={r['rate']:.4f} "
              f"+/-{r['ci95']:.4f}   [{flag}]")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
