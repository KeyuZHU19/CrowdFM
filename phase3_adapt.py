"""Phase-3 held-out-mechanism adaptation evaluation for CrowdSI-FM.

For each held-out world we report truth-aggregation accuracy for:
  crowdfm            frozen backbone truth head
  crowdsi_zeroshot   mechanism truth head at amortized mu_0
  crowdsi_always     latent adaptation on all audit tasks (no gate)
  crowdsi_egated     evidence-gated adaptation (run_crowdsi)
  oracle_latent      latent optimized against gold truths (headroom, cheats)
  fullnet_tta        full-network test-time fine-tuning on audit responses (baseline)
  majority_vote      classic aggregation baseline
  dawid_skene        classic EM aggregation baseline

Also records the e-value and whether the gate fired. The central quantities are
  Delta_arch = zeroshot - crowdfm
  Delta_adapt = egated  - zeroshot
and the always-vs-gated negative-transfer comparison.
"""
from __future__ import annotations

import copy
import dataclasses
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
from cfm.si.adaptation import (AdaptationConfig, MechanismPosterior, adapt_mechanism,
                               crossfit_e_value)
from cfm.si.likelihood import joint_annotation_log_likelihood
from cfm.si.pipeline import CrowdSIPipelineConfig, run_crowdsi
from cfm.si.split import make_task_crossfit_split


def _seed_all(seed):
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)


def _acc(pred, y, valid):
    return float((pred[valid] == y[valid]).float().mean().item())


def majority_vote_pred(data):
    answers = data.triple[1].cpu().numpy(); tasks = data.triple[2].cpu().numpy()
    pred = np.zeros(data.num_task, dtype=np.int64)
    for k in range(data.num_task):
        a = answers[tasks == k]
        if len(a):
            pred[k] = np.bincount(a, minlength=data.num_option).argmax()
    return torch.tensor(pred)


def dawid_skene(data, n_iter=30):
    M, N, K = data.num_worker, data.num_task, data.num_option
    w = data.triple[0].cpu().numpy(); a = data.triple[1].cpu().numpy(); t = data.triple[2].cpu().numpy()
    # init task posteriors by majority vote
    T = np.zeros((N, K)) + 1e-6
    for wi, ai, ti in zip(w, a, t):
        T[ti, ai] += 1
    T /= T.sum(1, keepdims=True)
    for _ in range(n_iter):
        # M-step: worker confusion + class prior
        pi = T.sum(0) + 1e-6; pi /= pi.sum()
        cm = np.zeros((M, K, K)) + 1e-6
        for wi, ai, ti in zip(w, a, t):
            cm[wi, :, ai] += T[ti]
        cm /= cm.sum(2, keepdims=True)
        # E-step
        logT = np.log(pi)[None, :].repeat(N, 0).copy()
        for wi, ai, ti in zip(w, a, t):
            logT[ti] += np.log(cm[wi, :, ai] + 1e-12)
        logT -= logT.max(1, keepdims=True)
        T = np.exp(logT); T /= T.sum(1, keepdims=True)
    return torch.tensor(T.argmax(1))


@torch.no_grad()
def oracle_latent_predict(model, data, base_post, device, y, valid, steps=150, lr=0.05):
    """Optimize the latent against gold-truth CE (upper bound on latent headroom)."""
    mean = torch.nn.Parameter(base_post.mean.detach().clone())
    opt = torch.optim.Adam([mean], lr=lr)
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    was_train = model.training; model.eval()
    yv = y[valid]
    try:
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad()
                out = model(data, mechanism_latent=mean, sample_mechanism=False)
                logits = out["hat_task_option"][valid]
                loss = torch.nn.functional.cross_entropy(logits, yv)
                loss.backward(); opt.step()
    finally:
        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)
        model.train(was_train)
    with torch.no_grad():
        out = model(data, mechanism_latent=mean.detach(), sample_mechanism=False)
    return out["hat_task_option"].argmax(-1)


def fullnet_tta_predict(model, data, split_seed, device, ctx_cfg, steps=30, lr=1e-3):
    """Full-network test-time fine-tuning on audit response NLL (self-supervised)."""
    m = copy.deepcopy(model)
    split = make_task_crossfit_split(
        data.triple, data.num_task, num_worker=data.num_worker,
        audit_task_fraction=ctx_cfg["audit_task_fraction"], context_fraction=ctx_cfg["context_fraction"],
        min_context_workers=ctx_cfg["min_context_workers"], min_audit_workers=ctx_cfg["min_audit_workers"],
        min_worker_context_edges=ctx_cfg["min_worker_context_edges"], seed=split_seed)
    ctx = masked_data(data, split.context_edge_mask)
    idx = torch.nonzero(split.audit_edge_mask).flatten()
    qw = data.triple[0, idx].long(); qt = data.triple[2, idx].long(); qa = data.triple[1, idx].long()
    for p in m.parameters():
        p.requires_grad_(True)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        out = m(ctx, query_workers=qw, query_tasks=qt, sample_mechanism=False)
        nll = -joint_annotation_log_likelihood(out["hat_task_option"], out["hat_annotation_given_truth"],
                                                qa, qt, reduction="mean")
        nll.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    m.eval()
    with torch.no_grad():
        out = m(data, sample_mechanism=False)
    return out["hat_task_option"].argmax(-1)


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

    ctx_cfg = dict(audit_task_fraction=1.0, context_fraction=0.5, min_context_workers=1,
                   min_audit_workers=2, min_worker_context_edges=1)
    pipe_cfg = CrowdSIPipelineConfig(
        alpha=float(cfg.get("alpha", 0.05)), num_predictive_samples=int(cfg.get("num_predictive_samples", 32)),
        adapt_only_with_evidence=True, require_trained_model=True,
        adaptation_steps=int(cfg.get("adaptation_steps", 100)),
        adaptation_learning_rate=float(cfg.get("adaptation_learning_rate", 0.05)),
        adaptation_kl_weight=float(cfg.get("adaptation_kl_weight", 1.0)),
        adaptation_num_elbo_samples=int(cfg.get("adaptation_num_elbo_samples", 4)),
        min_worker_context_edges=1)
    pipe_cfg_always = dataclasses.replace(pipe_cfg, adapt_only_with_evidence=False)

    num_worlds = int(cfg.get("num_worlds", 100))
    base_seed = int(cfg.get("base_seed", 60000))
    do_oracle = bool(cfg.get("do_oracle", True))
    do_fullnet = bool(cfg.get("do_fullnet", True))
    do_ds = bool(cfg.get("do_ds", True))
    records = []
    for i in range(num_worlds):
        _seed_all(base_seed + i)
        data = simulator.generate(); data.to(device)
        y = data.task_y.to(device).long(); valid = y >= 0
        seed = base_seed + i
        rec = {"family": data.mechanism_family_name, "num_task": int(data.num_task),
               "num_option": int(data.num_option), "num_worker": int(data.num_worker)}
        try:
            r_g = run_crowdsi(model, data, config=pipe_cfg, seed=seed)
            rec["crowdfm"] = r_g["crowdfm_accuracy"]
            rec["zeroshot"] = r_g["zero_shot_accuracy"]
            rec["egated"] = r_g["accuracy"]
            rec["e_value"] = r_g["e_value"]
            rec["gate_fired"] = bool(r_g["used_adaptation"])
            r_a = run_crowdsi(model, data, config=pipe_cfg_always, seed=seed)
            rec["always"] = r_a["accuracy"]
            rec["mv"] = _acc(majority_vote_pred(data).to(device), y, valid)
            if do_ds:
                rec["dawid_skene"] = _acc(dawid_skene(data).to(device), y, valid)
            base_post = MechanismPosterior(mean=r_g["mechanism_mean_base"], log_variance=r_g["mechanism_log_variance_base"])
            if do_oracle:
                rec["oracle_latent"] = _acc(oracle_latent_predict(model, data, base_post, device, y, valid), y, valid)
            if do_fullnet:
                rec["fullnet_tta"] = _acc(fullnet_tta_predict(model, data, seed, device, ctx_cfg), y, valid)
        except Exception as e:
            rec["error"] = str(e)
        records.append(rec)
        if (i + 1) % 10 == 0:
            ok = [r for r in records if "zeroshot" in r]
            def m(k): return float(np.mean([r[k] for r in ok if k in r])) if ok else float("nan")
            print(f"  {i+1}/{num_worlds} crowdfm={m('crowdfm'):.3f} zshot={m('zeroshot'):.3f} "
                  f"egated={m('egated'):.3f} always={m('always'):.3f} gatefire={m('gate_fired'):.2f}")

    ok = [r for r in records if "zeroshot" in r]
    keys = ["crowdfm", "zeroshot", "egated", "always", "oracle_latent", "fullnet_tta",
            "mv", "dawid_skene", "e_value", "gate_fired"]
    def agg(k):
        v = np.array([r[k] for r in ok if k in r], dtype=np.float64)
        if not len(v):
            return None
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1) if len(v) > 1 else 0),
                "ci95": float(1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0), "n": len(v)}
    summary = {k: agg(k) for k in keys}
    # paired deltas
    def paired(k1, k2):
        d = np.array([r[k1] - r[k2] for r in ok if k1 in r and k2 in r])
        if not len(d):
            return None
        return {"mean": float(d.mean()), "ci95": float(1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0), "n": len(d)}
    deltas = {"arch(zeroshot-crowdfm)": paired("zeroshot", "crowdfm"),
              "adapt(egated-zeroshot)": paired("egated", "zeroshot"),
              "egated-always": paired("egated", "always"),
              "egated-crowdfm": paired("egated", "crowdfm"),
              "oracle-egated": paired("oracle_latent", "egated")}
    out = Path(cfg.get("output_path", "log/phase3_adapt.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": cfg.checkpoint_path, "families": sim_kwargs["mechanism_families"],
                               "n": len(ok), "summary": summary, "deltas": deltas, "records": records}, indent=2))
    print("\n=== PHASE-3 SUMMARY (families=%s) ===" % sim_kwargs["mechanism_families"])
    for k in keys:
        s = summary[k]
        if s:
            print(f"  {k:16s} {s['mean']:.4f} +/- {s['ci95']:.4f} (n={s['n']})")
    print("  --- paired deltas ---")
    for k, d in deltas.items():
        if d:
            print(f"  {k:28s} {d['mean']:+.4f} +/- {d['ci95']:.4f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
