"""Train ResidualMechanismCFM on a FROZEN CrowdFM.

modes:
  residual : zero-shot residual correction. Trains encoder/basis/residual/gate/emission/
             assignment via truth + response + assignment + anchor + mechanism-KL losses.
  meta     : truth-aligned meta-adaptation. Inner loop adapts Z_resp by a response-NLL
             gradient (deployment-observable only); a learned update g_psi maps that to an
             aggregation-latent correction Z_agg; the outer loop optimises truth accuracy
             with synthetic gold labels. Deployment needs no labels.
"""
from __future__ import annotations
import json
from pathlib import Path
import dlwheel, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cfm.data.crowdsi_simulator import CrowdSIDataset
from cfm.model.ResidualCFM import ResidualMechanismCFM
from cfm.audit.pipeline import masked_data
from cfm.audit.predictive_split import make_annotation_audit_split
from cfm.si.likelihood import (joint_annotation_log_likelihood, supervised_emission_loss,
                               diagonal_gaussian_kl)


def anchor_kl(official_logits, corrected_logits):
    p = F.log_softmax(official_logits, -1)
    q = F.log_softmax(corrected_logits, -1)
    return (p.exp() * (p - q)).sum(-1).mean()


def response_and_assignment(model, data, split, ctx_data, cfg):
    idx = split.audit_edge_indices
    qw = data.triple[0, idx].long(); qt = data.triple[2, idx].long(); qa = data.triple[1, idx].long()
    out = model(ctx_data, query_workers=qw, query_tasks=qt, sample_mechanism=True)
    ty = getattr(data, "task_y", None)
    if isinstance(ty, torch.Tensor) and torch.any(ty >= 0):
        resp = supervised_emission_loss(out["hat_annotation_given_truth"], qa, qt, ty)
    else:
        resp = -joint_annotation_log_likelihood(out["hat_task_option"],
               out["hat_annotation_given_truth"], qa, qt, reduction="mean")
    return out, resp, qw, qt, qa


def step_residual(model, data, cfg, seed):
    split = make_annotation_audit_split(data.triple, data.num_task, num_worker=data.num_worker,
        audit_task_fraction=1.0, context_fraction=0.5, min_context_workers=1,
        min_audit_workers=2, min_worker_context_edges=0, seed=seed)
    ctx = masked_data(data, split.context_edge_mask)
    out, resp, qw, qt, qa = response_and_assignment(model, data, split, ctx, cfg)
    ty = data.task_y.to(out["hat_task_option"].device).long(); v = ty >= 0
    truth = F.cross_entropy(out["hat_task_option"][v], ty[v]) if v.any() else out["hat_task_option"].sum()*0
    anchor = anchor_kl(out["hat_task_option_base"], out["hat_task_option"])
    mkl = diagonal_gaussian_kl(out["mechanism_mean"], out["mechanism_log_variance"]) / out["mechanism_mean"].numel()
    loss = (cfg["truth"] * truth + cfg["resp"] * resp + cfg["anchor"] * anchor + cfg["mkl"] * mkl)
    return loss, {"truth": float(truth), "resp": float(resp), "anchor": float(anchor),
                  "gate": float(out["gate"].mean()), "mkl": float(mkl)}


def step_meta(model, data, cfg, seed):
    # context/audit split; inner response-adapt on audit annotations (label-free)
    split = make_annotation_audit_split(data.triple, data.num_task, num_worker=data.num_worker,
        audit_task_fraction=1.0, context_fraction=0.5, min_context_workers=1,
        min_audit_workers=2, min_worker_context_edges=0, seed=seed)
    ctx = masked_data(data, split.context_edge_mask)
    base, mean, logvar = model.encode_mechanism(ctx)
    z_resp = mean  # encoder posterior mean (keeps grad to phi)
    idx = split.audit_edge_indices
    qw = data.triple[0, idx].long(); qt = data.triple[2, idx].long(); qa = data.triple[1, idx].long()

    # inner: response-NLL gradient wrt a detached copy of z_resp (first-order meta)
    zc = z_resp.detach().clone().requires_grad_(True)
    o_in = model(ctx, query_workers=qw, query_tasks=qt, mechanism_latent=zc, sample_mechanism=False)
    resp_nll = -joint_annotation_log_likelihood(o_in["hat_task_option"],
               o_in["hat_annotation_given_truth"], qa, qt, reduction="mean")
    g = torch.autograd.grad(resp_nll, zc, create_graph=False)[0]
    z_resp_prime = (zc - cfg["inner_lr"] * g).detach()

    # learned truth-aligned update -> Z_agg
    ctx_pool = torch.cat([base["z_w"].mean(0), base["z_t"].mean(0), base["z_o"].mean(0)]) \
        if False else base["z_t"].mean(0)  # pooled context feature (dim)
    delta_agg = model.agg_update(torch.cat([z_resp_prime, z_resp.detach(), ctx_pool]))
    z_agg = z_resp_prime + delta_agg  # grad flows to psi (and delta magnitude)

    # outer: truth loss with agg latent; response ctx uses z_resp (grad to phi)
    out = model(ctx, mechanism_latent=z_resp, agg_latent=z_agg, sample_mechanism=False)
    ty = data.task_y.to(out["hat_task_option"].device).long(); v = ty >= 0
    truth = F.cross_entropy(out["hat_task_option"][v], ty[v]) if v.any() else out["hat_task_option"].sum()*0
    anchor = anchor_kl(out["hat_task_option_base"], out["hat_task_option"])
    # keep response head trained so Z_resp is meaningful
    o_r, resp, *_ = response_and_assignment(model, data, split, ctx, cfg)
    mkl = diagonal_gaussian_kl(mean, logvar) / mean.numel()
    loss = (cfg["truth"] * truth + cfg["resp"] * resp + cfg["anchor"] * anchor + cfg["mkl"] * mkl)
    return loss, {"truth": float(truth), "resp": float(resp), "anchor": float(anchor),
                  "gate": float(out["gate"].mean()), "dAgg": float(delta_agg.norm())}


def main():
    cfg = dlwheel.setup()
    seed = int(cfg.get("seed", 42))
    import random, numpy as np
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    device = cfg.device
    mode = cfg.get("mode", "residual")
    mkw = dict(cfg.model.to_dict()); mkw["use_mechanism"] = bool(cfg.get("use_mechanism", True))
    model = ResidualMechanismCFM(**mkw).to(device)
    model.load_crowdfm_checkpoint(torch.load(cfg.backbone_checkpoint_path, map_location=device, weights_only=False), strict=True)
    model.freeze_official()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loader = DataLoader(CrowdSIDataset(**cfg.simulator.to_dict()), batch_size=int(cfg.batch_size),
                        num_workers=int(cfg.num_workers), collate_fn=lambda v: v)
    it = iter(loader)
    lcfg = {"truth": float(cfg.get("truth_w", 1.0)), "resp": float(cfg.get("resp_w", 1.0)),
            "anchor": float(cfg.get("anchor_w", 0.1)), "mkl": float(cfg.get("mkl_w", 1e-3)),
            "inner_lr": float(cfg.get("inner_lr", 0.1))}
    out_dir = Path(cfg.get("output_dir", "log/residual")); out_dir.mkdir(parents=True, exist_ok=True)
    accum = int(cfg.gradient_accumulation_steps); epochs = int(cfg.epochs)
    stepfn = step_meta if mode == "meta" else step_residual
    opt.zero_grad(); model.train()
    from tqdm import tqdm
    with tqdm(range(1, epochs + 1), dynamic_ncols=True) as bar:
        for ep in bar:
            batch = [d.to(device) for d in next(it)]
            losses = []; logs = []
            for j, d in enumerate(batch):
                l, info = stepfn(model, d, lcfg, seed * 1_000_000_007 + ep * 1000 + j)
                losses.append(l); logs.append(info)
            loss = torch.stack(losses).mean()
            (loss / accum).backward()
            if ep % accum == 0 or ep == epochs:
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.gradient_clip))
                opt.step(); opt.zero_grad(); model.mark_crowdsi_trained()
            if ep % 50 == 0:
                agg = {k: sum(x[k] for x in logs) / len(logs) for k in logs[0]}
                bar.set_postfix({"loss": float(loss), **{k: round(v, 3) for k, v in agg.items()}})
            if ep % int(cfg.save_interval) == 0 or ep == epochs:
                torch.save({"model_state_dict": model.state_dict()}, out_dir / f"{ep}.pt")


if __name__ == "__main__":
    main()
