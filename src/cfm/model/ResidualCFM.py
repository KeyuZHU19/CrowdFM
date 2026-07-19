"""Residual mechanism correction on a FROZEN CrowdFM.

Design (per the revised plan):
  l_k^0   = frozen CrowdFM truth logits  (backbone + original head, never trained)
  l_k     = l_k^0 + gamma_k(G,Z) * Delta_theta(G,Z)_k     (residual correction)

Constraints:
  * backbone + original truth head are frozen;
  * the residual head's last layer is zero-initialised => at init l_k = l_k^0 exactly
    (the model reproduces official CrowdFM);
  * gamma_k in [0,1] (sigmoid), so the correction can only be *applied* where useful;
  * an anchor KL(q_CrowdFM || q_corrected) keeps the corrected head near official
    unless the mechanism signal earns a move.

Two latents are supported:
  * Z_resp = h_phi(G)  (the amortized mechanism posterior) predicts responses/assignment;
  * Z_agg  drives the truth residual; by default Z_agg = Z_resp, but a learned
    truth-aligned update g_psi can produce Z_agg = Z_resp + Delta (meta stage).

set use_mechanism=False for the capacity-matched no-mechanism residual baseline (the
residual head then receives a zero mechanism context but keeps identical capacity).
"""
from __future__ import annotations
from typing import Any
import torch

from .CFM import CFM
from .CrowdSIFM import (GaussianMechanismEncoder, CompositionalMechanismBasis,
                        ConditionalEmissionHead)


class ResidualMechanismCFM(torch.nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        dim = int(kwargs["dim"]); dropout = float(kwargs.get("dropout", 0.0))
        latent_dim = int(kwargs.get("mechanism_latent_dim", dim))
        num_primitives = int(kwargs.get("num_mechanism_primitives", 8))
        self.dim = dim
        self.mechanism_latent_dim = latent_dim
        self.use_mechanism = bool(kwargs.get("use_mechanism", True))

        self.backbone = CFM(**kwargs)
        self.mechanism_encoder = GaussianMechanismEncoder(dim, latent_dim)
        self.mechanism_basis = CompositionalMechanismBasis(latent_dim, num_primitives, dim)

        # residual truth head: [task, option, mechanism_context] -> scalar logit residual
        self.residual_head = torch.nn.Sequential(
            torch.nn.Linear(3 * dim, 2 * dim), torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(2 * dim, 1))
        # zero-init last layer -> residual = 0 at init -> reproduce official CrowdFM
        torch.nn.init.zeros_(self.residual_head[-1].weight)
        torch.nn.init.zeros_(self.residual_head[-1].bias)

        # per-task gate gamma in [0,1]; bias init negative so gamma~0 at start
        self.gate_head = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, dim), torch.nn.LeakyReLU(), torch.nn.Linear(dim, 1))
        torch.nn.init.zeros_(self.gate_head[-1].weight)
        torch.nn.init.constant_(self.gate_head[-1].bias, -2.0)  # sigmoid(-2)=0.12

        # response + assignment heads (train the mechanism via annotations)
        self.emission_head = ConditionalEmissionHead(dim, dropout)
        self.assignment_head = torch.nn.Sequential(
            torch.nn.Linear(3 * dim, 2 * dim), torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(2 * dim, 1))

        # learned truth-aligned update g_psi (meta stage): maps (Z_resp', Z_resp, ctx) -> Delta Z_agg
        self.agg_update = torch.nn.Sequential(
            torch.nn.Linear(2 * latent_dim + dim, 2 * dim), torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * dim, latent_dim))
        torch.nn.init.zeros_(self.agg_update[-1].weight)
        torch.nn.init.zeros_(self.agg_update[-1].bias)  # Delta Z_agg = 0 at init

        self.register_buffer("_trained", torch.tensor(False, dtype=torch.bool), persistent=True)

    @property
    def crowdsi_trained(self): return bool(self._trained.item())
    def mark_crowdsi_trained(self, v=True): self._trained.fill_(bool(v))

    def load_crowdfm_checkpoint(self, checkpoint, *, strict=True):
        sd = checkpoint.get("model_state_dict", checkpoint)
        return self.backbone.load_state_dict(sd, strict=strict)

    def freeze_official(self):
        """Freeze the entire CrowdFM backbone incl. its original truth head."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @staticmethod
    def reparameterize(mean, logvar, *, sample):
        return mean if not sample else mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def encode_mechanism(self, data):
        base = self.backbone(data)
        mean, logvar = self.mechanism_encoder(base["z_w"], base["z_t"], base["z_o"], data.triple)
        return base, mean, logvar

    def forward(self, data, *, query_workers=None, query_tasks=None,
                mechanism_latent=None, agg_latent=None, sample_mechanism=None):
        base = self.backbone(data)
        official_logits = base["hat_task_option"]  # FROZEN official CrowdFM truth logits
        mean, logvar = self.mechanism_encoder(base["z_w"], base["z_t"], base["z_o"], data.triple)
        if mechanism_latent is None:
            sample = self.training if sample_mechanism is None else bool(sample_mechanism)
            mechanism_latent = self.reparameterize(mean, logvar, sample=sample)
        mechanism_latent = mechanism_latent.to(base["z_w"].device)
        weights, ctx_resp = self.mechanism_basis(mechanism_latent)

        # aggregation latent for the truth residual (default = response latent)
        z_agg = mechanism_latent if agg_latent is None else agg_latent.to(base["z_w"].device)
        _, ctx_agg = self.mechanism_basis(z_agg)
        truth_ctx = ctx_agg if self.use_mechanism else torch.zeros_like(ctx_agg)

        N = base["z_t"].shape[0]; K = base["z_o"].shape[0]
        task = base["z_t"][:, None, :].expand(-1, K, -1)
        opt = base["z_o"][None, :, :].expand(N, -1, -1)
        mech = truth_ctx[None, None, :].expand(N, K, -1)
        residual = self.residual_head(torch.cat([task, opt, mech], -1)).squeeze(-1)  # [N,K]
        gamma = torch.sigmoid(self.gate_head(
            torch.cat([base["z_t"], truth_ctx.expand(N, -1)], -1))).squeeze(-1)  # [N]
        corrected_logits = official_logits + gamma[:, None] * residual

        out = dict(base)
        out.update(hat_task_option_base=official_logits, hat_task_option=corrected_logits,
                   residual=residual, gate=gamma, mechanism_mean=mean,
                   mechanism_log_variance=logvar, mechanism_latent=mechanism_latent,
                   mechanism_weights=weights, mechanism_context=ctx_resp,
                   z_worker_si=base["z_w"], z_task_si=base["z_t"])
        if query_workers is not None:
            qw = query_workers.to(base["z_w"].device, dtype=torch.long)
            qt = query_tasks.to(base["z_t"].device, dtype=torch.long)
            out["hat_annotation_given_truth"] = self.emission_head(
                base["z_w"], base["z_t"], base["z_o"], ctx_resp, qw, qt)
            af = torch.cat([base["z_w"][qw], base["z_t"][qt], ctx_resp.expand(qw.numel(), -1)], -1)
            out["hat_assignment_logit"] = self.assignment_head(af).squeeze(-1)
        return out
