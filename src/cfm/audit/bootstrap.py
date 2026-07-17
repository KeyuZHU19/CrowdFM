from __future__ import annotations

from dataclasses import dataclass

import torch

from .disagreement import build_residual_matrix


@dataclass(frozen=True)
class MonteCarloResult:
    observed_statistic: float
    bootstrap_statistics: torch.Tensor
    p_value: float


def conditional_monte_carlo_test(
    triple: torch.Tensor,
    task_posterior: torch.Tensor,
    confusion: torch.Tensor,
    *,
    audit_edge_mask: torch.Tensor,
    observed_statistic: float | None = None,
    num_bootstrap: int = 199,
    seed: int = 0,
    min_pair_count: int = 1,
    probability_clip: float = 1e-4,
    variance_ridge: float = 1e-8,
) -> MonteCarloResult:
    """Calibrate the residual statistic under the fitted predictive null.

    The observation mask, task posteriors and confusion matrices remain fixed.
    Each replicate samples one latent truth per item, then samples held-out
    annotator labels independently conditional on that truth.
    """

    if num_bootstrap < 1:
        raise ValueError("num_bootstrap must be positive")

    device = task_posterior.device
    triple = triple.to(device)
    audit_edge_mask = audit_edge_mask.to(device)

    if observed_statistic is None:
        observed_statistic = build_residual_matrix(
            triple,
            task_posterior,
            confusion,
            audit_edge_mask=audit_edge_mask,
            min_pair_count=min_pair_count,
            probability_clip=probability_clip,
            variance_ridge=variance_ridge,
        ).statistic

    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    statistics = torch.empty(num_bootstrap, dtype=torch.float64)

    audit_workers = triple[0, audit_edge_mask].long()
    audit_tasks = triple[2, audit_edge_mask].long()

    for index in range(num_bootstrap):
        latent_truth = torch.multinomial(
            task_posterior,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        answer_probabilities = confusion[
            audit_workers,
            latent_truth[audit_tasks],
        ]
        simulated_answers = torch.multinomial(
            answer_probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)

        simulated_triple = triple.clone()
        simulated_triple[1, audit_edge_mask] = simulated_answers
        statistics[index] = build_residual_matrix(
            simulated_triple,
            task_posterior,
            confusion,
            audit_edge_mask=audit_edge_mask,
            min_pair_count=min_pair_count,
            probability_clip=probability_clip,
            variance_ridge=variance_ridge,
        ).statistic

    exceedances = int((statistics >= observed_statistic - 1e-12).sum().item())
    p_value = (1.0 + exceedances) / (num_bootstrap + 1.0)
    return MonteCarloResult(
        observed_statistic=float(observed_statistic),
        bootstrap_statistics=statistics,
        p_value=float(p_value),
    )
