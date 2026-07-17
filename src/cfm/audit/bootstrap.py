from __future__ import annotations

from dataclasses import dataclass

import torch

from .disagreement import build_residual_matrix, estimate_confusion_matrices


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
    nuisance_edge_mask: torch.Tensor | None = None,
    refit_confusion: bool = False,
    prior_strength: float = 1.0,
    observed_statistic: float | None = None,
    num_bootstrap: int = 199,
    seed: int = 0,
    min_pair_count: int = 1,
    probability_clip: float = 1e-4,
    variance_ridge: float = 1e-8,
) -> MonteCarloResult:
    """Calibrate the residual statistic under a parametric predictive null.

    With ``refit_confusion=False``, the observation mask, task posteriors and
    confusion matrices remain fixed. This is the exact fitted-null Monte Carlo
    test used by the initial CbR implementation.

    With ``refit_confusion=True``, each replicate simulates both nuisance and
    audit labels, re-estimates the worker confusion matrices from the simulated
    nuisance labels, and then recomputes the audit statistic. This parametric
    refit bootstrap captures first-order uncertainty from estimating the
    confusion matrices instead of treating a noisy plug-in estimate as known.
    The task posterior remains fixed in both modes.
    """

    if num_bootstrap < 1:
        raise ValueError("num_bootstrap must be positive")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    if refit_confusion and nuisance_edge_mask is None:
        raise ValueError("nuisance_edge_mask is required when refit_confusion=True")

    device = task_posterior.device
    triple = triple.to(device)
    audit_edge_mask = audit_edge_mask.to(device)
    if nuisance_edge_mask is not None:
        nuisance_edge_mask = nuisance_edge_mask.to(device)
        if nuisance_edge_mask.shape != audit_edge_mask.shape:
            raise ValueError("nuisance_edge_mask must have one entry per edge")
        if torch.any(nuisance_edge_mask & audit_edge_mask):
            raise ValueError("nuisance and audit edge masks must be disjoint")

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

    simulation_edge_mask = audit_edge_mask
    if refit_confusion:
        simulation_edge_mask = audit_edge_mask | nuisance_edge_mask

    simulation_workers = triple[0, simulation_edge_mask].long()
    simulation_tasks = triple[2, simulation_edge_mask].long()

    for index in range(num_bootstrap):
        latent_truth = torch.multinomial(
            task_posterior,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        answer_probabilities = confusion[
            simulation_workers,
            latent_truth[simulation_tasks],
        ]
        simulated_answers = torch.multinomial(
            answer_probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)

        simulated_triple = triple.clone()
        simulated_triple[1, simulation_edge_mask] = simulated_answers

        replicate_confusion = confusion
        if refit_confusion:
            replicate_confusion = estimate_confusion_matrices(
                simulated_triple,
                task_posterior,
                num_worker=confusion.shape[0],
                num_option=confusion.shape[1],
                edge_mask=nuisance_edge_mask,
                prior_strength=prior_strength,
            )

        statistics[index] = build_residual_matrix(
            simulated_triple,
            task_posterior,
            replicate_confusion,
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
