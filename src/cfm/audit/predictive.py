from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PredictiveResidualResult:
    """Statistics computed from held-out annotation predictions."""

    marginal_statistic: float
    dependence_statistic: float
    residual_matrix: torch.Tensor
    pair_counts: torch.Tensor
    standardized_edge_residuals: torch.Tensor
    probabilities: torch.Tensor


def normalize_annotation_probabilities(
    probabilities: torch.Tensor,
    *,
    probability_clip: float = 1e-4,
) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [num_edges, num_options]")
    if probabilities.shape[1] < 2:
        raise ValueError("at least two response options are required")
    if probability_clip < 0.0 or probability_clip >= 0.5:
        raise ValueError("probability_clip must lie in [0, 0.5)")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if torch.any(probabilities < 0):
        raise ValueError("probabilities must be nonnegative")

    normalized = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    if probability_clip > 0.0:
        normalized = normalized.clamp_min(probability_clip)
        normalized = normalized / normalized.sum(dim=-1, keepdim=True)
    return normalized


def standardized_categorical_residuals(
    probabilities: torch.Tensor,
    observed_answers: torch.Tensor,
    *,
    probability_clip: float = 1e-4,
    variance_ridge: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Pearson-style residual vectors for held-out categorical labels."""

    if variance_ridge <= 0:
        raise ValueError("variance_ridge must be positive")
    probabilities = normalize_annotation_probabilities(
        probabilities,
        probability_clip=probability_clip,
    )
    observed_answers = observed_answers.to(
        device=probabilities.device,
        dtype=torch.long,
    )
    if observed_answers.shape != (probabilities.shape[0],):
        raise ValueError("one observed answer is required for every probability row")
    if torch.any((observed_answers < 0) | (observed_answers >= probabilities.shape[1])):
        raise ValueError("observed answer lies outside the option range")

    one_hot = F.one_hot(
        observed_answers,
        num_classes=probabilities.shape[1],
    ).to(probabilities.dtype)
    variance = probabilities * (1.0 - probabilities) + variance_ridge
    residuals = (one_hot - probabilities) / torch.sqrt(variance)
    return residuals, probabilities


def marginal_categorical_statistic(
    probabilities: torch.Tensor,
    observed_answers: torch.Tensor,
    *,
    variance_ridge: float = 1e-8,
) -> torch.Tensor:
    """Direction-sensitive standardized mean categorical residual.

    Log-score aggregation can be blind to class-direction shifts when every
    predicted row is uniform.  This statistic instead retains the full option
    direction.  Monte Carlo calibration handles the negative covariance among
    categorical coordinates, so only coordinate-wise variance scaling is used.
    """

    if variance_ridge <= 0:
        raise ValueError("variance_ridge must be positive")
    observed_answers = observed_answers.to(
        device=probabilities.device,
        dtype=torch.long,
    )
    one_hot = F.one_hot(
        observed_answers,
        num_classes=probabilities.shape[1],
    ).to(probabilities.dtype)
    centered = one_hot - probabilities
    coordinate_variance = (
        probabilities * (1.0 - probabilities)
    ).sum(dim=0) + variance_ridge
    standardized_mean = centered.sum(dim=0) / torch.sqrt(coordinate_variance)
    return torch.linalg.vector_norm(standardized_mean, ord=2)


def spectral_statistic(residual_matrix: torch.Tensor) -> float:
    if residual_matrix.ndim != 2 or residual_matrix.shape[0] != residual_matrix.shape[1]:
        raise ValueError("residual_matrix must be square")
    if residual_matrix.numel() == 0:
        return 0.0
    eigenvalues = torch.linalg.eigvalsh(residual_matrix)
    return float(eigenvalues.abs().max().item())


def build_predictive_residual(
    probabilities: torch.Tensor,
    observed_answers: torch.Tensor,
    query_workers: torch.Tensor,
    query_tasks: torch.Tensor,
    *,
    num_worker: int,
    min_pair_count: int = 1,
    probability_clip: float = 1e-4,
    variance_ridge: float = 1e-8,
) -> PredictiveResidualResult:
    """Build marginal and worker-dependence residual statistics.

    For an audit edge e=(i,k), let

        u_e = diag(r_e * (1-r_e) + ridge)^(-1/2) (onehot(A_e)-r_e).

    The marginal statistic checks the direction-sensitive aggregate categorical
    residual.  For each worker pair, the dependence residual averages u_ik^T
    u_jk over shared audit tasks and divides by sqrt(pair count).  Under a correct
    conditionally independent response law, every off-diagonal entry has
    conditional mean zero.
    """

    if num_worker < 1:
        raise ValueError("num_worker must be positive")
    if min_pair_count < 1:
        raise ValueError("min_pair_count must be positive")
    if query_workers.ndim != 1 or query_tasks.ndim != 1:
        raise ValueError("query_workers and query_tasks must be one-dimensional")
    if query_workers.shape != query_tasks.shape:
        raise ValueError("query_workers and query_tasks must have the same shape")
    if probabilities.shape[0] != query_workers.numel():
        raise ValueError("one probability row is required for every query edge")

    query_workers = query_workers.to(device=probabilities.device, dtype=torch.long)
    query_tasks = query_tasks.to(device=probabilities.device, dtype=torch.long)
    if torch.any((query_workers < 0) | (query_workers >= num_worker)):
        raise ValueError("query worker lies outside the worker range")

    edge_residuals, probabilities = standardized_categorical_residuals(
        probabilities,
        observed_answers,
        probability_clip=probability_clip,
        variance_ridge=variance_ridge,
    )
    observed_answers = observed_answers.to(
        device=probabilities.device,
        dtype=torch.long,
    )
    marginal = marginal_categorical_statistic(
        probabilities,
        observed_answers,
        variance_ridge=variance_ridge,
    )

    residual_sum = torch.zeros(
        (num_worker, num_worker),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    pair_counts = torch.zeros(
        (num_worker, num_worker),
        dtype=torch.long,
        device=probabilities.device,
    )

    for task in torch.unique(query_tasks).tolist():
        indices = torch.nonzero(query_tasks == task, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        local_pairs = torch.triu_indices(
            indices.numel(),
            indices.numel(),
            offset=1,
            device=probabilities.device,
        )
        left = indices[local_pairs[0]]
        right = indices[local_pairs[1]]
        worker_i = query_workers[left]
        worker_j = query_workers[right]
        distinct = worker_i != worker_j
        if not torch.any(distinct):
            continue
        left = left[distinct]
        right = right[distinct]
        worker_i = worker_i[distinct]
        worker_j = worker_j[distinct]
        pair_score = (edge_residuals[left] * edge_residuals[right]).mean(dim=-1)

        residual_sum.index_put_((worker_i, worker_j), pair_score, accumulate=True)
        residual_sum.index_put_((worker_j, worker_i), pair_score, accumulate=True)
        ones = torch.ones_like(worker_i, dtype=pair_counts.dtype)
        pair_counts.index_put_((worker_i, worker_j), ones, accumulate=True)
        pair_counts.index_put_((worker_j, worker_i), ones, accumulate=True)

    supported = pair_counts >= min_pair_count
    residual_matrix = torch.zeros_like(residual_sum)
    residual_matrix[supported] = residual_sum[supported] / torch.sqrt(
        pair_counts[supported].to(probabilities.dtype)
    )
    residual_matrix.fill_diagonal_(0.0)
    dependence = spectral_statistic(residual_matrix)

    return PredictiveResidualResult(
        marginal_statistic=float(marginal.item()),
        dependence_statistic=dependence,
        residual_matrix=residual_matrix,
        pair_counts=pair_counts,
        standardized_edge_residuals=edge_residuals,
        probabilities=probabilities,
    )
