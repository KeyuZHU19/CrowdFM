from __future__ import annotations

from dataclasses import dataclass

import torch

from .predictive import (
    PredictiveResidualResult,
    build_predictive_residual,
    normalize_annotation_probabilities,
)


@dataclass(frozen=True)
class PredictiveMonteCarloResult:
    observed: PredictiveResidualResult
    bootstrap_marginal_statistics: torch.Tensor
    bootstrap_dependence_statistics: torch.Tensor
    marginal_p_value: float
    dependence_p_value: float
    p_value: float


def conditional_predictive_test(
    probabilities: torch.Tensor,
    observed_answers: torch.Tensor,
    query_workers: torch.Tensor,
    query_tasks: torch.Tensor,
    *,
    num_worker: int,
    num_bootstrap: int = 199,
    seed: int = 0,
    min_pair_count: int = 1,
    probability_clip: float = 1e-4,
    variance_ridge: float = 1e-8,
) -> PredictiveMonteCarloResult:
    """Test a fixed held-out annotation predictive law.

    Conditional on the context graph and predicted probability rows, the null is

        A_e independently follows Categorical(r_e)

    for every held-out annotation edge e.  The observed labels and Monte Carlo
    replicates are therefore exchangeable under the null.  Separate plus-one
    Monte Carlo p-values are computed for marginal miscalibration and residual
    worker dependence; a Bonferroni combination controls the joint test without
    requiring the two statistics to share a numerical scale.
    """

    if num_bootstrap < 1:
        raise ValueError("num_bootstrap must be positive")

    probabilities = normalize_annotation_probabilities(
        probabilities,
        probability_clip=probability_clip,
    )
    observed = build_predictive_residual(
        probabilities,
        observed_answers,
        query_workers,
        query_tasks,
        num_worker=num_worker,
        min_pair_count=min_pair_count,
        probability_clip=0.0,
        variance_ridge=variance_ridge,
    )

    generator = torch.Generator(device=probabilities.device.type)
    generator.manual_seed(seed)
    marginal_statistics = torch.empty(num_bootstrap, dtype=torch.float64)
    dependence_statistics = torch.empty(num_bootstrap, dtype=torch.float64)

    for index in range(num_bootstrap):
        simulated_answers = torch.multinomial(
            probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(1)
        replicate = build_predictive_residual(
            probabilities,
            simulated_answers,
            query_workers,
            query_tasks,
            num_worker=num_worker,
            min_pair_count=min_pair_count,
            probability_clip=0.0,
            variance_ridge=variance_ridge,
        )
        marginal_statistics[index] = replicate.marginal_statistic
        dependence_statistics[index] = replicate.dependence_statistic

    marginal_exceedances = int(
        (
            marginal_statistics
            >= observed.marginal_statistic - 1e-12
        ).sum().item()
    )
    dependence_exceedances = int(
        (
            dependence_statistics
            >= observed.dependence_statistic - 1e-12
        ).sum().item()
    )
    denominator = num_bootstrap + 1.0
    marginal_p_value = (1.0 + marginal_exceedances) / denominator
    dependence_p_value = (1.0 + dependence_exceedances) / denominator
    joint_p_value = min(1.0, 2.0 * min(marginal_p_value, dependence_p_value))

    return PredictiveMonteCarloResult(
        observed=observed,
        bootstrap_marginal_statistics=marginal_statistics,
        bootstrap_dependence_statistics=dependence_statistics,
        marginal_p_value=float(marginal_p_value),
        dependence_p_value=float(dependence_p_value),
        p_value=float(joint_p_value),
    )
