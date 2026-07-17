from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from .bootstrap import conditional_monte_carlo_test
from .disagreement import build_residual_matrix, estimate_confusion_matrices
from .pipeline import CBRConfig, masked_data
from .split import CrossFitSplit, make_crossfit_split
from .synthetic import SyntheticWorld

CalibrationVariant = Literal[
    "oracle_q_oracle_p",
    "model_q_oracle_p",
    "oracle_q_estimated_p",
    "model_q_estimated_p",
]

CALIBRATION_VARIANTS: tuple[CalibrationVariant, ...] = (
    "oracle_q_oracle_p",
    "model_q_oracle_p",
    "oracle_q_estimated_p",
    "model_q_estimated_p",
)


@dataclass(frozen=True)
class CalibrationWorldResult:
    variant: CalibrationVariant
    statistic: float
    p_value: float
    reject: bool
    audit_posterior_accuracy: float
    confusion_mae: float
    num_supported_pairs: int
    num_audit_edges: int


def _oracle_posterior(world: SyntheticWorld, device: torch.device | str) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        world.truth.to(device),
        num_classes=world.config.num_option,
    ).float()


def _model_posterior(
    model: torch.nn.Module,
    world: SyntheticWorld,
    split: CrossFitSplit,
) -> torch.Tensor:
    model_data = masked_data(world.data, split.model_input_edge_mask)
    model.eval()
    with torch.no_grad():
        output = model(model_data)
    return torch.softmax(output["hat_task_option"], dim=-1)


def run_calibration_world(
    model: torch.nn.Module,
    world: SyntheticWorld,
    *,
    variant: CalibrationVariant,
    config: CBRConfig | None = None,
    seed: int = 0,
) -> CalibrationWorldResult:
    if variant not in CALIBRATION_VARIANTS:
        raise ValueError(f"Unknown calibration variant: {variant}")
    cfg = config or CBRConfig()
    data = world.data
    split = make_crossfit_split(
        data.triple,
        data.num_task,
        audit_fraction=cfg.audit_fraction,
        context_fraction=cfg.context_fraction,
        min_context_workers=cfg.min_context_workers,
        min_audit_workers=cfg.min_audit_workers,
        seed=seed,
    )

    use_model_q = variant.startswith("model_q")
    task_posterior = (
        _model_posterior(model, world, split)
        if use_model_q
        else _oracle_posterior(world, data.device)
    )

    use_oracle_p = variant.endswith("oracle_p")
    true_confusion = world.true_confusion.to(data.device)
    confusion = (
        true_confusion
        if use_oracle_p
        else estimate_confusion_matrices(
            data.triple,
            task_posterior,
            num_worker=data.num_worker,
            num_option=data.num_option,
            edge_mask=split.nuisance_edge_mask,
            prior_strength=cfg.prior_strength,
        )
    )

    residual = build_residual_matrix(
        data.triple,
        task_posterior,
        confusion,
        audit_edge_mask=split.audit_edge_mask,
        min_pair_count=cfg.min_pair_count,
        probability_clip=cfg.probability_clip,
        variance_ridge=cfg.variance_ridge,
    )
    calibration = conditional_monte_carlo_test(
        data.triple,
        task_posterior,
        confusion,
        audit_edge_mask=split.audit_edge_mask,
        nuisance_edge_mask=split.nuisance_edge_mask,
        refit_confusion=(not use_oracle_p and cfg.refit_confusion_bootstrap),
        posterior_predictive_confusion=(
            not use_oracle_p and cfg.posterior_predictive_confusion_bootstrap
        ),
        prior_strength=cfg.prior_strength,
        observed_statistic=residual.statistic,
        num_bootstrap=cfg.num_bootstrap,
        seed=seed,
        min_pair_count=cfg.min_pair_count,
        probability_clip=cfg.probability_clip,
        variance_ridge=cfg.variance_ridge,
    )

    audit_tasks = split.audit_task_mask
    posterior_prediction = task_posterior.argmax(dim=-1)
    posterior_accuracy = (
        posterior_prediction[audit_tasks] == world.truth.to(data.device)[audit_tasks]
    ).float().mean()
    confusion_mae = (confusion - true_confusion).abs().mean()

    return CalibrationWorldResult(
        variant=variant,
        statistic=residual.statistic,
        p_value=calibration.p_value,
        reject=calibration.p_value <= cfg.alpha,
        audit_posterior_accuracy=float(posterior_accuracy.item()),
        confusion_mae=float(confusion_mae.item()),
        num_supported_pairs=int(
            (residual.pair_counts >= cfg.min_pair_count).sum().item() // 2
        ),
        num_audit_edges=int(split.audit_edge_mask.sum().item()),
    )
