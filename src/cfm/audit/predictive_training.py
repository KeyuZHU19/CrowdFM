from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from cfm.data.crowd_data import CrowdData

from .pipeline import masked_data
from .predictive_split import AnnotationAuditSplit, make_annotation_audit_split


@dataclass(frozen=True)
class PredictiveTrainingConfig:
    audit_task_fraction: float = 1.0
    context_fraction: float = 0.5
    min_context_workers: int = 1
    min_audit_workers: int = 1
    min_worker_context_edges: int = 0
    truth_loss_weight: float = 1.0
    annotation_loss_weight: float = 1.0


def predictive_training_loss(
    model: torch.nn.Module,
    data: CrowdData,
    *,
    config: PredictiveTrainingConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute the joint truth-aggregation and masked-response objective.

    Audit edges are removed from the graph before the forward pass.  Their labels
    supervise ``hat_annotation_option``.  Synthetic task truth, when available,
    continues to supervise ``hat_task_option``.  Real datasets without gold task
    labels can set ``truth_loss_weight=0`` and train the response objective alone.
    """

    cfg = config or PredictiveTrainingConfig()
    if cfg.truth_loss_weight < 0 or cfg.annotation_loss_weight <= 0:
        raise ValueError("loss weights must be nonnegative and annotation weight positive")

    # The training objective permits one held-out edge per task; the deployment
    # dependence audit separately requires at least two.
    split_min_audit = max(2, cfg.min_audit_workers)
    split: AnnotationAuditSplit = make_annotation_audit_split(
        data.triple,
        data.num_task,
        num_worker=data.num_worker,
        audit_task_fraction=cfg.audit_task_fraction,
        context_fraction=cfg.context_fraction,
        min_context_workers=cfg.min_context_workers,
        min_audit_workers=split_min_audit,
        min_worker_context_edges=cfg.min_worker_context_edges,
        seed=seed,
    )
    context_data = masked_data(data, split.context_edge_mask)
    audit_indices = split.audit_edge_indices
    query_workers = data.triple[0, audit_indices].long()
    observed_answers = data.triple[1, audit_indices].long()
    query_tasks = data.triple[2, audit_indices].long()

    output = model(
        context_data,
        query_workers=query_workers,
        query_tasks=query_tasks,
    )
    if "hat_annotation_option" not in output:
        raise KeyError("model must output hat_annotation_option for masked edges")
    annotation_loss = F.cross_entropy(
        output["hat_annotation_option"],
        observed_answers.to(output["hat_annotation_option"].device),
    )

    truth_loss = annotation_loss.new_zeros(())
    task_y = getattr(data, "task_y", None)
    if cfg.truth_loss_weight > 0 and isinstance(task_y, torch.Tensor):
        valid = task_y != -1
        if torch.any(valid):
            truth_loss = F.cross_entropy(
                output["hat_task_option"][valid],
                task_y.to(output["hat_task_option"].device)[valid],
            )

    total = (
        cfg.annotation_loss_weight * annotation_loss
        + cfg.truth_loss_weight * truth_loss
    )
    return {
        "loss": total,
        "annotation_loss": annotation_loss,
        "truth_loss": truth_loss,
        "output": output,
        "split": split,
        "num_audit_edges": int(audit_indices.numel()),
    }
