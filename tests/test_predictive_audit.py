import torch

from cfm.audit.predictive import build_predictive_residual
from cfm.audit.predictive_bootstrap import conditional_predictive_test
from cfm.audit.predictive_pipeline import PredictiveAuditConfig, run_predictive_audit
from cfm.audit.predictive_split import make_annotation_audit_split
from cfm.audit.predictive_training import (
    PredictiveTrainingConfig,
    predictive_training_loss,
)
from cfm.data.crowd_data import CrowdData
from cfm.model.PredictiveCFM import PredictiveCFM


def _toy_data() -> CrowdData:
    workers = [0, 1, 2] * 6
    tasks = [task for task in range(6) for _ in range(3)]
    answers = [
        0, 0, 1,
        1, 1, 0,
        0, 1, 0,
        1, 0, 1,
        0, 0, 1,
        1, 1, 0,
    ]
    data = CrowdData(
        dim=4,
        num_worker=3,
        num_task=6,
        num_option=2,
        triple=torch.tensor([workers, answers, tasks], dtype=torch.long),
    )
    data.task_y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    data.setup()
    return data


def test_annotation_audit_split_is_leakage_free_and_supported():
    data = _toy_data()
    split = make_annotation_audit_split(
        data.triple,
        data.num_task,
        num_worker=data.num_worker,
        context_fraction=1 / 3,
        min_context_workers=1,
        min_audit_workers=2,
        min_worker_context_edges=1,
        seed=9,
    )
    assert torch.all(split.context_edge_mask | split.audit_edge_mask)
    assert not torch.any(split.context_edge_mask & split.audit_edge_mask)
    assert torch.any(split.audit_edge_mask)

    workers = data.triple[0]
    context_degree = torch.bincount(
        workers[split.context_edge_mask], minlength=data.num_worker
    )
    audited_workers = torch.unique(workers[split.audit_edge_mask])
    assert torch.all(context_degree[audited_workers] >= 1)

    tasks = data.triple[2]
    for task in torch.unique(tasks[split.audit_edge_mask]):
        assert int((split.audit_edge_mask & (tasks == task)).sum().item()) >= 2


def test_predictive_cfm_response_head_has_variable_k_interface():
    data = _toy_data()
    model = PredictiveCFM(dim=4, layer=1, head=1, dropout=0.0, device="cpu")
    query_workers = torch.tensor([0, 2], dtype=torch.long)
    query_tasks = torch.tensor([1, 4], dtype=torch.long)
    output = model(
        data,
        query_workers=query_workers,
        query_tasks=query_tasks,
    )
    logits = output["hat_annotation_option"]
    assert logits.shape == (2, data.num_option)
    probabilities = torch.softmax(logits, dim=-1)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_predictive_residual_is_finite_and_two_sided():
    probabilities = torch.tensor(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.3, 0.7],
        ],
        dtype=torch.float32,
    )
    answers = torch.tensor([0, 0, 1, 0])
    workers = torch.tensor([0, 1, 0, 1])
    tasks = torch.tensor([0, 0, 1, 1])
    result = build_predictive_residual(
        probabilities,
        answers,
        workers,
        tasks,
        num_worker=2,
    )
    assert result.marginal_statistic >= 0
    assert result.dependence_statistic >= 0
    assert torch.isfinite(result.residual_matrix).all()
    assert torch.allclose(result.residual_matrix, result.residual_matrix.T)
    assert int(result.pair_counts[0, 1].item()) == 2


def test_marginal_statistic_detects_direction_under_uniform_predictions():
    probabilities = torch.full((8, 2), 0.5)
    answers = torch.zeros(8, dtype=torch.long)
    workers = torch.tensor([0, 1] * 4)
    tasks = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    result = build_predictive_residual(
        probabilities,
        answers,
        workers,
        tasks,
        num_worker=2,
    )
    assert result.marginal_statistic > 3.0


def test_conditional_predictive_test_is_reproducible():
    probabilities = torch.tensor(
        [
            [0.75, 0.25],
            [0.65, 0.35],
            [0.25, 0.75],
            [0.35, 0.65],
        ],
        dtype=torch.float32,
    )
    answers = torch.tensor([0, 0, 1, 1])
    workers = torch.tensor([0, 1, 0, 1])
    tasks = torch.tensor([0, 0, 1, 1])
    kwargs = dict(
        probabilities=probabilities,
        observed_answers=answers,
        query_workers=workers,
        query_tasks=tasks,
        num_worker=2,
        num_bootstrap=19,
        seed=123,
    )
    first = conditional_predictive_test(**kwargs)
    second = conditional_predictive_test(**kwargs)
    assert torch.equal(
        first.bootstrap_marginal_statistics,
        second.bootstrap_marginal_statistics,
    )
    assert torch.equal(
        first.bootstrap_dependence_statistics,
        second.bootstrap_dependence_statistics,
    )
    assert first.p_value == second.p_value
    assert 0.0 < first.marginal_p_value <= 1.0
    assert 0.0 < first.dependence_p_value <= 1.0
    assert 0.0 < first.p_value <= 1.0


def test_predictive_training_loss_backpropagates_to_response_head():
    data = _toy_data()
    model = PredictiveCFM(dim=4, layer=1, head=1, dropout=0.0, device="cpu")
    result = predictive_training_loss(
        model,
        data,
        config=PredictiveTrainingConfig(
            context_fraction=1 / 3,
            min_audit_workers=2,
            truth_loss_weight=1.0,
            annotation_loss_weight=1.0,
        ),
        seed=5,
    )
    result["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in model.annotation_head.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert torch.isfinite(result["loss"])


def test_full_predictive_pipeline_runs_with_trained_head_marker():
    data = _toy_data()
    model = PredictiveCFM(dim=4, layer=1, head=1, dropout=0.0, device="cpu")
    model.mark_response_head_trained()
    result = run_predictive_audit(
        model,
        data,
        config=PredictiveAuditConfig(
            context_fraction=1 / 3,
            num_bootstrap=9,
            require_trained_response_head=True,
        ),
        seed=11,
    )
    assert 0.0 < result["p_value"] <= 1.0
    assert result["num_audit_edges"] > 0
    assert result["annotation_probabilities"].shape[1] == data.num_option
