# Predictive Audit Implementation Specification

## 1. Model contract

The primary audit no longer estimates a deployment confusion matrix.  A compatible model must accept

```python
output = model(
    context_data,
    query_workers=query_workers,
    query_tasks=query_tasks,
)
```

and return

```python
output["hat_annotation_option"]  # [num_query_edges, num_options]
```

These logits represent the conditional response distribution for each held-out `(worker, task)` edge.  `PredictiveCFM` implements this interface with a scorer shared across candidate options.

The response head must be trained by masked-annotation prediction.  `run_predictive_audit` refuses checkpoints whose `PredictiveCFM.response_head_trained` marker is false.

## 2. Context/audit split

`make_annotation_audit_split` partitions every edge into exactly one of:

- context: visible to the model;
- audit: hidden from the model and used only after response probabilities are fixed.

Each audited task retains at least:

- `min_context_workers` visible labels;
- `min_audit_workers >= 2` hidden labels.

Each worker remaining in the audit set has at least `min_worker_context_edges` visible annotations elsewhere in the context graph.  If support repair leaves a task with fewer than two audit labels, that task is returned entirely to context.

## 3. Training objective

For audit query edge `e=(i,k)` with observed label `A_e`, train

```math
\mathcal L_{resp}
=-\sum_e\log r_e(A_e).
```

When synthetic truth is present, add the original task classification loss:

```math
\mathcal L
=\lambda_a\mathcal L_{resp}+\lambda_y\mathcal L_{truth}.
```

The helper `predictive_training_loss` constructs a fresh masked split and returns both losses.  Frozen-backbone head training is the initial baseline; joint fine-tuning is an ablation, not an assumption of the audit.

## 4. Categorical residual

For predicted row `r_e` and observed answer `A_e`, define

```math
u_{e,a}
=
\frac{1[A_e=a]-r_e(a)}
{\sqrt{r_e(a)(1-r_e(a))+\epsilon}}.
```

Probabilities are clipped and renormalized using `probability_clip` before residual construction and Monte Carlo sampling.

## 5. Marginal statistic

For each edge, let `L_e=-log r_e(A_e)`.  Its exact conditional mean and variance under `A_e~Categorical(r_e)` are used to construct

```math
T_{marg}
=
\frac{|\sum_e(L_e-E L_e)|}
{\sqrt{\sum_e Var(L_e)+\epsilon}}.
```

This detects direction-specific marginal response error and does not reduce labels to a binary disagreement indicator.

## 6. Dependence statistic

For worker pair `(i,j)` sharing `n_ij` audit tasks,

```math
R_{ij}
=
\frac{1}{\sqrt{n_{ij}}}
\sum_k \frac{u_{ik}^T u_{jk}}{K},
\qquad R_{ii}=0.
```

Pairs with fewer than `min_pair_count` shared audit tasks are masked.  The statistic is

```math
T_{dep}=\lVert R\rVert_{op}
=\max(|\lambda_{max}(R)|,|\lambda_{min}(R)|).
```

The two-sided norm is required because both excess disagreement and excess agreement can indicate misspecification.

## 7. Conditional Monte Carlo test

Condition on the context graph, query identities, exact audit mask, and fixed response rows.  For every replicate sample each audit label independently from its row and recompute both statistics.

For `s in {marg, dep}`:

```math
p_s
=
\frac{1+\sum_b1[T_s^{(b)}\ge T_s^{obs}]}{B+1}.
```

The reported joint value is

```math
p_{joint}=\min(1,2\min(p_{marg},p_{dep})).
```

Under the fixed predictive null, plus-one Monte Carlo exchangeability makes each component p-value conditionally valid.  Bonferroni controls the joint false-rejection probability without assuming independence between statistics.

## 8. Output semantics

- `reject=True`: the checked held-out response law is rejected at `alpha`;
- `reject=False`: the audit lacks evidence against that law;
- neither result directly proves or disproves task-truth correctness.

The result returns separate p-values and statistics, response probabilities, support counts, split masks, and bootstrap samples.

## 9. Legacy implementation

`pipeline.py`, `disagreement.py`, `bootstrap.py`, and `calibration.py` implement the retired post-hoc confusion audit.  They remain importable only to reproduce the negative calibration study.  New experiments must use the `predictive_*` modules.
