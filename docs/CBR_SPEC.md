# CbR Implementation Specification

## Data split

For each seed:

1. Split eligible items into nuisance and audit items.
2. Use all labels on nuisance items to estimate annotator confusion matrices.
3. Split labels on every audit item into context and held-out audit workers.
4. CrowdFM receives nuisance plus context edges only.
5. Residual construction uses held-out audit edges only.

This separation is mandatory. It prevents the model from adapting its item posterior to the same labels used to judge its predictive fit.

## Nuisance estimates

Let `q_k(c)` be CrowdFM's softmax posterior over classes. Estimate confusion matrices by posterior expected counts with symmetric Dirichlet smoothing:

```math
\widehat P_i(a\mid c)
= \frac{\eta/K + \sum_{(i,k)\in\mathcal E_N}
q_k(c)\mathbf 1[A_{ik}=a]}
{\eta + \sum_{(i,k)\in\mathcal E_N}q_k(c)}.
```

The current implementation uses a fixed-nuisance estimator. A learned confusion head can be added later, but must preserve the same cross-fit boundary.

## Item-conditioned prediction

For held-out annotations by workers `i,j` on item `k`:

```math
\widetilde p_{ijk}
=1-\sum_c q_k(c)\sum_a
\widehat P_i(a\mid c)\widehat P_j(a\mid c).
```

Do not replace `q_k` with a dataset-level class prior. Worker assignment and item composition can be non-random.

## Residual matrix

For support indicator `m_ijk`, define

```math
R_{ij}=
\frac{\sum_k m_{ijk}(D_{ijk}-\widetilde p_{ijk})}
{\sqrt{\sum_k m_{ijk}\widetilde p_{ijk}(1-\widetilde p_{ijk})+\lambda}}.
```

Unsupported worker pairs and the diagonal are zero. Use

```math
T(R)=\lVert R\rVert_{op}
```

rather than `lambda_max(R)` because structured excess agreement can appear as a large negative eigenvalue.

## Conditional Monte Carlo calibration

Condition on:

- the observed worker-item mask;
- context labels and the resulting `q_k`;
- fitted confusion matrices;
- all split masks.

For each replicate, sample one latent truth per item from `q_k`, then sample each held-out audit label independently from its worker confusion row. Recompute `R` and `T`, and use

```math
\widehat p=(1+\sum_b\mathbf 1[T_b\ge T_{obs}])/(B+1).
```

The initial implementation is fixed-nuisance. Full refitting bootstrap is an optional robustness experiment, not the default.

## Output semantics

- `reject=True`: the held-out disagreement pattern is incompatible with the fitted predictive null at level `alpha`.
- `reject=False`: insufficient evidence against the fitted predictive null.
- Neither outcome directly certifies ground-truth accuracy.

## Planned extensions

1. Repeated cross-fitting and p-value aggregation.
2. Learned K-invariant confusion head from worker embeddings.
3. OOD generators for coalition dependence, temporal drift, item-dependent skill, and class-conditioned shift.
4. Selective routing and risk-coverage evaluation.
