# Residual-Audited Crowd Foundation Models

## One-sentence definition

We extend a crowd foundation model with masked-annotation prediction and use cross-fitted held-out annotations to test whether its deployment-time worker-response distribution is compatible with the observed crowd, deferring aggregation when that predictive law is rejected.

## 1. Research problem

A deployment dataset contains workers `i in [M]`, tasks `k in [N]`, options `a in [K]`, an observation mask `O`, and annotations `A_ik` wherever `O_ik=1`. The task truth `Y_k` is unavailable at deployment.

The original CrowdFM learns the discriminative aggregation map

```text
annotation graph G -> q_theta(Y_k | G).
```

A confident task posterior does not reveal whether the deployment annotation process resembles the synthetic worlds used in pretraining. Without gold labels, task-truth correctness is not identifiable in general: the same observed graph may arise from latent worlds with different truths.

The defensible label-free question is therefore:

> Can the model predict annotations that were not shown to it?

Given a context graph `G_C` and held-out edge set `H`, the revised model exposes

```math
r_{ik}(a)
=
\Pr_\theta(A_{ik}=a\mid G_C,i,k),
\qquad (i,k)\in H.
```

The deployment audit checks the factorized conditional predictive null

```math
H_0:
\mathcal L(A_H\mid G_C,H)
=
\prod_{(i,k)\in H}\operatorname{Categorical}(r_{ik}).
```

The alternative consists of detectable departures in marginal response probabilities or cross-worker dependence. Non-rejection means compatibility with this checked predictive law, not proof that the aggregated task labels are correct.

## 2. Why the previous implementation was abandoned

The first implementation used the frozen CrowdFM task posterior `q_k` and estimated a full worker-specific confusion matrix

```math
\widehat P_i(a\mid c)
```

from deployment annotations. It then manufactured response probabilities through

```math
\widehat r_{ik}(a)
=
\sum_c q_k(c)\widehat P_i(a\mid c).
```

This response law was neither produced nor trained by the official CrowdFM checkpoint. In sparse multiclass settings, each worker/class row had only a few effective observations but `K-1` free parameters. Refit bootstrap, posterior-predictive sampling, prior-strength sweeps, and hierarchical shrinkage did not restore null calibration. Oracle-confusion variants stayed near nominal, while estimated-confusion rejection remained as high as 65--75% in the ten-class setting.

The conclusion is structural:

> A post-hoc full confusion estimator cannot be treated as the predictive law of a discriminative crowd foundation model.

The old implementation remains only as a reproducible negative result.

## 3. Predictive CrowdFM

### 3.1 Backbone

The original backbone maps the visible annotation graph to worker, task, and option embeddings

```math
z_i^w,\qquad z_k^t,\qquad z_a^o,
```

and to task-truth logits.

### 3.2 Masked-annotation response head

For every held-out query edge `(i,k)`, a scorer shared across candidate options predicts

```math
\ell_{ika}=h_\phi(z_i^w,z_k^t,z_a^o),
\qquad
r_{ik}(a)=\operatorname{softmax}_a(\ell_{ika}).
```

Sharing the scorer over options permits variable `K`. Unlike a stationary confusion matrix, `r_ik` may depend on worker history, task context, task difficulty encoded by the graph, and option representations.

### 3.3 Training objective

Randomly hide annotation edges before the forward pass. Their labels supervise

```math
\mathcal L_{resp}
=
-\sum_{(i,k)\in H}\log r_{ik}(A_{ik}).
```

When synthetic task truth is available, retain the original aggregation objective

```math
\mathcal L_{truth}
=
-\sum_k\log q_k(Y_k),
```

and optimize

```math
\mathcal L
=
\lambda_a\mathcal L_{resp}
+
\lambda_y\mathcal L_{truth}.
```

Real annotation matrices without gold task truth can still contribute to `L_resp` as self-supervised training data. An official CrowdFM checkpoint may initialize the backbone, but the new response head must be trained before its probabilities are audited.

## 4. Leakage-free deployment split

For each auditable task, split observed workers into a visible context set and a held-out audit set. Every remaining audit worker must have a minimum amount of visible history elsewhere in the context graph. The model receives only context edges; audit labels are used only after all response probabilities have been fixed.

Cross-fitting is load-bearing. If an audited annotation participates in constructing its own probability row, a flexible model can shrink its residual artificially.

## 5. Predictive residuals

For audit edge `(i,k)`, define the standardized categorical residual vector

```math
u_{ik,a}
=
\frac{\mathbf 1\{A_{ik}=a\}-r_{ik}(a)}
{\sqrt{r_{ik}(a)(1-r_{ik}(a))+\epsilon}}.
```

The audit uses two complementary statistics.

### 5.1 Marginal categorical calibration

A scalar aggregate log score can be blind to directional shift when predicted rows are uniform. The implementation therefore retains the full option direction:

```math
z_a
=
\frac{\sum_{(i,k)\in H}
[\mathbf 1\{A_{ik}=a\}-r_{ik}(a)]}
{\sqrt{\sum_{(i,k)\in H}r_{ik}(a)(1-r_{ik}(a))+\epsilon}},
```

and defines

```math
T_{marg}=\lVert z\rVert_2.
```

The categorical coordinates are negatively correlated, but the method does not use a Gaussian approximation to calibrate this statistic. Its exact finite-instance null distribution is simulated from the fixed categorical rows.

### 5.2 Structured worker dependence

For worker pair `(i,j)`, accumulate residual alignment on shared audit tasks:

```math
R_{ij}
=
\frac{1}{\sqrt{n_{ij}}}
\sum_{k:(i,k),(j,k)\in H}
\frac{1}{K}u_{ik}^{\top}u_{jk},
\qquad R_{ii}=0.
```

Under the conditionally independent predictive null,

```math
\mathbb E[R_{ij}\mid G_C]=0.
```

A coalition, shared bias, temporal shock, or unmodelled dependence can create a coherent signed block. The dependence statistic is

```math
T_{dep}=\lVert R\rVert_{op}
=
\max\{|\lambda_{max}(R)|,|\lambda_{min}(R)|\}.
```

The leading absolute-eigenvalue eigenvector can localize the workers driving rejection.

## 6. Conditional Monte Carlo audit

Condition on the context graph, exact audit mask, query worker/task identities, and fixed response rows `r_ik`. For replicate `b`, independently sample

```math
A_{ik}^{(b)}\sim\operatorname{Categorical}(r_{ik})
```

and recompute both statistics. For `s in {marg,dep}` use the plus-one p-value

```math
p_s
=
\frac{1+\sum_{b=1}^{B}\mathbf 1\{T_s^{(b)}\ge T_s^{obs}\}}
{B+1}.
```

Combine the two valid component tests by

```math
p_{joint}=\min\{1,2\min(p_{marg},p_{dep})\}.
```

Under `H_0`, the observed audit labels and Monte Carlo replicates are exchangeable conditional on the fixed response rows. Each component p-value is therefore finite-sample conditionally valid, and Bonferroni controls the joint false-rejection probability without requiring independence between statistics.

The system emits the base aggregation only when `p_joint>alpha`; otherwise it defers.

## 7. What the paper can claim

### Valid claims

1. A direct label-free test of held-out response prediction rather than a heuristic confidence score.
2. Conditional finite-sample type-I control when the fixed response law is correct.
3. Complementary sensitivity to marginal annotation shift and structured cross-worker dependence.
4. Worker localization through the signed spectral residual.
5. A selective aggregation mechanism whose utility is evaluated through risk--coverage.

### Claims to avoid

1. Non-rejection does not certify task-truth correctness.
2. A process inducing the same conditional held-out response law is not detectable by this audit.
3. An untrained response head or the original CrowdFM checkpoint alone does not define a valid audit.
4. Rejection does not make any particular fallback automatically correct.
5. Exact Monte Carlo calibration cannot repair a systematically misspecified learned response predictor.

## 8. Experimental program

### A. Response-model quality

- masked-annotation NLL, Brier score, accuracy, and calibration;
- worker-frequency, Dawid--Skene, 3PL/GLAD, and unconditional baselines;
- frozen-backbone head training versus joint fine-tuning;
- synthetic-only versus synthetic plus real self-supervised training.

### B. In-prior null calibration

Evaluate empirical rejection at multiple nominal levels over unseen worlds stratified by `M,N,K`, sparsity, imbalance, worker support, and task difficulty. Include oracle response rows to isolate the statistical test from response-model error.

### C. Structured shift power

Hold out complete mechanisms from pretraining:

- class-conditioned errors;
- task-difficulty shifts;
- worker coalitions and dependence;
- non-random assignment;
- temporal drift;
- Sybil and targeted adversarial workers.

Report marginal and dependence power separately, joint power, localization, and severity curves.

### D. Selective aggregation

Measure coverage, accepted-set risk, AURC, accuracy at fixed coverage, fallback gain/penalty, and oracle-router gap. Compare against CrowdFM entropy/margin, worker-subsampling instability, latent OOD scores, and classical diagnostics.

## 9. Implementation map

Primary implementation:

- `src/cfm/model/PredictiveCFM.py`;
- `src/cfm/audit/predictive_split.py`;
- `src/cfm/audit/predictive.py`;
- `src/cfm/audit/predictive_bootstrap.py`;
- `src/cfm/audit/predictive_pipeline.py`;
- `src/cfm/audit/predictive_training.py`;
- `train_predictive.py`;
- `evaluate_predictive_audit.py`;
- `tests/test_predictive_audit.py`.

Historical implementation:

- `disagreement.py`, `bootstrap.py`, `pipeline.py`, `calibration.py`, and `run_cbr_calibration.py` reproduce the confusion-estimation study and are retained as documented negative controls.
