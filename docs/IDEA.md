# Residual-Audited Crowd Foundation Models

## One-sentence definition

We extend a crowd foundation model with masked-annotation prediction and use cross-fitted held-out annotations to test whether its deployment-time response distribution is compatible with the observed crowd, deferring aggregation when that predictive law is rejected.

## 1. Research problem

Let a deployment dataset contain workers `i in [M]`, tasks `k in [N]`, options `a in [K]`, an observation mask `O`, and observed annotations `A_ik` wherever `O_ik = 1`.  The task truth `Y_k` is unavailable at deployment.

A conventional CrowdFM backbone learns the discriminative aggregation map

```text
annotation graph G -> q_theta(Y_k | G).
```

A high-confidence `q_theta` does not reveal whether the deployment annotation process resembles the synthetic worlds used in pretraining.  Without gold labels, correctness itself is not identifiable: two latent worlds can induce the same observed graph while assigning different truths.  The defensible label-free question is therefore not

```text
Is the predicted truth certainly correct?
```

but

```text
Can the model predict annotations that were not shown to it?
```

Formally, given a context graph `G_C` and held-out edge set `H`, the model must expose

```math
r_{ik}(a)
=
Pr_theta(A_{ik}=a | G_C, i, k),
qquad (i,k) in H.
```

The deployment audit tests

```math
H_0:
\mathcal L(A_H | G_C,H)
=
\prod_{(i,k)\in H}\operatorname{Categorical}(r_{ik})
```

against detectable departures in either marginal response prediction or cross-worker dependence.  Non-rejection means compatibility with this checked predictive law, not proof that the aggregated labels are correct.

## 2. Why the previous implementation was not the intended method

The first code path used the frozen CrowdFM task posterior `q_k` and fitted a full worker-specific confusion matrix

```math
\widehat P_i(a|c)
```

from deployment annotations.  It then manufactured response predictions through

```math
\widehat r_{ik}(a)
=
\sum_c q_k(c)\widehat P_i(a|c).
```

This bridge was not produced or trained by the official CrowdFM checkpoint.  In sparse multiclass data, each worker/class row had only a few effective observations but `K-1` free parameters.  Hierarchical shrinkage reduced entrywise error but did not calibrate the audit: estimated-confusion null rejection remained 20--75% across the main settings, while oracle-confusion variants stayed close to the nominal level.  Increasing total task count also increased audit power, so small remaining nuisance bias continued to be detected.

The conclusion is structural:

> A post-hoc full confusion estimator cannot be treated as the predictive law of a discriminative crowd foundation model.

The old implementation remains in the repository only to reproduce this negative result.  It is not the primary method.

## 3. Predictive CrowdFM

### 3.1 Backbone

The original backbone maps the visible annotation graph to worker, task, and option embeddings

```math
z_i^w,\quad z_k^t,\quad z_a^o,
```

and task-truth logits.

### 3.2 Masked-annotation response head

For every held-out query edge `(i,k)`, a shared option scorer predicts

```math
\ell_{ika}
=
h_\phi(z_i^w,z_k^t,z_a^o),
\qquad
r_{ik}(a)=\operatorname{softmax}_a(\ell_{ika}).
```

The scorer is shared over options and therefore supports variable `K`.  Unlike a stationary confusion matrix, `r_ik` may depend on worker history, task context, item difficulty encoded by the graph, and the current option embeddings.

### 3.3 Training objective

During pretraining, randomly hide annotation edges before the forward pass.  The hidden responses supervise

```math
\mathcal L_{resp}
=
-\sum_{(i,k)\in H}
\log r_{ik}(A_{ik}).
```

When synthetic task truth is available, retain the original aggregation objective

```math
\mathcal L_{truth}
=
-\sum_k\log q_k(Y_k),
```

and train

```math
\mathcal L
=
\lambda_y\mathcal L_{truth}
+
\lambda_a\mathcal L_{resp}.
```

Real crowd matrices without gold truth can still contribute to `L_resp` as self-supervised data.  An official CrowdFM checkpoint may initialize the backbone, but the new response head must be trained before any audit claim is made.

## 4. Leakage-free deployment split

For each auditable task, split its observed workers into a visible context set and a held-out audit set.  Every remaining audit worker must have a minimum amount of visible history elsewhere in the context graph.  The model receives only context edges.  Audit labels are used only after all response probabilities have been fixed.

This cross-fitting condition is load-bearing: using an audited annotation to construct its own prediction would make residuals artificially small.

## 5. Predictive residuals

For audit edge `e=(i,k)`, define the standardized categorical residual vector

```math
u_{ik,a}
=
\frac{\mathbf 1\{A_{ik}=a\}-r_{ik}(a)}
{\sqrt{r_{ik}(a)(1-r_{ik}(a))+\epsilon}}.
```

The audit uses two complementary checks.

### 5.1 Marginal response calibration

Let `L_ik=-log r_ik(A_ik)`.  Under the fixed predictive row `r_ik`, its conditional mean and variance are available exactly.  The standardized aggregate log-score deviation is

```math
T_{marg}
=
\frac{\left|\sum_{(i,k)\in H}
[L_{ik}-\mathbb E_{r_{ik}}L_{ik}]\right|}
{\sqrt{\sum_{(i,k)\in H}\operatorname{Var}_{r_{ik}}(L_{ik})+\epsilon}}.
```

This detects direction-specific response misprediction that binary disagreement would discard.

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

Under the conditionally independent predictive null, `E[R_ij | G_C]=0`.  A coalition, shared bias, temporal shock, or unmodelled worker dependence can create a coherent signed block.  The dependence statistic is the two-sided operator norm

```math
T_{dep}=\lVert R\rVert_{op}.
```

The leading absolute-eigenvalue eigenvector can localize workers driving rejection.

## 6. Conditional Monte Carlo audit

Condition on the context graph, exact audit mask, query workers/tasks, and fixed response rows `r_ik`.  For replicate `b`, independently sample

```math
A_{ik}^{(b)}\sim\operatorname{Categorical}(r_{ik})
```

and recompute both statistics.  Plus-one Monte Carlo p-values are

```math
p_s
=
\frac{1+\sum_{b=1}^{B}\mathbf 1\{T_s^{(b)}\ge T_s^{obs}\}}
{B+1},
\qquad s\in\{marg,dep\}.
```

The joint test uses the Bonferroni value

```math
p_{joint}=\min\{1,2\min(p_{marg},p_{dep})\}.
```

Under `H_0`, each component p-value is finite-sample conditionally valid by exchangeability, and Bonferroni controls the joint false-rejection probability without assuming independence between the two statistics.

The system emits the base aggregation only when `p_joint > alpha`; otherwise it defers.

## 7. Claims and non-claims

### Valid claims

1. Conditional finite-sample type-I control when the fixed held-out response law is correct.
2. A direct label-free test of deployment response prediction rather than a heuristic confidence score.
3. Sensitivity to both marginal annotation shift and structured cross-worker dependence.
4. A selective aggregation mechanism whose utility can be measured through risk--coverage.

### Invalid claims

1. Non-rejection does not certify task-truth correctness.
2. The audit cannot detect a process that induces the same conditional held-out response law.
3. Random response-head weights or an untrained official CrowdFM checkpoint do not define a valid audit.
4. A rejected model does not imply that any particular fallback is automatically correct.

## 8. Required experimental program

### Phase A: predictive model quality

- masked-annotation NLL, Brier score, and calibration;
- comparison with worker-frequency, Dawid--Skene, 3PL/GLAD, and unconditional baselines;
- frozen-backbone head training versus joint fine-tuning;
- synthetic-only versus synthetic plus real self-supervised pretraining.

### Phase B: null calibration

Generate unseen in-prior worlds and report rejection at `alpha in {0.01,0.05,0.1}` across `M,N,K`, density, imbalance, and worker heterogeneity.  The primary requirement is that the learned response law itself is calibrated; an exact Monte Carlo test cannot repair a misspecified predictor.

### Phase C: structured shift power

Hold out complete mechanisms from pretraining:

- class-conditioned errors;
- task-difficulty shifts;
- worker coalitions and dependence;
- non-random assignment;
- temporal drift;
- Sybil or adversarial workers.

Report marginal/dependence detection separately, joint power, worker localization, and severity curves.

### Phase D: selective aggregation

Measure clean accuracy, accepted-set accuracy, coverage, risk--coverage AUC, and the gain or loss produced by each defer route.  Compare against CrowdFM entropy/margin, ensemble or split-view instability, latent OOD scores, and classical aggregation diagnostics.

## 9. Current implementation map

Primary implementation:

- `src/cfm/model/PredictiveCFM.py`: response head and CrowdFM wrapper;
- `src/cfm/audit/predictive_split.py`: leakage-free annotation split;
- `src/cfm/audit/predictive.py`: categorical and spectral residuals;
- `src/cfm/audit/predictive_bootstrap.py`: conditional Monte Carlo test;
- `src/cfm/audit/predictive_pipeline.py`: deployment audit;
- `src/cfm/audit/predictive_training.py`: joint masked-response/truth objective;
- `tests/test_predictive_audit.py`: unit tests for the new path.

Historical implementation:

- `disagreement.py`, `bootstrap.py`, `pipeline.py`, and `calibration.py` reproduce the confusion-estimation study and are retained as a documented negative control.
