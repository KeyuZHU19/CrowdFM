# Residual-Audited Crowd Foundation Models

## One-sentence definition

We extend CrowdFM to predict an edge-conditioned annotator emission law given each candidate task truth, then use cross-fitted held-out annotations and conditional Monte Carlo simulation to reject deployment crowds that the learned predictive law cannot reproduce.

## 1. Research problem

For workers `i in [M]`, tasks `k in [N]`, and options `a in [K]`, let `A_ik` be the observed crowd label and `Y_k` the unavailable task truth. The original CrowdFM only outputs

```math
q_k(c)=\Pr_\theta(Y_k=c\mid G_C),
```

where `G_C` is the visible annotation graph. A confident `q_k` does not show whether the deployment annotation process is compatible with the synthetic pretraining prior.

Without gold labels, task-truth correctness cannot be certified in general. The label-free research problem is instead:

> Can a crowd foundation model predict annotations that were hidden from it, including the dependence among workers induced by their shared unknown task truth?

## 2. Predictive model

For every held-out query edge `(i,k)`, the revised model outputs an edge-conditioned emission matrix

```math
P_{ik}(a\mid c,G_C)
=
\Pr_\phi(A_{ik}=a\mid Y_k=c,G_C,i,k).
```

A shared scorer consumes worker, task, candidate-truth-option, and candidate-reported-option embeddings, so one head supports variable `K`.

The conditional predictive null is

```math
Y_k\sim q_k,
\qquad
A_{ik}\mid Y_k=c,G_C\sim P_{ik}(\cdot\mid c,G_C),
```

independently across held-out workers only after conditioning on the shared `Y_k`. Marginally, workers on the same task are dependent whenever `q_k` is uncertain.

This shared latent-truth structure is essential. Treating the marginal response probabilities of different workers as independent would falsely flag legitimate agreement caused by uncertainty about `Y_k`.

## 3. Training

Hide annotation edges before the forward pass. If synthetic truth is known, train the correct emission row:

```math
\mathcal L_{emit}
=
-\sum_{(i,k)\in H}\log P_{ik}(A_{ik}\mid Y_k,G_C).
```

Retain the original task-truth loss

```math
\mathcal L_{truth}=-\sum_k\log q_k(Y_k).
```

For a real task without gold truth, use the marginal response likelihood

```math
\Pr(A_{ik}=a\mid G_C)
=
\sum_c q_k(c)P_{ik}(a\mid c,G_C).
```

An official CrowdFM checkpoint can initialize the backbone, but the new conditional annotation head must be trained before deployment auditing.

## 4. Cross-fitting

Each auditable task keeps at least one visible context subset and at least two held-out audit workers. Every worker remaining in the audit set must have a minimum visible history elsewhere. Audit labels are never used to construct `q_k` or `P_ik`.

## 5. Audit statistics

### 5.1 Marginal categorical residual

First integrate out task truth:

```math
r_{ik}(a)=\sum_c q_k(c)P_{ik}(a\mid c,G_C).
```

For each option,

```math
z_a
=
\frac{\sum_{(i,k)\in H}[\mathbf 1\{A_{ik}=a\}-r_{ik}(a)]}
{\sqrt{\sum_{(i,k)\in H}r_{ik}(a)(1-r_{ik}(a))+\epsilon}},
\qquad
T_{marg}=\lVert z\rVert_2.
```

This retains the direction of class-specific response shift.

### 5.2 Item-conditioned pairwise disagreement residual

For held-out workers `i,j` on task `k`, the model predicts

```math
p_{ijk}^{dis}
=
1-
\sum_c q_k(c)
\sum_a P_{ik}(a\mid c,G_C)P_{jk}(a\mid c,G_C).
```

The worker-pair residual matrix is

```math
R_{ij}
=
\frac{\sum_k m_{ijk}[\mathbf 1\{A_{ik}\ne A_{jk}\}-p_{ijk}^{dis}]}
{\sqrt{\sum_k m_{ijk}p_{ijk}^{dis}(1-p_{ijk}^{dis})+\epsilon}},
\qquad R_{ii}=0.
```

Use the two-sided operator norm

```math
T_{dep}=\lVert R\rVert_{op}.
```

The signed leading eigenvector can localize coherent worker groups whose agreement or disagreement is not explained by the learned law.

## 6. Conditional Monte Carlo test

Condition on the context graph, split, `q_k`, and all edge-conditioned emissions. For each replicate:

1. sample one shared `Y_k` per audit task from `q_k`;
2. sample each held-out response from its emission row given that shared truth;
3. recompute `T_marg` and `T_dep`.

Use plus-one p-values for both statistics and combine them by

```math
p_{joint}=\min\{1,2\min(p_{marg},p_{dep})\}.
```

The observed audit array and simulated arrays are exchangeable under the fixed predictive null, giving finite-sample conditional validity. Because Bonferroni doubles the minimum component p-value, the code requires enough Monte Carlo samples for the requested `alpha`.

## 7. What changed from the failed implementation

The retired path estimated a stationary full confusion matrix for every worker from sparse deployment labels and combined it with frozen CrowdFM posteriors. That matrix was not an output of CrowdFM and remained badly miscalibrated in multiclass settings despite refit bootstrap and hierarchical shrinkage.

The revised method:

- learns the emission law amortized during pretraining;
- allows emissions to depend on worker, task, context, and candidate truth;
- estimates no full confusion matrix at deployment;
- preserves dependence caused by a shared latent truth;
- audits the same predictive law that the model was trained to expose.

## 8. Claims and limitations

Valid claim: rejection means the learned held-out annotation law is incompatible with the checked deployment responses at the selected test level.

Invalid claim: non-rejection proves the aggregated task truth is correct.

Observationally equivalent processes remain undetectable without gold labels, trusted workers, metadata, temporal information, or another anchor.

## 9. Implementation

Primary files:

- `src/cfm/model/PredictiveCFM.py`;
- `src/cfm/audit/predictive_split.py`;
- `src/cfm/audit/predictive.py`;
- `src/cfm/audit/predictive_bootstrap.py`;
- `src/cfm/audit/predictive_pipeline.py`;
- `src/cfm/audit/predictive_training.py`;
- `train_predictive.py`;
- `evaluate_predictive_audit.py`;
- `tests/test_predictive_audit.py`.

Legacy confusion-based files are retained only to reproduce the negative calibration study documented in `CALIBRATION_RESULTS.md`.
