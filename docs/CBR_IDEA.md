# Residual-Audited Crowd Foundation Models

## Research question

Given a crowd foundation model pretrained on synthetic annotation worlds and a new deployment annotation graph without gold labels, can cross-fitted held-out annotations determine whether the model's conditional annotator-emission law remains compatible with the deployment process, and can rejection-based defer improve selective aggregation?

## Current idea

The official CrowdFM backbone predicts the task posterior

```math
q_k(c)=\Pr(Y_k=c\mid G_C),
```

but does not expose a predictive law for held-out worker responses. The revised model adds an edge-conditioned emission head

```math
P_{ik}(a\mid c,G_C)
=\Pr(A_{ik}=a\mid Y_k=c,G_C,i,k).
```

At deployment, the audit hides labels from the model. For each audit task it treats all held-out workers as sharing the same unknown truth `Y_k`, computes item-conditioned marginal and pairwise-disagreement residuals, and calibrates them by Monte Carlo simulation that first samples one truth per task and then samples worker responses conditional on that truth.

The two checks are:

1. a direction-sensitive marginal categorical residual;
2. a signed worker-pair disagreement matrix summarized by its two-sided operator norm.

Separate p-values are combined by Bonferroni. Rejection triggers defer.

## What changed

The previous implementation estimated full worker confusion matrices from sparse deployment labels and combined them with frozen CrowdFM task posteriors. That estimator was an external heuristic rather than an output of CrowdFM. Multiclass null rejection remained strongly inflated after refit bootstrap, posterior-predictive sampling, hierarchical shrinkage, and support diagnostics.

The revised method learns the response law amortized during pretraining, estimates no full confusion matrix at deployment, and preserves the dependence among workers induced by shared task truth.

The full failure history is in `CALIBRATION_RESULTS.md`; the formal definition is in `IDEA.md`; implementation details are in `CBR_SPEC.md`.

## Interpretation

- Rejection means the learned held-out annotation law cannot reproduce the checked deployment responses at the selected level.
- Non-rejection means compatibility with the checked law, not proof that task labels are correct.
- The conditional emission head must be trained; the original CrowdFM checkpoint alone is insufficient.
- Observationally equivalent failure processes remain undetectable without gold labels or additional anchors.
