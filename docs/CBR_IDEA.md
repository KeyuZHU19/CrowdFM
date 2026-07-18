# Residual-Audited Crowd Foundation Models

## Research question

Given a crowd foundation model pretrained on synthetic annotation worlds and a new deployment annotation graph without gold labels, can cross-fitted held-out annotations determine whether the model's conditional response distribution remains compatible with the deployment process, and can rejection-based defer improve selective aggregation?

## Current idea

The official CrowdFM backbone predicts task truth but does not predict how a specific worker will label a specific task.  The revised method therefore adds a masked-annotation response head

```math
r_{ik}(a)=Pr_\theta(A_{ik}=a\mid G_C,i,k),
```

trained by hiding annotation edges and predicting their labels from the remaining context graph.

At deployment, audit labels are never exposed to the model.  Their categorical residuals provide:

1. a marginal log-score calibration statistic;
2. a signed worker-pair residual covariance matrix whose two-sided operator norm detects coherent dependence or shared bias.

Conditional Monte Carlo samples audit responses directly from the fixed rows `r_ik`.  Separate marginal and dependence p-values are combined by Bonferroni.  Rejection triggers defer.

## What changed

The previous implementation estimated full worker confusion matrices from deployment labels and combined them with CrowdFM task posteriors.  That estimator was an external heuristic rather than an output of the frozen CrowdFM model.  Multiclass null rejection remained strongly inflated even after refit bootstrap, posterior-predictive sampling, hierarchical shrinkage, and higher-support diagnostics.

The full history is recorded in `CALIBRATION_RESULTS.md`.  The formal revised method is in `IDEA.md`.

## Interpretation

- Rejection means that the learned held-out response law cannot reproduce the checked deployment annotations at the selected level.
- Non-rejection means compatibility with the checked predictive law, not proof that task labels are correct.
- The response head must be trained; an official CrowdFM checkpoint alone is insufficient.
- Observationally equivalent failure processes remain undetectable without gold labels or additional information.
