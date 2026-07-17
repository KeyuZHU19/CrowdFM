# Certified-by-Residual for Crowd Aggregation

## Problem

A foundation aggregator can transfer across annotation matrices, but its confidence is only meaningful when the deployment noise process is compatible with the synthetic prior used during pretraining. CbR adds an instance-level posterior-predictive audit that asks whether the model can reproduce the worker-pair disagreement structure observed on held-out labels.

## Core idea

For each audit item `k`, CrowdFM receives only a context subset of its labels and produces an item posterior `q_k(c)`. Annotator confusion matrices `P_i(a | c)` are estimated from separate nuisance items. For two held-out workers `i,j`, the fitted model predicts

```math
\widetilde p_{ijk}
= 1 - \sum_c q_k(c)\sum_a P_i(a\mid c)P_j(a\mid c).
```

The observed disagreement is `D_ijk = 1[A_ik != A_jk]`. CbR aggregates signed standardized differences into a worker-pair matrix

```math
R_{ij}=
\frac{\sum_k m_{ijk}(D_{ijk}-\widetilde p_{ijk})}
{\sqrt{\sum_k m_{ijk}\widetilde p_{ijk}(1-\widetilde p_{ijk})+\lambda}}.
```

The audit statistic is the two-sided operator norm

```math
T=\lVert R\rVert_{op}
=\max(|\lambda_{max}(R)|,|\lambda_{min}(R)|).
```

A fixed-nuisance conditional Monte Carlo test preserves the deployment annotation mask, samples latent truths from `q_k`, samples held-out labels from the fitted confusion matrices, and computes an exact finite-bootstrap p-value under the fitted predictive null.

## Interpretation

- A large statistic is evidence that the fitted independent confusion model cannot explain the held-out disagreement structure.
- A small statistic is not a proof that the aggregate label is correct.
- Detectability is limited to alternatives that change the observable held-out disagreement law. Observationally equivalent coordinated behavior cannot be detected without extra information.

## Positioning

CbR is not a claim that a learned monotone link inherits every DGN theorem. Uniform symmetric noise is instead treated as a special case: its confusion matrices recover the standard DGN pairwise-disagreement identity. The primary contribution is a cross-fitted, item-conditioned, spectrally summarized posterior-predictive audit for an amortized crowd aggregator, followed by selective defer or routing.
