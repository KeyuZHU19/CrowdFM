# CrowdGuard — Project Status & Roadmap

*Active Expert Anchoring for Crowd Foundation Models.*
Consolidated status as of 2026-07-19. Detailed running log: `RESULTS.md`
(mirrored on GitHub as `docs/EXPERIMENT_LOG.md`, branch `agent/crowdguard`).

---

## 0. TL;DR

- **CrowdSI-FM** (mechanism system identification to improve aggregation) is **falsified**;
  earlier gains were baseline contamination. **STOPPED.**
- **CrowdGuard** (inject a new information source — a few trusted expert anchors — to
  actively identify annotator reliability and correct a frozen crowd FM) is validated to
  beat CrowdFM (+2–4pp), BUT a baseline reality-check (§3.5) shows it beats the best
  *classic* anchor-aggregator (no FM) **only under adversarial (Sybil) attack**.
- **DIRECTION DECIDED — plan C: adversarial deployment as the main battlefield, active
  reliability-identification (VOI) selection as the method core.** Story: *"Active expert
  anchoring for robust crowd aggregation under adversarial deployment shift."* This is the
  one setting where the full method + FM prior clearly wins, and VOI is a robust,
  model-agnostic novelty (beats uncertainty sampling everywhere).

---

## 1. The question

> When a crowd foundation model (CrowdFM) fails at deployment, how do we use a very small
> number of expert anchors to actively identify annotator reliability, and correct the
> remaining tasks at minimal expert cost?

Working title: **CrowdGuard: Active Expert Anchoring for Crowd Foundation Models.**

Given: crowd annotation graph `G`, frozen CrowdFM, expert-label budget `B ≪ N`, reliable
expert labels `Y*_k`. Decide (i) which `B` tasks to query, (ii) how to correct the rest.

Why this can work when CrowdSI could not: everything in CrowdSI read the *same* `G`, so it
could not create truth information CrowdFM had not already extracted. Expert anchors change
identifiability `G → (G, Y*_S)` and inject genuinely new signal.

---

## 2. Method (CrowdGuard)

Three components on a **frozen** CrowdFM:

1. **Anchor-conditioned correction** (starts = CrowdFM at B=0):
   `logit_k = log q_CrowdFM(k) + Σ_i wgt_i · onehot(vote_ik)`,
   where `wgt_i` = anchor-estimated worker reliability log-odds (negative for
   systematically-wrong / coalition / Sybil workers ⇒ their votes are subtracted).
   Current version is a no-training numpy estimator (a *lower bound*); a **learned** head
   `h_φ` is the next step (§4.1).

2. **Active VOI selection** (the novelty). Adaptive greedy: after each anchor, update
   worker reliability beliefs, then anchor the task that resolves the most
   *influential* AND *(reliability-uncertain OR consensus-deviating/suspect)* workers,
   boosted by the task's own risk (double benefit: fix a likely-wrong task for free AND
   inform the correction). Beats random and **decisively** beats uncertainty sampling.

3. **Escalation gate** (Stage A) — *open problem, reframed*. The task-disjoint e-value is a
   valid, calibrated *shift detector*, but shift ≠ correctable benefit, so it does **not**
   predict where anchoring pays off. Since the correction is low-risk, "always apply a
   small budget" is a safe default; a good gold-free spend predictor remains open.

---

## 3. Results (validated)

### 3.1 Anchor headroom (ceiling) — large
Oracle worker-model (full gold) vs frozen CrowdFM: **+14.1pp (mixed), +31.4pp (coalition),
+35.5pp (sparse)**. Confirms real headroom (opposite of CrowdSI's ~0 oracle-latent).

### 3.2 Active selection — VOI wins across regimes
Effective accuracy vs budget; VOI = ours, oracle = truth-greedy upper bound.

| regime | CrowdFM | VOI − random | VOI − entropy | AURC: VOI / random / entropy / oracle |
|---|---:|---|---|---|
| mixed (in-prior) | 0.851 | +0.7…+1.2pp | +3.2…+8.4pp | 0.111 / 0.119 / 0.153 / 0.103 |
| coalition | 0.675 | +0.5…+4.2pp | +1.1…+5.9pp | 0.254 / 0.271 / 0.284 / 0.247 |
| sparse | 0.549 | +0.6…+3.3pp | +0.2…+0.7pp | 0.388 / 0.403 / 0.392 / 0.389 |
| **sybil (adversarial)** | 0.687 | **+1.5…+4.4pp** | **+2.2…+4.9pp** | **0.238** / 0.257 / 0.265 / 0.226 |

**Key finding:** uncertainty sampling (entropy) is *worse than random* (negative gains at
low budget) — anchoring ambiguous tasks yields noisy reliability estimates. The value is in
reliability *identification*, not task uncertainty. Sybil is the showcase (widest VOI
margin; CrowdFM 0.687 → 0.855 at B=50).

### 3.3 Real crowds — validated
Mean gain over 14 valid real datasets: VOI **+1.8 / +2.4 / +4.1pp** at B=10/20/50; random
+0.7/+0.9/+2.1; entropy negative at low B. Standouts: Bird +20pp, CF +8.7pp, ZC_in +3.5pp.
(Trec, Senti excluded — CrowdFM below chance ⇒ dataset loading index-misalignment to fix.)

### 3.4 Escalation gate — honest negative
No gold-free signal predicts anchor benefit across datasets (Spearman: e-value −0.07,
CrowdFM entropy −0.04, disagreement −0.02; CrowdFM accuracy −0.42 but needs gold).
Reframed as low-risk always-apply.


### 3.5 ⚠ Baseline reality-check (important)
vs CLASSIC anchor-aggregators (same VOI anchors, no FM): CrowdGuard beats the best no-FM
baseline CLEARLY only under **Sybil/adversarial** (+2 to +4.5pp). On benign/mixed
semi-supervised DS+anchors is better (−2pp); on coalition/sparse a reliability-weighted
majority vote ties it (on sparse mv_rw 0.585 > CrowdFM 0.546, so the FM prior adds little).
**The FM prior only pulls weight under adversarial attack.** What robustly stands: (1) VOI
active selection beats uncertainty sampling for ALL aggregators (model-agnostic, novel);
(2) anchoring recovers accuracy at small budget. Framing must NOT claim broad "FM+anchors
wins"; options: center on active reliability-ID selection, and/or narrow to adversarial
deployment. **DECISION (2026-07-19): plan C — adversarial-deployment focus + VOI active
reliability-ID as the method core.**

---

## 4. Roadmap — PLAN C (adversarial focus)

### 4.1 Adversarial attack-strength sweep — MONEY PLOT (in progress)
Sweep Sybil fraction ρ ∈ {0.1…0.5} (and coalition strength) at fixed budget. Show:
(a) CrowdFM degrades with ρ; (b) CrowdGuard's advantage over the best classic
anchor-aggregator (mv_rw, DS+anchors) GROWS with ρ; (c) VOI beats random/entropy
throughout. This is the core figure for plan C.
### 4.2 Solidify: ≥3 seeds, tighten CIs, budget × attack-strength grid.
### 4.3 Baselines (partly done): CrowdFM, MV, reliability-MV, DS+anchors, honeypot/
   gold-injection; position novelty = active reliability-ID selection + FM-anchored
   correction under adversarial shift.
### 4.4 Real adversarial signal: identify real datasets with coordinated/low-reliability
   workers (or inject controlled attacks on real masks). Fix Trec/Senti loading.
### 4.5 Metrics: budget–accuracy, AURC, expert-cost-saved-at-target, attacker-detection
   (do we down-weight the true adversaries?).
### 4.6 Theory: reliability-identification objective; why uncertainty sampling fails here.
### 4.7 Write-up.

### Deprioritized: learned correction (bottleneck is anchor *information*, not model
capacity — scalar correction matches full-confusion DS; VOI within ~0.7% AURC of the
selection oracle ⇒ near the per-budget frontier). Escalation gate (no gold-free benefit
predictor; correction is low-risk so "always apply small budget" suffices).

---

## 5. What still holds from CrowdSI (reused)
- Task-disjoint cross-fitted **e-value**: correctly calibrated under its exact null
  (E[E]≈1.04, Pr(E≥1/α)≤α). Valid *shift detector* (Stage A component), not a benefit
  predictor.
- Conditional **emission head** predicts held-out annotations far better than frequency
  baselines (annotation-process modeling works; just doesn't transfer to truth alone).

## 6. Reproduction
- Cluster: `ssh keyuzhu@hive.hpc.ucdavis.edu`, `/nfs/hive/scratch/keyuzhu/crowdfm/CrowdFM`.
- Model: frozen official CrowdFM `checkpoint.pt`; `res_mech_ip_s42` = frozen backbone +
  trained emission head (for reliability/response signals).
- Code (GitHub `agent/crowdguard`): `crowdguard_eval.py` (synthetic 4 regimes),
  `crowdguard_real.py` (real), `anchor_headroom.py` (ceiling), `escalation_analysis.py`,
  `src/cfm/model/ResidualCFM.py`, `src/cfm/data/crowdsi_simulator.py` (+ `sybil` family).
- Submit: `sbatch -A publicgrp -p low --cpus-per-task=8 jobs/pyjob.sh <script> <cfg>`.
