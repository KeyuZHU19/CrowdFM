# ⛔ STATUS: CrowdSI-FM main line — STOPPED / FALSIFIED (2026-07-18)

The hypothesis "explicit crowd system identification improves label aggregation over
CrowdFM" is **empirically falsified** by three nested clean experiments culminating in an
oracle-ceiling proof (even a gold-tuned mechanism latent does not beat frozen CrowdFM;
Z−X ≈ 0 everywhere). Automatic hyperparameter/family/latent expansion on this line is
**halted**. Results, checkpoints, and reproduction scripts are retained. See the
GO/NO-GO sections below. Next line: response-predictive evidence for *selective /
risk-aware aggregation in regimes where CrowdFM fails* (probe in progress) — a NEW
research question, not a continuation of CrowdSI-FM.

---

# CrowdSI-FM Experiment Log & Analysis

Running record of experiments toward the AAAI submission, following
`docs/CROWDSI_AUTORESEARCH_PLAN.md`. Compute: UC Davis Hive (slurm, public
`low`/`high` GPU partitions + junshan `gpu-l40s`; junshan A100 avoided).
Working dir: `/nfs/hive/scratch/keyuzhu/crowdfm/CrowdFM`.

Model: CrowdSIFM `dim=32, layer=10, head=4, latent=16, primitives=8` initialised
from the official CrowdFM backbone `checkpoint.pt` (831K backbone + 58K SI params),
freeze_backbone=false. Training: 10000 iters, batch 1, grad-accum 16 (=625 updates),
AdamW lr 3e-4. ~22 min on one RTX A6000. Reproducibility seed added to `train_crowdsi.py`.

---

## Phase 0 — correctness & smoke (PASS)

- Full unit suite: **28/28 passed** (21 legacy + 7 CrowdSI), incl. numerical
  shared-truth-likelihood test, task-disjoint fold invariants, edge-permutation
  invariance of the mechanism encoder, gradient flow to all SI heads, and the
  weight-restoring latent adaptation.
- Smoke training (`crowdsi_smoke.yaml`, CPU): all losses finite and decreasing.
- Backbone checkpoint loads `strict=True` into the dim=32 model (0 missing / 0
  unexpected keys).

## Phase 1 — in-prior predictive validation (PASS, strong)

Model `p1_full_s42` (full family vocabulary: irt, class_bias, coalition,
assignment_bias, mixed). Held-out evaluation on **200 fresh in-prior worlds**
(seed 90000), `eval_synth.py`. All numbers mean ± 95% CI.

| Metric | Value | Read |
|---|---|---|
| CrowdFM accuracy | 0.7778 ± 0.0284 | frozen backbone truth head |
| **CrowdSI zero-shot accuracy** | **0.8132 ± 0.0266** | mechanism head at μ₀ |
| Majority vote | 0.7874 ± 0.0234 | classic baseline |
| **Δ_arch = zeroshot − crowdfm** | **+0.0354 ± 0.0058** | paired, CI excludes 0 |
| Response NLL (model, per edge) | 1.575 ± 0.085 | model marginal predictive |
| Response NLL (option-freq) | 2.248 ± 0.089 | baseline |
| Response NLL (worker-freq) | 2.411 ± 0.095 | baseline |
| Assignment AUROC | 0.684 ± 0.016 | mechanism head |
| Assignment AUROC (density-only) | 0.650 ± 0.012 | baseline |
| Primitive-weight entropy | 1.473 ± 0.014 | max = ln 8 = 2.079 |
| Mechanism logvar (mean) | −1.137 ± 0.009 | posterior variance ≈ 0.32 |
| Cross-view posterior L2 | 0.031 ± 0.003 | two masks → same μ |

**Analysis.**
1. **No in-prior loss — actually a gain.** The mechanism-conditioned architecture
   raises zero-shot accuracy over the original CrowdFM by **+3.5 pp** (paired CI
   [+2.4, +4.7]). This clears the paper's C2 gate (required ≥ −0.5 pp) decisively
   and, importantly, the gain exists *before any test-time adaptation*, so it is
   attributable to the architecture, not to TTA. It also beats majority vote.
2. **The emission model is real.** Held-out response NLL (1.58) is far below both
   the option-frequency (2.25) and worker-frequency (2.41) baselines, i.e. the
   conditional emission head P(a|c) captures structure that marginal frequencies
   cannot. This is the prerequisite for the likelihood-ratio e-value to have power.
3. **The assignment head beats a density-only baseline** (0.684 vs 0.650). The gap
   is modest but consistent, confirming the mask model learns selection structure
   beyond raw task popularity.
4. **No posterior/primitive collapse.** Primitive entropy 1.47/2.08 (primitives are
   used, not one-hot, not uniform); the mechanism posterior contracts below the
   N(0,I) prior (logvar −1.14 ⇒ σ²≈0.32) rather than collapsing to it; two
   independent masked views of the same world produce near-identical mechanism
   posteriors (L2 0.031). Together this is the first evidence for claim C1 (a
   stable, transferable dataset-level mechanism representation).

**Seed replication (Δ_arch, in-prior, n=200 each).** Three fully-trained seeds:

| seed | CrowdFM | Zero-shot | Δ_arch |
|---|---:|---:|---:|
| 42 | 0.778 | 0.813 | +0.0354 ± 0.0058 |
| 43 | — | — | +0.0064 ± 0.0022 |
| 44 | 0.787 | 0.814 | +0.0273 ± 0.0044 |

Mean Δ_arch ≈ **+0.023**, positive on all three seeds. Seed 43 is a low outlier
(+0.6pp): the magnitude of the architecture gain is seed-sensitive, but its *sign* is
robust. Reported honestly; a longer-trained / more-seeds study would tighten this.

**Decision:** Phase-1 gate PASSED on all criteria → proceed to adaptation and
calibration experiments.

---

## Phase 2 — exact-null e-value calibration (PASS; 300 worlds, GPU, K≤10)

`phase2_exactnull.py`: held-out audit answers regenerated from the model's own law at
μ₀ so the null H0: Z=μ₀ is exactly true, then run the cross-fitted e-value.

- **E[E] = 1.042**, median E = 1.016 (theory requires E₀[E] ≤ 1; observed ≈1, the
  small excess is Monte-Carlo noise in the mixture-numerator estimate).
- **Pr(E ≥ 10) = 0.000** (α=0.10), **Pr(E ≥ 20) = 0.000** (α=0.05),
  **Pr(E ≥ 100) = 0.000** (α=0.01) — all ≤ nominal.

**Analysis.** Claim C4 confirmed. Under the exact plug-in null the e-value concentrates
tightly around 1 and never exceeds the gate threshold in 300 null worlds, so the
false-adaptation rate is controlled (in fact conservative) at every α. This isolates
the gate's validity from any simulator misspecification — the theorem holds as
implemented. (Engineering: run on GPU with lighter adaptation, 30 steps / 16 predictive
/ 2 ELBO, which keeps the numerator a valid normalized predictive law, so the e-value
guarantee is unaffected. K restricted to ≤10 for a tractable emission tensor.)

## Phase 4 — compositional generalization (mixed), 20 worlds preliminary

Model `p4_train-singles_s42` trained on the four single families {irt, class_bias,
coalition, assignment_bias}; tested on held-out **mixed** (their composition).

Prelim (n=20): CrowdFM 0.879, zero-shot 0.881, e-gated 0.881, oracle-latent 0.884,
MV 0.824, Dawid–Skene 0.897. Δ_arch = +0.3pp (n=20, not yet significant), Δ_adapt = 0,
gate fired 0/20. Compositional architecture gain is modest here (mixed is a strong,
dense regime where Dawid–Skene is competitive); adaptation again adds nothing. Full
100-world result pending (5 chunks; slow at ~180 s/world).

Note: `abl_nomech` training was preempted on the `low` partition and restarted from
scratch (train loop has no mid-run resume), delaying the decisive mechanism-conditioning
ablation by ~20 min.

## Phase 3 — held-out mechanism adaptation, coalition (60 worlds; 3/5 chunks)

Model `p3_train-irt-cb_s42` trained only on {irt, class_bias}; tested on held-out
**coalition** worlds (a dependence mechanism never seen in training).

| Method | Accuracy | vs zero-shot |
|---|---:|---:|
| CrowdFM | 0.6356 | — |
| CrowdSI zero-shot | 0.6596 | — |
| CrowdSI always-adapt | 0.6598 | +0.0003 |
| CrowdSI e-gated | 0.6596 | 0 (gate never fires) |
| Oracle latent (gold-tuned) | 0.6633 | +0.0037 |
| Full-network TTA | 0.6471 | −0.0125 |
| Majority vote | 0.6630 | +0.0034 |
| Dawid–Skene | 0.6451 | −0.0145 |

Paired: **Δ_arch (zero-shot−CrowdFM) = +0.0240 ± 0.0071** (CI excludes 0);
Δ_adapt = 0; mean E-value ≈ 1.07 (gate fired 0/60).

**Analysis.**
1. **Architecture generalises to an unseen mechanism.** Zero-shot beats CrowdFM by
   +2.4pp on held-out coalition (never trained on), and beats Dawid–Skene; it matches
   majority vote (coalition is correlated noise, where MV is strong).
2. **Latent adaptation has almost no truth headroom here.** The gate correctly stays
   quiet (E≈1: the plug-in mechanism already predicts held-out coalition responses
   well), always-adapt changes accuracy by +0.03pp, and even the **oracle latent tuned
   on gold labels gains only +0.4pp**. So the flat Δ_adapt is not under-tuning — the
   zero-shot mechanism inference is already near-optimal for aggregation on this shift.
3. **Full-network TTA hurts (−1.3pp)** while latent-only adaptation does not — direct
   evidence that confining adaptation to the low-dimensional mechanism is the safe
   choice, supporting the design over unconstrained TTA.

Together with the real-data table, the consistent conclusion is: the **mechanism
architecture** is the robust source of gains; **latent adaptation is truth-neutral with
a safe, calibrated gate**; **full TTA is unsafe**.

## Phase 6/7 — real crowd benchmarks (13/16 done; SP, Senti, Trec pending)

Full `run_crowdsi` pipeline on the bundled real datasets (α=0.05, adaptation uses
no gold labels; gold used only for retrospective accuracy). One job per dataset.

| Dataset | K | E-value | Gate | CrowdFM | Zero-shot | Adapted | Δ_arch | Δ_adapt |
|---|---:|---:|:--:|---:|---:|---:|---:|---:|
| RTE | 2 | 3.2e1 | fire | 0.6075 | 0.7487 | 0.7500 | **+14.1** | +0.1 |
| Web | 5 | 1.5e51 | fire | 0.8153 | 0.9084 | 0.8948 | **+9.3** | −1.4 |
| CF | 5 | 0.97 | — | 0.8233 | 0.8833 | 0.8833 | +6.0 | 0 |
| LabelMe | 8 | 1.1 | — | 0.7430 | 0.7700 | 0.7700 | +2.7 | 0 |
| CF* | 5 | 2.2 | — | 0.8267 | 0.8500 | 0.8500 | +2.3 | 0 |
| ZC_in | 2 | 1.2e3 | fire | 0.7275 | 0.7466 | 0.7461 | +1.9 | −0.05 |
| PosSent | 2 | 3.0e18 | fire | 0.9230 | 0.9420 | 0.9340 | +1.9 | −0.8 |
| MS | 10 | 0.94 | — | 0.7543 | 0.7686 | 0.7686 | +1.4 | 0 |
| ZC_us | 2 | 1.0e2 | fire | 0.8603 | 0.8691 | 0.8608 | +0.9 | −0.8 |
| Dog | 4 | 2.2e1 | fire | 0.8228 | 0.8278 | 0.8216 | +0.5 | −0.6 |
| ZC_all | 2 | 1.1e64 | fire | 0.8314 | 0.8309 | 0.8397 | −0.05 | +0.9 |
| Bird | 2 | 5.6 | — | 0.8056 | 0.7963 | 0.7963 | −0.9 | 0 |
| Face | 4 | 2.2 | — | 0.6353 | 0.6216 | 0.6216 | −1.4 | 0 |

**Analysis (key, honest).**
1. **The mechanism architecture is a real-world win.** Zero-shot beats CrowdFM on
   10/13 datasets (mean Δ_arch ≈ **+2.9pp**), with large gains on RTE (+14.1),
   Web (+9.3), CF (+6.0). This is the strongest evidence that the mechanism-
   conditioned heads generalise beyond the simulator — they are not merely
   inverting synthetic worlds, since these are real crowds with unknown mechanisms.
2. **The e-value gate is discriminative on real data.** It fires on 7/13 datasets,
   precisely the large ones where an adapted mechanism predicts held-out annotations
   astronomically better (E up to 1e64). So the response-predictive evidence is real.
3. **But likelihood evidence ≠ truth-accuracy improvement.** When the gate fires,
   the mean Δ_adapt is ≈ **−0.4pp** (RTE +0.1, ZC_all +0.9, but Web −1.4, PosSent
   −0.8, ZC_us −0.8, Dog −0.6). Real crowds deviate from the plug-in emission law
   (hence high E), and adapting the *shared* mechanism latent to fit responses moves
   the truth head in a direction that is not reliably truth-improving. This is the
   `likelihood/accuracy misalignment` the master plan flags (F4/F7); it must be
   reported, not hidden.
4. **Safe gating still delivers value.** Even where adaptation slightly hurts, the
   worst adapted degradation is only −1.4pp — there is no catastrophic negative
   transfer, unlike unconstrained always-adapt/TTA. The gate + trust-region KL keep
   the deployed model within a safe neighbourhood of the strong zero-shot model.

**Implication for the paper's positioning.** The robust, defensible contributions
are (i) the amortized mechanism *architecture* (in-prior +3.5pp and real +2.9pp
zero-shot gains) and (ii) *safe evidence-gated* deployment. Latent adaptation under
the response-likelihood objective is a mechanism the gate keeps safe rather than a
source of large accuracy gains on real data. Phase-3/4 (held-out synthetic) will show
whether adaptation helps when the shift is a genuine mechanism the model can represent
but was not trained on — the discriminator between "useful adaptation" and
"audit-only" framing. Candidate fix per F4: an aggregation-relevant latent subspace or
truth-preserving regularisation so that response adaptation cannot drag the truth head.

## Phase 7 — ablations (in-prior, isolate source of Δ_arch)

Retrained variants, in-prior eval (200 worlds). Purpose: attribute the zero-shot
architecture gain to *mechanism conditioning* vs. merely added head capacity.

| Variant | Zero-shot | Δ_arch | note |
|---|---:|---:|---|
| Full CrowdSI-FM (seed 42) | 0.813 | +0.035 | 8 primitives, mechanism on |
| − compositional basis (R=1) | 0.831 | +0.018 | single primitive |
| − assignment head | 0.846 | +0.002 | assignment_weight=0 (AUROC 0.52) |
| − mechanism conditioning (C(Z)=0) | 0.835 | **+0.001** | decisive test |

**Analysis (complete).** The decisive `C(Z)=0` ablation (mechanism context zeroed, all
heads kept at full size) drops Δ_arch to **+0.06pp ± 0.20** — statistically zero. So the
architecture gain is genuinely from *conditioning on the inferred dataset mechanism*, not
from the extra retrained head parameters. Removing the assignment loss also collapses the
gain (+0.2pp, assignment AUROC → chance), because that loss supplies the gradient that
trains the mechanism pathway; and a single primitive (no composition) halves it (+1.8pp).
Consistent picture: mechanism conditioning drives the gain, the assignment process is
what makes the mechanism useful, and compositionality adds roughly half.

## Operational note
Phase-2 on CPU was ~80 s/world (emission tensor + ~900 adaptation/predictive forwards);
relaunched on A6000 GPU with lighter adaptation (30 steps, 16 predictive, 2 ELBO
samples — still a valid e-value, since the numerator is any normalized predictive law).
Phase-3 (3/5 chunks done) and Phase-4 (running) merge via a durable slurm finalizer job.

## In flight
- Phase 2 exact-null calibration — 5 GPU chunks (500 worlds), merge job queued.
- Phase 3 coalition adaptation — 3/5 chunks done; Phase 4 (`mixed`) running.
- Ablations abl_nomech, abl_noassign training → in-prior eval.
- Large real datasets SP/Senti/Trec requeued light (Senti has 569K labels).

---

# Summary, verdict, and analysis

**What was tested.** Following the master plan, we validated CrowdSI-FM across: unit
tests + smoke (P0), in-prior predictive quality over 3 seeds (P1), exact-null e-value
calibration (P2), held-out mechanism adaptation on coalition (P3), compositional
generalization to `mixed` (P4), 13 real crowd benchmarks (P6/7), and three architecture
ablations (P7). Compute: UC Davis Hive, public `low` GPUs/CPUs + junshan l40s;
**junshan A100 never used**.

**The three-legged claim (plan §19), assessed honestly:**

1. **Amortized mechanism architecture — STRONG PASS.** Zero-shot aggregation improves
   over CrowdFM in-prior (+2.3pp mean over seeds, +3.5pp best), on a held-out mechanism
   (+2.4pp on coalition, CI excludes 0), and on real crowds (+2.9pp mean, RTE +14, Web
   +9). The gain is present without any adaptation, so it is architectural. Ablations
   localize it: the **assignment-modeling loss** is essential (removing it → +0.2pp) and
   the **compositional basis** contributes about half.

2. **Safe evidence gating — PASS.** The cross-fitted e-value is calibrated under its
   exact null (E[E]≈1.04; Pr(E≥1/α)=0 ≤ α for all α). It fires only with strong
   task-disjoint evidence and never causes catastrophic negative transfer (worst real
   adapted case −1.4pp), unlike full-network TTA which actively hurts (−1.3pp on
   coalition).

3. **Unseen-mechanism *adaptation* that raises truth accuracy — DOES NOT HOLD.** This is
   the key honest finding. Latent adaptation is **truth-neutral**: on real data where the
   gate fires decisively (E up to 1e64), mean Δ_adapt ≈ −0.4pp; on held-out coalition,
   even the **oracle latent tuned on gold labels** gains only +0.4pp. The e-value soundly
   certifies *response-predictive* improvement, but response-likelihood and truth
   accuracy are misaligned in the shared mechanism subspace (plan risk F4/F7).

**Verdict vs. plan's AAAI bar.** Two of three legs hold robustly; the adaptation leg is
a documented negative result rather than a win. Per the plan this is *not* a slam-dunk
"adaptation helps everywhere" paper. The honest, defensible paper it supports is:
**"An amortized, compositional crowd-mechanism architecture that improves zero-shot
aggregation on real crowds, plus a calibrated task-disjoint e-value that makes
deployment adaptation provably safe — and a careful demonstration that likelihood-gated
latent adaptation is response-valid but truth-neutral, motivating aggregation-relevant
adaptation as future work."** This matches the idea (system identification + safe
adaptation) without overclaiming.

**Recommended next steps to strengthen (before submission):**
- Multi-seed the ablations (seed variance is large) to firm up the attribution.
- Implement the F4 remedy: split the mechanism into response- vs. aggregation-relevant
  subspaces (or a truth-preserving trust region) and test whether gated adaptation can
  then move truth accuracy positively — this could convert leg 3 into a win.
- Full-posterior null denominator (plan §6.4) to widen the theorem beyond the plug-in
  null.
- Finish the full 100-world P4 and the `disable_mechanism` ablation (in flight) for the
  final tables.

**Deliverables.** `paper/main.tex` (+ `tab_*.tex`) compiles to `paper/main.pdf` via
tectonic; this `RESULTS.md` is the full experiment log and analysis. All runs, configs,
checkpoints, and per-world JSONs are under `/nfs/hive/scratch/keyuzhu/crowdfm/`.

---

# ⚑ GO / NO-GO: decontaminated 3-way comparison (DECISIVE)

Motivated by a sharp reviewer-style critique: my earlier Δ_arch used
`hat_task_option_base` as the "CrowdFM baseline", but CrowdSI trains with
**backbone unfrozen**, so that head is a *co-trained, drifted* backbone — not the
official frozen CrowdFM — and each model was scored on a *different* eval bank. Both
are baseline contamination.

**Clean design.** X = official frozen CrowdFM (backbone loaded separately from
`checkpoint.pt`, never touched by CrowdSI training; pred = `hat_task_option_base`);
Y = capacity-matched no-mechanism (full architecture, `disable_mechanism=True`, same
training); Z = full CrowdSI (zero-shot). **3 matched seeds** (full & no-mech share
identical training worlds per seed — `disable_mechanism` consumes no RNG), **same eval
world bank** for all, official isolated. Full-s42 models were **retrained seeded** so
they share worlds with their no-mech counterparts.

| Setting | X official | Y no-mech | Z full | **Z−X** | Z−Y (mechanism) | threshold | verdict |
|---|---:|---:|---:|---:|---:|---:|:--:|
| in-prior (3×200) | 0.8084 | 0.8116 | 0.8131 | **+0.47pp** | +0.15pp | ≥1.5pp | **FAIL** |
| held-out coalition (3×150) | 0.6869 | 0.6864 | 0.6855 | **−0.14pp** | −0.09pp | ≥1.5pp | **FAIL** |
| real datasets (3×16) | 0.8233 | 0.8136 | 0.8096 | **−1.37pp** | −0.40pp | ≥1.0pp | **FAIL** |

**Verdict: the architecture contribution does NOT survive the fair comparison — on real data it is WORSE.** All three settings fail: in-prior +0.47pp, held-out −0.14pp, real **−1.37pp**. The most vivid inversion is RTE, earlier reported as **+14pp**: cleanly it is **−5.75pp** (official frozen 0.9125 vs full CrowdSI 0.8550) — the +14pp was entirely the co-trained backbone's own RTE head collapsing to 0.6075. On real data, fine-tuning the backbone on *synthetic* worlds during CrowdSI training actively degrades real transfer relative to the official checkpoint.

**Root-cause of the earlier inflated +3.5pp.** The full model's *own* base head degraded
during co-training (official 0.808 → co-trained base 0.778) because the unfrozen backbone
drifts to serve the mechanism heads. Comparing zero-shot (0.813) against that **degraded**
head (0.778), on a *different* eval bank, manufactured +3.5pp. Against the untouched
official CrowdFM (0.808), the true gain is **+0.47pp in-prior and −0.14pp held-out** — and
the *pure mechanism-conditioning* effect (Z−Y) is ~0 (+0.15 / −0.09pp).

**What this means for the paper.** The user's hypothesis was correct: the headline
"mechanism-conditioned zero-shot aggregation improves over CrowdFM" was an artifact of an
unfair head/backbone-training comparison. It does **not** hold. Combined with the earlier
result that likelihood-gated adaptation is truth-neutral, the honest overall conclusion is:

> **CrowdSI-FM as currently implemented does not demonstrably improve crowd aggregation
> over the frozen CrowdFM foundation model** — neither the architecture (fair comparison
> ≈ tie, <0.5pp) nor the adaptation (truth-neutral). The only surviving positive is the
> exact-null calibration of the e-value gate, a statistical property that currently gates
> an adaptation with no truth benefit.

This is a **NO-GO for the paper as framed**. It is a genuine, valuable negative result
(baseline contamination + likelihood≠truth in crowd-FM system identification), but it is
not a positive main-conference contribution without a method change that makes either the
architecture or the adaptation clear the bar under this clean protocol. Candidate fixes
before any resubmission attempt: (i) **freeze the backbone** and train only the mechanism
heads (so the fair baseline is the frozen head and the mechanism must add value on top),
(ii) the F4 aggregation-relevant latent so adaptation can move truth accuracy, (iii)
distill/anchor the base head to prevent degradation. None are validated yet.

---

# ⚑⚑ FROZEN-BACKBONE GO/NO-GO — the definitive, confound-free answer

To remove the backbone-drift confound entirely, we retrained with **`freeze_backbone=true`**
(only the SI heads train; the backbone stays exactly the official CrowdFM). Now X =
official frozen CrowdFM is, by construction, each model's own untouched base head — no
drift, no cross-bank mismatch. Same clean 3-way, 3 matched seeds, same eval banks.

| Setting | X official | Y no-mech (frozen) | Z full (frozen) | **Z−X** | Z−Y (pure mechanism) | verdict |
|---|---:|---:|---:|---:|---:|:--:|
| in-prior (3×200) | 0.8084 | 0.8032 | 0.8036 | **−0.49pp** | +0.04pp | FAIL |
| held-out coalition (3×150) | 0.6869 | 0.6812 | 0.6820 | **−0.49pp** | +0.09pp | FAIL |
| real (3×16) | 0.8210 | 0.8195 | 0.8196 | **−0.14pp** | −0.01pp | FAIL |

**Definitive verdict: the mechanism idea adds NO aggregation value, even with the
confound removed — and the pure mechanism-conditioning effect (Z−Y) is ~0 everywhere.**

Two independent clean experiments now agree:
- **Unfrozen** (backbone co-trained): Z−X = +0.47 / −0.14 / −1.37 pp (the "+3.5pp" was
  the *degraded* baseline).
- **Frozen** (backbone untouched): Z−X = −0.49 / −0.49 / −0.14 pp; **Z−Y ≈ 0**.

Interpretation: a fresh mechanism-conditioned truth head trained on top of frozen CrowdFM
embeddings is **slightly worse** than CrowdFM's own already-trained head, and conditioning
that head on the inferred dataset mechanism recovers **nothing** (Z−Y ≈ 0). CrowdFM's
amortized attention already extracts essentially all the aggregation signal the mechanism
latent could provide — consistent with the earlier finding that even the *oracle* latent
(tuned on gold labels) gained only +0.4pp.

**Bottom line for the paper.** The central hypothesis — that explicit crowd *system
identification* improves label aggregation over a strong amortized aggregator — is **not
supported**. Neither the architecture (mechanism conditioning ≈ 0) nor the adaptation
(truth-neutral) delivers. The e-value gate is correctly calibrated but gates an adaptation
with no accuracy benefit. This is a clean **NO-GO** for the paper as conceived. What
remains is honest: a negative result + a methodology contribution (the contamination trap;
likelihood≠truth; oracle-mechanism headroom ≈ 0), or a pivot of the *claim* away from
accuracy (e.g. selective/deferred aggregation, calibration, OOD flagging) where the
response-predictive machinery might still earn its keep — to be tested, not assumed.

---

# ⚑⚑⚑ REDESIGN GO/NO-GO — frozen CrowdFM + zero-init residual + truth-aligned meta (CONCLUSIVE STOP)

Per the revised plan, we removed every confound: **frozen** CrowdFM backbone + original
head; a **zero-init** mechanism residual (`l = l_official + γ·Δ_θ(G,Z)`, reproduces
CrowdFM exactly at init, verified max logit diff 0.0); an **anchor** KL to CrowdFM; and a
**truth-aligned meta-learner** (inner response-NLL step on Z_resp → learned g_ψ → Z_agg,
outer loss = synthetic truth accuracy). 6 arms, 90 worlds, same eval banks.

| Arm | in-prior | held-out (coalition) |
|---|---:|---:|
| A1 official frozen CrowdFM | **0.8260** | **0.6700** |
| A2 no-mechanism residual | 0.8251 | 0.6695 |
| A3 zero-shot mechanism residual | 0.8252 | 0.6696 |
| A4 direct response-likelihood adaptation | 0.8253 | 0.6697 |
| A5 **meta** truth-aligned adaptation | 0.8254 | 0.6692 |
| A6 **oracle** latent (tuned on GOLD truth) | 0.8254 | 0.6698 |

Deltas vs official: A5−A1 = **−0.06pp / −0.09pp**; A5−A4 = +0.01 / **−0.05**; A5−A2 =
+0.03 / **−0.04**; **A6(oracle)−A1 = −0.06pp / −0.03pp**.

**Verdict: CONCLUSIVE NO-GO. Every Go condition fails.** Official CrowdFM (A1) is the
best arm in both settings; meta beats neither direct adaptation nor the no-mechanism
control; held-out A5−A1 is −0.09pp (needed +1pp). All arms sit within ±0.1pp of each
other and of CrowdFM.

**The killer is the oracle (A6).** Even tuning the mechanism latent *directly on gold
truth* — the absolute upper bound of what any latent adaptation could ever achieve —
adds nothing (≈0pp, actually slightly negative). This is not a training, tuning, or
architecture-fairness problem: **there is essentially zero aggregation headroom for a
mechanism-latent correction on top of frozen CrowdFM.**

## Final scientific conclusion

Three independent clean experiments now agree, culminating in an oracle-ceiling proof:
1. unfrozen 3-way: mechanism gain ≈ 0 (the "+3.5pp" was baseline contamination);
2. frozen 3-way: Z−Y ≈ 0, all arms ≤ CrowdFM;
3. **frozen residual + meta + oracle: even the gold-tuned oracle latent ≈ CrowdFM.**

**The central hypothesis is falsified: explicit crowd system identification does not
improve label aggregation over the CrowdFM foundation model.** The deployment-observable
annotations carry no aggregation-relevant signal that CrowdFM's amortized attention has
not already extracted. Per the agreed decision rule, the CrowdSI main line is **STOPPED**.

### What still holds (verified, contamination-free)
- The task-disjoint cross-fitted e-value is correctly calibrated under its exact null
  (E[E]≈1.04, Pr(E≥1/α)≤α) — a valid statistical construction, but it gates an adaptation
  with no accuracy benefit.
- The conditional emission head predicts held-out annotations far better than frequency
  baselines — genuine annotation-structure learning, but it does not transfer to truth.

### What was ruled out (valuable negative knowledge)
- shared mechanism conditioning ⇒ better aggregation: FALSE.
- response-likelihood improvement ⇒ truth-risk improvement: FALSE.
- and now: even oracle mechanism-latent correction ⇒ better aggregation: FALSE (~0 headroom).

### Honest options from here
1. **Rigorous negative-result / methodology paper** (workshop or methods venue): the
   contamination trap, likelihood≠truth, and the oracle-ceiling analysis are genuinely
   useful to the crowd-FM / TTA community. This is the defensible publishable output.
2. **New research question** (a fresh project, not a fix): the finding says CrowdFM is
   near-ceiling on these regimes, so any value of mechanism modeling must come from
   settings where CrowdFM actually fails (extreme sparsity, adversarial/Sybil workers,
   strong non-stationarity) or from a non-accuracy objective (calibration, abstention).
   To be scoped and validated, not assumed.
3. **Stop and reassess the direction.**

---

# PIVOT probe #1 — response evidence for selective aggregation (error detection): NEGATIVE

New question (not accuracy): can per-task response-predictive mismatch identify WHERE
CrowdFM is wrong, better than / complementary to CrowdFM's own confidence, enabling
selective/risk-aware aggregation? Reused the frozen-CrowdFM+emission model (no training).
Pooled ~40K tasks/regime; AUROC for predicting CrowdFM error; 2-fold CV logistic to test
ORTHOGONAL value of the response signal over confidence.

| regime | CrowdFM acc | conf AUROC | resp_nll AUROC | learned(no resp) | learned(+resp) | Δ orthogonal |
|---|---:|---:|---:|---:|---:|---:|
| class_bias | 0.815 | 0.896 | 0.767 | — | — | — |
| mixed | 0.842 | 0.839 | 0.704 | — | — | — |
| coalition | 0.690 | 0.798 | 0.680 | 0.7935 | 0.7906 | **−0.0029** |
| **sparse (CrowdFM fails)** | **0.535** | 0.689 | 0.622 | 0.6868 | 0.6856 | **−0.0012** |

**Verdict: NEGATIVE.** The response-mismatch signal detects errors above chance
(AUROC 0.62–0.77) but is a *weaker, redundant* version of CrowdFM's own confidence. A
learned (logistic) combination shows the response signal adds **zero orthogonal value**
(Δ ≈ 0, slightly negative) — even in the sparse regime where CrowdFM genuinely fails
(acc 0.535). CrowdFM's posterior confidence already integrates the annotation signal
(including response patterns) near-optimally; the per-task response NLL is a lossy
re-derivation of information CrowdFM already used.

**Consolidated finding across everything:** CrowdFM's amortized attention extracts
essentially all annotation-derived signal — for the aggregate label, its uncertainty,
AND error detection. No task-level annotation-derived quantity (mechanism latent,
response likelihood, e-value) has been shown to add value for accuracy OR for
selective prediction, even at oracle/optimal-combination ceilings.

**Structurally-different angles NOT yet tested** (each a distinct problem, would need new
simulation + probe): (a) *worker-level* anomaly / Sybil localization (uses cross-task
worker patterns a task-posterior need not expose); (b) temporal non-stationarity /
drift detection; (c) calibrated selective aggregation with a few trusted expert anchors
(injects external signal). These are the only remaining places the machinery could earn
value; the task-level angle is exhausted.

---

# 🟢 NEW DIRECTION — CrowdGuard Anchor Headroom Test: POSITIVE (GO)

Per the revised plan, before building CrowdGuard we test the ceiling: do a few TRUSTED
expert anchors (gold Y on B tasks) — a NEW information source not derivable from G —
create real aggregation headroom over frozen CrowdFM on the remaining tasks?
`anchor_headroom.py`, frozen CrowdFM + numpy DS/worker-model, 120 worlds/scenario.
Metric: effective accuracy over all N tasks, anchored counted correct (bought).

| scenario | CrowdFM | MV | DS(unsup) | **oracle-worker (full gold)** | ceiling |
|---|---:|---:|---:|---:|---:|
| mixed | 0.850 | 0.822 | 0.889 | 0.991 | **+14.1pp** |
| coalition | 0.678 | 0.669 | 0.667 | 0.992 | **+31.4pp** |
| sparse (1–3 ann/task) | 0.554 | 0.554 | 0.438 | 0.908 | **+35.5pp** |

Best-of{anchor-DS-random, anchor-DS-entropy, reweight} minus CrowdFM, by budget B:

| B | mixed | coalition | sparse |
|---|---:|---:|---:|
| 5  | +5.6 | +1.6 | +1.1 |
| 10 | +6.5 | +4.2 | +3.0 |
| 20 | +8.2 | +7.8 | +5.9 |
| 50 | +11.4 | +16.6 | +14.5 |

**Verdict: GO.** (1) The oracle worker-model (full gold) beats CrowdFM by **+14 to
+35pp** — huge headroom, the exact opposite of the annotation-only oracle-latent test
(≈0). (2) At ≤5–10% budget (~10 anchors), anchor methods clear the +2pp bar on all three
shifts. (3) Entropy selection beats random (e.g. coalition B=50: 0.843 vs 0.811) →
active anchor selection carries value → the full CrowdGuard (e-value escalation + active
selection + anchor-conditioned correction) is warranted.

**Why this works when everything before failed:** gold anchors change identifiability
`G → (G, Y_S*)`. CrowdFM is far from the ceiling once trusted labels expose which workers
/ groups are (un)reliable on shifted data; on coalition/sparse (where CrowdFM fails), a
per-dataset worker model informed by a few anchors recovers many points.

**Honest caveats to address next:** (a) the anchor methods here are DS-based, not yet a
CrowdFM-*anchored residual* that equals CrowdFM at B=0 — so on `mixed` part of the gain
is "a fitted model beats the frozen FM" (DS 0.889 > CrowdFM 0.850 even unsupervised);
the clean CrowdGuard correction must start from CrowdFM so the gain is unambiguously from
anchors. (b) oracle-worker uses full gold (ceiling, not budget). (c) needs ≥3 seeds, real
data, and a real Sybil/temporal shift. (d) tiny budgets (5 anchors on small N) are
borderline on the hardest shifts. But the headroom is real and large — this is the first
validated positive direction.

**Next steps (validated build order):** (1) CrowdFM-anchored residual correction (starts
= CrowdFM, improves with anchors, trust-region); (2) worker-reliability posterior from
anchors as correction input; (3) active VOI/influence anchor selection vs entropy/random;
(4) e-value as dataset-level escalation gate only; (5) budget–accuracy / AURC curves,
3 seeds, real data + a Sybil regime.

## Clean confirmation — correction that STARTS from CrowdFM (caveat (a) resolved)

`cf_anchor_correct`: log-posterior = log q_CrowdFM + Σ_i wgt_i·onehot(vote), wgt_i =
anchor-estimated worker log-odds reliability (0 at B=0 ⇒ pure CrowdFM; negative for
systematically-wrong/coalition workers ⇒ their votes are subtracted). Gain is now
unambiguously from anchors, on top of frozen CrowdFM:

| B (anchors) | mixed | coalition | sparse |
|---|---:|---:|---:|
| 5  | +2.6 | +2.1 | +2.0 |
| 10 | +3.6 | +4.2 | +4.1 |
| 20 | +4.8 | +7.9 | +7.6 |
| 50 | +6.9 | +14.0 | +17.4 |

(best of random/entropy selection, pp over CrowdFM). Monotone in budget; clears +2pp at
~5–10 anchors on all three shifts. **Neither random nor entropy selection dominates**
(mixed→random better, sparse→entropy better) ⇒ the value is in *worker-coverage/influence-
aware* VOI selection (Stage B), not naive entropy — motivating the active-selection module.

**GO is airtight.** CrowdGuard is justified to build: (1) e-value dataset-level escalation
gate; (2) active VOI anchor selection (worker-influence/coverage aware); (3) learned
anchor-conditioned residual correction on frozen CrowdFM (this numpy reliability-vote is a
strong no-training lower bound; a learned h_φ using worker-reliability posteriors + CrowdFM
embeddings should approach the +14–35pp oracle ceiling); (4) budget–accuracy/AURC curves,
≥3 seeds, real data, + a real Sybil/temporal shift.

---

# 🟢 CrowdGuard core result — Active VOI anchor selection (VALIDATED)

Paper Q: when a crowd FM fails at deployment, use few expert anchors to actively identify
annotator reliability and correct remaining tasks at minimal cost.
`crowdguard_eval.py`, 80 worlds/scenario, correction = anchor-reliability-weighted on
frozen CrowdFM (B=0 ⇒ CrowdFM). Selection policies compared under identical correction+budget.

**VOI (ours)** = adaptive greedy: after each anchor, update worker reliability beliefs;
anchor the task that resolves the most *reliability-uncertain OR consensus-deviating
(suspect)* AND *influential* workers, boosted by the task's own risk (double benefit:
anchoring a likely-wrong task fixes it free AND informs the correction).

Effective accuracy vs budget (anchored counted correct), and AURC (area under budget-error, lower better):

| scenario | CrowdFM | metric | random | entropy | **VOI (ours)** | oracle |
|---|---:|---|---:|---:|---:|---:|
| mixed | 0.851 | AURC | 0.1186 | 0.1530 | **0.1108** | 0.1031 |
| coalition | 0.675 | AURC | 0.2712 | 0.2839 | **0.2539** | 0.2467 |
| sparse | 0.549 | AURC | 0.4026 | 0.3917 | **0.3880** | (≤B20) 0.3891 |

VOI − random (pp), by budget: coalition {B5 +0.5, B10 +2.0, B20 +2.0, B50 +4.2};
sparse {+0.6, +1.4, +2.1, +3.3}; mixed {+0.9, +1.2, +1.0, +0.7}. VOI − entropy is larger
(entropy/uncertainty-sampling is *worse than random* on mixed/coalition — a notable finding:
anchoring ambiguous tasks yields noisy worker-reliability estimates that hurt the correction).

**Findings (validated core):**
1. **Correction works:** a few anchors + reliability-weighted correction on *frozen* CrowdFM
   recover +2 to +14pp on failure regimes (coalition, sparse) — clean, starts = CrowdFM.
2. **Selection matters a lot:** large oracle gap (up to +10pp at B20 on coalition) — *which*
   tasks to anchor is a real problem.
3. **Uncertainty sampling FAILS** (entropy ≤ random) — the standard active-learning default is
   the wrong objective here; the value is reliability *identification*, not task uncertainty.
4. **Our reliability-ID VOI wins:** beats random and entropy on all three shifts, and nearly
   matches the oracle in AURC on the adversarial coalition regime (0.254 vs 0.247).

This is a validated, publishable core: (framework) evidence-triggered anchoring on a frozen
crowd FM; (novelty) reliability-identification VOI selection that beats uncertainty sampling;
(mechanism) anchor-conditioned correction. Remaining for the paper: ≥3 seeds, real crowd
datasets, a genuine Sybil/temporal shift, the e-value dataset-level escalation gate
(when to spend budget), and a learned correction toward the +14–35pp oracle-worker ceiling.

## CrowdGuard on REAL crowd datasets (validated)

`crowdguard_real.py`, frozen CrowdFM (res_mech), anchor-reliability correction + selection,
gold used only as purchased anchors + scoring. Mean gain over datasets (method − CrowdFM):

| B (anchors) | random | entropy | **VOI (ours)** |
|---|---:|---:|---:|
| 10 | +0.7 | **−3.2** | **+1.8** |
| 20 | +0.9 | −2.9 | **+2.4** |
| 50 | +2.1 | +1.1 | **+4.1** |

VOI beats random and (failing) entropy on real crowds too — uncertainty sampling is
negative at low budget, reproducing the synthetic finding. Per-dataset VOI gains (B10→B50):
Bird +13.9→+20.4, CF +1.7→+8.7, ZC_in +3.8→+2.8, Face +1.7→+7.0, CF* +1.7→+7.0, MS +0.6→+3.1,
Dog +0.6→+4.0, ZC_all/us +1.8→+2.0, PosSent/LabelMe/SP/Web/RTE small (+0–2pp, CrowdFM already
strong). Gains largest where CrowdFM has room (Bird, CF, ZC_in).

**Data caveat:** Trec (CrowdFM acc 0.077) and Senti (0.009) are far below chance ⇒ a
label/option-index misalignment in those two datasets' loading (not real CrowdFM behavior);
exclude/fix before the paper. The other 14 datasets are sensible (0.63–0.95).

**Status:** CrowdGuard is now validated on synthetic (all shifts) AND real crowds: few
VOI-selected expert anchors + reliability correction on a frozen CrowdFM give consistent
gains, beating random and uncertainty-sampling baselines. Remaining: e-value escalation gate
demo, ≥3 seeds, a genuine Sybil/temporal regime, learned correction toward the oracle-worker
ceiling, fix Trec/Senti loading, standard baselines.

**Code synced:** committed to branch `agent/crowdguard` (commit c54033a); push pending user
GitHub auth on the HPC.

## Escalation gate (Stage A) — honest negative; correction is low-risk instead

`escalation_analysis.py`: does any gold-free dataset-level signal predict WHERE expert
anchoring pays off? Correlation with realized VOI gain@B20 over 14 valid real datasets:

| signal (gold-free) | Spearman | Pearson |
|---|---:|---:|
| e-value (response-shift) | −0.07 | −0.15 |
| CrowdFM entropy | −0.04 | −0.09 |
| CrowdFM max-confidence | +0.16 | +0.05 |
| worker disagreement | −0.02 | +0.10 |
| CrowdFM accuracy (needs gold) | −0.42 | −0.24 |

**Finding:** no cheap signal reliably predicts anchor benefit; the e-value in particular
does NOT (it detects response-process *shift* — verified, calibrated — which is orthogonal
to *correctable truth error*, consistent with the likelihood≠truth result). So the planned
e-value escalation gate is the wrong trigger, and there is no good gold-free "where to
spend" predictor at the dataset level (gains depend on dataset-specific correctability that
aggregate stats miss; e.g. Bird +16.7pp is unremarkable on every signal).

**Reframing (what to do instead):** the correction is *low-risk* — at B=20 it is positive
or ≈0 on all 14 datasets (worst ≈ +0.3pp), never meaningfully negative. So a perfect
escalation gate is not required: "always apply a small anchor budget" is safe, and the
value concentrates on the shifted/failure regimes (synthetic coalition/sparse; real Bird/CF/
ZC_in) — exactly where CrowdFM has room. The paper should present the escalation gate as an
open problem (and the e-value as a valid shift *detector* but not a benefit predictor),
and center the contribution on the safe, budget-efficient active-anchoring + correction.

## Sybil / adversarial regime — CrowdGuard's showcase (strongest VOI win)

Added a `sybil` family to the simulator: a fraction (0.2–0.45) of workers report a FIXED
wrong target class regardless of truth (coordinated attack) — systematically wrong, only
detectable by cross-task reliability (i.e., expert anchors). This is the canonical scenario
for the paper's question.

CrowdFM drops to 0.687 under the attack. Effective accuracy vs budget:

| B | random | entropy | **VOI (ours)** | oracle | VOI−random | VOI−entropy |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 0.718 | 0.711 | 0.733 | 0.800 | +1.5 | +2.2 |
| 10 | 0.735 | 0.723 | 0.750 | 0.831 | +1.5 | +2.6 |
| 20 | 0.763 | 0.745 | 0.784 | 0.864 | +2.2 | +3.9 |
| 50 | 0.811 | 0.806 | 0.855 | — | +4.4 | +4.9 |

AURC: **VOI 0.2382** < random 0.2572 < entropy 0.2654 (oracle 0.2261). Widest VOI margin
over baselines of all regimes, and closest to oracle. CrowdGuard recovers CrowdFM
0.687→0.855 (+16.8pp) at B=50. This is the clean showcase: reliability-identification VOI
selection + correction is exactly what neutralises coordinated adversarial annotators.

### CrowdGuard status — validated core (multi-regime + real)
| regime | CrowdFM | VOI beats random | VOI beats entropy | VOI vs oracle (AURC) |
|---|---:|---|---|---|
| in-prior (mixed) | 0.851 | + | ++ | near |
| coalition | 0.675 | ++ | ++ | near |
| sparse | 0.549 | ++ | + | near |
| **sybil (adversarial)** | 0.687 | **+++** | **+++** | **closest** |
| real (14 datasets) | 0.63–0.95 | + (+2.4pp @B20 avg) | ++ (entropy negative) | — |

**Solid contributions:** (1) evidence that a frozen crowd FM leaves large *anchor* headroom
under deployment shift (oracle-worker +14–35pp); (2) a reliability-identification VOI active
selection that beats random and decisively beats uncertainty sampling (which is *negative*),
nearly matching the oracle on adversarial shifts; (3) a low-risk anchor-conditioned
correction on frozen CrowdFM; validated on 4 synthetic shifts + 14 real datasets.
**Open/next:** escalation gate (shown hard — reframe as low-risk always-apply), learned
correction toward the ceiling, ≥3 seeds, temporal-drift regime, standard baselines, fix
Trec/Senti loading, write-up.

## Correction strength — scalar is near the per-budget frontier (learned correction deprioritized)

`correction_test.py` (fixed VOI selection, isolate the correction): scalar reliability-
weighted correction vs CrowdFM-prior semi-supervised Dawid–Skene (full K×K confusions) vs
oracle-worker ceiling (all gold).

| regime | ceiling | scalar @B10/20/50 | cfds(full-confusion) @B10/20/50 |
|---|---|---|---|
| coalition | +34.1pp | +6.0 / +10.0 / +18.4 | +3.7 / +7.2 / +16.6 |
| sparse | +36.7pp | +4.1 / +8.0 / +17.3 | +3.6 / +6.9 / +16.2 |
| sybil | +29.7pp | +6.7 / +10.3 / +17.5 | +7.5 / +10.3 / +17.4 |

**Finding:** the simple scalar log-odds reliability correction **matches or beats** the
full-confusion DS at all budgets — with few anchors you can robustly estimate one
reliability scalar per worker but not full K×K confusions (which are noisy/undersampled).
The correction bottleneck is **anchor information, not model capacity**, so a *learned*
correction has limited upside at small budget → **deprioritized**.

**Frontier framing (important for the paper):** the oracle-worker ceiling (+30–37pp) is an
*information* bound (effectively uses ~all labels), not a method deficiency. At a *fixed
budget*, VOI selection is within ~0.7% AURC of the *selection* oracle (coalition: VOI 0.254
vs select-oracle 0.247), and the scalar correction captures ~44–58% of the ceiling by B=50.
So CrowdGuard is **near the achievable per-budget frontier**; the remaining gap to the
ceiling closes only by buying more labels. This makes the contribution clean: *expert-
efficient* correction that is near-optimal for its budget, not a chase after the ceiling.

**Re-prioritized roadmap:** (1) solidify — ≥3 seeds, temporal-drift regime, tighten CIs;
(2) standard baselines (MV, DS+anchors, honeypot/gold-injection, budget-allocation);
(3) theory: reliability-identification objective + the information frontier; (4) fix
Trec/Senti; (5) write-up. Learned correction downgraded to optional.

## ⚠ Baseline reality-check — the frozen-FM prior helps only under adversarial attack

`baseline_test.py` (same VOI anchors for all methods): does the frozen CrowdFM prior beat
CLASSIC anchor-using aggregators (no FM)? Effective accuracy at B=10/20/50:

| regime | CrowdFM(0) | mv_rw (reliab. MV, no FM) | ds_anchor (semi-sup DS, no FM) | CrowdGuard (FM) | CG − best-no-FM |
|---|---:|---|---|---|---:|
| mixed (+sybil) | 0.708 | 0.82/0.85/0.89 | **0.86/0.87/0.91** | 0.83/0.85/0.88 | **−2.1 to −2.6** |
| coalition | 0.653 | 0.71/0.75/0.84 | 0.70/0.74/0.83 | 0.71/0.75/0.84 | +0.7 / +0.2 / −0.6 |
| sparse | 0.546 | **0.585/0.624/0.720** | 0.47/0.51/0.62 (collapses) | 0.587/0.626/0.719 | +0.2 / +0.1 / −0.1 |
| **sybil** | 0.674 | 0.66/0.72/0.83 | 0.71/0.73/0.80 | **0.74/0.78/0.85** | **+3.1 / +4.5 / +2.1** |

**Honest conclusion:** CrowdGuard beats the best *classic anchor-aggregator* clearly ONLY
under heavy adversarial (Sybil) attack. On benign/mixed, semi-supervised DS+anchors is
*better* (−2pp); on coalition/sparse a simple reliability-weighted majority vote (no FM)
ties it. On sparse, mv_rw (0.585) already beats CrowdFM (0.546), so the FM prior contributes
little — the anchors' reliability estimate does the work. The foundation-model prior is NOT
pulling weight except when a global view resists coordinated local corruption (Sybil).

**What genuinely stands (independent of this caveat):**
1. **VOI active selection** — beats random and (failing) uncertainty-sampling for ALL
   aggregators, on all regimes + real data. This is the robust, novel methodological result.
2. **Anchor-based reliability weighting recovers accuracy under deployment shift** at small
   budget (framework-level positive), but classic reliability-MV is a strong baseline.
3. **Sybil / adversarial** is where the FM prior + the full method clearly win.

**Implication for framing (needs a decision):** the "crowd *foundation model*" angle is
weaker than hoped — the FM prior only helps under adversarial attack. Stronger honest
framings: (a) center on **active reliability-identification selection** (VOI beats
uncertainty sampling — robust, novel, model-agnostic); (b) narrow to **adversarial/Sybil
deployment** where the full approach shines; or (c) both. NOT a broad "FM + anchors beats
everything" claim.

## Plan C — Adversarial attack-strength sweep (the money plot)

`baseline_test.py` over Sybil fraction ρ (fixed via `sybil_fraction_range`), budget B=20,
all methods use the same VOI anchors. CrowdGuard = log CrowdFM + anchor-reliability votes.

| ρ (adversary frac) | CrowdFM | best classic no-FM | **CrowdGuard** | CG − best-no-FM |
|---:|---:|---:|---:|---:|
| 0.1 | 0.822 | 0.807 | **0.863** | **+5.6** |
| 0.2 | 0.791 | 0.774 | **0.839** | **+6.6** |
| 0.3 | 0.723 | 0.748 | **0.798** | **+5.0** |
| 0.4 | 0.611 | 0.712 | **0.736** | **+2.4** |
| 0.5 | 0.415 | 0.673 | 0.635 | **−3.8** |

**Result (plan-C headline):** under low-to-moderate coordinated attack (ρ ≤ 0.4 — the
realistic regime), **CrowdGuard beats the best classic anchor-aggregator by +2.4 to +6.6pp**,
peaking at moderate attack. Only under extreme attack (ρ=0.5, half the workers adversarial)
does it reverse — because CrowdFM *itself* is destroyed (0.415) and anchoring its corrupted
prior hurts, while FM-free methods recover.

**Attempted fix + insight (honest):** "down-weight the FM prior when it is corrupted" using
CrowdFM's accuracy on the anchors FAILED (worse everywhere) — because **VOI deliberately
selects tasks where CrowdFM is likely wrong**, so FM accuracy *measured on the selection
anchors is biased low* and cannot gauge true FM trust. Correct fix needs an *unbiased* probe
(spend a few random anchors to estimate FM reliability) or a learned trust signal —
noted as future work. The clean, defensible result is the fixed CrowdGuard above.

**Plan-C narrative (validated):** *Active expert anchoring for robust crowd aggregation
under adversarial deployment shift.* (1) VOI reliability-identification selection beats
uncertainty sampling (which fails) for all aggregators; (2) under realistic coordinated
attack, FM-anchored correction beats classic anchor methods by +2–7pp; (3) documented
failure at extreme attack + the selection-bias insight for FM-trust calibration.
