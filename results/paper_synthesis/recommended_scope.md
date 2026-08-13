# Recommended scope and contributions

**Scope B** — a controlled characterization of when direct and factorized representations of
intermittent demand differ, how far that characterization survives contact with real demand,
and where the conditional advantage stops being usable.

---

## C1 — Controlled characterization of relative Point/Hurdle inductive bias

**What is new.** ADI and CV² describe the marginal distribution of a demand series and say
nothing about the order of its zeros. The controlled study manipulates occurrence-interval and
magnitude dependence while holding the marginals fixed, so the relative advantage of the two
representations is attributed to temporal structure rather than to sparsity or scale.

**Evidence.** The synthetic study (Stage 1 / Stage 3 / Stage 4). **Its artifacts are not in
this repository** — see `artifact_audit.json`, entry `A_numerical`. Every downstream document
in this repo references it, and the frozen H1/H2/H3 definitions in
`pre_analysis_spec.json` were derived from it, but no number from it can currently be cited
from an artifact here.

**How it differs from existing work.** Croston-family research studies estimators under
intermittency; this studies which *representation of the conditional mean* is favoured, given
the temporal structure of occurrence, with the marginals controlled.

**Boundary.** One backbone family. Synthetic delta is measured against an exact oracle and is
not numerically comparable to the real-data delta measured against realized y.

**Status: BLOCKED on artifact recovery.** Not blocked on new science.

---

## C2 — Real-data validation and the entanglement boundary

**What is new.** The structural relationship transfers directionally, and a rule derived from
the controlled study works as an out-of-sample selector, but the association it exploits does
not survive balancing on scale and sparsity. That gap — a good predictor that is not an
explanation — is the finding, and it is measured rather than asserted.

**Evidence (strongest first).**
- Frozen H2 rule on an independent M5 population, 675 candidates against 5,018 controls:
  effect **−0.0230** CI [−0.0294, −0.0163]; Point win rate **+11.87 pp** CI [+7.85, +15.81].
  Reproduced under seeds 0/1/2 with every CI excluding zero.
- Overlap-adjusted association on the same population: **+0.0032** CI [−0.0033, +0.0094];
  1:2 matching fails at worst SMD **0.614** on `log_train_scale`.
- H1 marginal association: M5 **+0.1065** CI [+0.0437, +0.1652], Favorita **+0.0789**
  CI [+0.0205, +0.1405]; six scale estimates across two datasets, all positive, all CI clear of
  zero; intermittent regime relative **+0.153** CI [+0.052, +0.261].
- Non-replications reported as results, not omitted: H3 (both signs opposite, both CI include
  zero) and the occurrence gate's Brier skill against a constant rate
  (M5 −0.008, Favorita −0.091 CI [−0.140, −0.044]).

**How it differs from existing work.** External-validity screens of a synthetic finding are
rare in this literature, and the selector/mechanism split is normally not attempted at all.

**Boundary.** Two retail datasets; a regime-balanced sample, not a population sample; the
adjusted partial association for H1 is small and not separated from zero.

**Status: SUPPORTED, ready to write.**

---

## C3 — The adaptive-use boundary

**What is new.** A measurable oracle opportunity does not imply a learnable routing function.
The paper shows the opportunity being *enlarged* on purpose and still not converted, with the
usual explanations eliminated one at a time.

**Evidence.**
- Opportunity: per-origin convex oracle is **4.11%** better than the best static mixture on M5;
  deliberately diversifying the expert pair multiplies the ceiling by **2.15**.
- Non-conversion: the frozen gate's first external test is **−2.43%** CI [−2.74, −2.13] on the
  dataset pre-declared as the primary external confirmation.
- Elimination: target (Gate-v3 diagnosis), loss and parameterisation (2×2 factorial),
  aggressiveness (Safe-P0L1: better tails on 4/4, worse mean on 4/4), capacity
  (HGB wins 1 of 4 on identical features), representation (raw-history GRU: FreshRetailNet
  −0.506% → **+2.648%** CI [+2.068, +3.287], UCI **−193.9%**).
- The realized gain even where routing "worked" was small: +0.43% over the better expert on the
  diverse pair.

**How it differs from existing work.** Mixture-of-experts papers report the wins. This reports
a pre-registered stop rule being triggered, with the oracle gap quantified so the reader can
see exactly how much was on the table and was not collected.

**Boundary.** One expert family throughout. Single canonical seed for the sequence gate (frozen
before training, matching the handcrafted gate's single-seed structure). UCI's availability
semantics are recorded as `AVAILABILITY_UNKNOWN` in `dataset_audit.json`, which weakens it as a
benchmark though not as a counterexample.

**Status: SUPPORTED, ready to write.**

---

## Non-overlap check

C1 is about a manipulated cause under known ground truth. C2 is about whether the same
structure carries predictive information in observational data and whether it survives
adjustment. C3 is about whether a conditional advantage that exists can be exploited. No two
share a primary artifact: C1's is the synthetic study, C2's is
`rule_replication/primary_result.json`, C3's is `multi_benchmark/external_benchmark.json`.
