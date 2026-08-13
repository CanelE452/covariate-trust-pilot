# Frozen claim ledger

Frozen 2026-08-11, after the synthetic source was recovered and verified. Every
number is from `results/synthetic_source_verification/` or from an artifact named
inline. This file fixes **wording**, not just findings, so the story cannot drift
between here and the manuscript.

`results/paper_synthesis/` is preserved unmodified as the pre-recovery record.

---

## 0. Frozen terminology

These six are settled. Anything written later that contradicts them is a defect.

**F1 — stage names.**
```
Stage 1                     fixed-marginal 2x2x2 factorial
                            (decomposition_when_helps, run_20260802_112655)
Stage 2                     stationary rho sweep, 18 cells
                            (temporal_dependence/rho_sweep, pilot_20260803_051713)
Stage 1 component           the hybrid-column decomposition inside the Stage 1 run
  diagnostic                (p_true x mu_hat, p_hat x mu_true, p_hat x mu_hat)
```
"Stage 3" and "Stage 4" do not exist in the source and are retired.

**F2 — one performance quantity.**
```
G = 100 * (1 - RMSE_Hurdle / RMSE_Point)          G > 0 favours Hurdle
                                                   G < 0 favours Point
```
Verified: recomputing Stage 1's main effects from cell means in this unit gives
+7.831 / −4.581 / −6.264 pp against stored +7.831 / −4.581 / −6.264. Stage 1's
`effect_pp` column is already exactly this quantity.

Raw `Δ` is never used across studies: Stage 1's absolute delta is
`RMSE_hurdle − RMSE_point`, Stage 2's and the real data's are `RMSE_point − RMSE_hurdle`.
Opposite signs. If an absolute delta must appear, its formula goes on the same line.

**F3 — Stage 2 carries the generalization.**
Stage 1's predictable arm is deterministic period-2 alternation (ρ = −1) and its
status is `CONDITIONALLY_VALID`; its own audit blocks the claims "temporal
predictability in general drives the hurdle advantage" and "the Stage 1 interval
effect measures a graded predictability axis". The source's report states that
Stage 2 "overturns the reading Stage 1 invited". Stage 1 establishes that the
structure matters and that the two axes interact; **Stage 2 establishes what the
operative quantity is.** Stage 1 is never cited alone for a graded claim.

**F4 — H2 is two claims.**
```
predictive selector transfer        supported
isolated mechanism                  not replicated
```
Never "the synthetic mechanism replicated".

**F5 — mechanism strength.**
`COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT`, not `MECHANISM_AGGREGATE_SUPPORT`.
Allowed: "diagnostic evidence that occurrence estimation carries the error under
specific conditions." Not allowed: any claim that the factorial effect's causal
mechanism has been demonstrated. The hybrid columns substitute a true component for
an estimated one; they attribute error, they do not identify a cause.

**F6 — transport audit label.**
`PRIMARY SOURCE PAYLOAD INTEGRITY PASS`.
Precisely: the Ubuntu run produced 20 files; 19 were uploaded, the twentieth being
the un-split 6.9 GB tar, which was reconstructed on Windows from its four parts and
matched its Ubuntu SHA-256 byte for byte (7,380,474,226 bytes). So the scientific
payload is bit-identical. The additional inventory artifacts specified in the
original recovery protocol (`scientific_candidate_paths.txt`,
`synthetic_critical_artifact_inventory.md`, `working_tree_scientific_manifest.csv`,
`recovery_report.md`) were **never generated on Ubuntu** — the paste-safe script
dropped them — and the equivalent analysis was performed directly against the
recovered source on Windows, which is strictly better than transcribing a remote
inventory. Do not write "recovery package full metadata parity PASS".

---

## 1. Contribution 1 — frozen wording

> **Controlled characterization of the finite-sample relative inductive bias of
> direct and factorized forecasting under temporal occurrence and magnitude
> dependence.**

`finite-sample` is load-bearing and is supported by the design: both arms are
parameter-matched at 5,856, share one trainer and a 30-epoch budget, and are scored
against an **exact** oracle conditional mean on 80 series per cell. What is measured
is estimation behaviour under a budget, not the asymptotic capability of two
function classes that share a target.

Status: **CONFIRMED** (C1-G1…G8 all pass, `c1_gate_report.json`).

---

## 2. Stage 1 — what it does and does not license

Per-cell G, M1 vs M0, with paired series bootstrap:

```
cell  d  interval      magnitude       G (pp)        95% CI (pp)
C01   4  alternating   alternating       8.47    [  6.36,  10.57]
C02   4  alternating   independent      27.81    [ 25.46,  30.27]
C03   4  independent   alternating      18.76    [ 17.04,  20.50]
C04   4  independent   independent       8.55    [  7.14,   9.82]
C05   8  alternating   alternating       3.57    [  2.18,   5.00]
C06   8  alternating   independent      26.87    [ 23.78,  30.18]
C07   8  independent   alternating      11.10    [  8.37,  13.78]
C08   8  independent   independent      -3.01    [ -6.79,   0.23]
```

Contrasts:

```
interval_dependence      +7.83  [ +6.09,  +9.45]
magnitude_dependence     -4.58  [ -6.28,  -2.92]
sparsity                 -6.26  [ -8.01,  -4.57]
sparsity x interval      +3.35  [ +1.62,  +5.09]
sparsity x magnitude     -0.01  [ -1.70,  +1.69]      spans zero
interval x magnitude    -16.74  [-18.45, -15.05]
three_way                -1.96  [ -3.64,  -0.31]
```

**Licensed:** relative performance moves substantially with temporal structure while
the marginals are held fixed; the two axes interact strongly, and the interaction is
larger in magnitude than either main effect.

**Not licensed:** any graded reading of the interval axis; any claim about
persistence, since the predictable arm is alternation only.

**New, and it matters for H2:** Stage 1 contains **no** cell where Point wins with an
interval clear of zero. C08 is −3.01 with the interval touching zero. The
Point-favourable region is a Stage 2 result, not a Stage 1 one.

---

## 3. Stage 2 — the generalization, and the asymmetry between the two axes

```
term            estimate    95% CI              excludes 0
abs_rho_I       +0.1904  [+0.1699, +0.2119]        yes
rho_I           +0.0667  [+0.0533, +0.0807]        yes
abs_rho_M       -0.0228  [-0.0441, -0.0020]        yes
rho_M           -0.0711  [-0.0851, -0.0570]        yes
d               -0.0239  [-0.0322, -0.0153]        yes
rho_I x rho_M   -0.0886  [-0.1115, -0.0654]        yes
d x rho_I       +0.0332  [+0.0194, +0.0471]        yes
d x rho_M       -0.0124  [-0.0262, +0.0012]         no
```

**The asymmetry, stated as a frozen claim.** For the occurrence axis the *magnitude*
of dependence is what matters — `|ρ_I|` carries 2.9 times the coefficient of signed
`ρ_I`, and both signs give a positive G (C_neg +0.127, C_pos +0.201, both intervals
clear of zero). For the magnitude axis the *sign* is what matters — signed `ρ_M`
carries 3.1 times the coefficient of `|ρ_M|`, and it is persistence, not alternation,
that moves the comparison toward the direct model.

This single asymmetry explains the apparent tension between the two stages: Stage 1's
magnitude arm was alternation, and its magnitude-alternating cells are strongly
Hurdle-favourable (C03 +18.76, C07 +11.10). Only Stage 2's positive-persistence arm
produces a Point-favourable region.

**The Point-favourable region, exactly one cell of eighteen:**
```
d = 8, rho_I = 0.0, rho_M = +0.8      G = -19.76  [-26.00, -14.53]
```
sparse, occurrence-unpredictable, magnitude-persistent.

Classification `CLASS_A_GENERAL_PREDICTABILITY_SUPPORT`; five integrity gates pass,
including a no-signal control whose out-of-sample Markov gain is significantly
negative.

---

## 4. Empirical transfer — frozen statuses

```
H1                      SUPPORTED_WITH_BOUNDARY        EMPIRICAL_ANALOGUE
  external uses abs(rho_interval); Stage 2 shows that is the finding, not a
  convention. M5 +0.1065, Favorita +0.0789; six of six scale estimates with
  intervals clear of zero; present in the intermittent regime, absent in lumpy;
  adjusted partial association not separated from zero.

H2 predictive selector  CONFIRMED                      EMPIRICAL_ANALOGUE
  synthetic origin is the Stage 2 cell above; the frozen rule's three axes
  correspond one to one, sign included. Independent M5 population: -0.0230
  [-0.0294, -0.0163], Point win rate +11.87 pp, reproduced under three seeds.

H2 isolated mechanism   NOT_REPLICATED
  overlap-weighted association +0.0032 [-0.0033, +0.0094]; matching failed at
  SMD 0.614 on log scale. The synthetic design has no scale axis, so it cannot
  arbitrate this either way.

H3                      NOT_REPLICATED at the pre-registered split,
                        CONSTRUCT_MISMATCH on the contrast
  synthetic sparsity x interval +3.35 pp and d x rho_I +0.0332, both intervals
  clear of zero, at ADI 4 vs 8; external split at the ADI median, 1.304 and 1.317.

Occurrence mechanism    COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT (synthetic)
                        LEARNED_OCCURRENCE_HEAD_NOT_SUPPORTED (real data)
  two questions, two labels, never merged.

Routing                 opportunity CONFIRMED, stable learned routing NOT_REPLICATED
```

---

## 5. The reader chain, frozen

1. ADI and CV² summarize the marginal distribution and discard temporal ordering.
2. With the marginals held fixed, the finite-sample behaviour of direct and
   factorized forecasts still diverges according to the temporal structure of
   occurrence and magnitude.
3. A fixed-marginal 2×2×2 factorial establishes that both axes matter and that they
   interact strongly (interaction −16.74 pp, larger than either main effect).
4. A stationary ρ sweep establishes what the operative quantities are: `|ρ_I|` for
   occurrence, signed `ρ_M` for magnitude.
5. The same sweep contains a clear counterexample — sparse, occurrence-unpredictable,
   magnitude-persistent — where the direct model wins by about twenty percent.
6. In real data the occurrence direction reproduces and the Point-favourable selector
   transfers out of sample, but the synthetic axes are entangled with scale and
   sparsity, so the isolated mechanism does not transfer.

This is not a paper claiming that factorization helps on intermittent demand. It is a
paper characterizing **when factorization helps or hurts**, and **how far that
characterization survives contact with real demand.**

---

## 6. What remains open, and is written as open

```
single backbone family      every arm is DLinear, in the synthetic study and the
                            routing chain alike. All claims are conditional on it.
no scale axis in the        the controlled study cannot address the confound that
  synthetic design          removed the external H2 association.
Stage 1 conditional         its predictable arm is alternation; the graded claim
  validity                  rests on Stage 2 alone.
H3 untested at its          synthetic contrast ADI 4 vs 8; support exists in the
  own contrast              real data (M5 127 vs 52, Favorita 84 vs 45) but was
                            never used as a primary test.
```

---

## 7. Frozen status

```
C1                          CONFIRMED
paper readiness             PAPER_READY_FOR_OUTLINE
MUST_HAVE                   NONE
second backbone             NICE_TO_HAVE, not an outline blocker
routing                     ROUTING_MODEL_DEVELOPMENT_STOP, unchanged
next                        detailed outline -> Figure 1/2 and Table spec -> manuscript
```
