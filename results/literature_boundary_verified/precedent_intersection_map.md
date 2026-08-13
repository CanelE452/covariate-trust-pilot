# Precedent intersection map

**Revised 2026-08-12 (second build).** The first build recorded component 3 as `PRIOR`
while the evidence matrix recorded the same fact as `UNKNOWN` — the two files disagreed.
The cause was a missing distinction, now fixed: see `evidence_policy_separation.md`.

Every component carries **two independent fields**.

```
evidence_status   WHAT THE LITERATURE SHOWS
                  PRIOR | PARTIAL_OVERLAP | NOT_FOUND_IN_AUDIT | UNKNOWN

novelty_policy    WHAT WE DECIDE TO CLAIM
                  NOT_CLAIMED | EXCLUDED_FROM_NOVELTY |
                  CLAIM_ONLY_IN_CONJUNCTION | CLAIMED_IN_CONJUNCTION
```

A conservative policy is never recorded as a finding. `NOT_FOUND_IN_AUDIT` is a statement
about **this search**, never about the world, and is never rendered as "no prior work
exists".

Machine-readable twin: `novelty_component_map.csv`, checked against
`literature_evidence_matrix.csv` by `verify_consistency.py`.

---

## Component table

```
#    component                                    evidence_status      novelty_policy               closest
------------------------------------------------------------------------------------------------------------------------
1    ADI / CV^2 classification                    PRIOR                NOT_CLAIMED                  [SBC05] [KH06]
2    temporal dependence manipulated              PRIOR                NOT_CLAIMED                  [ALR12]
3    controlled marginal characteristics +        UNKNOWN              EXCLUDED_FROM_NOVELTY        [ALR12]
     dependence variation
4    occurrence / magnitude decomposition         PRIOR                NOT_CLAIMED                  [Cro72] [SB05] [TSB11]
4b   the occurrence-PROBABILITY form              PRIOR                NOT_CLAIMED                  [TSB11]
5    neural direct vs decomposed neural           PRIOR                NOT_CLAIMED                  [Kou13]
6a   direct / single-stage vs hurdle / two-       PRIOR                NOT_CLAIMED                  [NAR26]
     stage PRODUCT form, identical features
6b   the same, under matched capacity and         NOT_FOUND_IN_AUDIT   CLAIM_ONLY_IN_CONJUNCTION    [NAR26], match
     matched training                                                                               status UNKNOWN
7    occurrence and magnitude dependence on       NOT_FOUND_IN_AUDIT   CLAIMED_IN_CONJUNCTION       [ALR12] and [Kou13]
     SEPARATE axes, CROSSED with representation                                                     on opposite sides
8    finite-sample relative inductive bias at     NOT_FOUND_IN_AUDIT   CLAIMED_IN_CONJUNCTION       [Kou13]
     a fixed budget
9    controlled synthetic condition followed      NOT_FOUND_IN_AUDIT   CLAIMED_IN_CONJUNCTION       [TJWC21]
     to a real-data transfer boundary
```

### Component 3 — the one that changed

```
evidence_status  UNKNOWN.  [ALR12]'s full text could not be obtained; which marginal
                 characteristics are held constant across correlation levels is not
                 established.  Six access routes failed (alr12_fulltext_verification.md).
                 NOT promoted to PRIOR: recording an unverified gap as a precedent is
                 not conservatism, it is fabrication in the safe-looking direction.
                 NOT read as NOT_FOUND: that would favour us.
novelty_policy   EXCLUDED_FROM_NOVELTY, and this does NOT depend on [ALR12].  Holding
                 the marginals fixed is an EXPERIMENTAL CONTROL in this paper's design:
                   claim_ledger_frozen.md reader chain 1-2 -- the control is the PREMISE
                     of step 2, not its result
                   final_outline_freeze.md -- "related work: no novelty claim over the
                     correlation literature", frozen before this audit ran
consequence      no sentence may say what [ALR12] holds constant, AND no sentence may
                 claim marginal control as ours.  Both directions closed.
```

### Component 6 — the one that split

```
6a   evidence_status PRIOR.  [NAR26] verified at full text: a LightGBM regressor
     "trained directly on the full feature set" against a two-stage LightGBM classifier
     x Tweedie regressor -- P(sale) x E[qty | sale] -- with "identical data
     preprocessing, feature construction, and evaluation protocols".
     The comparison of the two FORMS has been done.  Conceded without qualification.
     Not weakened by [NAR26] being non-neural.

6b   evidence_status NOT_FOUND_IN_AUDIT.  No located paper STATES a match on capacity or
     training between a direct arm and a product-form arm.
       [NAR26]  NAR-E, NAR-F, NAR-G all NOT STATED -> U, not N
       [Kou13]  parameters N (two output nodes vs one; per-arm configuration selection);
                training P (shared trainer, unmatched capacity)
     novelty_policy CLAIM_ONLY_IN_CONJUNCTION.  Because the nearest neighbour is UNKNOWN
     rather than known-absent, 6b must never carry a claim standalone.  It appears only
     as one condition inside the intersection.
```

Guard: **"same model family" and "same feature set" do not establish matched capacity or
matched training.** Matching is recorded only where the source states it
(`LIT-W-NAR26`).

---

## Where the three nearest precedents sit

```
[ALR12]   has    component 2, both dependence axes
          lacks  any representation contrast -- the compared objects are Croston-family
                 estimators inside one already-factorized form
          U on   component 3

[Kou13]   has    component 5, and it is a real precedent, conceded in full
          lacks  component 2 (one simulated population, no dependence factor)
                 matched capacity (per-arm tuning; output widths differ by construction)
          note   its decomposed arm is a size / interval RATIO, de-biased afterwards,
                 not a probability x magnitude PRODUCT

[NAR26]   has    component 6a in the product form, at an identical feature set
          lacks  component 2 (NAR-H verified negative: no dependence factor, no regime
                 or dependence breakdown), neural models, any synthetic study
          U on   component 6b (capacity, training budget, per-formulation tuning)
```

The dependence question, the representation question and the product-form comparison
have each been asked. This audit located no work in which the representation question is
asked **as a function of** the dependence question.

---

## The intersection, stated as the contribution

Four conditions, none of them a claim on its own:

```
I1  a matched comparison -- one backbone, 5,856 = 5,856 parameters, one trainer, one
    30-epoch budget, one evaluation target
      artifact  claim_ledger_frozen.md section 1; T1 in final_outline_freeze.md
      policy    CLAIM_ONLY_IN_CONJUNCTION (component 6b)
I2  occurrence dependence and magnitude dependence varied on SEPARATE axes, marginals
    held fixed AS AN EXPERIMENTAL CONTROL
      artifact  Stage 1 fixed-marginal 2x2x2 factorial; Stage 2 eighteen-cell rho sweep
      policy    the control itself is EXCLUDED_FROM_NOVELTY (component 3);
                the separate axes are CLAIMED_IN_CONJUNCTION (component 7)
I3  the outcome read as the RELATIVE finite-sample behaviour of the two representations,
    not as a method ranking
      artifact  G = 100(1 - RMSE_Hurdle/RMSE_Point); |rho_I| for occurrence, signed
                rho_M for magnitude
      policy    CLAIMED_IN_CONJUNCTION (component 8)
I4  the controlled pattern followed into real demand until it breaks -- an empirical
    analogue, a frozen selector that transfers, an isolated mechanism that does not
    survive overlap adjustment
      artifact  H1 / H2 statuses in claim_ledger_frozen.md section 4
      policy    CLAIMED_IN_CONJUNCTION (component 9)
```

---

## Why this is not "old ideas glued together"

The honest objection: *dependence has been varied before, representations have been
compared before, hurdle and direct have been compared before; the conjunction is
arithmetic.*

It would be, if the crossing returned what the separate strands predict. It does not.

```
what a naive combination predicts
    dependence makes series more predictable, so the more structured representation
    benefits monotonically along both axes.

what the crossing returns
    the two axes respond to DIFFERENT FEATURES of dependence.  Occurrence tracks the
    STRENGTH and is largely insensitive to sign -- |rho_I| carries 2.9x the coefficient
    of signed rho_I, and both signs give a positive G.  Magnitude tracks the DIRECTION --
    signed rho_M carries 3.1x the coefficient of |rho_M| -- and it is persistence, not
    structure, that moves the comparison toward direct prediction.  Within the
    eighteen-cell sweep this yields a direct-favourable configuration -- sparse,
    occurrence-unpredictable, magnitude-persistent -- at about -19.8%.
      artifact  claim_ledger_frozen.md section 3
```

That shape is not recoverable by reading [ALR12], [Kou13] and [NAR26] together. It rests
on our own artifacts, which is where a contribution argument belongs.

**No "first to combine" claim is made.** Recall is unquantified (`WARN_FAIL.md` G2, G3),
and a combination claim is exactly as unfalsifiable as a precedence claim.

---

## What the contribution is NOT

```
NOT   the first study of temporal dependence in intermittent demand      component 2
NOT   the first to control marginals while varying dependence            component 3
NOT   the first comparison of direct and decomposed neural forecasters   component 5
NOT   the first comparison of direct and hurdle / two-stage forms        component 6a
NOT   the introduction of occurrence / magnitude decomposition           component 4
NOT   the introduction of the occurrence-probability parameterization    component 4b
NOT   a claim that either representation is generally preferable
```
