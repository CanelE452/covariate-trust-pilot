# H1 / H2 / H3 provenance

For each external hypothesis: what the synthetic source actually shows, how the
external version operationalizes it, and whether that is a direct replication, an
empirical analogue, or a construct mismatch.

Empirical definitions are read from `experiments/external_validity_screen/prereg.py`
and `docs/external_validity_STATUS.md` in this repository; synthetic definitions
from the recovered source.

---

## H1 — occurrence-interval dependence and the factorized advantage

```
                    synthetic                          external
──────────────────────────────────────────────────────────────────────────────────
construct           dependence in the occurrence       serial correlation of
                    interval sequence Q_j              observed inter-arrival gaps
manipulation        designed: rho_I in {-0.8, 0, +0.8} measured: rho_interval_train
                    (Stage 2); alternation vs iid      per series, then |.|
                    (Stage 1)
quantity that       |rho_I|, coefficient +0.1904        abs(rho_interval_train)
  matters           vs signed rho_I +0.0667
outcome             gain = 1 - RMSE_h/RMSE_p against    delta = RMSE_Point - RMSE_Hurdle
                    the exact DP oracle                 against realized y
statistic           factor model over 18 cells          Spearman across ~1,200 series
```

**Verdict: EMPIRICAL_ANALOGUE, with an unusually tight construct match.**

The external hypothesis uses the **absolute value**, and Stage 2 is precisely the
experiment that justifies it: the coefficient on `|rho_I|` is about three times the
coefficient on signed `rho_I`, and both signs of dependence give a positive gain
(C_neg +0.127, C_pos +0.201, both CI clear of zero). So the external `abs()` is not
an arbitrary choice — it is the synthetic finding.

It is an analogue rather than a direct replication because the manipulation becomes
a measurement: synthetic `rho_I` is set by design at three levels, external
`rho_interval_train` is estimated from each observed series and has no controlled
support. The outcome scales also differ (oracle target vs realized y), which is why
only the direction may be compared.

**Allowed:** "the external analysis measures, in observational data, the same
quantity the controlled study manipulates."
**Forbidden:** "H1 replicates the synthetic result." / any numerical comparison of
the synthetic gain and the external Spearman.

---

## H2 — the Point-favourable condition

```
axis                synthetic (Stage 2 cell)          external frozen rule
──────────────────────────────────────────────────────────────────────────────────
sparsity            d = 8  (the sparser of {4, 8})    HIGH_ADI: ADI above the
                                                       dataset median
occurrence signal   rho_I = 0.0  (no dependence)      LOW_OCC: |rho_interval| in the
                                                       lower tertile
magnitude           rho_M = +0.8  (positive           MAG_PERSISTENT: signed
  persistence         persistence, not alternation)    rho_magnitude in the upper
                                                       tertile
result              gain -19.76% [-26.00, -14.53],    rule effect -0.0230
                    the only Point-favourable cell     [-0.0294, -0.0163];
                    of 18                              Point win rate +11.87 pp
```

**Verdict: EMPIRICAL_ANALOGUE with a one-to-one axis correspondence.**

All three axes correspond, including the sign: the synthetic cell uses **positive**
rho_M, and the external rule uses the **signed** upper tertile, not the absolute
value. That distinction was carried over correctly.

Two things stop it being a direct replication. First, the synthetic condition is a
designed corner of a factorial (a single cell at fixed levels), while the external
rule is a conjunction of three sample-relative cutoffs (median, tertile, tertile) on
estimated descriptors. Second, and more importantly, the external analysis found
that once scale, sparsity and occurrence dependence are balanced, the association is
**+0.0032 [−0.0033, +0.0094]** — it does not survive adjustment. The synthetic study
has no scale axis at all, so it cannot speak to that confound.

**Therefore the two readings must stay separate:**

```
predictive selector   synthetic corner exists AND the frozen rule transfers
                      out of sample                              -> supported
isolated mechanism    synthetic isolates rho_M by design, but the external
                      association does not survive overlap weighting
                                                                 -> NOT replicated
```

**Allowed:** "the controlled study contains exactly one Point-favourable
configuration, and its three axes are the three the external rule encodes."
**Forbidden:** "the external result confirms that magnitude persistence causes the
Point advantage."

---

## H3 — sparsity interaction

```
                    synthetic                          external
──────────────────────────────────────────────────────────────────────────────────
construct           does sparsity amplify the          does the |rho_interval|-delta
                    occurrence effect?                 correlation differ between
                                                       high- and low-ADI groups?
contrast            Stage 1: d = 4 vs d = 8            split at the ADI median
                    Stage 2: d x rho_I interaction        (M5 1.304, Favorita 1.317)
result              Stage 1 sparsity x interval        M5 -0.0305 [-0.1418, +0.0912]
                      +3.345 pp [+1.619, +5.091]       Favorita -0.0428 [-0.1587, +0.0704]
                    Stage 2 d x rho_I                  both signs opposite to the
                      +0.0332 [+0.0194, +0.0471]       prediction, both CI span zero
                    both CI clear of zero
```

**Verdict: CONSTRUCT MISMATCH on the contrast, and the external test is therefore
not a test of the synthetic claim.**

The synthetic effect is real and appears in **both** studies independently: sparsity
amplifies the occurrence effect. But the synthetic contrast is d = 4 versus d = 8,
i.e. ADI 4 versus 8, while the external split is at the ADI **median**, which is
1.304 on M5 and 1.317 on Favorita. Both external groups sit far below the lower end
of the synthetic contrast. The external analysis compared ADI ≈ 1.1 against ADI ≈ 2,
not 4 against 8.

The external posthoc recorded that series at the intended contrast do exist —
M5 has 127 series with ADI 3–5 and 52 with ADI ≥ 8; Favorita 84 and 45 — but they
were never used as a primary test.

**Allowed:** "the sparsity interaction is present and significant in both controlled
studies; the external test did not evaluate it at the contrast it was derived from,
and at the pre-registered median split it did not replicate."
**Forbidden:** "sparsity does not modulate the occurrence effect." / presenting the
external H3 result as a refutation of the synthetic interaction.

---

## Summary

```
hypothesis           synthetic support        external status              relation
─────────────────────────────────────────────────────────────────────────────────────
H1                   strong, |rho_I| is the   reproduces in 2 datasets,   EMPIRICAL_ANALOGUE
                     operative axis           6/6 CI clear of zero
H2 as selector       exactly one cell, and    frozen rule transfers,      EMPIRICAL_ANALOGUE
                     it is decisive           3 seeds, +11.87 pp
H2 as mechanism      isolated by design       vanishes under overlap      NOT_REPLICATED
                                              weighting
H3                   present in both Stage 1  not tested at the synthetic CONSTRUCT_MISMATCH
                     and Stage 2              contrast; null at median
```
