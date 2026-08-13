# NAR26 — matched-comparison verification

Nathan, B.S., Aravinth, P.M., Reddy, B.V.S., Sastry, C.C., Salunkhe, S. & Cep, R. (2026).
Primacy of feature engineering over architectural complexity for intermittent demand
forecasting. *Scientific Reports* 16, 4792. doi:10.1038/s41598-026-35197-y

Source re-read: PMC full text, PMC12873174. No new literature search was run.
**Status: FULL-TEXT VERIFIED, with three items the article does not state.**

This file exists because the previous matrix recorded values for NAR26 that the article
does not support. Three defects are corrected below.

---

## NAR-A … NAR-H

```
NAR-A  direct / single-stage arm                                          VERIFIED
       "a LightGBM regressor is trained directly on the full feature set" to
       "produce a direct estimate of expected demand for each time period."
       -> a direct conditional-mean arm.  NOT NEURAL.

NAR-B  two-stage / hurdle arm                                             VERIFIED
       "Stage 1 (occurrence model): a LightGBM classifier estimates the probability of
       observing non-zero demand"; "Stage 2 (conditional size model): a LightGBM
       regressor with a Tweedie objective predicts the expected demand quantity."
       -> this IS the P(Y>0) x E[Y | Y>0] product form.  NOT NEURAL.

NAR-C  same feature set                                                   VERIFIED  Y
       "both frameworks use the raw monthly demand quantity (PartQty) as the prediction
       target and operate under identical data preprocessing, feature construction, and
       evaluation protocols."  Both stages receive the full base feature set augmented
       with SHOS.

NAR-D  same model family                                                  PARTIAL  P
       both are LightGBM, but stage 2 uses a Tweedie objective and stage 1 is a
       classifier.  The article does not state that the family is uniformly identical.

NAR-E  matched parameter count / capacity                                 *** NOT STATED ***
       no statement that tree counts, depths or capacity were matched between the
       single-stage and two-stage formulations.

NAR-F  matched training budget / optimizer / early stopping               *** NOT STATED ***
       no discussion of matched training budgets or checkpoint protocols between
       formulations.

NAR-G  hyperparameters tuned separately per formulation                   *** NOT STATED ***
       "All models are trained using pre-tuned hyperparameters obtained from the initial
       tuning stage."  Whether tuning differed by formulation is not stated.

NAR-H  temporal dependence as an experimental factor                      VERIFIED  N
       autocorrelation of intervals or of demand sizes is neither manipulated nor
       reported as a breakdown variable.  Results are dataset-wide aggregates.
```

---

## Three corrections to the previous matrix row

```
field                              was    now   why
--------------------------------------------------------------------------------------
direct_neural_prediction           Y      N     NAR26 is LightGBM / XGBoost / RF /
                                                Ridge / ElasticNet.  The row's own note
                                                already said "not neural"; the flag
                                                contradicted it.  EC2-class defect.
matched_parameter_budget           P      U     NAR-E is NOT STATED.  "P" implied a
                                                partial match the article never claims.
matched_training_protocol          Y      U     NAR-F is NOT STATED.  "Y" was inferred
                                                from "identical preprocessing and
                                                evaluation protocols", which is a
                                                DATA-pipeline statement, not a TRAINING
                                                statement.  EC3-class defect.
```

Two fields added so the distinction cannot collapse again:

```
neural_model            N     explicit, so "direct arm" and "neural" are never conflated
matched_feature_set     Y     NAR-C is verified and is genuinely matched
```

`occurrence_probability_magnitude_hurdle` stays **Y**: the product form is verified
(NAR-B). It is the *form* that matches ours, not the learner.

---

## Consequence — component 6 splits

```
6a  a direct / single-stage arm compared against a hurdle / two-stage
    occurrence-probability x magnitude arm, at an identical feature set
      evidence_status  PRIOR
      closest          [NAR26]  -- verified at NAR-A, NAR-B, NAR-C
      note             this is a real precedent and is conceded without qualification.
                       It is NOT weakened by NAR26 being non-neural: the comparison of
                       the two FORMS has been done.

6b  the same comparison under MATCHED CAPACITY and MATCHED TRAINING, with the
    representation isolated as the only varied factor
      evidence_status  NOT_FOUND_IN_AUDIT
      nearest          [NAR26], whose matching status is UNKNOWN (NAR-E, NAR-F, NAR-G
                       all NOT STATED) -- not verified absent, merely unstated
      novelty_policy   CLAIM_ONLY_IN_CONJUNCTION
      note             because the nearest paper's match status is UNKNOWN rather than
                       known-absent, 6b MUST NOT be claimed standalone.  It may appear
                       only as one condition inside the intersection.
```

This is the same discipline applied to `[ALR12]` component 3, in the other direction:
an unstated fact is recorded as unstated, and the claim policy is set so that the gap
cannot be spent in our favour.

---

## Guard — same family is not matched capacity

```
GUARD   "same model family" and "same feature set" are DATA and MODEL-CLASS statements.
        They do not establish matched parameter count, matched training budget, or
        equal tuning effort.
        A paper may only be recorded as matched when it STATES the match.
scope   permanent; applies to any future addition to the matrix.
```

Registered as **LIT-W-NAR26** in `WARN_FAIL.md`.

---

## Collision grade

Re-assessed from N2 to **N3**. NAR26 overlaps two components, not one: 6a fully, and the
occurrence-probability form (4b) in its learner. Calling it N2 understated a genuine
direct-versus-hurdle precedent.

It is **not N4**: NAR-H is verified negative — no dependence factor, no regime or
dependence breakdown, real data only, aggregates only. It cannot have performed the
intersection.

---

## What NAR26 changes in the manuscript

```
Introduction P2   must acknowledge the hurdle / two-stage precedent, otherwise the gap
                  sentence is reachable by "but NAR26 compares exactly those two forms".
                  -> introduction_v6.md
Related Work 2.4  NAR26 promoted to required content, described by NAR-A/NAR-B/NAR-C and
                  by what is NOT STATED (NAR-E/F/G).
novelty wording   6b may never carry a claim on its own.
P1                NAR26's headline -- two-stage hurdle models do not outperform the
                  single-stage model -- is independent third-party support for P1's
                  "neither formulation is uniformly superior in aggregate".  Citation
                  still not added; P1 is frozen and rests on our artifacts.
```
