# Warnings and failures — literature boundary audit

**Revised 2026-08-12 (second build).** One new guard added (`LIT-W-NAR26`), one warning
reopened and re-closed (`LIT-W4`), and `LIT-W3`'s status vocabulary clarified: it is a
**submission follow-up**, not a blocker.

---

## LIT-W3 — [ALR12] full text not obtained — **OPEN / SUBMISSION_FOLLOWUP_DESIRABLE**

Not a blocker for Related Work drafting. It blocks exactly one thing: any sentence that
states what [ALR12] holds constant.

Escalated from "not read" to "attempted and failed". Six independent routes:

```
ScienceDirect article page                  HTTP 403
academia.edu hosted copy                    HTTP 403
works.bepress.com/nezih_altay/5             repository decommissioned; serves a
                                            Helpjuice notice page, not a PDF
Semantic Scholar openAccessPdf              points at the same dead URL; the
                                            isOpenAccess=true flag is stale
Unpaywall                                   is_oa = false
OpenAlex                                    oa_status = "closed"
CORE API                                    redirect wall, no record
European Scientific Journal follow-up       HTTP 403
```

**Unknown:** whether ALR12 holds ADI, CV², zero proportion, mean demand, the size
distribution or the interval distribution constant across correlation levels (ALR-B), and
whether "correlation alone" is an explicit isolation goal in its methodology (ALR-C).

**Handling — CORRECTED in this build.** The previous build recorded
`marginal_characteristics_controlled = U` in the matrix but graded novelty component 3 as
`PRIOR` in the component map, calling that "grading against ourselves". The reasoning was
right and the encoding was wrong: recording UNKNOWN as PRIOR manufactures a precedent
that was never verified, and it made two files disagree about one fact (EC1).

Now:

```
evidence_status   UNKNOWN     (component 3, unchanged from the matrix)
novelty_policy    EXCLUDED_FROM_NOVELTY
```

Two fields, not one. See `evidence_policy_separation.md`.

**Why it is not a blocker.** The fixed-marginal novelty claim is excluded by a policy
that does not depend on ALR12 at all — holding the marginals fixed is an experimental
control, and the project's frozen sources already treat it as one
(`claim_ledger_frozen.md` reader chain 1-2; `final_outline_freeze.md` "no novelty claim
over the correlation literature"). Whichever way LIT-W3 resolves, **no claim we make gets
weaker.**

**Prohibition while open:** no sentence anywhere in the manuscript may describe what
ALR12 holds constant. Enforced in `related_work_outline.md` 2.3.

**Action:** obtain the published PDF by institutional subscription or ILL; close with page
and table numbers for ALR-B and ALR-C.

---

## LIT-W-KOU13 — terminology guard *(new)* — **ACTIVE, permanent**

```
fact      Kourentzes' NN-Dual forecasts non-zero demand size z' and inter-demand
          interval x' from one network with two linear outputs, then computes z'/x'
          (Croston's division) and removes the resulting inversion bias with a
          coefficient c fitted from z_t/x_t = c Y_t.
fact      this paper's factorized arm computes P(Y>0|h) * E[Y|Y>0,h] -- a PRODUCT of a
          probability and a conditional mean, period-indexed, with no inversion.
risk      calling NN-Dual "decomposed" and stopping there lets a reader equate it with
          our hurdle, which would concede the paper rather than a precedent.
risk      the opposite error is equally bad: overstating the difference would hide a
          real precedent.
guard     write "a Croston-style network that forecasts size and interval and divides
          them".  Never "hurdle".  Never "the same two representations".
scope     permanent; applies to Introduction, Related Work, Discussion and any rebuttal.
```

Source: `kou13_representation_verification.md` (full text verified).

---

## LIT-W5 — fixed-marginal novelty claim withdrawn — **RESOLVED at source**

```
was       earlier drafts treated "marginals held fixed while dependence varies" as a
          novelty component (K4).
now       evidence_status = UNKNOWN (component 3, unchanged from the matrix)
          novelty_policy  = EXCLUDED_FROM_NOVELTY
why       it is a design property, not a finding, and the frozen sources already say so:
          claim_ledger_frozen.md reader chain 1-2; final_outline_freeze.md "no novelty
          claim over the correlation literature".
          This does NOT rest on what [ALR12] does -- LIT-C5 is UNRESOLVED.
check     "fixed marginals" appears in no LEVEL A/B/C wording and in no v6 sentence as
          a claim; NBF16 checks it mechanically.
```

---

## DERIVATIVE-ARTIFACT ERROR POLICY — frozen 2026-08-12

Generalized from `LIT-W6`. Three kinds of inconsistency are handled differently, and the
distinction is what decides whether work stops.

### TYPE A — primary-literature / source contradiction

```
example    a paper's own text disagrees with what the verified audit records about it
           (e.g. full text shows [ALR12] compares representations after all)
action     STOP.  Do not adjust the prose to fit, and do not adjust the audit to fit the
           prose.  Report the contradiction with the source location.
status     RELATED_WORK_REVEALED_LITERATURE_MISMATCH
rationale  the audit's authority rests on the primary source.  If they disagree, every
           downstream artifact is suspect until the source is re-read.
```

### TYPE B — verified audit ↔ derived metadata inconsistency

```
example    literature_evidence_matrix.csv records [NAR26] as N3 while
           reference_metadata.csv still carries N2  (this is LIT-W6 exactly)
action     1. record the defect in WARN_FAIL.md, with cause and both values
           2. correct the DERIVATIVE to match the verified audit source, never the
              reverse
           3. re-run verify_consistency.py
           4. confirm 0 violations, and add a check that makes the class detectable
              (LIT-W6 added EC6)
status     no stop required; the work continues in the same session
rationale  the fact was already verified once.  What failed is propagation, not
           evidence.  Silence is still forbidden -- the record must show the correction.
guard      the direction is fixed: verified audit source wins, derivative is rewritten.
           A derivative may never be used to justify changing the audit.
```

### TYPE C — prose wording issue

```
example    a sentence claims more than its source supports; defensive or declarative
           tone; ambiguous terminology (e.g. "decomposed" where "ratio" is meant)
action     revise the prose, then re-run the sentence-level claim audit and the
           reference map.  No change to the evidence layer.
status     no stop; the revision is recorded in the version diff
rationale  the evidence is intact; only its expression was wrong.
```

### Which files are authoritative

```
authoritative  the primary sources, and the verification files written directly from
               them: alr12_fulltext_verification.md, kou13_representation_verification.md,
               nar26_matched_comparison_verification.md, literature_evidence_matrix.*,
               novelty_component_map.csv, precedent_intersection_map.md
derivative     core_reference_list.md, reference_metadata.csv, related_work_*.md,
               introduction_*.md, section summaries, reference maps
```

A `grade`, a status label or a scope description that appears in both layers is a
duplication hazard. Where duplication cannot be avoided, a mechanical check must tie the
copies together — that is why `EC6` exists.

---

## LIT-W6 — two stale derived fields, found while drafting Related Work — **CORRECTED, recorded**

Found on 2026-08-12 by a cross-file grade check run before any prose was written. Both are
bookkeeping duplicates in derived files that were not propagated when [NAR26] was regraded;
neither is a disagreement with a literature source, and neither changed any prose.

```
defect 1   reference_metadata.csv  NAR26 grade = N2
           literature_evidence_matrix.csv and novelty_collision_audit.md = N3
           cause   the N2 -> N3 upgrade was applied to the matrix and the collision audit
                   but not to the metadata table, which carries a duplicate grade column.
           fix     reference_metadata.csv NAR26 grade N2 -> N3.  Recorded here, not
                   silently applied.
defect 2   core_reference_list.md described [NAR26] as a comparison "at equal features and
           equal model family".
           cause   stale wording from the first build.  NAR-D is PARTIAL: both arms are
                   LightGBM but the second stage uses a Tweedie objective, and the article
                   does not state that the family is uniformly identical.
           fix     rewritten to "at an identical feature set ... Both arms are LightGBM;
                   the article does NOT state a capacity, training-budget or
                   per-formulation tuning match (LIT-W-NAR26)".
```

**Why the grade duplication existed at all.** `reference_metadata.csv` is a bibliographic
table that also carried a `grade` column, so the same fact lived in two files with no
check tying them together. `EC6` was added to `verify_consistency.py` to make grade
disagreement across files a hard failure.

**Scope check.** Neither file is in the preservation list for this step
(`introduction_v1..v6`, `contributions_v1..v3`, the three frozen scientific files,
`novelty_boundary_freeze.md`, `literature_evidence_matrix.*`, `novelty_component_map.csv`,
`precedent_intersection_map.md`). Those eleven were verified untouched: the six preserved
writing files by sha256, the rest by modification time.

**Effect on Related Work v1.** None. The prose is driven by the evidence matrix and the
component map, both of which already carried N3 and the correct NAR26 flags. The
correction removes a contradiction in the record rather than changing a conclusion.

---

## LIT-W1 — Croston (1972) container name — **OPEN, cosmetic**

Published in *Operational Research Quarterly* 23(3), 1972; Crossref and both publishers
index it under the post-1978 name *Journal of the Operational Research Society*.
`core_reference_list.md` prints the historical name; `reference_metadata.csv` keeps the
Crossref string. DOI, volume, issue and pages agree across all three sources.

## LIT-W2 — [GDTP25] no volume or pages — **OPEN, cosmetic**

Published online 31 Oct 2025; no print issue assigned. Re-check before submission.

## LIT-W-NAR26 — matched-comparison guard *(new)* — **ACTIVE, permanent**

```
fact      [NAR26] states a match on FEATURES and on the data and evaluation pipeline:
          "identical data preprocessing, feature construction, and evaluation protocols".
fact      it does NOT state a match on capacity, on training budget, or on
          per-formulation tuning (NAR-E, NAR-F, NAR-G).
fact      it is LightGBM / XGBoost / RF / Ridge / ElasticNet -- not neural.
risk      reading a feature match as a capacity match records a paper as "matched" that
          never claimed it, and lets us claim a matched-comparison first on an evidence
          gap.
risk      the opposite error is flagging a non-neural model as neural, which the previous
          matrix did (direct_neural_prediction = Y against its own note).
guard     "same model family" and "same feature set" are DATA and MODEL-CLASS statements.
          Matching is recorded ONLY where the source states it.
guard     component 6b carries novelty_policy = CLAIM_ONLY_IN_CONJUNCTION.
scope     permanent; applies to every future matrix row.
```

Source: `nar26_matched_comparison_verification.md` (full text verified).

---

## LIT-W4 — Introduction P2 length — **REOPENED, then CLOSED at 167**

```
v4   209 words   flagged; the concessions were worth the length but the paragraph was
                 the longest in the Introduction
v5   165 words   closed, inside the 130-165 target
v6   167 words   REOPENED by adding the [NAR26] concession, then closed at 167
```

The band was widened to <= 170 for this build, and the two extra words are stated rather
than shaved: a fourth precedent citation was added because component 6a is PRIOR on
verified full text, and dropping it to hit a self-imposed number would have been the
wrong trade. Three sentences were tightened to absorb most of the cost, and [WSS04] was
moved out of P2 into Related Work 2.3. Closed.

---

## Tool-level obstructions

```
OBS-1  nature.com redirected to an identity provider (HTTP 303).  Resolved via the PMC
       mirror PMC12873174, which was re-read in full for NAR-A .. NAR-H.
OBS-2  ScienceDirect HTTP 403 on three articles.  Resolved for [WSS04] and [Kou13] via
       publisher-independent records and, for [Kou13], the author-deposited PDF.
       NOT resolved for [ALR12] -- LIT-W3.
OBS-3  the [Kou13] PDF was unreadable by the fetch tool (binary stream).  Resolved by
       downloading and extracting locally with pypdf: 28 pages, 60,858 characters.
```

---

## Coverage limits — declared, not fixed

```
G1  paywalled full texts verified at abstract + metadata level, EXCEPT [Kou13]
    (full text) and [NAR26] / [TJWC21] (publisher/PMC full text).
G2  no Scopus / Web of Science sweep; recall unquantified.
G3  2025-2026 preprints thinly indexed; a same-idea preprint cannot be excluded.
G4  [ALR12]'s "no representation contrast" rests on title and abstract at high
    confidence, not on methodology.  Labelled at every point of use.
```

These are why every absence claim is hedged with "we are not aware of", and why no
"first" or "first to combine" claim is made.

---

## Prohibitions — all observed in this revision

```
citation invented                                    0
author / year / venue guessed                        0
arXiv-only item presented as peer-reviewed           0    [MC26] labelled; never cited
                                                          in manuscript text
collision verdict on an abstract alone               0    [Kou13] full text; [ALR12]
                                                          graded against us
unverified evidence graded in our favour             0    see below
unverified evidence graded AGAINST the record        0    ALR12 marg_ctrl stays U;
                                                          NAR26 match fields U, not P/Y
evidence status conflated with claim policy          0    two fields, EC5 enforced
"first" / "first to combine" used                    0
new literature sweep restarted                       0    targeted ALR12/Kou13 only
Related Work prose drafted                           0    headings + roles only
Methods / Results / Discussion drafted               0
new experiment, training run, TEST scoring           0
frozen scientific artifacts modified                 0    claim_ledger_frozen,
                                                          final_outline_freeze,
                                                          final_rendering_freeze
introduction_v1..v4, contributions_v1..v3 modified   0
commit / push / merge                                0
```
