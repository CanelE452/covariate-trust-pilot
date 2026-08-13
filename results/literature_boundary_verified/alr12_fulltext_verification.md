# ALR12 — full-text verification attempt

Altay, N., Litteral, L.A. & Rudisill, F. (2012). Effects of correlation on intermittent
demand forecasting and stock control. *International Journal of Production Economics*
135(1), 275-283. doi:10.1016/j.ijpe.2011.08.002

**Status: FULL-TEXT VERIFICATION FAILED. `LIT-W3` remains OPEN.**

Metadata is verified (Crossref). Methodology is **not** verified. Nothing in this file
infers what the paper's simulation design fixes.

---

## Access routes attempted, and how each failed

```
ScienceDirect, article page                         HTTP 403
academia.edu hosted copy                            HTTP 403
works.bepress.com/nezih_altay/5                     repository decommissioned; the
                                                    URL now serves a Helpjuice notice
                                                    page (61 KB of HTML, not a PDF)
Semantic Scholar openAccessPdf                      points at the same dead bepress URL;
                                                    isOpenAccess=true is stale
Unpaywall API                                       is_oa = false
OpenAlex                                            oa_status = "closed";
                                                    any_repository_has_fulltext = false
CORE API                                            redirect wall, no record returned
European Scientific Journal follow-up study         HTTP 403
citing-work descriptions                            only restate the abstract's findings
```

Six independent routes, no full text. This is a hard access failure, not an
insufficiently persistent search.

---

## ALR-A … ALR-G — what is and is not established

```
ALR-A  dependence types manipulated
       ESTABLISHED (abstract + record).  Three: demand-size autocorrelation,
       inter-demand-interval autocorrelation, and size-interval cross-correlation.

ALR-B  what marginal / intermittency characteristics are held fixed
       *** NOT ESTABLISHED. ***
       Known only that demand is generated from a compound Poisson process and that
       intermittency appears as a separate factor ("higher intermittency intensifies
       these service-level changes").  Whether ADI, CV^2, zero proportion, mean demand,
       the size distribution or the interval distribution are individually held
       constant across correlation levels is UNKNOWN.
       No inference is recorded here.  A compound Poisson construction makes marginal
       control plausible, but plausible is not verified, and this audit does not grade
       on plausibility.

ALR-C  is "correlation alone" an explicit isolation goal in the methodology?
       *** NOT ESTABLISHED. ***  The title and abstract are consistent with it.  The
       methodology was not read.

ALR-D  forecasting methods compared
       ESTABLISHED at family level: Croston-family estimators, selected via the
       ADI / CV^2 categorization (Croston for smoother series, SBA for the
       intermittent / erratic / lumpy regions).  The exact method list is UNKNOWN.

ALR-E  representation-level direct vs factorized comparison
       ESTABLISHED AS ABSENT, at the level the abstract and title permit.  The compared
       objects are estimators inside an already-factorized (Croston) representation.
       There is no direct conditional-mean arm to contrast against.
       Confidence: high.  A representation contrast would be the paper's headline and
       would appear in its title and abstract; neither mentions one.

ALR-F  neural, matched-capacity comparison
       ESTABLISHED AS ABSENT.  Confidence: high, same reasoning; a 2012 IJPE paper
       introducing neural models would say so.

ALR-G  stock-control outcomes
       ESTABLISHED.  Service level and cost; negative autocorrelation gives higher
       service levels than positive, with cost not significantly changed; cross-
       correlation acts in the opposite direction to autocorrelation.
```

---

## How this is handled in the audit

Two decisions, both taken so that an unverified fact cannot help us.

**1. The matrix grades ALR12 in our disfavour where evidence is missing.**
`marginal_characteristics_controlled` is recorded as `UNKNOWN`, and every downstream
statement treats `UNKNOWN` as if it were `PRIOR`. Concretely: the novelty component
"controlled marginal characteristics while varying dependence" is graded **PRIOR**, not
`NOT_FOUND_IN_AUDIT`, even though we could not confirm it.

**2. The fixed-marginal novelty claim is dropped outright, independently of ALR12.**
This does not rest on ALR12 at all. Holding the marginals fixed is an **experimental
control** in this paper's design, not a finding, and the project's own frozen sources
already treat it that way:

```
claim_ledger_frozen.md, reader chain steps 1-2
    ADI and CV^2 summarize the marginal distribution and discard temporal ordering;
    WITH THE MARGINALS HELD FIXED, the finite-sample behaviour still diverges.
    -- the control is the premise of step 2, not its result.

final_outline_freeze.md, wording decisions
    "related work: no novelty claim over the correlation literature"
    -- already frozen before this audit ran.
```

So the removal is a claim reduction consistent with the frozen source, and it would be
correct even if ALR12 turned out never to control anything.

---

## What would change if the full text were obtained

```
if ALR12 DOES hold marginals fixed while varying correlation
    matrix field moves UNKNOWN -> Y.  Nothing else changes: the component was already
    graded PRIOR, and the novelty claim was already dropped.  LIT-W3 closes.

if ALR12 does NOT hold marginals fixed
    matrix field moves UNKNOWN -> N, and one additional distinction becomes available
    to us.  We would still not claim it, because the outline freeze forbids a novelty
    claim over the correlation literature.  LIT-W3 closes.
```

Either way the current wording stands. That is the point of grading it against
ourselves: **the outcome of LIT-W3 cannot weaken any claim we are making.** It is
therefore a warning, not a blocker.

---

## Action

Obtain the published PDF through an institutional subscription or interlibrary loan
before submission, and close LIT-W3 with the page and table numbers for ALR-B and
ALR-C. Until then, no sentence anywhere in the manuscript may describe what ALR12 holds
constant.
