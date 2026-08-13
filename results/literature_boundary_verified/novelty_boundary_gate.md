# Novelty boundary gate — NB1 … NB25

> **SUPERSEDED 2026-08-12 by `novelty_boundary_freeze.md` (NF1 … NF27).** Retained as the
> record of the first pass, which resolved the two `[CITATION NEEDED]` placeholders. Two
> of its gradings were later corrected: NB15/NB16 treated "marginals held fixed" as a
> novelty component (it is an experimental control), and the `[Kou13]` grading did not
> distinguish a Croston-style size/interval ratio from an occurrence-probability ×
> magnitude product. See `precedent_intersection_map.md` and
> `kou13_representation_verification.md`. Where the two files disagree, the NF gate wins.

Twenty-five checks. Any FAIL blocks the citation freeze.

---

## Reference integrity

```
NB1   every reference has a DOI or a stable identifier                        PASS  12/12
NB2   every peer-reviewed DOI resolves through Crossref                       PASS  11/11 OK
NB3   Crossref title matches the title as cited                               PASS
NB4   Crossref authors match the authors as cited                             PASS
NB5   volume / issue / pages match, or the mismatch is recorded               PASS  LIT-W2
NB6   no reference was written from memory                                    PASS
NB7   no author, year or venue was inferred                                   PASS
NB8   arXiv-only items labelled as preprints                                  PASS  [MC26], twice
NB9   no preprint is cited in the manuscript text                             PASS  [MC26] absent
NB10  container-name discrepancies recorded, not silently normalised          PASS  LIT-W1
```

## Collision handling

```
NB11  every located paper graded N0-N4                                        PASS  12/12
NB12  no paper graded N4                                                      PASS
NB13  every N3 paper read beyond its abstract, or the gap is declared         PASS  [Kou13]
                                                                                    full text;
                                                                                    [ALR12]
                                                                                    LIT-W3
NB14  N3 papers graded conservatively where evidence is incomplete            PASS  [ALR12]
                                                                                    K4 = "P"
NB15  each conceded component removed from the novelty claim                  PASS  K1, K2, K3
NB16  the surviving claim is a conjunction no located row satisfies           PASS  K3 ∧ K6
NB17  the nearest miss is named rather than omitted                           PASS  [TJWC21]
NB18  a review was checked for a prior statement that the question is
      settled                                                                 PASS  [GDTP25]
```

## Wording

```
NB19  "first" not used as a novelty claim                                     PASS
NB20  "no prior work" / "nobody has" / "has never been" absent                PASS
NB21  absence claims hedged with "we are not aware of"                        PASS  option A
NB22  the in-text gap states what prior work DID                              PASS  option B
NB23  rejected wordings recorded with the paper that refutes each             PASS  options D, E
```

## Scope discipline

```
NB24  no Related Work prose, Methods, Results or Discussion drafted           PASS
NB25  no new experiment, training run or TEST scoring; frozen artifacts
      unmodified; no commit, push or merge                                    PASS
```

---

```
PASS 25    FAIL 0    WARN 4   (LIT-W1 .. LIT-W4, see WARN_FAIL.md)
```

Status: **`INTRODUCTION_CITATIONS_RESOLVED`**. The v3 status
`INTRODUCTION_STRUCTURE_FROZEN_PENDING_CITATIONS` is discharged.

One follow-up is outstanding and is not a blocker: obtain the [ALR12] full text and
re-check K4 (LIT-W3).
