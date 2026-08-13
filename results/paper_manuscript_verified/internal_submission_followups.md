# Internal submission follow-ups

Tasks that are **not** scientific limitations and must not appear in the manuscript's
Limitations section. Each is bookkeeping, verification hygiene or venue logistics.

---

## FU-1 · ALR12 full text — OPEN

```
what        Altay, Litteral & Rudisill (2012), IJPE 135(1) 275-283.  Six access routes
            failed: ScienceDirect 403, academia.edu 403, works.bepress decommissioned,
            Semantic Scholar openAccessPdf points at the dead URL, Unpaywall is_oa=false,
            OpenAlex closed, CORE redirect wall.
unknown     which marginal characteristics that study holds constant across its
            correlation levels (ALR-B), and whether correlation isolation is an explicit
            methodological goal (ALR-C).
why it is   no empirical or synthetic result in this paper depends on it.  The novelty
NOT a       boundary was drawn with the corresponding component graded PRIOR - i.e.
scientific  against us - so the resolution cannot weaken any claim we make.
limitation
manuscript  ZERO.  No sentence states what that study holds constant.  Checked
dependency  mechanically (L1 / RWF11).
action      obtain via institutional subscription or interlibrary loan; close LIT-W3 with
            page and table numbers.
removed     Discussion 6.5, in v2.  It was listed there in v1 and should not have been.
from
```

## FU-2 · Bibliography

```
twelve verified records in core_reference_list.md, all DOI-resolved through Crossref
except the one arXiv preprint, which is labelled.
no .bib generated.  Two metadata items to re-check at typesetting:
  LIT-W1  Croston (1972) - published in Operational Research Quarterly; Crossref and both
          publishers index it under the post-1978 name.  Decide which to print.
  LIT-W2  Giannopoulos et al. (2025) - online first, no volume or pages assigned.
```

## FU-3 · Venue-dependent formatting

```
abstract limit, page limit, reference style, figure dimensions and fonts, anonymization,
supplementary packaging, code and data availability statement.
None can be settled before a venue is chosen.
```

## FU-4 · Artifact hygiene

```
the recovery narrative in the verification artifacts names a machine, a username and a
local path.  The manuscript does not, but the supplementary material would if the
artifacts were attached unedited.  Scrub before packaging.
```

## FU-5 · Git

```
no commit, no push, no merge has been performed at any point in the drafting or auditing
of this manuscript.  To be done only on explicit user instruction.
```

---

## Explicitly not on this list

Second-backbone robustness is a **scientific** option, not an operational follow-up; its
decision rule lives in `submission_readiness.md`. Confusing the two is what this file
exists to prevent.
