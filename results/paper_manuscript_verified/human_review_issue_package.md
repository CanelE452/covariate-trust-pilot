# Human review issue package — manuscript v1 → v2

Read this first. It is the short version of `section_flow_audit.md`.

```
CRITICAL  2      scientific meaning, claim or source conflict
MAJOR     4      reader flow or scope moves substantially
MINOR     6      sentence, redundancy or transition
STYLE     3      readability only
```

Nothing found in this pass changed a scientific conclusion. Both CRITICAL items were
presentation defects that would have misled a reader, not errors in the evidence.

---

## A. Fixed in v2 — no decision needed

### A1 · CRITICAL · section order ran backwards
**Section.** Whole manuscript.
**What v1 did.** A reader met the sections in the order
`… 4.3, 4.4, 4.5, **5.1, 5.2, 5.3, 4.6, 4.7, 4.8**, 5.4 …`. Section 5's design subsections
appeared *before* Section 4's results.
**Why it happened.** The design/result split proposed in `methods_structure.md` was applied
at the level of whole files: all Methods came from one file and all Results from another, so
assembly could not interleave them.
**Problem.** A reviewer reading linearly is asked to accept the empirical protocol before
seeing the synthetic result that motivates it, then sent backwards. This is the single worst
reader-flow defect in v1.
**Action taken.** The assembler now splits `methods_v1.md` at Section 5 and `results_v1.md`
at 5.4, and emits strict numeric order: `3.1–3.4 → 4.1–4.5 → 4.6–4.8 → 5.1–5.3 → 5.4–5.9`.
**Scientific impact.** None. No sentence changed.
**Verified.** `V2-R order strictly increasing` PASS, zero backward jumps.

### A2 · CRITICAL · the same number appeared in three renderings
**Section.** Abstract, Introduction, 4.8.
**What v1 did.** The direct-favourable cell was "about 20%" in the Abstract, "about −19.8%"
in the Introduction and "−19.76" in Results.
**Problem.** A reviewer checking the abstract against the results table sees three numbers.
"20%" also drops the sign, which is the part that says *which* formulation wins.
**Action taken.** Abstract now reads "about −19.8%", matching the Introduction; Results keeps
the full precision. Both renderings map to the same row of `manuscript_v2_number_map.csv`.
**Scientific impact.** None; a rounding-presentation fix.

### A3 · MAJOR · ALR12 appeared as a scientific limitation
**Section.** 6.5 Limitations.
**What v1 did.** Listed "one of the closest prior studies could not be obtained in full text"
among the study's limitations.
**Problem.** It is not a limitation of this study. No empirical or synthetic result depends
on it; it is an internal literature-verification follow-up. Placing it in 6.5 invites a
reviewer to treat a bibliographic gap as a scientific weakness.
**Action taken.** Removed from 6.5. Recorded in `internal_submission_followups.md` as
`FU-1`, with `LIT-W3` still OPEN in the literature audit.
**Scientific impact.** None. The novelty boundary was already drawn with that study graded
conservatively, and that grading is unchanged.

### A4 · MAJOR · the abstract did not admit the absolute-accuracy result
**Section.** Abstract.
**What v1 did.** Reported the conditional findings without saying that both formulations are
outranked by classical estimators. Section 5.4 and 6.5 said so, but a reader who only sees
the abstract could infer a competitiveness claim.
**Action taken.** One sentence added: *"Both formulations are outranked by classical
estimators in absolute accuracy; the study isolates relative behaviour rather than
establishing forecasting accuracy."*
**Scientific impact.** Reduces what the abstract implies. Costs 20 words, paid for by trims
elsewhere.

### A5 · MINOR · abstract length
265 → **254 words**. The target was 240–250; 254 is four over and is the price of A4. Named
here rather than met by deleting the baseline sentence.

### A6 · MINOR · table-caption code fence was unterminated
`table_captions_v1.md` ended a fenced block without closing it, which would swallow following
content in any Markdown renderer. Closed.

### A7 · MINOR · a caption phrase was broken by blockquote wrapping
The Figure 2 caption wrapped as `… is not a confidence` / `> interval.`, so any tool
flattening the text saw `not a confidence > interval`. The caption was reflowed and the
audit's text extractor now strips blockquote markers before matching.

---

## B. Needs your decision

### B1 · MAJOR · keep the design/result split, or revert to the frozen outline?
**Context.** `final_outline_freeze.md` merges Stage 1 design and Stage 1 results into 4.3,
and the same for 4.5. `methods_structure.md` proposed splitting them so the Methods/Results
boundary can be audited mechanically, and v2 implements that.
**Trade-off.** The split makes "Methods contains no outcome" checkable and prevented one
mixed paragraph during drafting. The merged form reads more naturally in a
design-then-immediately-result journal style, and matches the frozen outline exactly.
**Recommendation.** Keep the split. It costs nothing structurally now that ordering is fixed,
and it is why `METH-P1 result leakage = 0` is a real check rather than a claim.
**If you disagree:** the prose recombines without editing a sentence.

### B2 · MAJOR · working title
**Recommendation.** `WORKING_TITLE`:
*When Does Factorized Forecasting Help for Intermittent Demand? Temporal Dependence,
Finite-Sample Behaviour, and Empirical Boundaries.*
**Why over T1.** T1 ("When Does Factorizing Intermittent Demand Help?") is ambiguous about
*what* is being factorized — a reader can parse it as factorizing the series. The variant
names the formulation, which is the paper's subject.
**Guards.** No SOTA, no "optimal", no "universal", no "first", no routing headline.
Five candidates and their trade-offs remain in `../paper_writing_verified/title_candidates.md`.

### B3 · MINOR · 6.6 "What would move this forward"
It is 80 words and overlaps 6.5's forward-looking sentences. It can stay as its own
subsection, or fold into the end of 6.5 to give Discussion five subsections instead of six.
**Recommendation.** Fold it, if the venue counts subsections. Left as-is in v2 because the
choice is cosmetic and reversible.

### B4 · MINOR · Section 5.4 is the shortest results subsection (110 words)
It carries the classical-baseline honesty, which is load-bearing for scope. It is short
enough to skim past between the synthetic results and H1.
**Options.** (i) leave it; (ii) move the classical ranks into Table 4 and cut 5.4 to two
sentences; (iii) promote one sentence of it into 5.1 so the scope caveat arrives with the
datasets.
**Recommendation.** (i). It is correctly placed — the reader needs absolute context before
the conditional claims — and lengthening it would over-weight a non-contribution.

---

## C. Blocked until a venue is chosen

```
C1  abstract word limit          254 now; venues range 150-300.  Trim targets are marked
                                 in abstract_v1_v2_diff.md.
C2  main-text page limit         9,818 words plus 3 figures and 4 tables.  If the venue is
                                 tight, 6.6 and 5.4 are the first candidates, then the
                                 appendix caption stubs.
C3  reference style + BibTeX     twelve verified records; no .bib generated yet.
C4  figure template              drafts exist at 3 sizes; dimensions and fonts not set.
C5  anonymization                no author block written; the recovery narrative in the
                                 artifacts names a machine and a user path.
C6  code / data availability     the synthetic source is recovered and hash-verified; the
                                 statement text is not written.
```

---

## D. Kept in the appendix, deliberately

```
D1  the full routing chronology (Gate-v1/v2/v3, anchor, shrinkage, HGB, every gate
    feature variant, every failure trace, the internal kill-test process)
    Main text keeps four facts only: complementarity exists; an oracle opportunity is
    measurable; learned routing did not transfer across domains or time; the search was
    stopped by a pre-registered rule.  Main-text routing is 421 words against 984 for C1
    and 625 for C2 - 20.7% of the three results blocks.
D2  FreshRetailNet-LT and UCI Online Retail II protocol, including AVAILABILITY_UNKNOWN
D3  Stage 1 per-cell values and the component-attribution columns
D4  Stage 2 per-cell values for all 18 cells
D5  classical baseline ranks in full
```

---

## E. Controversial results kept at full strength

```
E1  UCI Online Retail II at -193.9%.  In 5.9 and 6.4, at full scale, not a footnote.
E2  H3 non-replication with the wrong sign on both datasets, with both intervals.
E3  the overlap-adjusted association crossing zero, reported as a separate result from the
    selector's success rather than merged into it.
E4  the occurrence head having no skill advantage on real data (-0.0084, -0.0908), which
    directly limits the synthetic component-attribution story.
E5  both neural formulations outranked by SBA, Croston, TSB and SES.
```

None of these was softened in v2. Each is a place a reviewer will press, and each is
answered in `reviewer_attack_audit.md`.

---

## F. Optional robustness experiment

```
second backbone      NICE_TO_HAVE, not a submission blocker.
                     Decision rule and the exact permitted scope are in
                     submission_readiness.md.  Not run for v2.
scale axis in the    would settle the entanglement question of 6.3 rather than bound it.
synthetic design     Larger than a robustness check; a separate study.
per-observation      would test whether the synthetic attribution has any empirical
occurrence           counterpart.  Currently the mechanism story is synthetic-only and
diagnostic           says so.
H3 at ADI 3-5        the support exists (M5 127 vs 52 series) and was not used as a
vs >= 8              primary test.  Would convert a construct mismatch into a real test.
```

All four are `NICE_TO_HAVE`. `MUST_HAVE` remains **NONE**.
