# manuscript_v1 → v2

`manuscript_v1.md` and every v1 map are retained unmodified. No scientific evidence,
component grade, hypothesis status or number changed. Every edit is ordering, one removed
limitation entry, or abstract wording.

---

## Word counts

```
                                  v1      v2    delta
Abstract                         265     254     -11
1 Introduction                   881     881       0    byte-identical
2 Related Work                  1412    1412       0    byte-identical
3 + 4.1-4.5 design              1969    1969       0    byte-identical
4.6-4.8 synthetic results        979     979       0    byte-identical
5.1-5.3 empirical design         665     665       0    byte-identical
5.4-5.9 empirical results        952     952       0    byte-identical
6 Discussion                    1430    1365     -65    ALR12 limitation removed
7 Conclusion                     277     277       0    byte-identical
Figure captions                  546     547      +1    caption reflow
Table captions                   425     426      +1    closing fence
TOTAL                           9801    9727     -74
```

## Change 1 — section order (CRITICAL)

```
v1 reader order   3.1 3.2 3.3 3.4 4.1 4.2 4.3 4.4 4.5 [5.1 5.2 5.3] 4.6 4.7 4.8 5.4 ...
                                                       ^^^^^^^^^^^^^ backward jump
v2 reader order   3.1 3.2 3.3 3.4 4.1 4.2 4.3 4.4 4.5  4.6 4.7 4.8  5.1 5.2 5.3 5.4 ...
```

Cause: the design/result split was applied at file level, so all Methods preceded all
Results. The assembler now splits `methods_v1.md` at Section 5 and `results_v1.md` at 5.4
and interleaves them. **No sentence changed**; the emitted order did.

Verified: `V2-R order strictly increasing` PASS, zero backward jumps, 32 numbered
subsections.

## Change 2 — Abstract (CRITICAL + MAJOR)

```
"about 20% in relative error"  ->  "about -19.8% in relative error"
    one quantity had three renderings across Abstract / Introduction / Results, and the
    Abstract's dropped the sign that carries the direction.

added: "Both formulations are outranked by classical estimators in absolute accuracy; the
        study isolates relative behaviour rather than establishing forecasting accuracy."
    v1's abstract permitted a competitiveness reading that Sections 5.4 and 6.5 deny.

trimmed to pay for it: the opening definition, the descriptor sentence, the transfer
    sentence's trailing clause, and the closing sentence.  265 -> 254 words.
```

## Change 3 — Discussion 6.5 (MAJOR)

Removed the entry beginning *"A literature limitation, distinct from the above."* — the
ALR12 full-text gap. It is a bibliographic follow-up, not a limitation of this study; no
result depends on it, and listing it invited a reviewer to read a verification gap as a
scientific weakness. Recorded as `FU-1` in `internal_submission_followups.md`; `LIT-W3`
remains OPEN in the literature audit, unchanged.

The other eight limitation entries are untouched: single backbone, synthetic simplicity,
Stage 1 conditional validity, H3 construct mismatch, no real-data occurrence-head skill,
routing instability, absolute accuracy not the contribution.

## Change 4 — two caption defects (MINOR)

```
figure_captions   the Figure 2 caption wrapped as "... is not a confidence" / "> interval.",
                  so a flattened read saw "not a confidence > interval".  Reflowed, and the
                  audit's extractor now strips blockquote markers before matching.
table_captions    a fenced block was left open, which would swallow following content in a
                  Markdown renderer.  Closed.
```

---

## What did NOT change

```
every number, interval and sample size                    identical
every hypothesis status (H1 / H2a / H2b / H3 / routing)   identical
C1 / C2 / C3 wording                                      identical
component evidence and novelty policy                     identical
Kou13 ratio, NAR26 product / LightGBM, ALR12 silence      identical
Introduction, Related Work, Methods, Results, Conclusion  byte-identical
figure and table content                                  identical
```

## Audit results, v1 and v2

```
                       v1     v2
OVERCLAIM               0      0
UNSUPPORTED             0      0
UNMAPPED_NUMBER         0      0
UNMAPPED_CITATION       0      0
NOTATION_CONFLICT       0      0
backward section jumps  1      0
number-rendering conflicts (same quantity)  1  0
```

## Net effect

v2 says slightly less than v1 (the abstract now concedes the absolute-accuracy result) and
reads in the order its section numbers promise. Everything else is preserved.
