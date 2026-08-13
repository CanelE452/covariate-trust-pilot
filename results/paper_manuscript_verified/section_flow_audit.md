# Section flow audit — manuscript v1, read in order

A linear read of `manuscript_v1.md`, one entry per section. Severity uses
`CRITICAL / MAJOR / MINOR / STYLE / NO_ISSUE`. Fixes landed in `manuscript_v2.md`.

---

## Reader-question map

Where a first-time reviewer's thirteen questions are answered, and how early.

```
      question                                            answered in        timing
------------------------------------------------------------------------------------
R-Q1  what is the problem?                                Abstract; 1 P1     immediate
R-Q2  why are ADI / CV^2 not enough?                      1 P2; 2.2; 3.1     immediate
R-Q3  why compare direct against factorized?              1 P1; 3.2-3.3      immediate
R-Q4  is the comparison fair?                             3.4; 4.2; Table 1  early
R-Q5  what did Stage 1 show?                              4.6; Table 2       on time
R-Q6  why is Stage 2 needed?                              4.3 closing; 4.5;  on time
                                                          4.6 closing
R-Q7  how do the two axes differ?                         4.8; Fig 2B/2C;    on time
                                                          6.2
R-Q8  is there really a direct-favourable region?         4.8 closing;       on time
                                                          Fig 2A
R-Q9  does it transfer to real data?                      5.5; 5.6; Fig 3    on time
R-Q10 selector versus mechanism?                          5.6 / 5.7 split;   on time
                                                          6.3
R-Q11 can routing exploit it?                             5.9; 6.4           on time
R-Q12 why does this matter if classical wins?             5.4; 6.5 last;     LATE in v1
                                                          Abstract (v2)
R-Q13 what exactly is new, and what is not?               2.5; 6.1; 6.5      on time
```

**Only R-Q12 was answered late.** In v1 a reader met the absolute-accuracy result for the
first time in Section 5.4, after the entire synthetic study. Fixed in v2 by putting one
sentence in the Abstract (issue A4). 5.4 stays where it is, because the reader also needs
it immediately before the conditional claims.

---

## Section-by-section

### Abstract — MAJOR (2 issues)
Purpose: the whole paper in one paragraph. Strongest claim: the asymmetry plus the
transfer boundary.
`MAJOR` no mention that classical estimators outrank both arms → a competitiveness reading
was available to anyone reading only the abstract. **Fixed (A4).**
`CRITICAL` "about 20%" for a quantity written "−19.8%" in the Introduction and "−19.76" in
Results; the sign, which carries the direction, was dropped. **Fixed (A2).**
Transition out: strong.

### 1 Introduction — NO_ISSUE
Seven paragraphs, one role each; the concessions to prior work arrive in P2 before the gap
sentence. Redundancy with the Abstract is normal and not excessive. Strongest claim is
scoped ("within the eighteen-cell sweep"). No change in v2.

### 2.1–2.5 Related Work — NO_ISSUE
Each subsection concedes its own topic and narrows by one dimension; 2.5 positions without
an absence claim. `STYLE` 2.4 remains the longest at 324 words, which is correct — it holds
both representation precedents.

### 3.1–3.4 Problem Setup — MINOR
`MINOR` 3.2 is 76 words against 3.3's 179. The asymmetry is defensible (the direct arm is
one equation) but reads as if the paper cares less about it. Left as-is; lengthening 3.2
would pad.
3.4 does the heavy lifting — it is where "finite-sample" is defined and where `G` is fixed —
and it earns its 254 words.

### 4.1–4.5 Synthetic design — MINOR
`MINOR` 4.2 at 351 words is the longest design subsection. It is dense because the leakage
prohibitions (checkpoint rule, split continuity, oracle conditioning) all live there, and
those are the reviewer's first attack surface. Kept.
4.3's closing paragraph carries the Stage 1 → Stage 2 handoff and does it explicitly:
"this is a deliberate limitation of the design and it is what Stage 2 exists to relax".

### **Ordering — CRITICAL**
`CRITICAL` after 4.5 the v1 manuscript ran 5.1, 5.2, 5.3 and only then 4.6. **Fixed (A1).**
This is the defect that most damaged a linear read, and it was invisible to every
string-level gate because every individual section was correct.

### 4.6 Stage 1 results — NO_ISSUE
391 words. States the contrasts, then two boundaries in the same subsection: no
direct-favourable cell with an interval clear of zero, and the alternation-only limitation
with its `CONDITIONALLY_VALID` label. The control-integrity paragraph pre-empts "your
control is fake".

### 4.7 Component attribution — NO_ISSUE
119 words, correctly short, and labels itself as attribution rather than causation.

### 4.8 Stage 2 results — NO_ISSUE
The paper's centre, 439 words. The asymmetry, the Figure 2B/2C spread disclaimer, the
Stage 1 reconciliation and the single direct-favourable cell with its scope qualifier all
land here. Strongest section in the manuscript.

### 5.1–5.3 Empirical design — MINOR
`MINOR` 5.2 at 378 words is long for a design subsection, because each hypothesis carries
its construct boundary at definition time. That placement is deliberate — H3's mismatch is
declared before its result — and is what makes 5.8 read as a report rather than an excuse.

### 5.4 Overall comparison — MAJOR in v1, resolved
`MAJOR` in v1 this was the reader's first contact with the absolute-accuracy result, at
110 words, after ~5,000 words of synthetic material. Resolved by A4 (the Abstract now says
it); 5.4 itself is unchanged and correctly placed.

### 5.5–5.8 H1, H2, mechanism, H3 — NO_ISSUE
The 5.6 / 5.7 split is the manuscript's most important structural decision and it holds:
two subsections, two statuses, no merged sentence. 5.8 declares the construct mismatch and
does not argue with its own negative.

### 5.9 Routing — NO_ISSUE
196 words. Opportunity, failure, stop rule, and the UCI number at full scale. Proportionate.

### 6.1–6.4 Discussion — MINOR
`MINOR` 6.2 (276 words) offers a mechanism *account*; it is labelled as interpretation twice
and closes by noting the real-data occurrence head has no skill. Correct but it is the
paragraph most likely to be quoted out of context.
6.3 and 6.4 restate the boundaries without repeating the numbers. No redundancy with
Results beyond what a Discussion needs.

### 6.5 Limitations — MAJOR
`MAJOR` included the ALR12 full-text gap as a scientific limitation. It is a bibliographic
follow-up, not a limitation of the study. **Fixed (A3).** The other eight entries are
correct and complete against the required list.

### 6.6 What would move this forward — STYLE
80 words, partially overlapping 6.5's forward-looking sentences. Kept; flagged as a
foldable subsection (issue B3).

### 7 Conclusion — NO_ISSUE
277 words, no digits, no new claim. Ends on "a characterization and its boundary rather
than a recommendation", which is the correct last sentence for this paper.

### Figure and table captions — MINOR (2 issues)
`MINOR` the Figure 2 caption wrapped across a blockquote marker, breaking the phrase "not a
confidence interval" for any text extractor. **Fixed (A7).**
`MINOR` the table-caption file left a code fence open. **Fixed (A6).**
Otherwise the captions introduce no claim absent from the body, and Figure 2's caption
carries both the sign convention and the spread-is-not-a-CI warning.

---

## Redundancy scan

```
Abstract vs Introduction         normal overlap; the Abstract is not a précis of P4-P7
Introduction P4 vs 4.8           the Introduction previews, Results quantify.  No duplicated
                                 sentence.
2.5 vs 6.1                       both position the study; 2.5 does it against the
                                 literature, 6.1 against the evidence.  Different work.
5.7 vs 6.3                       6.3 adds the "take the weaker sentence" instruction and
                                 the entanglement account; not a restatement.
6.5 vs 6.6                       partial overlap - the only real redundancy found (B3).
```

## Excessive detail scan

```
4.2 leakage prohibitions         kept; first attack surface
5.2 construct boundaries         kept; declaring them late would read as excuse-making
4.6 control-integrity paragraph  kept; pre-empts a specific reviewer attack
routing chronology               already in the appendix; main text holds four facts
```

Nothing in the main text was found that belongs in the appendix and is not already there.
