# related_work_v1 → v2

`related_work_v1.md`, `related_work_claim_audit.md`, `related_work_reference_map.csv` and
`related_work_section_summary.md` are retained unmodified.

**Style-only revision.** The verified literature boundary, the citation set and the
novelty intersection are unchanged. No evidence status moved; no component grade moved;
no citation was removed from the section.

---

## Word counts

```
        v1     v2    delta   change
2.1    247    247       0    byte-identical
2.2    234    234       0    byte-identical
2.3    276    276       0    byte-identical
2.4    387    333     -54    compressed
2.5    262    265      +3    two verb/wording edits
TOTAL 1406   1355     -51
```

Sentences: 52 → 50. Citation occurrences in prose: 21 → 20, over the same 11 keys
(12 including the footnote's `[MC26]`).

---

## 2.1 / 2.2 / 2.3 — unchanged

Verified byte-identical by direct string comparison, not by inspection. The decomposition
concession, the ADI/CV² treatment and the temporal-dependence concession all stand as
written in v1.

---

## 2.4 — compressed 387 → 333 words

### What the compression removed

```
1  the inversion-bias correction in [Kou13]'s decomposed arm
   v1  "...combined as a ratio in the manner of Croston's method and then corrected for
        the resulting inversion bias [Kou13]."
   v2  "...combines them as a ratio, in the manner of Croston's method [Kou13]."
   why  redundant.  Section 2.1 already states that the ratio introduces an inversion
        bias and that SBA supplies the correction.  Repeating it in 2.4 added length
        without adding a distinction.
   kept in  related_work_v2_reference_map.csv (Kou13 notes) and
            kou13_representation_verification.md

2  the hyperparameter grid detail
   v1  "each at its own best configuration of input lags and hidden nodes, selected on
        in-sample error rank"
   v2  "each at its own selected configuration"
   why  the load-bearing fact is that each arm is reported at its OWN configuration.
        The grid range and the selection criterion are audit detail.
   kept in  the same two files

3  [Kou13]'s inventory-metric conclusion
   v1  "...and favours the directly predicted rate once service levels are considered."
   v2  removed; the sentence now ends at the accuracy-versus-inventory divergence.
   why  the paper's own preferred method is not load-bearing for this section, which is
        about which comparisons exist.
   kept in  kou13_representation_verification.md

4  one redundant setting marker in the closing paragraph
   "in a tree-ensemble setting on real data [NAR26]" -> "on real data [NAR26]"
   why  the gradient-boosting / non-neural marker is already stated at the start of the
        [NAR26] paragraph, where it does the guard work.
```

### What the compression added — one bounding clause

```
v2  "...both under identical data preprocessing, feature construction and evaluation
     protocols; capacity and training budget across the two arms are not reported
     [NAR26]."
```

**Why added rather than removed.** The instruction for this revision assumed that the
three `NOT STATED` items (capacity, training budget, per-formulation tuning) were
enumerated in v1's prose and needed compressing into one sentence. **They were not in v1
at all** — v1 stated only what [NAR26] does match. That left a residual risk in the other
direction: a reader could take "identical data preprocessing, feature construction and
evaluation protocols" as a fully matched experiment. The added clause bounds it.

Wording discipline: **"not reported", never "not matched".** `NAR-E`, `NAR-F` and `NAR-G`
are `NOT STATED` in the article, which is `UNKNOWN`, not a negative finding. Checked
mechanically (RWF10).

### What the compression retained — verified by check, not by reading

```
Kou13 direct-rate arm                        retained
Kou13 separate size + interval outputs       retained
Kou13 RATIO combination                      retained  <- the distinction that must survive
Kou13 neural representation precedent        retained
NAR26 direct LightGBM arm                    retained
NAR26 probability x conditional-size product retained
NAR26 two-stage precedent                    retained
NAR26 non-neural marker                      retained  ("gradient-boosting rather than a
                                                        neural setting")
both-precedent concession                    retained
not-our-novelty statement                    retained
```

### One regression caught during compression

Trimming P2 detached `[Kou13]` from the sentence carrying the ratio distinction, so that
sentence became unmapped in the regenerated reference map. The citation was restored
before the map was accepted. This is the exact failure mode the reference-map rebuild
exists to catch, and it is recorded rather than quietly fixed.

---

## Concession wording — two sentences, less declarative

```
v1  "Both comparisons therefore already exist: a directly predicted rate against a
     Croston-style ratio in a neural setting [Kou13], and a directly predicted
     conditional mean against a probability-times-size product in a tree-ensemble
     setting on real data [NAR26]."
v2  "These studies provide clear precedents for both comparisons: a directly predicted
     rate against a Croston-style ratio in a neural setting [Kou13], and a directly
     predicted conditional mean against a probability-times-size product on real data
     [NAR26]."
```

*"Both comparisons therefore already exist"* announces a verdict; *"These studies provide
clear precedents"* attributes it to the studies. Same concession, standard register.

```
v1  "Neither the factorized formulation nor the act of comparing it against a direct one
     originates here."
v2  "Accordingly, neither decomposition itself nor the direct-versus-factorized
     comparison is the focus of the contribution reported here."
```

*"originates here"* reads as pre-empting a challenge; *"is the focus of the contribution
reported here"* states scope. The concession is not weakened — it now names both things
being conceded (decomposition, and the comparison) rather than one thing twice.

---

## 2.5 — two wording edits

```
v1  "...how the *relative* behaviour of the two representations moves as temporal
     dependence changes."
v2  "...how the *relative* behaviour of the two representations changes as temporal
     dependence varies."

v1  "...while the temporal dependence of occurrence and that of positive magnitude are
     varied along separate axes..."
v2  "...while the temporal dependence of occurrence and the temporal dependence of
     positive magnitude are varied along two separate axes..."
```

The first replaces a colloquial verb. The second spells out both axes rather than relying
on "that of", so the two-axis design is unambiguous on a single reading. No claim changed.

---

## Nothing unexpected

```
citations removed from the section        0     (21 -> 20 occurrences, same 11 keys;
                                                 the drop is one duplicate [NAR26])
concession elements lost                  0     (10/10 retained, checked mechanically)
evidence status changed                   0
component grade changed                   0
absence claims introduced                 0
Introduction v6 modified                  0
2.1 / 2.2 / 2.3 substantive change        0     (byte-identical)
```
