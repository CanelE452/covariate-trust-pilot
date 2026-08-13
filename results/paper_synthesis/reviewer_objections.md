# Reviewer objections

Each is marked **ANSWERABLE NOW** / **NEEDS EXPERIMENT** / **REDUCE SCOPE**.

---

**1. "This is just a Point vs Hurdle comparison."** — REDUCE SCOPE (of the framing, not the work)

The comparison itself is not the contribution and the paper must not open as if it were.
The contribution is the *conditioning*: which temporal structure makes one representation
preferable, and whether that conditioning survives observational data. The abstract should lead
with the conditioning question, and Table 2's overall row (mean delta −0.00066 on M5) should be
presented as the motivation — the aggregate is a tie, so the interesting variation is
conditional.

---

**2. "Correlation effects on intermittent demand are already studied."** — ANSWERABLE NOW

The related work is about estimators under intermittency (Croston, SBA, TSB, iETS). This is
about which *representation of the conditional mean* a neural forecaster should use, with the
marginals held fixed so the effect is attributable to occurrence ordering rather than to
sparsity. The positioning must be explicit in related work, not implied.

---

**3. "Why are Croston and TSB not in the mechanism study?"** — ANSWERABLE NOW

They are in the paper, as Table 3, and the answer is uncomfortable and should be stated
plainly: on M5 by mean rank, SBA 3.152, Croston 3.260, TSB 3.411, SES 3.483 all beat
dlinear_point 4.202 and dlinear_hurdle 4.220. The mechanism study is a controlled comparison of
two representations of one backbone and is not a claim about the best forecaster. Hiding
Table 3 would be the fatal version of this objection; publishing it converts it into a scope
statement.

---

**4. "The synthetic setting is too small."** — NEEDS EXPERIMENT (deferred), REDUCE SCOPE now

Cannot be assessed from this repository at all, since the synthetic artifacts are not here.
Once recovered, the defensible position is that the grid is a designed contrast rather than a
survey, and the external screen is what tests generality. Expanding the grid is
NICE_TO_HAVE, not MUST_HAVE.

---

**5. "The real-data H1 effect is tiny."** — ANSWERABLE NOW, with a concession

Spearman +0.1065 and +0.0789 are small. Three things make them worth reporting: they are the
same sign in two independent datasets, they survive three error scales and three eligibility
thresholds with six of six intervals clear of zero, and they strengthen in the intermittent
regime (relative +0.153 CI [+0.052, +0.261]). The concession that must be written: the adjusted
partial association is +0.032 standardized with a CI containing zero, so this is a marginal
association, not a controlled effect, and the paper says so.

---

**6. "H2 is a scale confound."** — ANSWERABLE NOW, and the paper agrees

This is the paper's own finding rather than a vulnerability. Matching failed at SMD 0.614 on
`log_train_scale`, and after overlap weighting to SMD 0.0004 the association is +0.0032
CI [−0.0033, +0.0094]. The claim is therefore restricted to "predictive selector", and the
mechanism claim is explicitly withdrawn. A reviewer raising this is confirming the paper's
Section on entanglement.

---

**7. "Why include a routing method that failed externally?"** — ANSWERABLE NOW

Because it is not presented as a method. It is presented as a measured gap between an oracle
opportunity (4.11% on M5, multiplied 2.15× by deliberate diversification) and what any of five
successive designs could collect (−2.43% on the first external dataset). The stop rule was
frozen before the results, and the paper reports it being triggered. Removing the section would
leave the "adaptive selection is suggested" implication of Section 5 unexamined, which is worse.

---

**8. "Why no comparison against a modern state-of-the-art model?"** — ANSWERABLE NOW

The claim is comparative between two representations sharing one backbone and one training
budget; a stronger backbone changes both arms and does not test the claim. The classical
benchmark already establishes the absolute position honestly. This must be stated as a scope
limitation in the discussion rather than defended as a virtue.

---

**9. "UCI availability is unclear — is it a valid benchmark?"** — REDUCE SCOPE

`dataset_audit.json` records `availability_status: AVAILABILITY_UNKNOWN` for UCI Online Retail
II, and it was admitted with role `EXTERNAL_ROBUSTNESS`, not as a confirmation dataset. Its
role in the paper should be exactly that: a counterexample that shows a routing failure mode,
not a benchmark whose absolute numbers are interpreted. The −193.9% is used to say "the gate
left the protection of the static weight", supported by the mechanism numbers (expert A error
3249.2 versus expert B 86.7, gate weight 0.295 against a fitted alpha of 0.65), not to rank
methods.

---

**10. "Why is DLinear the only backbone?"** — NEEDS EXPERIMENT (NICE_TO_HAVE) + REDUCE SCOPE

There is no artifact addressing this anywhere in the project. Every gate and every expert in
the routing chain is from one family. The honest response is a stated limitation: all claims
are conditional on this backbone family. A second backbone would materially strengthen C1 and
C3 and is the single most valuable optional experiment, but no contribution collapses without
it provided the limitation is stated in the abstract-level scope, not buried.

---

## Top five by risk

```
rank  objection                        current answer                          residual weakness
─────────────────────────────────────────────────────────────────────────────────────────────────
 1    #10 single backbone              stated limitation                       real; unaddressed
 2    #3  classicals beat the backbone Table 3 published, scope restated       uncomfortable but honest
 3    #5  H1 effect is small           2 datasets, 3 scales, regime split      adjusted effect ~0
 4    #4  synthetic scope              cannot be assessed - artifacts absent   blocking for C1
 5    #9  UCI validity                 demoted to a failure-mode counterexample  weakens C3 slightly
```
