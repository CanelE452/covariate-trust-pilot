# Reader chain, checked against the artifacts

The proposed chain is reproduced step by step with a verdict on whether the evidence actually
supports that link. Two links needed rewriting.

---

**1. Problem — intermittent-demand forecasts can represent the conditional mean directly or
through occurrence/magnitude factorization, and when one is preferable is unclear.**

Supported. `stage_a_results.json`: M5 mean delta −0.00066 with Hurdle winning 47.2% of series;
Favorita −0.03197 with 49.3%. RMSE favours Point, MAE favours Hurdle, on both. The question is
open in the data, not only in the literature.

---

**2. Why existing descriptors are insufficient — ADI and CV² describe marginals, not temporal
ordering.**

Supported as motivation, and it is also *why* the controlled study is needed: the two axes the
study manipulates are invisible to ADI/CV² by construction. The paper should state this as a
design rationale rather than as an empirical finding, because no artifact here measures
"ADI/CV² fail to predict delta" directly.

---

**3. Controlled study — manipulate temporal dependence while controlling marginals.**

Supported in principle; **artifacts absent from this repository**. This link is currently a
citation to work that cannot be audited here.

---

**4. Finding — relative Point/Hurdle performance changes systematically.**

Same status as link 3.

---

**5. Mechanism — occurrence predictability can make the Hurdle gate useful, while decomposition
can impose a cost when occurrence provides little value.**

**Needs rewriting.** This is true only inside the controlled study. In the real datasets the
fitted occurrence head has no skill against a per-series constant rate: BSS −0.0084 on M5
(CI includes zero) and −0.0908 on Favorita (CI [−0.140, −0.044], significantly worse). If the
chain presents the mechanism as general, the paper contradicts its own appendix.

*Rewritten link.* "Within the controlled study the occurrence head is where the difference
originates. Whether that mechanism operates in observational data is a separate question, and
in the two datasets examined the fitted occurrence head does not beat a constant rate."

---

**6. Real data — some structural relationships transfer, but factors are entangled and not all
synthetic interactions replicate.**

Supported, and this is the strongest link in the chain. H1 transfers (six of six estimates
positive with intervals clear of zero); H2 transfers as a selector on an independent population
(−0.0230 CI [−0.0294, −0.0163]; +11.87 pp win rate) and across three seeds; H3 does not
replicate (both signs opposite, both CI spanning zero); and the entanglement is quantified
rather than asserted — matching fails at SMD 0.614 on scale, and the overlap-adjusted
association is +0.0032 with a CI containing zero.

---

**7. Adaptive implication — no universal factorization dominates, suggesting adaptive
selection.**

Supported as a motivation. `convex_oracle.json` M5: the per-origin convex oracle is 4.11%
better than the best static mixture, with the oracle weight at an endpoint for 70% of origins.
The suggestion is well founded; whether it is realizable is link 8.

---

**8. Boundary — expert complementarity can be increased, but stable learned routing does not
generalize reliably across domains.**

Supported, with one addition the chain currently omits. The diversity multiplier is 2.15× but
the realized gain on that pair is only +0.43% over the better expert — so the honest statement
is that the *ceiling* was raised, not the performance. The external failure is −2.43%
CI [−2.74, −2.13] on the pre-declared primary external dataset, and the final representation
experiment splits: +2.648% on FreshRetailNet against −193.9% on UCI.

*Added clause.* "Raising the oracle ceiling did not raise realized performance."

---

**9. Conclusion — understanding conditional inductive bias is more robust than claiming one
universal forecasting architecture.**

Supported, but it should not be the last word, because as written it is unfalsifiable. A
sharper closing claim that the artifacts do support: **a measurable oracle opportunity is not
evidence that a routing function is learnable, and the gap between the two is large enough to
be worth reporting.** That is a claim a future paper could refute, which the softer version
is not.

---

## Net changes to the chain

- Link 5 is scoped to the controlled study and the real-data non-replication is stated in the
  same breath.
- Link 8 gains "the ceiling rose, the realized gain did not".
- Link 9 is replaced with the falsifiable version.
- Links 3 and 4 are flagged as currently unauditable in this repository.
