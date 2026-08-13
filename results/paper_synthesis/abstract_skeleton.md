# Abstract skeleton

Six sentences. Numbers appear only where an artifact supports them, and each sentence carries
its evidence source. Sentence 4 is a placeholder because the controlled study's artifacts are
not in this repository.

---

**1 — Problem.**
Intermittent demand can be forecast by predicting the conditional mean directly or by
factorizing it into occurrence and positive magnitude, and across 1,200 series in each of two
public retail datasets neither choice dominates — the mean difference is −0.0007 on M5 and
−0.0320 on Favorita, with RMSE favouring one representation and MAE the other.
→ `stage_a_results.json`

**2 — Gap.**
The descriptors normally used to characterize intermittent series, ADI and the squared
coefficient of variation, summarize the marginal distribution and are silent about the order in
which zeros and positive values arrive, so they cannot say which representation a given series
should use.
→ design rationale; no empirical artifact, and the abstract must not imply one

**3 — Controlled method.**
We manipulate occurrence-interval and magnitude dependence while holding the marginal
distribution fixed, so that any change in relative performance is attributable to temporal
structure rather than to sparsity or scale.
→ the synthetic study; **artifacts not present in this repository**

**4 — Main numerical result.**
*[PLACEHOLDER — the direction and magnitude of the relative-gain surface, and the location of
the direct-favorable region, must be filled from the recovered synthetic artifacts. No number
is asserted here until then.]*
→ blocked on MUST_HAVE M1

**5 — Real-data result and boundary.**
In two public datasets the association between occurrence-interval dependence and the relative
advantage of the factorized model is positive on three error scales (Spearman +0.107 and
+0.079, all six intervals excluding zero) and strengthens in the intermittent regime, while a
rule frozen from the controlled study shifts 675 unseen M5 series toward the direct model by
11.9 percentage points of win rate — yet the same association is +0.003 with an interval
containing zero once scale and sparsity are balanced, so the rule predicts without explaining.
→ `stage_a_results.json`, `posthoc_diagnostic.json`, `regime_h1.json`,
`rule_replication/primary_result.json`, `rule_replication/secondary_overlap.json`

**6 — Contribution and implication.**
A per-origin oracle over the two representations is 4.1% better than the best static mixture and
deliberately diversifying the expert pair multiplies that ceiling by 2.15, but a gate frozen
after development was 2.4% worse than a static mixture on the first external dataset and five
successive redesigns did not recover it — so we report the conditional inductive bias as the
transferable finding and the routing gap as a measured limit rather than a method.
→ `structure_gate/convex_oracle.json`, `expert_diversity/expert_set_spec.json`,
`multi_benchmark/external_benchmark.json`, `temporal_routing_encoder/aggregate_results.json`

---

## Notes for whoever drafts the real abstract

- Sentence 5 is the longest and carries the most weight; it will probably need to split.
- The word "causes" must not appear anywhere in the abstract.
- The UCI −193.9% does not belong in the abstract; it belongs in Section 7 with its mechanism.
- Do not put the FreshRetailNet sequence-gate recovery (+2.6%) in the abstract without the UCI
  failure beside it. If both do not fit, leave both out.
