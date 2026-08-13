# 6. Discussion — draft v1

No number appears here that is not already in Section 4 or 5. This section states what the
results license and what they do not.

---

## 6.1 A finite-sample reading

The two formulations estimate the same quantity. The product `p_t · mu_t` and a directly
regressed conditional mean describe the same function of the history, so nothing in these
results can be read as one formulation being able to represent something the other cannot.
What differs is how a fixed parameter budget, a fixed objective and a fixed number of
gradient steps are spent.

That is the whole content of the comparison, and it constrains the conclusions in a
specific way. When the occurrence sequence carries exploitable structure, dedicating part of
the budget to an explicit occurrence head appears to pay for itself; when the magnitude
sequence is persistent, the same split appears to cost more than it returns, because a
directly regressed mean can absorb a slowly varying level without maintaining two heads.
Both statements are about **estimation under a budget**, not about asymptotic superiority,
and neither is offered as a general claim about intermittent demand.

The practical reading is correspondingly narrow. These results give no reason to prefer one
formulation over the other in the abstract; they give a reason to expect the choice to
matter conditionally, and they indicate which conditions to look at.

## 6.2 Why the two axes behave differently

The most informative pattern is that the two dependence axes respond to different features
of dependence: for occurrence it is the strength of dependence that tracks the comparison,
almost regardless of sign, while for magnitude it is the direction, with persistence
specifically moving the comparison toward direct prediction.

A plausible account is that the two heads face different estimation problems. The occurrence
head estimates a bounded probability, and dependence of either sign — clustering or
alternation — makes that probability more predictable from recent history than the marginal
rate; the *amount* of structure is what helps, not its polarity. The magnitude head
estimates an unbounded conditional level, and persistent magnitude means the level drifts
slowly, which is precisely the regime in which a single directly regressed mean tracks well
without paying for a second head.

This is an interpretation of a controlled observation, not a demonstrated mechanism. The
component-attribution diagnostic of Section 4.7 localizes error to the occurrence head under
one synthetic configuration; it does not identify a cause, and on real data the learned
occurrence head shows no skill advantage at all. The account above should be read as the
hypothesis the results make most natural, and as something a future design could test
directly.

The asymmetry also dissolves an apparent contradiction between the two stages. Stage 1's
structured magnitude arm is alternation, which sits at negative `ρ_M`, and those cells are
factorized-favourable; only persistence, which Stage 1 never generated, produces a
direct-favourable region. Reading Stage 1 alone would have suggested that structured
magnitude is uniformly unfavourable to factorization. It is not; the sign is what matters,
and only the signed sweep can see that.

## 6.3 Synthetic structure and its entanglement in observed demand

Two things transfer to real demand and one does not, and the split is the most useful part
of the empirical section.

What transfers is *conditional information*. The occurrence relationship appears as an
analogue on both datasets and strengthens within the intermittent regime, and a rule frozen
from the controlled sweep selects previously unseen series toward direct prediction on an
independent population, reproducibly across seeds. A pattern found by manipulating a
generating parameter turns out to carry usable signal about which observed series behave
which way.

What does not transfer is the *isolated association*. After adjusting for overlap in scale,
sparsity, positive-demand variability and occurrence dependence, the association is no longer
distinguishable from zero. The reason is visible in the covariates rather than in the
outcome: in the controlled study the three descriptors are set independently by
construction, whereas in observed demand they arrive together, and the selected and control
populations differ substantially in scale as well as on the intended axis.

It is important not to over-read this in either direction. The disappearance is **not**
evidence that the synthetic mechanism is wrong, and the paper does not attribute it to scale
as a cause; the controlled design contains no scale axis and cannot arbitrate the question.
Equally, the selector's success is **not** evidence that the mechanism replicated. These are
two results, reported separately and labelled separately, and a reader who wants one summary
sentence should take the weaker one: the configuration is a useful predictor whose
underlying association is confounded in observed demand.

## 6.4 Complementarity is not a routing function

Section 5.9 reports a measurable opportunity and a failure to convert it. Both belong in the
paper.

The opportunity is genuine: an origin-level oracle over the two forecasts beats the best
static mixture of them, and the ceiling grows when the expert pair is deliberately chosen to
be less correlated. That is the strongest available evidence that the conditional
differences documented in Sections 4 and 5 are real and exploitable in principle.

The conversion did not survive contact with new domains. A gate frozen after development
lost to a static mixture on the first external dataset; successive changes to the target,
the loss, the aggressiveness, the capacity and finally the input representation did not
recover cross-domain transfer, and the final representation experiment improved one external
domain while failing severely on another. A pre-registered stopping rule ended the search
rather than allowing it to continue until something worked, which is the reason this appears
as a boundary and not as an unreported negative.

The general lesson is worth stating plainly because it is easy to get wrong: an oracle
gap measures how much a perfect selector *could* gain, and says nothing about whether a
selector estimable from data will generalize. In this study the gap between the two was the
whole story.

## 6.5 Limitations

**One backbone family.** Every arm in the synthetic study and in the routing chain is
DLinear. The comparison is therefore between two representations *within* one function
class, and it is possible that a different backbone would redistribute the finite-sample
advantage. This is the study's largest single exposure. Testing a second backbone would
strengthen the characterization and is not required to state it; no such experiment was run
for this paper.

**Synthetic simplicity.** The generating process excludes trend, calendar seasonality,
hidden regime switching, heavy tails, test-time shift, phase jitter and interval–magnitude
cross-correlation. That exclusion is what makes the marginal control interpretable, and it
is also why the controlled results are a characterization of a mechanism rather than a
forecast of behaviour on any particular real catalogue.

**Stage 1's conditional validity.** Stage 1's structured arm is deterministic alternation.
It establishes that ordering matters under fixed marginals and that the two axes interact;
every graded statement about the strength or the sign of dependence rests on Stage 2 alone.

**H3's construct mismatch.** The synthetic sparsity contrast is a mean interval of 4 against
8; the pre-registered external test splits at an ADI near 1.3. The non-replication is
reported, and it is not read as a refutation, because the tested contrast is not the
synthetic one. Re-testing at a contrast the real data can support would settle it; the
support exists but was not used as a primary test here.

**No occurrence-head skill on real data.** The synthetic component-attribution result has no
counterpart in the empirical data, where the learned occurrence head shows no skill
advantage. Any mechanism story is therefore synthetic-only.

**Routing instability.** Learned routing is reported as failed, not as unfinished. The
stopping rule was triggered deliberately.

**Absolute accuracy is not the contribution.** Both neural formulations are outranked by
classical estimators on both datasets. The paper compares two representations under one
budget and makes no competitiveness claim.

**A literature limitation, distinct from the above.** One of the closest prior studies could
not be obtained in full text, so what it holds constant across its correlation levels is
unverified. Nothing in this paper's empirical or synthetic results depends on that, and no
statement is made about that study's control design; it is recorded because the novelty
boundary was drawn with it graded conservatively.

## 6.6 What would move this forward

A second backbone would test whether the finite-sample characterization is a property of the
representation pair or of DLinear. A design with a scale axis would let the entanglement
question of Section 6.3 be settled rather than bounded. A per-observation occurrence
diagnostic on real data would test whether the synthetic attribution has any empirical
counterpart. None of these is promised here, and none is required for the claims that are
made.
