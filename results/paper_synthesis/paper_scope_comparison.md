# Paper scope comparison

Judged on the evidence that exists in artifacts, not on what would be nicest to write.

---

## Scope A — factorization diagnostic paper

Controlled numerical characterization of Point vs Hurdle inductive bias under temporal
occurrence and magnitude structure. Real data as a short external check. Routing in the
discussion only.

**Strength.** The cleanest possible story. The controlled study is the part where the causal
manipulation is genuine, and the mechanism claim survives there because the ground truth is
known.

**Weakness.** It discards the strongest independent evidence in the whole project: the frozen
H2 selector replicating on 5,693 held-out M5 series across three seeds. It also discards the
routing chain entirely, which is roughly half the work and contains the only genuinely
surprising negative results. What is left is a synthetic study with a modest real-data
appendix, and the real-data section as written would then be *weaker* than the artifacts
support.

**Needed experiments.** A second backbone becomes close to mandatory, because with routing
removed the paper's only defence against "this is a DLinear quirk" is more backbones.

**Reviewer risk.** "Why should I believe the synthetic axes matter in practice?"

**Evidence completeness.** Low — the synthetic artifacts are not in this repository.

---

## Scope B — factorization plus empirical boundary paper

Controlled characterization, then real-data conditional validation, then the entanglement that
real demand imposes on the synthetic axes, and finally the adaptive-use boundary as the closing
experiment.

**Strength.** Every result in the ledger has a place, including the negatives, and the
negatives are load-bearing rather than embarrassing. The two strongest artifacts —
the H2 independent replication (`−0.0230` CI [−0.0294, −0.0163], +11.87 pp win rate, three
seeds) and the routing elimination sequence — are both first-class content. The overlap
adjustment (`+0.0032`, CI includes zero, matching fails at SMD 0.614 on scale) turns what could
look like a weakness into the paper's sharpest observation: a rule can be a good selector and a
bad explanation at the same time.

**Weakness.** Three sections is a lot of surface. The occurrence-gate BSS result
(M5 −0.008, Favorita −0.091) cuts against the mechanism narrative and must be shown, which
costs a paragraph and some comfort.

**Needed experiments.** None that are experiments. One recovery task: the controlled study's
artifacts.

**Reviewer risk.** "Too many things in one paper." Manageable if the routing chain is
compressed to one section with three numbers.

**Evidence completeness.** High for everything except the controlled study itself.

---

## Scope C — adaptive routing method paper

Expert diversity plus a learned gate as the contribution.

**Strength.** Would be the most citable if it worked.

**Weakness.** It does not work. The first external test of the frozen gate was −2.43%
(CI [−2.74, −2.13]) against a static mixture on the dataset that had been declared the primary
external confirmation before results existed. Every subsequent repair failed: shrinkage cost
mean accuracy on 4/4, a much stronger learner on the same features won on 1/4, and the raw
sequence gate was −193.9% on UCI. The realized gain even on the diverse pair was +0.43%.
There is no version of this paper that is honest and also a method paper.

**Needed experiments.** A working router. That is exactly what
`ROUTING_MODEL_DEVELOPMENT_STOP` forbids, and the stop rule was frozen before the results.

**Reviewer risk.** Fatal. The paper would be rejected on its own appendix.

**Evidence completeness.** Negative — the evidence actively contradicts the claim.

---

## Recommendation

**Scope B.**

Scope C is ruled out by evidence, not by preference. Scope A is defensible but throws away the
project's two most defensible results and would need *more* new experiments than Scope B, not
fewer, because it removes the empirical spine that currently protects the synthetic claims from
"so what". Scope B is the only scope where the failures do work: the routing chain stops being
a failed method and becomes a demonstration that a measurable oracle opportunity does not
imply a learnable routing function — which is a claim nobody in the ledger can contradict.
