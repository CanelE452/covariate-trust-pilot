# Figure plan

Five main figures. No figure is rendered in this task; this is the plan only.
Existing plots in `results/external_validity_screen/fig{A,B,C}_*.{png,pdf,svg}` are reusable
starting points for Fig 3 and Fig 4.

---

## Fig 1 — Two representations of the same conditional mean

**Purpose.** Define the object of study before any result.

**One sentence the reader should leave with.** The same conditional mean can be produced
directly or as occurrence times positive magnitude, and the two differ in what temporal
structure they can exploit.

**Panels.** (a) a schematic intermittent series with the interval and magnitude sequences
marked; (b) the two computational graphs side by side — direct head versus occurrence head ×
positive-magnitude head; (c) the two temporal-dependence axes the controlled study manipulates,
with ADI/CV² shown as marginal descriptors that are constant along both axes.

**Source.** Conceptual. No artifact.

**New plot needed.** Yes, entirely.

---

## Fig 2 — Controlled numerical result

**Purpose.** Show that relative performance moves systematically with temporal dependence while
the marginals are held fixed.

**One sentence.** With sparsity and marginal variability controlled, the advantage shifts from
one representation to the other as occurrence-interval and magnitude dependence change.

**Panels.** Heatmap of relative gain over the (rho_interval × rho_magnitude) grid, one panel per
sparsity level; the Point-favorable cell marked, since it is what H2's rule was derived from.

**Source.** The synthetic study — **artifacts not present in this repository**
(`artifact_audit.json`, entry `A_numerical`). This figure cannot be produced until they are
recovered.

**New plot needed.** Yes, and it is blocked.

---

## Fig 3 — Real-data validation of H1, with its regime boundary

**Purpose.** Show the structural relationship transferring, and show where it stops.

**One sentence.** The association between occurrence-interval dependence and relative
factorization advantage is positive in both public datasets and on all three error scales, and
it is present in the intermittent regime but absent in the lumpy one.

**Panels.** (a) M5 and (b) Favorita scatter of |rho_interval| against delta with the Spearman
estimate and CI annotated (+0.1065 [+0.0437, +0.1652] and +0.0789 [+0.0205, +0.1405]);
(c) a forest panel of the twelve regime × scale estimates from `regime_h1.json`, so the reader
sees the intermittent intervals clear of zero and the lumpy ones straddling it.

**Source.** `stage_a_results.json`, `posthoc_diagnostic.json`, `regime_h1/regime_h1.json`.
Panels (a)–(b) can start from `figA_H1_rho_interval_vs_delta.*`.

**New plot needed.** Panel (c) is new; (a)–(b) are a restyle.

---

## Fig 4 — A good selector that is not an explanation

**Purpose.** Carry C2's central point in one image.

**One sentence.** The frozen rule shifts unseen series toward the direct model by about twelve
points of win rate, and that shift disappears once scale and sparsity are balanced.

**Panels.** (a) candidate versus control delta distributions on the independent M5 population
with the effect −0.0230 [−0.0294, −0.0163] and the win-rate difference +11.87 pp [+7.85, +15.81];
(b) the three seeds side by side; (c) covariate balance before and after overlap weighting —
the `log_train_scale` SMD going from 0.614 (matching failed) to 0.0004, next to the
overlap-adjusted association +0.0032 [−0.0033, +0.0094] crossing zero.

**Source.** `rule_replication/primary_result.json`, `seed_robustness/seed_robustness.json`,
`rule_replication/secondary_overlap.json`, `h2_confirmatory/matching_failed.json`.
Panel (a) can start from `figB_H2_point_candidate_vs_control.*`.

**New plot needed.** Panel (c) is new and is the most important panel in the paper.

---

## Fig 5 — The adaptive-use boundary

**Purpose.** Show the opportunity, the deliberate enlargement of it, and the failure to convert.

**One sentence.** An origin-level oracle is about four percent better than the best static
mixture and diversifying the experts roughly doubles that ceiling, yet the learned gate is
worse than a static weight on the first external dataset and catastrophic on the second.

**Panels.** (a) loss ladder on M5 — point, hurdle, 50:50, hard oracle, convex oracle — with the
4.11% gap marked; (b) ceiling multiplier for the reference pair versus the diverse pair (2.15×)
with the realized gain (+0.43%) overlaid to show the gap between opportunity and collection;
(c) external result per dataset: FreshRetailNet −2.43% [−2.74, −2.13] for the handcrafted gate
and +2.648% [+2.068, +3.287] for the sequence gate, against UCI −193.9% — plotted on a broken
axis so the catastrophe is visible rather than hidden by scaling.

**Source.** `structure_gate/convex_oracle.json`, `expert_diversity/expert_set_spec.json`,
`expert_diversity/pair_gate_result.json`, `multi_benchmark/external_benchmark.json`,
`temporal_routing_encoder/aggregate_results.json`.

**New plot needed.** Yes, all three panels.

---

## Deliberately not a figure

A mechanism figure showing the occurrence gate masking magnitude error was on the candidate
list. It is dropped for real data: the fitted occurrence head has **no** skill against a
constant per-series rate (M5 BSS −0.008; Favorita −0.091, CI [−0.140, −0.044]). If such a
figure is drawn at all it belongs inside Fig 2 as part of the controlled study, where the
mechanism is observable, and it must not be captioned as if it were validated on M5 or Favorita.
