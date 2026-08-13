# Routing chronology

Fourteen steps, in the order they happened, each with the number that decided what came next.
Numbers are from the artifact named on the line.

```
 #  step                          artifact                              headline
────────────────────────────────────────────────────────────────────────────────────────────
 1  complementarity confirmed     structure_gate/gate_potential.json    M5 point 1.0226,
                                                                       hurdle 1.0214
 2  static mixture is strong      same                                  50:50 -> 1.0087,
                                                                       better than both;
                                                                       fitted alpha = 0.50
 3  oracle potential measured     structure_gate/convex_oracle.json     hard 0.9689,
                                                                       convex 0.9673;
                                                                       convex gain 4.11%;
                                                                       g* at 0 / interior / 1
                                                                       = 29.6 / 30.1 / 40.3%
 4  Gate-v1 kill test             structure_gate/killtest.json          GATE_KILL_YELLOW
 5  Gate-v2 on cross-fitted OOF   structure_gate/gate_v2_oof_result.json GATE_V2_OOF_GREEN;
                                                                       M5 gate vs point
                                                                       +1.99% [+1.69,+2.30]
 6  Gate-v2 fresh holdout         structure_gate/fresh_confirmatory.json GATE_V2_CONFIRM_GREEN
                                                                       on 23,513 unseen M5
                                                                       series
 7  expert diversity raised       expert_diversity/expert_set_spec.json pair point_plain|naive,
                                                                       geometric ceiling
                                                                       multiplier 2.15,
                                                                       worst 1.86,
                                                                       max residual corr 0.855
    frozen gate on that pair      expert_diversity/pair_gate_result.json DIVERSE_GATE_GREEN,
                                                                       but only +0.43%
                                                                       [+0.06,+0.81] over
                                                                       expert A alone
 8  FIRST EXTERNAL TEST           multi_benchmark/external_benchmark.json
                                                                       FreshRetailNet
                                                                       -2.43% [-2.74,-2.13]
                                                                       UCI +0.13% [-0.06,+1.13]
                                                                       -> EXTERNAL_VALIDATION
                                                                          _NOT_REPLICATED
 9  regret-target diagnosis       gate_v3/                              the regret-BCE target
                                                                       was NOT the cause;
                                                                       corr(q, g*) 0.84-0.86
10  direct loss, 2x2 factorial    gate_v3/aggregate_results.json        DIRECT_LOSS_SUPPORTED,
                                                                       ALPHA_ANCHOR_SUPPORTED
                                                                       = False.
                                                                       GATE_V3_OOF_STRONG is
                                                                       an OOF label only.
11  P0L1 across more folds        gate_p0l1_robustness/                 m5 +0.774%*,
                                                                       favorita +1.350%,
                                                                       fresh -0.506%,
                                                                       uci +67.924%*
                                                                       operational STRONG,
                                                                       interpretation B
12  Safe-P0L1 shrinkage           gate_safe_p0l1/                       tail p95 better on 4/4,
                                                                       mean worse than raw on
                                                                       4/4; Fresh -0.809%
                                                                       -> SAFE_P0L1_TEMPORAL
                                                                          _MIXED
13  same features, stronger       routing_information_ceiling/          HGB beats the MLP on
    learner                                                            1/4 (M5 +0.286%, CI
                                                                       includes 0); Fresh
                                                                       -1.308% CI clear of 0
                                                                       -> CURRENT_FEATURE
                                                                          _INFORMATION_LIMITED
14  raw-history GRU               temporal_routing_encoder/             Fresh -0.506% -> +2.648%
                                                                       CI [+2.068,+3.287],
                                                                       3/3 folds over P0L1;
                                                                       UCI -193.9%,
                                                                       one fold -265.9%
                                                                       -> SEQUENCE_ROUTING_RED
```

Binding state after step 14:

```
HANDCRAFTED_FEATURE_GATE_STOP
RAW_SEQUENCE_GATE_STOP
ROUTING_MODEL_DEVELOPMENT_STOP
DO_NOT_CONSUME_NEW_CONFIRMATORY_DATASET
```

## What the chain is evidence for

The sequence is not a story of a method being improved. Steps 1–7 raise the *opportunity*
(oracle ceiling, expert diversity) and steps 8–14 repeatedly fail to convert it into
cross-domain gain, each time ruling out one explanation:

- step 9 rules out the training target,
- step 10 rules out the loss and the parameterisation,
- step 12 rules out "the gate is simply too aggressive" (shrinking it costs mean accuracy),
- step 13 rules out "the gate is too small",
- step 14 rules out "the features are the only problem" only partly: raw history helps on one
  dataset and fails badly on another.

That elimination sequence is the contribution. The method is not.

## The one thing steps 8–14 never ruled out

No step tested a different **expert backbone**. Every gate routed between variants of the same
DLinear family (plus `naive`). Whether the instability is a property of routing or a property
of this expert family is untested, and the paper must say so.
