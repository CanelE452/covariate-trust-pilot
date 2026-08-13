# Claim to evidence map

Every claim the main text will make, with the artifact behind it and the figure or
table that carries it. Claims are numbered so the manuscript can cite this file.

---

## Contribution 1 — controlled characterization

```
M1  Neither representation dominates on real intermittent demand; the aggregate is
    close to a tie and its sign depends on the metric.
    evidence  stage_a_results.json: M5 mean delta -0.00066, factorized wins 47.2%;
              Favorita -0.03197, 49.3%; RMSE favours direct, MAE favours factorized
    carried   Section 1, Table 4 Block A

M2  Under fixed marginals, changing only the temporal ordering changes the relative
    finite-sample behaviour of the two representations substantially.
    evidence  stage1_verified_contrasts.csv: interval +7.83 [+6.09,+9.45],
              magnitude -4.58 [-6.28,-2.92], sparsity -6.26 [-8.01,-4.57]
    carried   Section 4.3, Table 2

M3  The two axes interact, and the interaction is larger than either main effect.
    evidence  interval x magnitude -16.74 [-18.45,-15.05]
    carried   Section 4.3, Table 2

M4  Sparsity amplifies the occurrence effect and does not modulate the magnitude
    effect.
    evidence  sparsity x interval +3.35 [+1.62,+5.09];
              sparsity x magnitude -0.01 [-1.70,+1.69], spans zero;
              Stage 2 d x rho_I +0.0332 [+0.0194,+0.0471], d x rho_M spans zero
    carried   Sections 4.3 and 4.5, Table 2

M5  Error attribution: substituting the true component isolates the occurrence head
    as the dominant error source under selected conditions.
    evidence  stage1_verified_cells.csv, cell C03 model M1:
              p_true x mu_hat 0.2900, p_hat x mu_true 0.9030, full 0.9652
    carried   Section 4.4, Appendix E
    strength  COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT only

M6  The factorized advantage rises with occurrence dependence in both directions.
    evidence  stage2_scientific_classification.json:
              C_neg +0.1266 [+0.0870,+0.1660], C_pos +0.2008 [+0.1554,+0.2439]
    carried   Section 4.5, Figure 2A and 2B

M7  Occurrence effects are primarily associated with dependence strength;
    magnitude effects primarily with dependence direction.
    evidence  stage2_verified_factor_effects.csv:
              |rho_I| +0.1904 vs signed +0.0667  (2.9x)
              signed rho_M -0.0711 vs |rho_M| -0.0228  (3.1x)
              level means: rho_I U-shaped +12.10/+3.57/+16.47;
                           rho_M monotone +14.10/+11.94/+6.11
    carried   Section 4.6, Figures 2B and 2C
    note      this is the paper's sharpest synthetic finding

M8  Within the eighteen-cell grid, only one condition shows statistically clear
    superiority for direct prediction: sparse, occurrence-unpredictable,
    magnitude-persistent.
    evidence  stage2_verified_cells.csv: d=8, rho_I=0, rho_M=+0.8,
              G = -19.76 [-26.00, -14.53]
    carried   Section 4.7, Figure 2A
    note      Stage 1 has no such cell; C08 is -3.01 with CI [-6.79, +0.23]

M9  The controlled comparison is matched, not merely similar.
    evidence  5,856 parameters each; one trainer; identical optimizer, budget, split,
              seeds and checkpoint rule; per-cell tuning prohibited
    carried   Section 4.2, Table 1
```

## Contribution 2 — empirical transfer and boundary

```
M10 An empirical analogue of the occurrence-dependence relationship appears on two
    public datasets and is robust to scale and threshold choices.
    evidence  M5 +0.1065 [+0.0437,+0.1652], Favorita +0.0789 [+0.0205,+0.1405];
              six of six scale estimates positive with intervals clear of zero;
              thresholds 15/20/30 move it by <0.001
    carried   Section 5.3, Figure 3A
    relation  EMPIRICAL_ANALOGUE

M11 It is present in the intermittent regime and absent in the lumpy one.
    evidence  regime_h1.json: intermittent relative +0.153 [+0.052,+0.261],
              scaled +0.123 [+0.013,+0.227]; lumpy +0.014/+0.032/+0.028, all spanning
              zero
    carried   Section 5.3, Figure 3A inset

M12 A rule frozen from the controlled configuration shows predictive transfer:
    it selects unseen series toward direct prediction.
    evidence  rule_replication/primary_result.json: 675 vs 5,018,
              effect -0.0230 [-0.0294,-0.0163], Point win rate +11.87 pp
              [+7.85,+15.81]; seed_robustness.json: three seeds, all intervals clear
    carried   Section 5.4, Figure 3B

M13 The three axes of that rule correspond one to one with the synthetic cell,
    including the sign of the magnitude axis.
    evidence  h1_h2_h3_provenance.md; synthetic rho_M = +0.8 against external signed
              upper tertile
    carried   Section 5.4

M14 The association does not survive balancing, so the rule predicts without
    explaining.
    evidence  secondary_overlap.json: +0.0032 [-0.0033,+0.0094] after weighting to a
              worst SMD of 0.0004; matching_failed.json: worst SMD 0.614 on
              log_train_scale
    carried   Section 5.5, Figure 3C

M15 The sparsity interaction was not tested at the contrast it was derived from.
    evidence  synthetic ADI 4 vs 8; external split at ADI median 1.304 / 1.317;
              support exists (M5 127 at ADI 3-5, 52 at >= 8)
    carried   Section 5.6, Appendix C

M16 Both representations are outranked by SBA on both datasets.
    evidence  classical_benchmark/benchmark.json mean ranks
    carried   Section 5.2, Table 4 Block B
```

## Contribution 3 — adaptive-use boundary

```
M17 A per-origin oracle over the two experts is about 4% better than the best static
    mixture, and diversifying the pair roughly doubles that ceiling.
    evidence  convex_oracle.json: convex gain 4.11% on M5;
              expert_set_spec.json: geometric ceiling multiplier 2.15
    carried   Section 5.7

M18 The ceiling rose; realized performance did not.
    evidence  pair_gate_result.json: +0.43% [+0.06,+0.81] over the better expert
    carried   Section 5.7

M19 A gate frozen after development was worse than a static mixture on the first
    external dataset it saw.
    evidence  external_benchmark.json: -2.43% [-2.74,-2.13] on FreshRetailNet,
              declared the primary external confirmation before results existed
    carried   Section 5.7

M20 Successive redesigns did not recover it; the representation change recovered one
    dataset and failed catastrophically on another.
    evidence  temporal_routing_encoder: FreshRetailNet +2.648% [+2.068,+3.287],
              UCI -193.9%; pre-registered stop rule triggered
    carried   Section 5.7
```

## Limitations, each stated in the text

```
L1  one backbone family throughout          Sections 4.2, 6.5
L2  no scale axis in the controlled design  Sections 5.5, 6.3, 6.5
L3  Stage 1 CONDITIONALLY_VALID; its
    structured arm is alternation only      Sections 4.3, 4.5, 6.5
L4  H3 untested at its own contrast         Sections 5.6, 6.5
L5  learned occurrence head has no skill
    on real data                            Section 6.5, Appendix B
L6  point metrics only                      Section 6.5
L7  Stage A sample is regime-balanced,
    not representative                      Section 5.1
```

---

## Coverage check

Every main claim M1–M20 has a named artifact. No main claim rests on a history file,
on the conversation record, or on a number that has not been read from a result file.
Claims M2–M9 rest on artifacts recovered and hash-verified on 2026-08-11; claims
M10–M20 on artifacts that were already local and were re-read for this map.
