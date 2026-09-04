FINAL VERDICT: ONLINE_MEMORY_NO_GO

## 1. What was tested

[관찰] {"executed_model_seeds":[0],"experiment":"PH-ONLINE-MEMORY-GONO-v1","stage":"FULL_PROTOCOL","transfers":["m5_to_favorita","favorita_to_m5"]}

## 2. What was frozen before results

[관찰] {"frozen_preregistration":{"bootstrap":{"cluster_contents":"all six evaluation origins for a sampled series","confidence_interval_percent":[2.5,97.5],"direction_sampling":"resample series independently inside each target dataset","draws":2000,"macro_sampling":"independently resample M5 and Favorita series clusters within each draw, compute both direction RIs, then average the two RIs","paired_policy_resampling":"all compared policy losses for a sampled series use the same cluster multiplicity within a draw","seed":20260904,"seed_average_rule":"compute policy loss within each model seed, average series-origin losses across seeds, then series-cluster bootstrap; never average predictions","uncertainty_scope":"series uncertainty conditional on the six observed origins; not origin uncertainty","unit":"series cluster"},"conditional_model_seeds":[1,2],"experiment_name":"PH-ONLINE-MEMORY-GONO-v1","frozen_at_utc":"2026-09-04T09:53:17.418099+00:00","frozen_before_any_new_model_fit":true,"model_seed":0,"policy_grids":{"b3_alpha":[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],"b4_eta":[0.5,2.0,8.0,32.0],"b4_half_life_origins":[1,3],"m1_k":[32,128],"m1_lambda_max":[0.25,0.5]},"preregistration_sha256":"2b7462ff0fb85068af0155bcc9014b6fed0c985bf9e1147361c7a0de1ada7ef5","repository":{"branch":"main","dirty":true,"dirty_path_count":36,"git_commit":"222c82cbf91c4313e77c531742831d211c756b02","head_equals_origin_main":true,"path":"E:/CODING/proj/covariate-trust-pilot","remote_origin":"https://github.com/CanelE452/covariate-trust-pilot.git"},"splits":{"favorita":{"evaluation_origins":[1520,1548,1576,1604,1632,1660],"last_interval":[1660,1688],"model_train":[0,1464],"model_validation":[1464,1492],"warmup_interval":[1492,1520],"warmup_origin":1492},"interval_encoding":"half_open","m5":{"evaluation_origins":[1773,1801,1829,1857,1885,1913],"last_interval":[1913,1941],"model_train":[0,1717],"model_validation":[1717,1745],"warmup_interval":[1745,1773],"warmup_origin":1745},"warmup_excluded_from_final_metrics":true}},"terminal_artifact_bindings":{"final_gate_report":{"file_sha256":"3603e4cea3ebe16c4ec6bfaad8b61b29d04a750abf7af55a7166435a533bcfb4","path":"final_gate_report.json","payload_sha256":"b92028b63d811cadeb5740dd61c4e03cb56cf45cb4b66a11ab6b451f8c84befc"},"preregistered_spec":{"file_sha256":"b53ab0b6df02dc1683960fa09f370aba2f944024cef98c20df691a6291a63314","path":"preregistered_spec.json","payload_sha256":"2b7462ff0fb85068af0155bcc9014b6fed0c985bf9e1147361c7a0de1ada7ef5"},"runtime_estimate":{"file_sha256":"d214d220f11b5e9effe37f3e911ce686bd1b96b8b27abdac7466d2e5c9c86f9a","path":"runtime_estimate.json"},"tables_a_to_g":{"file_sha256":"19a7b6a5d780313e2f3c317aaad2e1a81a81b0430ffb727e435c73c16935e1f4","path":"tables_a_to_g.json","payload_sha256":"675b5a34b60485a57f587e70584202eeaaaf83ddbef7f10f413d839f0282f8a9"}}}

## 3. Data and split

[관찰] [{"actual_synchronized_end_to_end_wall_by_seed_seconds":{"0":{"hurdle":1352.4282940998673,"point":1023.269578400068}},"actual_synchronized_end_to_end_wall_seconds":2375.6978724999353,"dataset":"m5","eligible_n":29059,"evaluation_origins":[1773,1801,1829,1857,1885,1913],"executed_model_seeds":[0],"execution_device_type":"cuda","hurdle_params":{"model_id":"M1_factorized_mean","n_parameters":7056},"interval_encoding":"half_open","point_params":{"model_id":"M0PM_point_mse_param_matched","n_parameters":7056},"status":"OBSERVED","train_end":1717,"training_runtime_by_seed_seconds":{"0":{"hurdle":1349.8145625591278,"point":1020.4039692878723}},"training_runtime_seconds":2370.218531847,"validation_interval":[1717,1745],"warmup_origin":1745},{"actual_synchronized_end_to_end_wall_by_seed_seconds":{"0":{"hurdle":3642.700227199821,"point":2033.1338698000181}},"actual_synchronized_end_to_end_wall_seconds":5675.834096999839,"dataset":"favorita","eligible_n":55561,"evaluation_origins":[1520,1548,1576,1604,1632,1660],"executed_model_seeds":[0],"execution_device_type":"cuda","hurdle_params":{"model_id":"M1_factorized_mean","n_parameters":7056},"interval_encoding":"half_open","point_params":{"model_id":"M0PM_point_mse_param_matched","n_parameters":7056},"status":"OBSERVED","train_end":1464,"training_runtime_by_seed_seconds":{"0":{"hurdle":3638.146287202835,"point":2028.035808801651}},"training_runtime_seconds":5666.182096004486,"validation_interval":[1464,1492],"warmup_origin":1492}]

## 4. Runtime

[관찰] [{"actual_synchronized_end_to_end_wall_by_seed_seconds":{"0":{"hurdle":1352.4282940998673,"point":1023.269578400068}},"actual_synchronized_end_to_end_wall_seconds":2375.6978724999353,"dataset":"m5","training_runtime_by_seed_seconds":{"0":{"hurdle":1349.8145625591278,"point":1020.4039692878723}},"training_runtime_seconds":2370.218531847},{"actual_synchronized_end_to_end_wall_by_seed_seconds":{"0":{"hurdle":3642.700227199821,"point":2033.1338698000181}},"actual_synchronized_end_to_end_wall_seconds":5675.834096999839,"dataset":"favorita","training_runtime_by_seed_seconds":{"0":{"hurdle":3638.146287202835,"point":2028.035808801651}},"training_runtime_seconds":5666.182096004486}]

## 5. Expert quality

[관찰] [{"always_hurdle_loss":1.5797869068316117,"always_point_loss":1.7261906985345432,"dataset":"m5","half_half_loss":1.5909909502738298,"origin_convex_oracle_loss":1.4848752107493735,"origin_hard_oracle_loss":1.493513637970448,"source_dataset":"favorita","source_static_alpha":0.55,"source_static_status":"OBSERVED","status":"OBSERVED","target_oracle_static_alpha":0.8,"target_oracle_static_loss":1.5693890396302792},{"always_hurdle_loss":1.8386150244386605,"always_point_loss":1.8467329923807234,"dataset":"favorita","half_half_loss":1.8258155135908642,"origin_convex_oracle_loss":1.7810604333889304,"origin_hard_oracle_loss":1.7828078381853827,"source_dataset":"m5","source_static_alpha":0.8,"source_static_status":"OBSERVED","status":"OBSERVED","target_oracle_static_alpha":0.55,"target_oracle_static_loss":1.8255782001419496}]

## 6. Oracle opportunity

[관찰] {"gains_percent":{"favorita":2.4385570965712478,"m5":5.385142035961699},"heterogeneous_diagnostic":null,"macro_gain_percent":3.911849566266473}

## 7. Temporal recurrence

[관찰] [{"ci95":[0.2156065527607962,0.2317840438152952],"dataset":"m5","gate_1a":"PASS","lag1_spearman":0.22381171915131903,"real_minus_shuffled":0.22597358084622243,"shuffled_spearman":-0.0021618616949033885,"status":"OBSERVED"},{"ci95":[0.3261469673929036,0.33682505909538035],"dataset":"favorita","gate_1a":"PASS","lag1_spearman":0.33143629448671985,"real_minus_shuffled":0.3308780099346586,"shuffled_spearman":0.0005582845520612447,"status":"OBSERVED"}]

## 8. B4 simple memory

[관찰] [{"b3_loss":1.584290739212776,"b4_loss":1.584432999654078,"ri_b4_vs_b3_percent":-0.008979440312373299,"source":"favorita","status":"PARTIAL_OBSERVED_M1_NOT_RUN_AFTER_GATE1B","target":"m5"},{"b3_loss":1.8294491813430234,"b4_loss":1.8145240764275583,"ri_b4_vs_b3_percent":0.8158250618641616,"source":"m5","status":"PARTIAL_OBSERVED_M1_NOT_RUN_AFTER_GATE1B","target":"favorita"}]

## 9. M1 retrieval memory

[관찰] [{"ci95_percent":null,"m1_loss":null,"ri_m1_vs_b3_percent":null,"ri_m1_vs_b4_percent":null,"source":"favorita","status":"PARTIAL_OBSERVED_M1_NOT_RUN_AFTER_GATE1B","target":"m5"},{"ci95_percent":null,"m1_loss":null,"ri_m1_vs_b3_percent":null,"ri_m1_vs_b4_percent":null,"source":"m5","status":"PARTIAL_OBSERVED_M1_NOT_RUN_AFTER_GATE1B","target":"favorita"}]

## 10. Controls

[관찰] [{"control_over_real_ratio":null,"random_neighbor_ri_percent":null,"real_retrieval_ri_percent":null,"shuffled_value_ri_percent":null,"status":"NOT_RUN_AFTER_GATE1B","target":null}]

## 11. Safety

[관찰] {"observed_gate_metrics":{"status":"NOT_RUN_AFTER_GATE1B"},"worst_origin":{"status":"NOT_RUN_AFTER_GATE1B"}}

## 12. Seed robustness

[관찰] {"executed_model_seeds":[0],"observed_gate_metrics":{"status":"NOT_RUN_AFTER_GATE1B"}}

## 13. Gate table

[판정] [{"gate":"GATE0","observed":{"gains_percent":{"favorita":2.4385570965712478,"m5":5.385142035961699},"heterogeneous_diagnostic":null,"macro_gain_percent":3.911849566266473},"pass_fail":"PASS","scientific_interpretation":"Point/Hurdle convex oracle opportunity is sufficient","status":"OBSERVED","threshold":{"each_dataset_gain_percent":">= 1.0","macro_origin_convex_oracle_gain_percent":">= 2.0"}},{"gate":"GATE1A","observed":{"favorita":{"bootstrap_draws":2000,"bootstrap_valid_draws":2000,"gap_above_0_05":true,"n_adjacent_pairs":277805,"passed":true,"real_minus_shuffled_rho":0.3308780099346586,"real_pvalue":0.0,"real_rho":0.33143629448671985,"real_status":"OK","rho_above_0_10":true,"rho_ci95":[0.3261469673929036,0.33682505909538035],"shuffled_rho":0.0005582845520612447,"shuffled_status":"OK"},"m5":{"bootstrap_draws":2000,"bootstrap_valid_draws":2000,"gap_above_0_05":true,"n_adjacent_pairs":145295,"passed":true,"real_minus_shuffled_rho":0.22597358084622243,"real_pvalue":0.0,"real_rho":0.22381171915131903,"real_status":"OK","rho_above_0_10":true,"rho_ci95":[0.2156065527607962,0.2317840438152952],"shuffled_rho":-0.0021618616949033885,"shuffled_status":"OK"}},"pass_fail":"PASS","scientific_interpretation":"past expert advantage recurs temporally in both datasets","status":"OBSERVED","threshold":{"each_dataset_lag1_spearman":"> 0.10","each_dataset_real_minus_shuffled":"> 0.05"}},{"gate":"GATE1B","observed":{"ri_percent":{"favorita":0.8158250618641616,"m5":-0.008979440312373299}},"pass_fail":"FAIL","scientific_interpretation":"simple online combination does not beat source-static in both transfers","status":"OBSERVED","threshold":{"each_transfer_b4_vs_b3_percent":">= 0.30"}},{"gate":"GATE2","observed":{"status":"NOT_RUN_AFTER_GATE1B"},"pass_fail":"NOT_RUN_AFTER_GATE1B","scientific_interpretation":"not evaluated after the prior gate stop","status":"NOT_RUN_AFTER_GATE1B","threshold":{"at_least_one_dataset_ci95_lower_percent":"> 0","each_transfer_m1_vs_b3_percent":">= 0.30","each_transfer_m1_vs_b4_percent":">= -0.10","macro_ci95_lower_percent":"> 0","macro_m1_vs_b3_percent":">= 0.70","macro_m1_vs_b4_percent":">= 0.20"}},{"gate":"GATE3_SAFETY","observed":{"status":"NOT_RUN_AFTER_GATE1B"},"pass_fail":"NOT_RUN_AFTER_GATE1B","scientific_interpretation":"not evaluated after the prior gate stop","status":"NOT_RUN_AFTER_GATE1B","threshold":{"each_dataset_q95_m1_over_b4":"<= 1.01","worst_origin_m1_vs_b3_percent":">= -0.50"}},{"gate":"GATE3_CONTROL","observed":{"status":"NOT_RUN_AFTER_GATE1B"},"pass_fail":"NOT_RUN_AFTER_GATE1B","scientific_interpretation":"not evaluated after the prior gate stop","status":"NOT_RUN_AFTER_GATE1B","threshold":{"random_over_real":"<= 0.50","real_macro_ri_percent":"> 0","shuffled_over_real":"<= 0.25"}},{"gate":"GATE4","observed":{"status":"NOT_RUN_AFTER_GATE1B"},"pass_fail":"NOT_RUN_AFTER_GATE1B","scientific_interpretation":"not evaluated after the prior gate stop","status":"NOT_RUN_AFTER_GATE1B","threshold":{"rule":"frozen conditional seed-0/seed-1/seed-2 sign, retention, and seed-average series-cluster CI rules"}}]

## 14. Final interpretation

[판정] B4 simple online combination이 source-static을 양방향에서 이기지 못했다.

## 15. What must NOT be claimed

[판정] 이 development-dataset 결과를 untouched external confirmatory validation, 최종 논문 성능, 인과 효과, 또는 일반적 우월성으로 주장해서는 안 된다.

## 16. Exact next action

[최종] ONLINE_MEMORY_NO_GO → online memory mechanism 개발 종료
