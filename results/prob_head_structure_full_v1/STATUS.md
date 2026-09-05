FINAL RECOMMENDATION: ALL_NEW_METHOD_BRANCHES_NO_GO

## 1. What was attempted

PROB-HEAD-STRUCTURE-FULL-v1 screened whether NB, hurdle-shifted NB and full Tweedie heads specialize by temporal structure, and whether that supports distribution-space distillation (A), structure-conditioned routing (B) or a disagreement sensor (C).

## 2. What was frozen

The authoritative preregistration, its companion hash, the protected-artifact baseline, every gate threshold, every seed and the runtime-tier rule were frozen before any fit.
```
item                          status
────────────────────────────────────
protected_manifest            PASS  
confirmatory_synthetic_cells  14    
unbalanced_dgp_cells          3     
```

## 3. Runtime tier

tier: MINIMAL-COMPLETE

## 4. Environment

```
python   torch        cuda  device                 
───────────────────────────────────────────────────
3.10.20  2.1.1+cu118  11.8  NVIDIA GeForce RTX 4070
```

## 5. Dataset support

```
dataset        eligible_pool  panel_shape    runtime_tier      sampled
──────────────────────────────────────────────────────────────────────
m5             29233          [29290, 1941]  MINIMAL-COMPLETE  1000   
online_retail  2036           [2036, 374]    MINIMAL-COMPLETE  1000   
```

## 6. Numerical likelihood validation

```
comparisons  finite_fraction  zero_relative_error  median_abs_log_difference  branch
────────────────────────────────────────────────────────────────────────────────────
600          1.0              0.0                  8.881784197001252e-16      PASS  
```

## 7. Synthetic specialization

```
cell_id           d  head          rho_I  rho_M  sCRPS              
────────────────────────────────────────────────────────────────────
d4_rI-0.8_rM-0.8  4  HSNB          -0.8   -0.8   0.23531913105780367
d4_rI-0.8_rM-0.8  4  NB            -0.8   -0.8   0.23858717259380344
d4_rI-0.8_rM-0.8  4  TWEEDIE_FULL  -0.8   -0.8   0.26955399572448485
d4_rI-0.8_rM0.0   4  HSNB          -0.8   0.0    0.2522426168317653 
d4_rI-0.8_rM0.0   4  NB            -0.8   0.0    0.2563990612686374 
d4_rI-0.8_rM0.0   4  TWEEDIE_FULL  -0.8   0.0    0.28823082289636837
d4_rI-0.8_rM0.8   4  HSNB          -0.8   0.8    0.20688602371256615
d4_rI-0.8_rM0.8   4  NB            -0.8   0.8    0.21421127905269557
d4_rI-0.8_rM0.8   4  TWEEDIE_FULL  -0.8   0.8    0.24312380472747908
d4_rI0.0_rM-0.8   4  HSNB          0.0    -0.8   0.3327246366738671 
d4_rI0.0_rM-0.8   4  NB            0.0    -0.8   0.3390296520777159 
d4_rI0.0_rM-0.8   4  TWEEDIE_FULL  0.0    -0.8   0.4035356304151587 
d4_rI0.0_rM0.0    4  HSNB          0.0    0.0    0.341343273953532  
d4_rI0.0_rM0.0    4  NB            0.0    0.0    0.34528060028555563
d4_rI0.0_rM0.0    4  TWEEDIE_FULL  0.0    0.0    0.41452597855738127
d4_rI0.0_rM0.8    4  HSNB          0.0    0.8    0.3213651780110709 
d4_rI0.0_rM0.8    4  NB            0.0    0.8    0.32517234001006884
d4_rI0.0_rM0.8    4  TWEEDIE_FULL  0.0    0.8    0.39465107953643763
d4_rI0.8_rM-0.8   4  HSNB          0.8    -0.8   0.24847996492225918
d4_rI0.8_rM-0.8   4  NB            0.8    -0.8   0.25508185732738714
d4_rI0.8_rM-0.8   4  TWEEDIE_FULL  0.8    -0.8   0.3005366217780888 
d4_rI0.8_rM0.0    4  HSNB          0.8    0.0    0.2642767421109936 
d4_rI0.8_rM0.0    4  NB            0.8    0.0    0.2673840111204704 
d4_rI0.8_rM0.0    4  TWEEDIE_FULL  0.8    0.0    0.31664482105256836
d4_rI0.8_rM0.8    4  HSNB          0.8    0.8    0.23912128500887053
d4_rI0.8_rM0.8    4  NB            0.8    0.8    0.24185138357785146
d4_rI0.8_rM0.8    4  TWEEDIE_FULL  0.8    0.8    0.2914120272543684 
d8_rI-0.8_rM-0.8  8  HSNB          -0.8   -0.8   0.14229773032792303
d8_rI-0.8_rM-0.8  8  NB            -0.8   -0.8   0.14471610991779663
d8_rI-0.8_rM-0.8  8  TWEEDIE_FULL  -0.8   -0.8   0.16230956063262175
d8_rI-0.8_rM0.0   8  HSNB          -0.8   0.0    0.15321178184475717
d8_rI-0.8_rM0.0   8  NB            -0.8   0.0    0.15780438333186056
d8_rI-0.8_rM0.0   8  TWEEDIE_FULL  -0.8   0.0    0.17394178838161603
d8_rI-0.8_rM0.8   8  HSNB          -0.8   0.8    0.14502079207249718
d8_rI-0.8_rM0.8   8  NB            -0.8   0.8    0.147111046773844  
d8_rI-0.8_rM0.8   8  TWEEDIE_FULL  -0.8   0.8    0.16790016462386637
d8_rI0.0_rM-0.8   8  HSNB          0.0    -0.8   0.2325345830338158 
d8_rI0.0_rM-0.8   8  NB            0.0    -0.8   0.24296060947511716
d8_rI0.0_rM-0.8   8  TWEEDIE_FULL  0.0    -0.8   0.2781752713771544 
d8_rI0.0_rM0.0    8  HSNB          0.0    0.0    0.23871361641412772
d8_rI0.0_rM0.0    8  NB            0.0    0.0    0.244170523659786  
d8_rI0.0_rM0.0    8  TWEEDIE_FULL  0.0    0.0    0.28465470782901386
d8_rI0.0_rM0.8    8  HSNB          0.0    0.8    0.2528231075740457 
d8_rI0.0_rM0.8    8  NB            0.0    0.8    0.25233691103405254
d8_rI0.0_rM0.8    8  TWEEDIE_FULL  0.0    0.8    0.29474035428207146
d8_rI0.8_rM-0.8   8  HSNB          0.8    -0.8   0.19540799902278908
d8_rI0.8_rM-0.8   8  TWEEDIE_FULL  0.8    -0.8   0.22800245992825963
d8_rI0.8_rM0.0    8  HSNB          0.8    0.0    0.21613354549811953
d8_rI0.8_rM0.0    8  NB            0.8    0.0    0.22589947502298824
d8_rI0.8_rM0.0    8  TWEEDIE_FULL  0.8    0.0    0.23387102360707746
d8_rI0.8_rM0.8    8  HSNB          0.8    0.8    0.21161465780228714
d8_rI0.8_rM0.8    8  NB            0.8    0.8    0.22248796823984202
d8_rI0.8_rM0.8    8  TWEEDIE_FULL  0.8    0.8    0.24200109490714744
```
```
confirmatory_cells  total_cells  cell_oracle_gain        series_origin_oracle_gain  S3_status      best_head_cell_counts                     practical_winner_share                                                                     
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
14                  18           0.00014883250840747042  0.030716368215661927       NOT_EVALUATED  {'HSNB': 17, 'NB': 1, 'TWEEDIE_FULL': 0}  {'HSNB': 0.6107843137254902, 'NB': 0.4352941176470588, 'TWEEDIE_FULL': 0.06127450980392156}
```

## 8. Temporal structure effect

```
NOT_PRODUCED
```

## 9. Real teacher quality

```
dataset        head          seconds             best_epoch
───────────────────────────────────────────────────────────
m5             NB            197.8742914199829   22        
m5             HSNB          166.80423998832703  14        
m5             TWEEDIE_FULL  201.24754691123962  16        
online_retail  NB            184.1995575428009   28        
online_retail  HSNB          112.93111300468445  12        
online_retail  TWEEDIE_FULL  207.29237484931946  24        
```
```
dataset        head          sCRPS              zero_brier           tail_sQL            relative_to_best
─────────────────────────────────────────────────────────────────────────────────────────────────────────
m5             HSNB          9.777219935028333  0.19355305042839377  5.396737479215325   None            
m5             NB            7.845563804829036  0.18073659423351557  6.646977298580728   None            
m5             TWEEDIE_FULL  7.532817210499032  0.1821886034606037   5.1740883316042865  None            
online_retail  TWEEDIE_FULL  4269.61951716686   0.3295994528368457   3834.701868605186   None            
```

## 10. Real complementarity

```
oracle_family  macro_oracle_gain  dataset_oracle_gains  dataset_best_heads
──────────────────────────────────────────────────────────────────────────
HARD           None               None                  None              
```

## 11. CDF pooling

```
dataset  P0    P2_weights       primary_pool  outer_pool_sCRPS   outer_best_single_head  relative_improvement
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
m5       HSNB  [0.0, 1.0, 0.0]  P2            9.777219941093016  TWEEDIE_FULL            -0.2979499811817814 
```

## 12. A distillation

```
dataset  variant  outer_sCRPS        selected_lambda  primary_student
─────────────────────────────────────────────────────────────────────
m5       A0       7.26697084615988   0.0              A3             
m5       A1       7.389259989518947  0.25             A3             
m5       A2       7.372978133720947  0.5              A3             
m5       A3       7.389259989518947  0.25             A3             
```

## 13. B structure-conditioned routing

```
NOT_PRODUCED
```
```
NOT_PRODUCED
```

## 14. C disagreement sensor

```
d  origins  series  shift_type       status
───────────────────────────────────────────
4  16       24      rho_I_positive   SCORED
4  16       24      rho_I_negative   SCORED
4  16       24      rho_M_positive   SCORED
4  16       24      rho_M_negative   SCORED
4  16       24      rho_I_and_rho_M  SCORED
4  16       24      no_change        SCORED
8  16       24      rho_I_positive   SCORED
8  16       24      rho_I_negative   SCORED
8  16       24      rho_M_positive   SCORED
8  16       24      rho_M_negative   SCORED
8  16       24      rho_I_and_rho_M  SCORED
8  16       24      no_change        SCORED
```
```
NOT_PRODUCED
```
```
NOT_PRODUCED
```

## 15. Controls

```
NOT_PRODUCED
```

## 16. Compression/runtime

```
NOT_PRODUCED
```

## 17. Confirmatory vs diagnostic evidence

```
branch                  upstream_required_gates                                  upstream_gate_status                                                                                                                                                                             confirmatory_eligible  scientific_role                          
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
HEAD_SPECIALIZATION     ['DGP_BALANCE', 'S1', 'S2', 'S3']                        {'DGP_BALANCE': 'FAIL', 'S1': 'PASS', 'S2': 'FAIL', 'S3': 'NOT_EVALUATED'}                                                                                                                       False                  DIAGNOSTIC_CONTINUATION_AFTER_DGP_BALANCE
REAL_DISTRIBUTION_POOL  ['R1', 'R2', 'R3']                                       {'R1': 'NOT_EVALUATED', 'R2': 'NOT_EVALUATED', 'R3': 'NOT_EVALUATED'}                                                                                                                            True                   None                                     
A_DISTILLATION          ['R1', 'R2', 'R3', 'A1', 'A2', 'A3', 'A4', 'CONTROL_A']  {'R1': 'NOT_EVALUATED', 'R2': 'NOT_EVALUATED', 'R3': 'NOT_EVALUATED', 'A1': 'NOT_EVALUATED', 'A2': 'NOT_EVALUATED', 'A3': 'NOT_EVALUATED', 'A4': 'NOT_EVALUATED', 'CONTROL_A': 'NOT_EVALUATED'}  True                   None                                     
B_STRUCTURE_ROUTING     ['R1', 'R2', 'B1', 'B2', 'CONTROL_B']                    {'R1': 'NOT_EVALUATED', 'R2': 'NOT_EVALUATED', 'B1': 'NOT_EVALUATED', 'B2': 'NOT_EVALUATED', 'CONTROL_B': 'NOT_EVALUATED'}                                                                       True                   None                                     
C_DISAGREEMENT_SENSOR   ['R1', 'C1', 'C2', 'C3', 'CONTROL_C']                    {'R1': 'NOT_EVALUATED', 'C1': 'NOT_EVALUATED', 'C2': 'FAIL', 'C3': 'NOT_EVALUATED', 'CONTROL_C': 'NOT_EVALUATED'}                                                                                False                  DIAGNOSTIC_CONTINUATION_AFTER_C2         
```

## 18. Gate table

```
DGP_BALANCE  S1    S2    C2  
─────────────────────────────
FAIL         PASS  FAIL  FAIL
```

## 19. Final recommendation

[판정] ALL_NEW_METHOD_BRANCHES_NO_GO
```
NOT_PRODUCED
```

## 20. What must not be claimed

- M5 and the other development datasets are not external confirmation
- a diagnostic continuation is not confirmatory evidence
- raw NLL cannot rank different distribution families
- Tweedie deviance is not the full density
- no continuous target was rounded into a count likelihood
- a teacher pool advantage is not by itself a distillation success
- a disagreement correlation is not a distribution-shift cause
- a synthetic structure effect is not a real-data causal effect
- one seed is not a general effect

## 21. Exact next research action

stop method development on this axis and reconsider the expert set
