FINAL RECOMMENDATION: RECOMMEND_CHARACTERIZATION_ONLY

## 1. What was attempted

PROB-HEAD-STRUCTURE-FULL-v1 screened whether NB, hurdle-shifted NB and full Tweedie heads specialize by temporal structure, and whether that supports distribution-space distillation (A), structure-conditioned routing (B) or a disagreement sensor (C).

## 2. What was frozen

The authoritative preregistration, its companion hash, the protected-artifact baseline, every gate threshold, every seed and the runtime-tier rule were frozen before any fit.
```
item                          status
────────────────────────────────────
protected_manifest            PASS  
confirmatory_synthetic_cells  17    
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
dataset  panel_shape    eligible_pool  sampled  sampling                                                        
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
m5       [29290, 1941]  29233          1000     train-only stratified quantile bins under the preregistered seed
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
confirmatory_cells  total_cells  cell_oracle_gain        series_origin_oracle_gain  best_head_cell_counts  practical_winner_share                                                                     
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
17                  18           0.00012052142868357318  0.029889632723719917       {'HSNB': 16, 'NB': 1}  {'HSNB': 0.6107843137254902, 'NB': 0.4352941176470588, 'TWEEDIE_FULL': 0.06127450980392156}
```

## 8. Temporal structure effect

```
effect                factor  high_level  high_mean_gap        low_level  low_mean_gap         pair                
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
5.951011141929179     d       8.0         -9.729300622096336   4.0        -15.680311764025515  NB_vs_TWEEDIE_FULL  
4.824692513570454     d       8.0         -12.05280147912418   4.0        -16.877493992694635  HSNB_vs_TWEEDIE_FULL
2.294754940443514     d       8.0         4.005395821445009    4.0        1.710640881001495    NB_vs_HSNB          
-1.3368254911907993   rho_I   0.8         -14.275266393381619  -0.8       -12.93844090219082   HSNB_vs_TWEEDIE_FULL
1.2938114338985667    rho_M   0.8         -14.602844506211422  -0.8       -15.896655940109989  HSNB_vs_TWEEDIE_FULL
1.0135582662956       rho_I   0.8         3.84889131829953     -0.8       2.83533305200393     NB_vs_HSNB          
-0.7566143120997628   rho_M   0.8         2.682779428123926    -0.8       3.439393740223689    NB_vs_HSNB          
-0.5229605396654318   rho_I   0.8         -11.75100793972422   -0.8       -11.228047400058788  NB_vs_TWEEDIE_FULL  
-0.05403932061057226  rho_M   0.8         -13.482930058488172  -0.8       -13.4288907378776    NB_vs_TWEEDIE_FULL  
```

## 9. Real teacher quality

```
head          seconds             best_epoch
────────────────────────────────────────────
HSNB          93.99631762504578   14        
NB            218.3116717338562   22        
TWEEDIE_FULL  162.31079769134521  16        
```
```
head          sCRPS              relative_to_best   
────────────────────────────────────────────────────
HSNB          9.777219935028333  0.2979499783163615 
NB            7.845563804829036  0.04151787911355487
TWEEDIE_FULL  7.532817210499032  0.0                
```

## 10. Real complementarity

```
best_global_loss   oracle_family  origin_oracle_gain    origin_oracle_loss  series_oracle_gain    series_oracle_loss
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
7.532817210499032  HARD           0.008793392721318738  7.466578190469205   0.001060636187501629  7.524827631971742 
```

## 11. CDF pooling

```
NOT_PRODUCED
```

## 12. A distillation

```
NOT_PRODUCED
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
changepoint  d  post_zero_rate      pre_zero_rate       series  shift_type       status   
──────────────────────────────────────────────────────────────────────────────────────────
288          4  0.7475405092592593  0.7488425925925926  24      rho_I_positive   GENERATED
288          4  0.7498553240740741  0.7488425925925926  24      rho_I_negative   GENERATED
288          4  0.7507233796296297  0.7488425925925926  24      rho_M_positive   GENERATED
288          4  0.7507233796296297  0.7488425925925926  24      rho_M_negative   GENERATED
288          4  0.7475405092592593  0.7488425925925926  24      rho_I_and_rho_M  GENERATED
288          4  0.7507233796296297  0.7488425925925926  24      no_change        GENERATED
288          8  0.8744212962962963  0.8729745370370371  24      rho_I_positive   GENERATED
288          8  0.8747106481481481  0.8729745370370371  24      rho_I_negative   GENERATED
288          8  0.875               0.8729745370370371  24      rho_M_positive   GENERATED
288          8  0.875               0.8729745370370371  24      rho_M_negative   GENERATED
288          8  0.8744212962962963  0.8729745370370371  24      rho_I_and_rho_M  GENERATED
288          8  0.875               0.8729745370370371  24      no_change        GENERATED
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
branch                  upstream_required_gates            upstream_gate_status                                                        confirmatory_eligible  scientific_role                          
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
HEAD_SPECIALIZATION     ['DGP_BALANCE', 'S1', 'S2', 'S3']  {'DGP_BALANCE': 'FAIL', 'S1': 'PASS', 'S2': 'FAIL', 'S3': 'PASS'}           False                  DIAGNOSTIC_CONTINUATION_AFTER_DGP_BALANCE
REAL_DISTRIBUTION_POOL  ['R1', 'R2', 'R3']                 {'R1': 'FAIL', 'R2': 'FAIL', 'R3': 'FAIL'}                                  False                  DIAGNOSTIC_CONTINUATION_AFTER_R1         
A_DISTILLATION          ['R2', 'R3', 'A1', 'A2']           {'R2': 'FAIL', 'R3': 'FAIL', 'A1': 'NOT_EVALUATED', 'A2': 'NOT_EVALUATED'}  False                  DIAGNOSTIC_CONTINUATION_AFTER_R2         
B_STRUCTURE_ROUTING     ['R2', 'B1', 'B2']                 {'R2': 'FAIL', 'B1': 'NOT_EVALUATED', 'B2': 'NOT_EVALUATED'}                False                  DIAGNOSTIC_CONTINUATION_AFTER_R2         
C_DISAGREEMENT_SENSOR   ['R1', 'C1']                       {'R1': 'FAIL', 'C1': 'NOT_EVALUATED'}                                       False                  DIAGNOSTIC_CONTINUATION_AFTER_R1         
```

## 18. Gate table

```
S1    S2    S3    R1    R2    R3    DGP_BALANCE
───────────────────────────────────────────────
PASS  FAIL  PASS  FAIL  FAIL  FAIL  FAIL       
```

## 19. Final recommendation

[판정] RECOMMEND_CHARACTERIZATION_ONLY
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

write up the characterization result and stop new method development on this axis
