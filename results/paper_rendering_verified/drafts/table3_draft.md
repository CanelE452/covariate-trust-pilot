# Table 3 draft - core empirical validation datasets (M5 and Favorita only)

```
dataset   series  T      train_end  test origins    lookback/horizon  availability                            sampling                    
--------  ------  -----  ---------  --------------  ----------------  --------------------------------------  ----------------------------
M5        1,200   1,941  1,829      1857/1885/1913  96/28             sell_prices-derived availability mask   SBC-balanced, 300 per regime
Favorita  1,200   1,688  1,576      1604/1632/1660  96/28             raw (loader restricts first day <= 90)  SBC-balanced, 300 per regime
```

Eligibility: n_positive_train >= 20.  Spec frozen 18:06:38, results 18:11:16.
M5 full pool for reference: {'intermittent': 23053, 'lumpy': 5942, 'smooth': 984, 'erratic': 496, 'excluded': 15}
FreshRetailNet-LT and UCI Online Retail II are stress tests for Section 5.7 and
appear only in the appendix dataset table.
