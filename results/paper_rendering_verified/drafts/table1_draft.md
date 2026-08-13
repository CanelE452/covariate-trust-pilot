# Table 1 draft - controlled design and fairness

```
item                   direct                     factorized                 shared / matched?                       
---------------------  -------------------------  -------------------------  ----------------------------------------
input history          lookback 96                lookback 96                matched                                 
forecast horizon       24                         24                         matched                                 
backbone               DLinear                    DLinear                    matched                                 
parameters             5,856                      5,856                      matched by construction                 
optimizer              Adam                       Adam                       matched                                 
learning rate          1e-3                       1e-3                       matched                                 
max epochs / patience  30 / 5                     30 / 5                     matched                                 
batch size             256                        256                        matched                                 
normalization          train split only           train split only           matched                                 
checkpoint criterion   validation realized-y MSE  validation realized-y MSE  identical; oracle and test forbidden    
per-cell tuning        prohibited                 prohibited                 matched                                 
evaluation target      exact DP conditional mean  exact DP conditional mean  matched                                 
seeds                  data (0,1) x model (0,1)   data (0,1) x model (0,1)   matched; model seeds averaged per series
series per cell        80                         80                         matched                                 
```
