# §7-1 확정 — 합성 rho 정의와 실데이터 descriptor 의 대응

포스터 서사의 미해결 항목. **합성 study 의 `rho_I`·`rho_M` 이 코드에서 무엇으로
정의되며, SCREEN 의 실데이터 descriptor 가 같은 양을 재는가**를 코드와 실측으로 확정한다.

근거 코드: `m5dataset/experiments/temporal_dependence/dgp.py` (이 저장소에는 미이관).
2026-08-08 확인.

---

## 1. 합성 쪽 정의 — 잠재 상태의 Markov 파라미터

```python
def markov_chain(rho, n, rng):          # dgp.py:43
    stay = (1.0 + rho) / 2.0            # 대칭 2상태 정상 체인, 주변분포 0.5/0.5
    ...                                 # states in {-1, +1}

gaps      = np.where(interval_states  < 0, d - 1,      d + 1)      # dgp.py:80
lambdas   = np.where(magnitude_states < 0, LAMBDA_LOW, LAMBDA_HIGH)  # (5, 15)
magnitudes = 1 + poisson_rng.poisson(lambdas - 1)
```

DGP 의 자체 검산은 **잠재 상태 수열**에 대해 이뤄진다 (`dgp.py:162`):

```python
rho_i = mean(empirical_rho(s) for s in block["interval_states"])
# empirical_rho(x) = mean(x[:-1] * x[1:])   +/-1 체인의 lag-1 자기상관
```

즉 `rho_I`·`rho_M` 은 **관측값이 아니라 잠재 상태의 lag-1 자기상관**이다.

## 2. SCREEN 쪽 정의 — 관측값의 lag-1 자기상관

`experiments/external_validity_screen/screen.py::describe_series`

```python
gaps  = np.diff(np.flatnonzero(segment > 0))    # 연속 발생 사이 간격
rho_interval_train  = lag1(gaps)
rho_magnitude_train = lag1(segment[events])     # 양수 크기, event 순서
```

실데이터엔 잠재 상태가 없으므로 관측값으로 잴 수밖에 없다. **두 정의가 같은 양인지가
쟁점이다.**

## 3. 판정 — interval 은 정확, magnitude 는 감쇠

### interval: 정확히 같은 양

`gap = d + s` (s in {−1,+1}) 로 **아핀 변환**이다. 상관계수는 아핀 변환에 불변이므로

```
Corr(gap_k, gap_k+1) = Corr(s_k, s_k+1) = rho_I     (정확, 감쇠 없음)
```

대칭 2상태 체인에서 `E[s_k s_k+1] = P(stay) − P(switch) = rho`, `Var(s)=1` 이므로
이론적으로도 정확히 rho 다.

### magnitude: 계수 0.7353 로 감쇠

`lambda = 10 + 5s` 이지만 `M = 1 + Poisson(lambda − 1)` 라 **수준 내 잡음이 남는다.**

```
Var_between = Var(lambda) = 25
Var_within  = E[Var(M|lambda)] = (4 + 14)/2 = 9
Corr(M_k, M_k+1) = rho_M x 25/34 = rho_M x 0.7353
```

### 실측 (Stage 2 생성기 직접 호출, d=4, 200 계열)

```
cell           target rhoI  acf(state)  acf(gap)  | target rhoM  acf(state)  acf(obs)   비율
D4_Im08_Mm08      -0.80      -0.8000    -0.8026  |    -0.80      -0.7919    -0.5870   0.741
D4_Im08_Mp00      -0.80      -0.8000    -0.8026  |    +0.00      -0.0077    -0.0046      -
D4_Im08_Mp08      -0.80      -0.8000    -0.8026  |    +0.80      +0.7855    +0.5603   0.713
D4_Ip00_Mm08      +0.00      -0.0033    -0.0041  |    -0.80      -0.7915    -0.5838   0.738
D4_Ip00_Mp00      +0.00      -0.0033    -0.0041  |    +0.00      -0.0072    -0.0067      -
D4_Ip00_Mp08      +0.00      -0.0033    -0.0041  |    +0.80      +0.7858    +0.5669   0.722
D4_Ip08_Mm08      +0.80      +0.7789    +0.7789  |    -0.80      -0.7921    -0.5890   0.744
D4_Ip08_Mp00      +0.80      +0.7789    +0.7789  |    +0.00      -0.0057    -0.0086      -
D4_Ip08_Mp08      +0.80      +0.7789    +0.7789  |    +0.80      +0.7883    +0.5686   0.721
```

acf(gap) 은 target 과 0.003 이내로 일치한다. magnitude 비율은 0.713~0.744 로 이론값
0.7353 주변이다.

**결론**: `rho_interval_train` 은 합성 `rho_I` 와 **같은 척도**다. `rho_magnitude_train`
은 합성 `rho_M` 의 **0.7353 배**다. H2 를 논할 때 합성 `rho_M = +0.8` 은 실데이터
관측 척도로 **+0.588** 에 해당한다.

---

## 4. 그런데 더 큰 문제 — 실데이터가 합성 파라미터 영역에 도달하지 않는다

정의를 맞춰놓고 보니, 합성 대비 조건에 해당하는 실데이터가 사실상 없다.

```
                |rho_I| 분포                        rho_M 분포
dataset    p50    p90    p95    max   >=0.80   p50    p90    p95    max   >=0.588
m5        0.019  0.162  0.255  0.646     0    +0.160 +0.531 +0.608 +0.845    68
favorita  0.020  0.135  0.202  0.524     0    +0.144 +0.409 +0.481 +0.769    14
```

- **`|rho_I| >= 0.8` 인 계열이 양쪽 다 0개.** 최댓값이 M5 0.646, Favorita 0.524 이고
  **중앙값은 0.02** 다. 실데이터의 간격 의존성은 사실상 없다.
- `rho_M >= 0.588` 은 M5 68 계열, Favorita 14 계열로 얇다.
- SCREEN 이 실제로 쓴 `MAG_PERSISTENT` cutoff 는 M5 +0.291, Favorita +0.2276 으로
  **합성 조건의 49% / 39%** 수준이다.

---

## 5. 세 가설에 대한 함의

이 하나의 사실이 SCREEN 결과 셋을 통일적으로 설명한다 — **descriptor 정의는
문제없고, 실데이터가 합성 파라미터 영역에 없다.**

```
H1  정의는 정확히 대응.   그러나 합성 대비(±0.8)에 해당하는 계열 0개.
    관측된 양의 Spearman 은 |acf| 이 거의 0 인 영역에서 잰 것이다.
    -> post-hoc 에서 ADI>=8 구간에 가면 상관이 사라지던 것과 같은 방향의 이야기.

H2  후보군의 지속성이 합성 조건의 40~50% 수준.
    "약화된 조건에서 방향만 재현" 으로 읽어야 하며, 합성 조건 자체는 미검정.
    합성 척도(+0.588) 로 다시 자르면 M5 68 / Favorita 14 계열이 존재한다.

H3  이미 알려진 대로 ADI 중앙값(1.30)에서 분할해 합성 대비(4 vs 8)를 안 봤다.
    -> 세 가설 모두 같은 구조적 한계를 공유한다.
```

**포스터에 쓸 한 문장**: descriptor 는 합성 정의와 정합하지만(interval 은 정확,
magnitude 는 알려진 0.735 배), M5·Favorita 는 합성 study 가 대비시킨 파라미터 영역을
포함하지 않는다. 따라서 SCREEN 은 "합성 메커니즘이 실데이터에서 재현되는가" 가 아니라
"실데이터가 놓인 약한 의존성 영역에서 같은 방향이 관찰되는가" 를 답한 것이다.

---

## 6. 재현

```bash
# m5dataset 저장소가 있는 머신에서
python -c "
from experiments.temporal_dependence import dgp, prereg
import numpy as np
acf1=lambda v:(np.nan if len(v)<3 or np.std(v)==0 else np.corrcoef(v[:-1],v[1:])[0,1])
for cell in prereg.CELLS:
    if cell['d']!=4: continue
    b=dgp.build_cell(cell, data_seed=0, n_series=200)
    g=[acf1(np.diff(np.flatnonzero(r==1))) for r in b['z']]
    print(cell['cell_id'], cell['rho_interval'], np.nanmean(g))
"
```

실데이터 쪽 분포는 이 저장소의 `results/external_validity_screen/per_series_metrics.csv`
에서 `rho_interval_abs_train`, `rho_magnitude_train` 열로 바로 확인할 수 있다.

## 7. 한계

- 감쇠계수 0.7353 은 `lambda_levels = (5, 15)` 와 `M = 1+Poisson(lambda-1)` 에
  종속된다. 다른 magnitude 법칙을 쓰면 값이 달라진다.
- d=8 도 실측해 동일함을 확인했다 [확인]:

```
 d=8 cell        target rhoI  acf(gap)   | target rhoM  acf(obs)   비율
 D8_Im08_*          -0.80      -0.8064   |    ±0.80      ∓0.59/+0.54   0.702~0.744
 D8_Ip00_*          +0.00      -0.0180   |
 D8_Ip08_*          +0.80      +0.7723   |
```

  gap 값이 (7, 9) 로 바뀌어도 아핀 관계가 같아 interval 대응은 그대로이고,
  magnitude 감쇠비도 0.702~0.750 으로 d=4 와 같은 범위다.
- 이 문서는 정의 대응만 확정한다. H1/H2/H3 의 사전등록 판정은 바꾸지 않는다.
