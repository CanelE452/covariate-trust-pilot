# 2026-08-07 (3) — External-validity SCREEN post-hoc diagnostic

같은 날 세 번째 문서. Stage A 결과에 대한 post-hoc 진단. **재학습 0회, 예측 재생성 0회,
Stage A artifact 수정 0건** (sha256 before == after 로 확인).

## 신규 파일

```
experiments/external_validity_screen/posthoc.py
reports/external_validity_screen/posthoc_diagnostic/
  posthoc_diagnostic.json            전 단계 결과
  posthoc_diagnostic_spec.json       진단 스펙
  posthoc_gate_report.json           G1~G8 + WARN
  h3_synthetic_like_sensitivity.json POSTHOC_SENSITIVITY
```

기존 `reports/external_validity_screen/` 4개 파일은 손대지 않았다.

## 판정: RECOMMEND_STAGE_B_H1

### H1 scale robustness — 6/6 양수, 부호 반전 없음

```
dataset    delta       Spearman   95% CI
m5         raw        +0.1065  [+0.0437, +0.1653]
m5         relative   +0.1231  [+0.0697, +0.1775]
m5         scaled     +0.0966  [+0.0390, +0.1536]
favorita   raw        +0.0789  [+0.0204, +0.1405]
favorita   relative   +0.1178  [+0.0603, +0.1727]
favorita   scaled     +0.0928  [+0.0387, +0.1473]
```

relative 는 오히려 raw 보다 강하다. RMSE_Point <= 1e-6 로 제외된 계열 0개.
train_scale 은 Stage A 학습에 실제 쓴 `train_scale()` 을 그대로 재사용.
-> **H1_SCALE_ROBUST**. scale confound 가능성 낮음.

### H1 confound [탐색]

```
dataset    abs_rho_interval  coef      std coef   95% CI               cond
m5                          +0.3407    +0.0317   [-0.0211, +0.7627]   2.3
favorita                    +0.2248    +0.0168   [-0.3507, +0.9050]   2.1
```

부호는 조정 후에도 양수로 유지되나 CI 가 0 을 포함하고 표준화 계수가 다른 항
(log_ADI, CV2, log_scale) 과 같은 크기다. condition number 가 낮아
`WARN_REGRESSION_UNSTABLE` 은 아니다. 선형 모형이라 delta outlier 에 민감하며
순위 기반 primary 와 다른 질문에 답한다 — primary 로 승격하지 않는다.

### H3 support — 지원은 있으나 사전등록 검정이 다른 구간을 봤다

```
dataset    ADI p50   p90    p95    max     N>=4  N>=6  N>=8   LOW(3~5)  HIGH(>=8)
m5           1.30    4.93   7.07   32.41    171    91    52       127        52
favorita     1.32    3.96   6.24   75.05    119    63    45        84        45
```

두 그룹 모두 >= 30 -> `H3_EXTERNAL_SUPPORT_AVAILABLE` -> **H3_NOT_REPLICATED 유지**.

**단, 중요한 관찰**: Stage A 의 H3 는 ADI **중앙값**(1.30 / 1.32)에서 분할했다.
synthetic 대비는 d=4 vs d=8, 즉 ADI 4 vs 8 이다. 중앙값 분할은 synthetic 대비
구간을 전혀 건드리지 않는다.

`[POSTHOC_SENSITIVITY]` synthetic 대응 구간에서 다시 보면:

```
dataset    median split diff    synthetic-like diff    corr(ADI>=8)  corr(ADI 3~5)
m5              -0.0306              -0.0031             -0.0079        -0.0047
favorita        -0.0427              -0.0007             -0.0409        -0.0402
```

**희소 구간에서는 H1 상관 자체가 0 근처로 사라진다.** 즉 H1 의 양의 신호는
ADI ~1-2 의 준-조밀 계열이 만들고 있고, synthetic 이 모델링한 간헐 구간에서는
관측되지 않는다. n 이 45~127 로 얇아 노이즈가 크지만 두 데이터셋에서 같은 양상이다.

### occurrence gate skill — 기존 해석을 철회하지 않고 오히려 강화

```
dataset    Brier const  Brier Hurdle   BSS        95% CI                판정
m5           0.1802       0.1817     -0.0084  [-0.0411, +0.0241]  WEAK_SKILL
favorita     0.1640       0.1789     -0.0908  [-0.1401, -0.0438]  WORSE_THAN_CONSTANT
```

계열별 BSS>0 비율 m5 36.8% / favorita 30.8%. p_const 는 **train** prevalence 로만
정의했다(test prevalence 미사용).

discrimination gap (mean p_hat|y>0 - mean p_hat|y=0) 은 m5 +0.070, favorita +0.053 으로
**양수** — 판별력이 아예 없는 건 아니고, calibration 이 나빠 전체 Brier 가 상수보다
못하다. Favorita 는 CI 가 0 을 제외해 상수보다 유의하게 나쁘다.

ROC-AUC / PR-AUC / Hurdle log loss 는 **계산 불가** — Stage A 가 계열별 집계만
저장했고 관측치 단위 p_hat 복원은 재학습이 필요하다(금지).

### H2 full-pool expansion

```
dataset    source      eligible    candidate   control    판정
m5         30,490      30,406           714     5,142    H2_EXPANSION_HIGH_VALUE
favorita    1,200       1,195            35       115    H2_EXPANSION_LOW_YIELD
```

적용한 frozen cutoff (Stage A pool 에서 결정론적으로 복원, full pool 에서 재계산 안 함):
m5 `ADI>=1.304, |rhoI|<=0.0074, rhoM>=0.2910` /
favorita `ADI>=1.317, |rhoI|<=0.0088, rhoM>=0.2276`.

**WARN**: Favorita 의 "full pool" 은 이미 1,200 으로 subsample 된 parquet 이다.
`prepare_favorita` 메타데이터는 eligible 56,918 계열을 기록하지만 rho descriptor 를
계산하려면 raw grid 를 물질화해야 해서 이번 범위 밖이다. Favorita 의
LOW_YIELD 판정은 **하한**이지 실제 수율이 아니다.

## Gate

G1~G8 전부 PASS. WARN 5건 (Favorita full pool 미물질화, AUC/log-loss 불가,
adjusted coef CI 0 포함, synthetic 구간에서 H1 소실, ADI>=8 표본 얇음).

## 상태

`EXTERNAL_VALIDITY_POSTHOC_READY_FOR_REVIEW`. Stage B 자동 실행 안 함.
