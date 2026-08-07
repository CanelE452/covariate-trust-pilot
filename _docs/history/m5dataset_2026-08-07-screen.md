# 2026-08-07 (2) — External-validity SCREEN Stage A

0단계 감사(같은 날 첫 문서)에 이어, 사용자 승인 후 결정사항을 적용하고 Stage A 실행.

## 감사 결과 정정 — 내 이전 주장이 틀렸다

0단계 감사에서 "M5 선행 0 이 inter-arrival 수열 맨 앞에 거대한 가짜 gap 을 넣어
`rho_interval` 을 오염시킨다"고 보고했으나 **틀렸다.**

`gaps = np.diff(positive_event_indices)` 는 첫 이벤트부터 시작하므로 t=0 과 첫
이벤트 사이 구간은 gap 을 만들지 않는다. 실측으로 확인:

```
descriptor          raw 대비 값이 다른 계열   max|diff|
ADI_train                   470 / 1200        82.60
rho_interval_train            0 / 1200         0.00
rho_magnitude_train           0 / 1200         0.00
```

즉 선행 0 이 바꾸는 건 **ADI 뿐**이고, H1 은 leading-zero 처리에 완전히 불변이다.
영향은 H2/H3 의 HIGH_ADI 분할에만 있다.

## 결정 1 — availability 감사 (사용자 지시로 first-positive trim 을 바로 쓰지 않음)

M5 는 `sell_prices.csv` 가 유일한 availability 신호다. (store_id, item_id) 의 최초
`wm_yr_wk` -> 그 주 첫 day_idx 를 first availability 로 정의.

```
매칭 실패                                      0 / 1200
first_positive - first_availability            median 0, p90 5, max 22일, 음수 0건
미취급 구간 (제거 대상)                         평균 208.9일, >100일 331 계열
availability 이후 진짜 zero demand (보존 대상)  평균 1.4일, 총 1,725일
```

음수가 0건이라는 점이 정의의 내적 정합성을 확인해 준다. **first-positive trim 은 저
1,725 일의 실제 0 수요를 잘못 버리므로**(지시 규칙 4 위반) primary 로 쓰지 않았다.

- M5 primary = `availability_aware`
- Favorita primary = `raw` (prepare_favorita 가 `first_day <= 90` 으로 이미 제한)
- sensitivity = raw / availability_aware / first_positive 3종 모두 계산

## 신규 파일

```
experiments/external_validity_screen/__init__.py
experiments/external_validity_screen/prereg.py     동결 스펙 원본
experiments/external_validity_screen/screen.py     데이터·descriptor·split·평가·H1~H3
experiments/external_validity_screen/cli.py        freeze / stage-a
experiments/external_validity_screen/figures.py    Figure A/B/C
```

출력: `reports/external_validity_screen/` (12개 파일)

### 기존 코드 재사용과 한 곳의 불가피한 복제

`build_splits` 가 `block["y"]`, `block["z"]` 만 요구해 M5 적용이 가능했다.
다만 `train_one` 은 내부에서 `build_splits` 를 호출하므로 availability 마스크가 적용된
split 을 받을 수 없다. 그래서 `cli.train_on_split` 에 **train_one 의 루프를 그대로 복제**
했다 — 시드 순서·optimizer·batch·epoch·patience·checkpoint 기준·predict 호출 모두 동일,
차이는 split 출처뿐. 기존 파일은 수정하지 않았다.

`series_metrics` 도 oracle 키를 무조건 참조해 쓸 수 없어, 같은 `_per_series` reducer 만
가져다 realized-y 지표를 계산했다(집계 방식 동일, 기준만 다름).

### SCREEN 전용 compute 결정

dense daily origin 이면 M5 만 1200 x 1706 = 200만 window (합성의 26배). 주간 stride 7 로
292,800 window 로 낮췄다. Point/Hurdle 동일 적용, test window 미접촉, 예측 생성 전 동결.
confirmatory run 에서는 제거해야 한다.

## Stage A 결과

```
dataset    n      RMSE_P   RMSE_H   MAE_P    MAE_H    mean_delta  median    H승%   P승%
m5        1200    2.9209   2.9215   2.2890   2.1797    -0.0007   -0.0073   47.2   52.8
favorita  1200    5.1124   5.1443   3.3017   3.1660    -0.0320   -0.0028   49.2   50.7
```

전체 평균은 사실상 무승부 (RMSE 기준 Point 미세 우위, MAE 기준 Hurdle 우위).
**전체 평균과 mechanism 은 별개 질문이다.**

```
H1  Spearman(|rho_interval|, delta)      thr=20
  m5        +0.1064  CI [+0.0437, +0.1652]  n 1200   재현, CI 0 제외
  favorita  +0.0789  CI [+0.0205, +0.1405]  n 1195   재현, CI 0 제외
  -> thr 15/30 및 descriptor variant 3종에서 부호 불변

H2  PointCandidate vs Control            thr=20
  m5        cand -0.0223 (n39) vs ctrl +0.0081 (n124)  diff -0.0303  CI [-0.0954,+0.0588]
            Point win rate  cand 0.667  ctrl 0.444
  favorita  cand +0.0510 (n35) vs ctrl +0.0734 (n116)  diff -0.0224  CI [-0.1618,+0.1335]
            Point win rate  cand 0.400  ctrl 0.474      <- win rate 는 반대 방향
  -> 평균 차이 방향은 양쪽 재현, 단 CI 가 모두 0 포함

H3  corr(high_ADI) - corr(low_ADI)       thr=20   예측 > 0
  m5        hi +0.0754  lo +0.1060  diff -0.0305  CI [-0.1418,+0.0912]   반대
  favorita  hi +0.0234  lo +0.0661  diff -0.0428  CI [-0.1587,+0.0704]   반대
  -> 양쪽 데이터셋에서 방향 실패 (variant 3종 모두 동일)
```

Hurdle 진단 (test): m5 Brier 0.1817, p_hat|y=0 0.5615, p_hat|y>0 0.6318 /
favorita Brier 0.1784, 0.5255, 0.5770. **occurrence head 의 판별력이 약하다**
(y=0 과 y>0 에서 p_hat 차이가 0.05~0.07 에 불과).

## Gate

G1~G7 PASS, G8 PASS_NO_NEW_FAILURES (baseline 과 동일하게 2 failed / 412 passed,
실패 2건은 기존 `unified_temporal_27_v3`).

`WARN_DESCRIPTOR_INSTABILITY`·`WARN_LEADING_ZERO_SENSITIVITY` 모두 미발동
(threshold 3종·variant 3종에서 H1/H2/H3 부호 불변).

## 판정

**GREEN — 단, 사전등록된 GREEN 기준이 H1/H2 만 다룬다.** H3 는 양쪽에서 방향 실패이며
이는 GREEN 기준에 포함되지 않은 별도의 부정적 발견이다. 결과를 보고 기준을 바꾸지
않기 위해 GREEN 을 유지하되 H3 실패를 같은 비중으로 보고한다.

상태: `EXTERNAL_VALIDITY_STAGE_A_READY_FOR_REVIEW`. Stage B 3-seed 는 자동 실행하지 않음.
