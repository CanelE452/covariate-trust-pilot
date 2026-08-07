# M5 / Favorita 데이터 유도 사슬

external-validity SCREEN(`experiments/external_validity_screen/`)이 쓰는 데이터가
원본 Kaggle 파일에서 학습 입력까지 어떻게 만들어졌는지 기록한다. 이 저장소로 이관하기
전 `m5dataset` 저장소에서 만들어진 것이며, 여기서 새로 생성하지 않았다.

---

## 1. 원본

```
data/sales_train_evaluation.csv   117M   30,490행 x d_1..d_1941, item x store 단위 일별 판매량
data/sell_prices.csv              194M   (store_id, item_id, wm_yr_wk, sell_price)
data/calendar.csv                 104K   d_* -> date, wm_yr_wk 매핑
```

Kaggle *M5 Forecasting - Accuracy*. `sales_train_evaluation.csv` 를 쓴다
(`sales_train_validation.csv` 아님) — 마지막 28일이 포함된 판이다.

Favorita 원본은 Kaggle *Corporación Favorita Grocery Sales Forecasting*.

---

## 2. 집계 수준과 series 정의

**집계하지 않는다.** M5 원본의 `id` 열(예: `HOBBIES_1_001_CA_1_evaluation`)이 그대로
series_id 이며, 이는 **item x store** 조합이다. dept/cat/state 로의 상향 집계는 없다.

Favorita 는 `item_nbr` + `_` + `store_nbr` 을 series_id 로 쓴다.

---

## 3. 시간 분할 — rolling origin 1번을 고정 사용

`rolling_origin_boundaries(num_periods, validation_horizon=28, test_horizon=28,
num_origins=3, origin_stride=28)` 의 **첫 origin** 을 쓴다.

```
locked_test_start        = T - 28
earliest_validation_start = locked_test_start - 28 - 2*28
train_end                = earliest_validation_start
```

실제 값:

```
             T       train_end   val         test window        날짜
M5         1,941       1,829     [1829,1857) [1857,1941)  train ~2016-01-31,
                                                          val 2016-02-01~02-28,
                                                          test 2016-02-29~2016-05-22
Favorita   1,688       1,576     [1576,1604) [1604,1688)  train ~2017-04-25,
                                                          val 2017-04-26~05-23,
                                                          test 2017-05-24~2017-08-15
```

test window 는 84일 = horizon 28 x 3 origin (stride 28, 겹치지 않음).
SCREEN 은 이 단일 origin 집합만 쓴다. confirmatory run 은 전체 rolling origin 이 필요하다.

---

## 4. series 선정 — SBC regime 층화표본, train 구간만 사용

`src/rcoi/data/prepare_m5.py`. **selection descriptor 는 train 구간에서만 계산**한다
(`train_cols = day_cols[:train_cutoff]`) — 이것이 selection lookahead 를 막는 지점이다.

계열별로 train 구간에서:

```
ADI  = len(train_segment) / n_positive
CV2  = (std(positive, ddof=1) / mean(positive)) ** 2
regime = classify_sbc(ADI, CV2, adi_threshold=1.32, cv2_threshold=0.49)
         ADI<1.32 & CV2<0.49 -> smooth      ADI<1.32 & CV2>=0.49 -> erratic
         ADI>=1.32 & CV2<0.49 -> intermittent  그 외 -> lumpy
```

제외 규칙: `min_positive_train=2`, `min_train_length=184`.

M5 실적:

```
원본 30,490 -> 제외 15 (all_zero_train 10, too_few_positive_train 5) -> eligible 30,475

regime 별 eligible        smooth 984   erratic 496   intermittent 23,053   lumpy 5,942
regime 당 300개 추출      smooth 300   erratic 300   intermittent 300      lumpy 300
                          -> 선정 1,200
sampling_seed = 42, manifest_hash = 5b463fdb77afe2b4
```

Favorita 도 동일 패턴 (`prepare_favorita.py`), eligible 56,918 -> 1,200.
추가 필터로 `first_day <= 90` (full-life series 만) 을 걸어 신제품 출시 전 0 padding 이
intermittency 를 왜곡하지 않게 한다.

> **중요한 편향**: intermittent 가 23,053개인데 300개만 뽑았으므로, 이 1,200 표본은
> M5 의 자연 희소도 분포를 대표하지 않고 **regime 간 균형을 맞춘 것**이다. 그 결과
> 표본 ADI 중앙값이 1.30 으로 낮다. SCREEN 의 H3 가 ADI 중앙값에서 분할했기 때문에
> synthetic 의 d=4 vs d=8 대비 구간을 검정하지 못한 원인이 여기 있다.

산출물: `data/processed/series.parquet` (long format:
`series_id, timestamp, target, item_id, dept_id, category_id, store_id, state_id`).

---

## 5. SCREEN 이 읽는 방식

`experiments/external_validity_screen/screen.py::load_dataset`

```python
frame = pd.read_parquet(...)
wide  = frame.pivot_table(index="series_id", columns="timestamp", values="target").sort_index()
y = wide.to_numpy(np.float32)        # (1200, T)
z = (y > 0).astype(np.float32)       # occurrence indicator
```

검증: NaN 이나 음수가 하나라도 있으면 `ScreenFailure` 로 중단한다. 두 데이터셋 모두 0건.

`build_splits` 가 요구하는 건 `y` 와 `z` 두 배열뿐이라, 합성 study 의 학습 코드를
수정 없이 재사용할 수 있었다.

---

## 6. M5 availability — 선행 0 을 실제 0수요와 구분

M5 에서 상품 도입 전 구간은 **미취급**이지 zero demand 가 아니다. `sell_prices.csv` 가
M5 가 제공하는 유일한 availability 신호다.

```
first_availability(store, item) = calendar 에서 min(wm_yr_wk with a price row) 의 첫 day_idx
```

실측 (`screen.py::m5_availability`):

```
매칭 실패                                       0 / 1200
first_positive - first_availability             median 0, p90 5, max 22일, 음수 0건
미취급 구간 (제거 대상)                          평균 208.9일, >100일 331 계열
availability 이후 진짜 zero demand (보존 대상)   평균 1.4일, 총 1,725일
```

음수가 0건이라는 사실이 정의의 내적 정합성을 확인해 준다. **첫 양수 이전을 무조건
자르는 방식(first-positive trim)은 저 1,725일의 실제 0수요를 잘못 버리므로 쓰지 않았다.**

적용 방식: pre-availability 구간의 **target_mask 를 0 으로** 만들어 학습 손실과 검증에서
제외한다(`_mask_before_availability`). 최대 availability day 는 1,715 로 test 시작
1,857 보다 앞서므로 **test 구간은 전 계열이 판매 가능 상태이고 평가 지표는 영향받지
않는다.**

Favorita 는 `first_day <= 90` 필터 덕에 선행 0 이 구조적으로 90일 이내라 raw 를 그대로 쓴다.

한계: history window 는 여전히 pre-availability 0 을 볼 수 있다. 창 생성이 전 계열 공통
origin 으로 벡터화돼 있어 계열별로 origin 을 다르게 줄 수 없기 때문이다.

---

## 7. 학습 window

```
lookback 96, horizon 28
train origin  dense origins 를 stride 7 (주간) 로 솎음  -> 244 origin x 1200 계열 = 292,800 window
val origin    dense                                     -> 1 origin  x 1200      = 1,200
test origin   stride 28                                 -> 3 origin  x 1200      = 3,600  (t=1857,1885,1913)
normalization train 구간 계열별 평균 (train_scale), train only
```

train stride 7 은 **SCREEN 전용 compute 결정**이다. dense daily 면 1200 x 1706 = 200만
window 로 합성 study 의 26배다. Point/Hurdle 에 동일 적용했고 test window 는 건드리지
않았으며 예측 생성 전에 동결했다. confirmatory run 에서는 제거해야 한다.

---

## 8. 재현

```bash
# 이 저장소에서 (data/ 가 채워져 있어야 함)
python -m experiments.external_validity_screen.cli freeze
python -m experiments.external_validity_screen.cli stage-a
python -m experiments.external_validity_screen.figures
python -m experiments.external_validity_screen.posthoc
```

`data/` 는 `.gitignore` 대상이라 clone 시 비어 있다. 필요한 파일:

```
Stage A 재현에 필수      data/processed/series.parquet, favorita_series.parquet   4.4M
                        data/calendar.csv                                        104K
posthoc 5단계·availability data/sell_prices.csv                                   194M
posthoc 5단계 (M5 full pool) data/sales_train_evaluation.csv                      117M
```

전처리 자체를 다시 돌리려면 `m5dataset` 저장소의 `src/rcoi/data/prepare_m5.py` 와
`prepare_favorita.py` 가 필요하다. 이 저장소에는 이관하지 않았다 — SCREEN 은 완성된
parquet 만 소비하기 때문이다.
