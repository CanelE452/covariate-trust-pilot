# 사전등록 — Study 4: Budgeted Premium Forecast Slot Allocation

작성: 2026-08-01. **test portfolio loss를 계산하기 전에** 확정한 문서다.
기계가 읽는 사본은 `runs/<run_id>/preregistration.json`이며 SHA-256이 manifest에 기록된다.

기존 Study 3의 사전등록·Gate H/I·D7 threshold는 **수정하지 않는다**. 별도 가설이다.

---

## 0. 시작 상태 (고정)

```
repo         /home/minjae/Documents/github/timeseries   (= covariate-trust-pilot 프로젝트)
branch       main
HEAD         8b655da6dc60ab0463cd8af3b73b6fb71828e502
diff hash    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  (빈 diff)
기존 pytest   238 passed
```

Study 3 artifact의 SHA-256은 `runs/<run_id>/provenance/study3_assets.json`에 기록하고
run 종료 후 재검증한다.

## 1. 질문

매일 07 UTC origin에서 4개 NYISO zone 중 **최대 K개**(K=1, 2)에만 premium forecast를
적용할 수 있을 때, 미래 부하를 보기 전에 대상을 골라 portfolio loss를 줄일 수 있는가.

허용 판정: `BUDGETED_ACQUISITION_NO_GO` / `SIMPLE_RULE_OPERATIONAL_RESULT` /
`VALUE_NOT_PREDICTABLE` / `FORECAST_VALUE_ROUTING_GO` /
`DECISION_VALUE_ACQUISITION_CANDIDATE` / `INCONCLUSIVE`.

## 2. Base와 premium (Study 3 정의 그대로 재사용)

```
M1 (base)     context: load + verified temperature + calendar,  future: calendar only
M3 (premium)  context: M1과 동일,                                future: calendar + 00Z ECMWF forecast temperature
```

`real_vintage.assert_fair_comparison`이 context 동일·future calendar 동일·temperature
컬럼만 차이를 강제한다. Study 4는 이 정의와 기존 forecast cache를 재사용하며
**재추론하지 않는다**.

premium slot = 비싼 weather-conditioned 실행 슬롯(추상 단위 1). 금전가격·GPU 과금·
재고예산이 아니다.

## 3. 기간

```
train                     2024-04-01 ~ 2024-12-31
validation                2025-01-01 ~ 2025-06-30
retrospective test        2025-07-01 ~ 2026-06-30     <- RETROSPECTIVE_HELDOUT_PILOT
fresh confirmation        2026-07-01 ~ 2026-07-31     (>= 20 complete day일 때만 평가)
```

retrospective test 기간의 aggregate M1/M3 결과는 Study 3에서 이미 관찰됐다.
따라서 이를 **untouched confirmation이라고 부르지 않는다**.

## 4. Budget과 정책

`at_most_k`, K ∈ {1, 2}, slot cost 1, abstention 허용,
예측 value가 양수가 아니면 slot을 쓰지 않는다.

Portfolio = 같은 UTC date의 07 UTC origin × 4 zone. **정확히 4개 zone**이어야 하며
한 zone이라도 M1/M3가 결측이면 그날 portfolio 전체를 primary analysis에서 제외하고
제외 사유를 기록한다. 가변 크기 portfolio를 만들지 않는다.

정책: P0 NO_PREMIUM / P1 ALL_PREMIUM(참고) / P2 RANDOM_K(2000회) / P3 ROUND_ROBIN /
P4 BASE_UNCERTAINTY / P5 REVISION_MAGNITUDE / P6 RECENT_BASE_ERROR /
P7 REPORTED_RELIABILITY / P8 RECENT_FULL_INFORMATION_UTILITY /
P9 VALUE_PREDICTOR_WQL / P10 VALUE_PREDICTOR_Q90 / P11 ORACLE_K(upper bound).

P8은 full-information 기준선이며 deployable baseline이 아니다.

## 5. Value 정의

```
V_wql(i,t) = L_base_wql(i,t) − L_premium_wql(i,t)
V_q90(i,t) = L_base_q90(i,t) − L_premium_q90(i,t)
```

`wql_m1`/`wql_m3`는 Study 3 `task_metrics.parquet`에서 그대로 쓴다.
q90 pinball은 Study 3 `predictions.parquet`의 q0.9와 `load_hourly.parquet`의 realized
load로 새로 계산하며, WQL과 같은 정규화(`2·Σpinball / (Σ|y| + ε)`, 단일 quantile)를 쓴다.

train의 V는 label, validation의 V는 모델 선택, test의 V는 평가에만 쓴다.
**현재 origin의 V는 어떤 selector feature에도 들어가지 않는다.**

이 pilot은 full-information offline setting이다 — 과거에는 모든 M1/M3 결과를 안다고 둔다.
partial-feedback/bandit learning은 범위 밖이며 보고서에 명시한다.

## 6. Feature 시점 무결성

허용 feature는 config `features.allowed_current_features` 26개로 고정한다.
전부 07 UTC origin 시점에 이용 가능해야 한다.

금지: origin 이후 load, future weather verification, 현재 realized weather error,
현재 M3 forecast/quantile/loss, 현재 ex-post gain, test-fitted normalization·imputation.

**feature frame에서 금지 컬럼은 존재 자체를 차단**하고 테스트로 검증한다.
결측 history는 `train_zone_month_median`으로만 대체한다(train에서만 적합).

## 7. Value predictor와 선택 규칙

후보: ridge / huber / hist_gradient_boosting / two_part_expected_gain.
two-part는 `P(V>0)·E[V|V>0] − P(V≤0)·E[−V|V≤0]`이며 양·음 표본이 부족하면
**실패 처리하고 다른 후보로 조용히 대체하지 않는다**.

모델 선택은 **validation K=1 portfolio WQL 하나로만** 한다. 선택된 모델은
validation K=2, retrospective test K=1·2, fresh confirmation K=1·2에 그대로 고정한다.
**test에서 다시 선택하지 않는다.**

## 8. 통계

Bootstrap unit은 **ISO calendar week**. 같은 주의 모든 날짜·zone·K·policy를 한 cluster로
resample한다. 5,000 resamples, 95% CI. zone·day를 독립 표본으로 bootstrap 하지 않는다.
Random policy의 무작위 변동(2000회)과 week bootstrap 불확실성을 별도로 보고한다.

## 9. Gate (threshold는 config `gates` 블록에서 고정)

**BA0 무결성** — 4 zone, M1/M3 context/calendar equality, leakage 없음, 시간순 분리,
현재 M3가 feature에 없음, complete portfolio day train≥150·validation≥100·test≥250,
`cross_learning=False`, Study 3 artifact hash 불변. 하나라도 실패 시 `INVALID_PILOT`.

**BA1 headroom** — K=1·2 각각에서 Oracle_K가 NO_PREMIUM 대비 WQL 2%↑, RANDOM_K 대비 1%↑,
CI가 개선 방향, oracle의 단일 zone 선택비율 <80%, premium-positive·negative task 공존.
FAIL: oracle vs random ≤0.5% 또는 한 zone 고정정책이 oracle gain의 95%↑ 회수.

**BA2 단순 heuristic 충분성** — best simple heuristic(P3~P7 중 validation K=1 WQL 최저)의
test oracle recovery ≥90%이고 VALUE_PREDICTOR와 WQL 차이 <1%면
`SIMPLE_HEURISTIC_SUFFICIENT` → 복잡한 value model 기여 No-Go.

**BA3 ex-ante 예측가능성** — P9가 K=1·2 모두에서 RANDOM 대비 1.5%↑, NO_PREMIUM 대비 개선,
oracle headroom 35%↑ 회수, CI 개선 방향, 최소 3개 zone에서 개선 방향,
단일 zone 집중 <80%. FAIL: random 대비 ≤0 또는 recovery ≤15% 또는 CI가 random 우위.

**BA4 decision-specific value** — q90 objective에서 P10이 P9보다 1%↑ 개선,
CI 개선 방향, 선택 overlap <95%, WQL을 1% 초과 악화시키지 않음.

**BA5 fresh** — complete day ≥20일 때만. P9가 random 대비 개선 방향,
oracle recovery ≥20%, P10이 q90에서 개선 방향, budget 위반 없음.
표본이 작으므로 고정 1~2% 임계를 두지 않는다. 20일 미만이면 `NOT_EVALUABLE_LOW_COUNT`.

## 10. 변경 금지

test 결과를 본 뒤 feature, value model 후보, validation rule, K, objective,
Gate threshold, test 기간, zone, abstention rule을 바꾸지 않는다.
구현 오류는 기존 run을 `INVALID_IMPLEMENTATION`으로 보존하고 새 run ID로 재실행한다.
Gate 실패 후 feature·model·threshold를 변경하지 않는다.

## 11. 실행 명령

```
.venv/bin/python -m covariate_trust.cli acquisition-audit  --config configs/study4_budgeted_acquisition.yaml
.venv/bin/python -m covariate_trust.cli acquisition-build  --config configs/study4_budgeted_acquisition.yaml
.venv/bin/python -m covariate_trust.cli acquisition-run    --config configs/study4_budgeted_acquisition.yaml
.venv/bin/python -m covariate_trust.cli acquisition-report --run-dir runs/<study4_run_id>
.venv/bin/python -m covariate_trust.cli acquisition-pilot  --config configs/study4_budgeted_acquisition.yaml
```
