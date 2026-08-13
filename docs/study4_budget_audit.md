# Study 4 시작 상태 audit — Budgeted Premium Forecast Slot Allocation

작성: 2026-08-01. **Study 4 코드를 쓰기 전에** 실제 파일에서 확인한 사실만 적는다.
기억이나 파일명 추측은 쓰지 않았다.

---

## 0. Repository 경로

지시문이 지정한 root `/home/minjae/Documents/github/covariate-trust-pilot`는 **존재하지 않는다**.
실제 covariate-trust-pilot 프로젝트는 다음 경로다.

```
/home/minjae/Documents/github/timeseries
```

근거 [확인]: 패키지가 `src/covariate_trust/`, egg-info가 `src/covariate_trust_pilot.egg-info`,
`configs/study3_real_vintage.yaml`과 Study 0~3 산출물이 모두 이 디렉터리에 있다.
동일 프로젝트로 판단하고 이 경로에서 작업한다.

## 1. 시작 git 상태 [확인]

```
branch      main
HEAD        8b655da6dc60ab0463cd8af3b73b6fb71828e502
status      (clean — 출력 없음)
diff --stat (비어 있음)
diff hash   e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855   (빈 diff의 SHA-256)
```

## 2. 기존 테스트 [확인]

```
.venv/bin/python -m pytest   →   238 passed in 15.84s   (exit 0)
```

`BLOCKED_EXISTING_REGRESSION` 조건에 해당하지 않으므로 진행한다.

## 3. 디스크 사용량 [확인]

```
data    39M      runs    1.1G      .cache  456M      .venv  31M
```

## 4. Study 3 정의 — 실제 파일에서 확인한 것

출처: `configs/study3_real_vintage.yaml`, `src/covariate_trust/real_vintage.py`,
`src/covariate_trust/schemas.py`.

| 항목 | 확인된 값 | 출처 |
|---|---|---|
| decision origin | `decision_origin_hour_utc: 7` | study3 config |
| prediction length | 24 | study3 config |
| quantile levels | 0.1 … 0.9 (9개) | study3 config |
| zones | NYC, LONG_ISLAND, CAPITAL, WEST (4개) | study3 config |
| primary weather run | `primary_run_hour_utc: 0` (00Z) | study3 config |
| revision run | `revision_run_hour_utc: 12` (전일 12Z) | study3 config |
| model | `amazon/chronos-2`, `frozen: true` | study3 config |
| cross_learning | `false` | study3 config |
| context length | 512 | study3 config |

**M1 / M3 정의** (`schemas.py` 상수, `real_vintage.py` docstring) [확인]

```
M1 = "M1_past_covariate_only"
     context: target + verified temperature + calendar,  future: calendar only
M3 = "M3_forecasted_future_covariate"
     context: M1과 동일,                                  future: calendar + 00Z ECMWF forecast temperature
```

**무결성 가드** — `real_vintage.py`에 이미 구현되어 있다 [확인]

- `assert_fair_comparison(in1, in2, in3)` — M1/M2/M3의 `context_df`가 `.equals`로 동일해야 함,
  future timestamp 동일, future calendar 블록 동일, M1은 future temperature 컬럼을 가질 수 없음
- `assert_real_task_invariants` — M1에 future temperature가 있으면 `SchemaError`
- M2(verified future temperature)는 oracle bound이며 method가 아님

Study 4는 이 정의를 **그대로 재사용**하고 어떤 것도 수정하지 않는다.

## 5. 재사용할 Study 3 artifact

Study 3 run: `runs/20260731_123122_real_vintage/`

| 파일 | shape | Study 4에서 쓰는 것 |
|---|---|---|
| `tables/task_metrics.parquet` | (2900, 23) | `wql_m1`, `wql_m3`, `nmae_m1`, `nmae_m3` → **value label** |
| `tables/origin_metadata.parquet` | (2900, 24) | `revision_rms`, `mean_load_mw`, temp 통계, `e_*` |
| `tables/proxy_{train,validation,test}.parquet` | 956 / 656 / 1288 | `reported_reliability_ratio` |
| `predictions/predictions.parquet` | (278400, 16) | M1 quantile → interval width, **q90 pinball** |
| `manifest.json` | — | model id/revision, preregistration hash |

**핵심**: `task_metrics.parquet`에 `wql_m1`과 `wql_m3`가 **이미 계산되어 있다** [확인].
따라서 Study 4는 **M1/M3 재추론이 필요 없다**. 지시문 §4 우선순위 1(기존 processed data와
prediction cache 재사용)에 해당한다.

`q90 pinball`은 `task_metrics`에 없으므로 `predictions.parquet`의 `q0.9`와
`data/processed/load_hourly.parquet`의 realized load로 새로 계산한다 [확인].

## 6. 데이터 커버리지 [확인]

```
task_metrics origin      2024-04-23 07:00Z ~ 2026-06-29 07:00Z
origin 수                2900 = 725 x 4 zones   (모든 zone이 동일 개수)
load_hourly              2024-04-01 04:00Z ~ 2026-07-01 03:00Z
weather_runs_v2_07utc    origin 2024-04-01 ~ 2026-06-30, valid_time ~ 2026-07-01 06:00Z
```

Study 3의 split 경계에서 실측한 complete portfolio 후보일 수:

```
split         기간                        rows   /4 = days
train         2024-04-23 ~ 2024-12-17      956      239
validation    2025-01-01 ~ 2025-06-28      656      164
test          2025-07-01 ~ 2026-06-29     1288      322
합계                                      2900      725
```

BA0의 최소 요구(train 150 / validation 100 / test 250)를 **모두 충족한다** [확인].
단 실제 complete portfolio day 수는 결측 zone 제외 후 Study 4 build 단계에서 다시 센다.

### Fresh confirmation (2026-07-01 ~ 2026-07-31)

**로컬 asset이 없다** [확인]. `load_hourly`는 2026-07-01 03:00Z까지, `weather_runs`는
valid_time 2026-07-01 06:00Z까지다. 즉 7월 portfolio day는 0개다.

따라서 fresh confirmation은 Study 4의 config·feature·model·Gate가 고정된 뒤에
새로 다운로드해야 하며(지시문 §14), 20일 미만이면 `NOT_EVALUABLE_LOW_COUNT`로 보고한다.
데이터가 없다고 해서 기간을 조용히 바꾸지 않는다.

## 7. Study 4가 건드리지 않는 것

- `configs/study3_real_vintage.yaml`, `src/covariate_trust/real_vintage.py`,
  `followup_gates.py`, `external_gates.py` 등 기존 Study 0~3 코드
- D7 threshold (`proxy.lower_threshold 0.75`, `upper_threshold 1.25`) — 재사용도 재튜닝도 하지 않음
- Gate H / Gate I 결과 (`runs/20260731_123122_real_vintage/tables/gate_h.json`, `gate_i.json`)
- 기존 run 디렉터리와 `data/` 전체 (read-only 참조만)

Study 4는 절대 reliability threshold를 쓰지 않는다. `reported_reliability_ratio`는
D7의 threshold 규칙이 아니라 **연속 feature 하나로만** 사용한다.

## 8. 기록한 hash

Study 3 artifact의 SHA-256은 run 시작 시
`runs/<study4_run_id>/provenance/study3_assets.json`에 기록하고, run 종료 후 재검증한다.
