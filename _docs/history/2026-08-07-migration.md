# 2026-08-07 — external-validity SCREEN 을 m5dataset 에서 이 저장소로 이관

## 사유

`~/Documents/github/m5dataset` 은 **git remote 가 0개**인 순수 로컬 저장소다
(마지막 커밋 2026-06-25). SCREEN 작업 전체가 백업되지 않은 상태여서, GitHub 에
연결된 이 저장소(`github.com/CanelE452/covariate-trust-pilot.git`)로 옮긴다.

## 무엇을 옮겼나 — 런타임 import 폐포로 결정

패키지 전체가 아니라 **실제로 로드되는 파일만** 옮겼다. 세 진입 모듈을 import 한 뒤
`sys.modules` 를 저장소 경로로 필터링해 폐포를 측정했다 (21개 파일, 172KB).
`unified_temporal_27_v3` 는 전체가 60개 넘는 모듈이지만 SCREEN 이 실제로 끌어오는 건
6개뿐이다.

```
experiments/__init__.py
experiments/external_validity_screen/   {__init__,prereg,screen,cli,figures,posthoc}.py
experiments/om_factorization_killtest/  {__init__,evaluate,models,prereg,train}.py
experiments/unified_temporal_27_v3/     {__init__,conditional_targets,config,model,scenarios,training}.py
rcoi/__init__.py                        src/rcoi 를 __path__ 에 붙이는 shim
src/rcoi/{__init__.py, seed.py}
src/rcoi/models/decomposition.py        MovingAverage
```

데이터:

```
data/processed/series.parquet            2.4M   M5 1200 계열 x 1941일
data/processed/favorita_series.parquet   2.0M   Favorita 1200 계열 x 1688일
data/calendar.csv                        104K   d_* -> wm_yr_wk 매핑
data/sell_prices.csv                     194M   M5 availability 정의에 필요
data/sales_train_evaluation.csv          117M   H2 full-pool descriptor 에 필요
```

결과·문서:

```
results/external_validity_screen/        3.3M  Stage A + posthoc 산출물 17개
_docs/history/m5dataset_2026-08-07*.md    3개  원본 저장소 history 사본
```

## 변경한 것 — 출력 경로 2줄

이 저장소는 `reports/` 가 아니라 `results/` 를 쓰므로 맞췄다. 중복 폴더를 만들지 않기 위함.

```
screen.py:28   Before  OUT = REPO / "reports" / "external_validity_screen"
               After   OUT = REPO / "results" / "external_validity_screen"
posthoc.py:20  Before  OUT = screen.REPO / "reports" / "external_validity_screen" / "posthoc_diagnostic"
               After   OUT = screen.OUT / "posthoc_diagnostic"
```

부수 효과: m5dataset 에서는 `.gitignore` 가 `reports/` 를 통째로 제외해 결과가 **추적조차
안 됐다.** 이 저장소의 `results/` 는 무시 대상이 아니므로 3.3M 산출물이 GitHub 에 백업된다.

## 이관 검증 — 수치 일치로 확인

timeseries 에서 post-hoc 진단을 재실행해 m5dataset 결과와 대조했다 (33초, 재학습 없음).

```
             H1 raw     H1 relative   BSS        full-pool candidate
m5          +0.1065      +0.1231     -0.0084            714
favorita    +0.0789      +0.1178     -0.0908             35
```

원본과 소수 4자리까지 동일. `stage_a_unmodified=True` 도 유지.

## git 안전성

```
data/sell_prices.csv            무시됨  (기존 .gitignore 의 `data/` 규칙)
data/sales_train_evaluation.csv 무시됨
data/processed/*.parquet        무시됨
100MB 초과 추적 파일             없음
새로 추적될 파일                 42개, 합계 3.5M
```

기존 `.gitignore` 의 `data/` + `!data/.gitkeep` 규칙이 대용량 원본을 이미 막고 있어
`.gitignore` 를 수정할 필요가 없었다. GitHub 파일당 100MB 한도에 걸리는 파일 없음.

## 남은 것

- m5dataset 원본은 **삭제하지 않았다** — 양쪽에 존재한다.
- 데이터가 gitignore 대상이라 다른 머신에서 clone 하면 `data/` 를 따로 채워야 한다.
  `series.parquet`·`favorita_series.parquet`·`calendar.csv` (4.5M) 만 있으면 Stage A 재현
  가능하고, `sell_prices.csv`·`sales_train_evaluation.csv` 는 posthoc 5단계(full-pool)와
  M5 availability 계산에만 필요하다.
- commit / push 는 하지 않았다.
