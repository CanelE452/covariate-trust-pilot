# External-validity SCREEN — 현재 상태

> **새 세션은 이 파일부터 읽으세요.** 나머지 문서는 여기서 링크합니다.
> 최종 갱신 2026-08-07. 상태 `EXTERNAL_VALIDITY_POSTHOC_READY_FOR_REVIEW`.

## 한 줄 요약

합성 study(`m5dataset` 저장소)에서 찾은 Point 대 Hurdle 상대우위 조건이
M5·Favorita 두 공개 벤치마크에서도 같은 방향인지 확인하는 **SCREEN**.
새 방법 개발이 아니라 외적 타당성 확인이며, 아직 확정적 결론 단계가 아니다.

## 진행 상황

```
[완료] 0단계  저장소·데이터 감사
[완료] Stage A  M5 + Favorita x Point + Hurdle, canonical seed 1개        판정 GREEN
[완료] post-hoc 진단  재학습 없이 scale/confound/support/gate skill       판정 RECOMMEND_STAGE_B_H1
[완료] m5dataset -> 이 저장소로 이관 (수치 일치 검증됨)
[미실행] Stage B  3-seed 견고성 — 사용자 승인 대기
[미실행] H2 확대 실험 — M5 full pool candidate 714 (재학습 필요)
```

## 결과 (바꾸지 말 것 — 사전등록됨)

`delta = RMSE_Point − RMSE_Hurdle` (realized y). **양수 = Hurdle 우위.**

```
가설  내용                                    M5              Favorita        판정
H1   |rho_interval| ↑ -> Hurdle 우위 ↑    +0.1065*        +0.0789*        재현 (양쪽 CI 0 제외)
H2   high ADI + weak occ + persist mag   -0.0303         -0.0224         방향만 재현 (CI 0 포함)
     -> Point 방향                        cand -0.0223    cand +0.0510
H3   희소할수록 occ 효과 강화              -0.0305         -0.0428         재현 실패 (양쪽 반대)

전체 평균: 사실상 무승부. RMSE 는 Point 미세 우위, MAE 는 Hurdle 우위.
```

post-hoc 진단:

```
H1 scale robustness   raw/relative/scaled 6개 전부 양수, 부호 반전 없음 -> H1_SCALE_ROBUST
H1 confound [탐색]     조정 후에도 부호 유지, 단 CI 0 포함, 표준화 계수 작음
H3 support            LOW(ADI 3~5)/HIGH(>=8) 둘 다 >=30 -> H3_NOT_REPLICATED 유지
occurrence gate       BSS M5 -0.008 / Favorita -0.091 -> 상수 baseline 대비 skill 없음
H2 확대 여력           M5 candidate 714 / control 5,142 -> HIGH_VALUE
                      Favorita 35 / 115 -> LOW_YIELD (단, full pool 미물질화라 하한)
```

## 반드시 알아야 할 세 가지 함정

1. **H1 은 견고하지만 희소 구간에서 사라진다.** ADI>=8 에서 상관이 M5 −0.008,
   Favorita −0.041 로 0 근처다. H1 의 양의 신호는 ADI 1~2 의 준-조밀 계열이 만든다.
   즉 synthetic 이 모델링한 간헐 구간에서는 관측되지 않는다.

2. **H3 검정이 synthetic 대비 구간을 보지 않았다.** Stage A 는 ADI **중앙값**(1.30)에서
   분할했는데 synthetic 은 d=4 vs d=8, 즉 ADI 4 vs 8 대비다. 표본이 SBC regime
   균형표본이라 ADI 중앙값이 낮은 게 원인. 자세한 건 `m5_favorita_data_derivation.md` §4.

3. **합성 delta 와 실데이터 delta 는 숫자를 비교하면 안 된다.** 합성은 exact DP oracle
   대비 RMSE, 실데이터는 realized y 대비 RMSE. **방향만** 비교한다.

## 동결된 것 (결과 보고 바꾸면 안 됨)

`results/external_validity_screen/pre_analysis_spec.json` — 예측 생성 전 동결
(spec 18:06:38 < results 18:11:16 로 확인됨).

```
estimand        delta = RMSE_Point − RMSE_Hurdle on realized y
eligibility     n_positive_train >= 20 (sensitivity 15/30)
subgroup cutoff HIGH_ADI = dataset median / LOW_OCC = |rho_I| lower tertile
                MAG_PERSISTENT = signed rho_M upper tertile
bootstrap       series 단위 2000 draws, seed 20260807
models          M0PM_point_mse_param_matched vs M1_factorized_mean, model seed 0
split           M5 train_end 1829 val_end 1857 / Favorita 1576, 1604, horizon 28, lookback 96
train stride    7 (SCREEN 전용 compute 결정, confirmatory 에서는 제거)
```

## 다음에 할 수 있는 것

**A. Stage B — 3 seed 견고성** (권고됨, 저비용)
Stage A 를 model seed 3개로 반복해 H1 이 seed 에 안정적인지 확인.
비용 약 9분(GPU). 단, H1 이 이미 CI 0 제외라 얻는 정보가 크지 않을 수 있다.

**B. H2 확대 실험** (정보량 큼, 재학습 필요)
M5 full pool 의 candidate 714 / control 5,142 로 H2 를 제대로 검정. Stage A 는
n=39 라 CI 가 너무 넓었다. frozen cutoff 는 이미
`posthoc_diagnostic.json > datasets.m5.frozen_cutoffs_recovered` 에 있다.

**C. 함정 1 추적** — H1 이 왜 희소 구간에서 사라지는지.
occurrence gate 가 skill 이 없다는 진단과 직접 연결된다. 가장 과학적으로 중요하지만
설계가 필요하다.

> 우선순위는 미팅에서 정하기로 되어 있다. **자동 실행하지 말 것.**

## 파일 지도

```
experiments/external_validity_screen/
  prereg.py     동결 스펙 원본        screen.py   데이터·descriptor·split·평가·H1~H3
  cli.py        freeze / stage-a      figures.py  Figure A/B/C
  posthoc.py    post-hoc 진단

results/external_validity_screen/
  pre_analysis_spec.json      동결 스펙
  stage_a_results.json        Stage A 전체 (manifest + H1/H2/H3 + variant 민감도)
  per_series_metrics.csv      계열별 descriptor + 지표 + delta (2400행)
  gate_report.json            Stage A G1~G8
  figA/B/C.{png,svg,pdf}
  posthoc_diagnostic/
    posthoc_diagnostic.json           전 단계 결과
    posthoc_gate_report.json          G1~G8 + WARN 5건
    h3_synthetic_like_sensitivity.json
    posthoc_diagnostic_spec.json

docs/m5_favorita_data_derivation.md   원본 CSV -> 학습 텐서 전 과정
_docs/history/m5dataset_2026-08-07-screen.md    Stage A 상세
_docs/history/m5dataset_2026-08-07-posthoc.md   post-hoc 상세
_docs/history/2026-08-07-migration.md           이관 내역
```

## 재현

```bash
python -m experiments.external_validity_screen.cli freeze     # 이미 동결됨, 덮어쓰지 않음
python -m experiments.external_validity_screen.cli stage-a    # 약 3분 (GPU)
python -m experiments.external_validity_screen.figures
python -m experiments.external_validity_screen.posthoc        # 약 33초, 재학습 없음
```

`data/` 는 gitignore 대상이라 clone 시 비어 있다. 필요한 파일 목록은
`m5_favorita_data_derivation.md` §8 참조. Stage A 재현에는 4.4M 만 있으면 된다.

## 알려진 WARN

```
Favorita full pool 미물질화        H2 LOW_YIELD 판정은 하한이지 실제 수율 아님
ROC-AUC / PR-AUC / Hurdle logloss  Stage A 가 관측치 단위 p_hat 미저장 -> 계산 불가
H1 adjusted coefficient CI 0 포함  선형 모형, 탐색적
ADI>=8 표본 얇음                   M5 52 / Favorita 45
validation 단일 origin (28일)      early stopping 이 얇음
기존 테스트 실패 2건               unified_temporal_27_v3, SCREEN 과 무관한 기존 실패
```
