# covariate-trust-pilot — 프로젝트 전체 기록

작성 2026-08-13. 이 문서는 **진입점**이다. 날짜별 상세는 `history/`에 있고, 여기에는
"무엇을 했고, 지금 무엇이 확정이며, 무엇이 미해결인가"만 둔다.

숫자는 전부 `results/` 아래 실제 아티팩트에서 나온 것이다. 이 문서 자체는 새 수치를
만들지 않는다.

---

## 0. 한 줄 요약

간헐 수요(intermittent demand)를 **직접 조건부평균(Point)** 으로 예측할 것인가
**발생확률 × 양수크기(Hurdle)** 로 분해할 것인가 — 어느 쪽이 언제 유리한지를 통제된
합성 실험으로 특성화하고, 그 결과가 실제 데이터에서 어디까지 전이되는지 그 **경계**까지
보고하는 연구.

```
성격     controlled diagnostic study + empirical transfer study + adaptive-use boundary study
아님     SOTA 예측 논문 / universal Point 논문 / universal Hurdle 논문 /
         새 hurdle 아키텍처 논문 / robust gating 방법 논문
현재     manuscript v2 사람 리뷰 대기 + 실데이터 회수가능성 분석 Gate 1 통과
```

---

## 1. 시간순 궤적

```
08-01        Study 4 budgeted premium forecast slot allocation (이 연구축의 전신)
08-02~08-05  이 저장소 변경 없음 (Stop hook 오탐 3일 연속 기록)
08-07        최대 작업일 (1,733줄).  SBC regime H1 -> H2 두 층 분리 -> seed robustness /
             Favorita transfer / classical benchmark -> Favorita full pool 복원 ->
             structure gate kill test -> Gate-v2 -> expert diversity -> 5-dataset
             external benchmark -> Gate-v3 -> P0L1 temporal robustness
08-08        포스터 2단계 수치 -> Safe-P0L1 shrinkage -> routing information ceiling ->
             raw-history sequence gate -> 포스터 그림
08-09        논문 관점 전면 재정리(실험 0회).  합성 study 원본이 이 머신에 없음을 확인
08-11        합성 원본 회수 완료 -> C1_CONFIRMED -> claim ledger freeze -> outline ->
             figure/table rendering -> Introduction v1~v3
08-12        문헌 novelty 경계 확정(v4~v6) -> Related Work v1~v3 -> Methods/Results/
             Discussion/Conclusion/Abstract -> manuscript v1 -> v2
08-13        실데이터 조건 발견(Gate 0/1) -> 회수가능성 분해 -> oracle family 분리 및
             leakage 교정
```

---

## 2. 확정된 과학적 결과

### 2.1 합성 통제 실험 — C1 CONFIRMED

원본은 별도 저장소(`m5dataset`)에 있었고 08-11에 해시 체인을 끊지 않고 회수했다.
provenance 4개 독립 식별자로 paper DGP를 확정했다.

```
Stage 1   fixed-marginal 2x2x2 factorial, 8 cells (C01~C08), run_20260802_112655
Stage 2   stationary rho sweep, d x rho_I x rho_M = 18 cells, pilot_20260803_051713
공정성    5,856 = 5,856 파라미터, 동일 backbone(DLinear)/trainer/optimizer/30 epoch/
          seed, checkpoint 는 validation realized-y MSE (oracle·test 선택 금지)
평가      exact DP oracle conditional mean, 80 series/cell, paired bootstrap 2000
지표      G = 100(1 - RMSE_H / RMSE_P),  G>0 이면 Hurdle 우세
```

**Stage 1 결과** — marginal 고정 하에서 시간 구조가 상대 성능을 크게 움직이고, 두 축이
상호작용한다.

```
interval_dependence   +7.83 pp [+6.09, +9.45]
magnitude_dependence  -4.58 pp [-6.28, -2.92]
sparsity              -6.26 pp [-8.01, -4.57]
sparsity x interval   +3.35 pp [+1.62, +5.09]
sparsity x magnitude  -0.01 pp [-1.70, +1.69]   0 포함
interval x magnitude -16.74 pp [-18.45,-15.05]  <- 주효과보다 큼
three_way             -1.96 pp [-3.64, -0.31]
```

한계가 결과와 같은 무게로 기록되어 있다: Stage 1의 구조 arm은 **결정론적 교대(rho=-1)**
한 점이라 `CONDITIONALLY_VALID`이며, Stage 1에는 **CI가 0을 벗어난 Point 우위 셀이 없다**
(C08 = -3.01 [-6.79, +0.23]).

**C1 게이트 8개 전부 PASS** (`synthetic_source_verification/c1_gate_report.json`,
verdict = `C1_CONFIRMED`):

```
C1-G1  paper DGP 식별 (run_20260802_112655, factorial_contrasts.csv 를 가진 유일한 run)
C1-G2  marginal control (양 arm 이 같은 2점 support·같은 장기 빈도)
C1-G3  dependence manipulation (Stage1 교대 vs iid, Stage2 rho ∈ {-0.8,0,+0.8})
C1-G4  Point/Hurdle 공정성 (5,856 = 5,856, 동일 trainer·optimizer·budget)
C1-G5  Stage 1 검증 (8 cells x 80 series, 7 contrasts, paired bootstrap 2000)
C1-G6  Stage 2 검증 (18 cells x 80 series, CLASS_A_GENERAL_PREDICTABILITY_SUPPORT)
C1-G7  metric·부호 (exact DP oracle 대비 rmse_mean_truth, gain = 1 - RMSE_h/RMSE_p)
C1-G8  seed·복제 (cell·data seed 당 40 series, data_seeds (0,1), model_seeds (0,1))
```

**Stage 1 component-attribution diagnostic** — 추정 성분 하나를 참값으로 바꿔 오차의
소재를 특정한다. C03(희소·독립 occurrence·구조 magnitude)에서 factorized arm 오차 0.9652
중 occurrence head 가 0.9030, magnitude head 가 0.2900 을 진다. 이는
`COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT` 이지 인과 규명이 아니며, 실데이터에는
대응물이 없다(§2.2 occurrence head skill 없음).

**Stage 2 결과** — 두 축이 서로 다른 성질에 반응한다.

```
abs_rho_I      +0.1904 [+0.1699, +0.2119]     signed rho_I  +0.0667  -> 2.9배
signed rho_M   -0.0711 [-0.0851, -0.0570]     abs_rho_M    -0.0228   -> 3.1배
d              -0.0239   d x rho_I  +0.0332   d x rho_M  -0.0124 (0 포함)
```

즉 **occurrence 축은 의존성의 세기, magnitude 축은 방향**에 반응하고, magnitude
persistence 가 direct 쪽으로 민다. 18셀 중 **정확히 1셀**만 통계적으로 뚜렷한 Point 우위:
`d=8, rho_I=0, rho_M=+0.8` 에서 `G = -19.76 [-26.00, -14.53]`.

### 2.2 실데이터 전이 — C2 SUPPORTED (두 개의 분리된 주장)

```
H1  occurrence-dependence analogue    SUPPORTED_WITH_BOUNDARY
    M5 +0.1064 [+0.0437,+0.1652] · Favorita +0.0789 [+0.0205,+0.1405]
    intermittent regime 에서 강화 +0.1529 [+0.0519,+0.2613]
    "replicates" 아님. EMPIRICAL_ANALOGUE 로만 표기
H2a frozen selector predictive transfer   CONFIRMED
    독립 M5 모집단 675 vs 5,018, -0.0230 [-0.0294,-0.0163], Point win rate +11.87 pp,
    3 seed 재현
H2b isolated mechanism                    NOT_REPLICATED
    overlap 가중 후 +0.0032 [-0.0033,+0.0094].  matching 은 SMD 0.614 로 실패
H3  sparsity interaction                  NOT_REPLICATED / CONSTRUCT_MISMATCH
    M5 -0.0305 [-0.1418,+0.0912] · Favorita -0.0428 [-0.1587,+0.0704]
    합성은 d=4 vs d=8, 외부는 ADI 중앙값(1.304 / 1.317) 분할 -> 다른 construct
occurrence head 실데이터 skill  없음.  Brier skill -0.0084(M5) / -0.0908(Favorita)
```

**실데이터 screen 게이트 8개** (`external_validity_screen/gate_report.json`):

```
G1 data validity        PASS   y = 일 단위 판매량, NaN 0 · 음수 0
G2 no leakage           PASS   모든 descriptor 가 y[start:train_end] 에서만 계산
G3 fair comparison      PASS   Point/Hurdle 이 동일 Split 객체 사용
G4 descriptor validity  PASS   eligibility n_positive_train >= 20 (민감도 15/20/30)
G5 numerical->real 매핑 PASS   rho_interval = 양수 이벤트 간격의 lag-1 autocorrelation
G6 statistical unit     PASS   모든 CI 가 series 를 재표집, 2000 draws
G7 hypothesis freeze    PASS   pre_analysis_spec.json 18:06:38 -> 결과 18:11:16
G8 repository 회귀      PASS_NO_NEW_FAILURES  (기존 실패 2건 유지, 신규 0)
                        WARN: unified_temporal_27_v3 test_loss_com... 기존 실패 존재
```

표본 구성: Stage A 는 **SBC regime-balanced 300/regime × 4 regime × 2 dataset = 2,400
series** 이고 자연분포가 아니다. H2 replication 과 08-13 분석은 **독립 자연 모집단**
(M5 5,693 · Favorita 5,405)을 쓴다. 둘을 섞어 보고하지 않는다.

### 2.3 적응적 활용 경계 — C3 SUPPORTED (음성 결과)

이 축은 **8개 실험의 연속 사슬**이며, 서사는 한 문장으로 요약된다:
**내부 게이트는 계속 통과했는데 외부에서 계속 실패했다.**

판정 사슬 (시간순, 각 판정은 해당 아티팩트에 문자열로 기록되어 있다):

```
#  실험                          판정                                artifact
1  structure_gate kill test      GATE_KILL_YELLOW                    structure_gate/killtest.json
2  gate potential                GATE_POTENTIAL_PASS                 structure_gate/gate_potential.json
3  Gate-v2 OOF                   GATE_V2_OOF_GREEN                   structure_gate/gate_v2_oof_result.json
4  Gate-v2 fresh holdout         GATE_V2_CONFIRM_GREEN               structure_gate/fresh_confirmatory.json
5  expert diversity screen       DIVERSE_GATE_GREEN                  expert_diversity/pair_gate_result.json
6  5-dataset external benchmark  EXTERNAL_VALIDATION_NOT_REPLICATED  multi_benchmark/external_benchmark.json
                                 relative_improvement -0.02428 (= -2.43%)  <- 첫 외부 실패
7  Gate-v3 mechanism             GATE_V3_OOF_STRONG                  gate_v3/gate_report.json
                                 단 anchor 에 근거 없음 (08-07 기록)
8  P0L1 temporal robustness      P0L1_TEMPORAL_STRONG                gate_p0l1_robustness/temporal_gate_report.json
                                 단 근거가 두 dataset 에 몰림
9  Safe-P0L1 shrinkage           SAFE_P0L1_TEMPORAL_MIXED            gate_safe_p0l1/gate_report.json
                                 하방은 지키지만 Fresh 회복 못 함
10 routing information ceiling   FRESH_ROUTING_NOT_RECOVERED         routing_information_ceiling/fresh_critical.json
                                 더 강한 learner 를 줘도 같은 feature 로는 일반화 실패
11 raw-history sequence gate     SEQUENCE_ROUTING_RED                temporal_routing_encoder/gate_report.json
                                 Fresh +2.648% [+2.068,+3.287] 처음 회복 / UCI -193.9%
12 최종                          ROUTING_MODEL_DEVELOPMENT_STOP      사전 등록 stop rule 발동
```

기회 자체는 실재한다: convex oracle **4.11%**, expert diversity 배수 **2.15**
(`structure_gate/convex_oracle.json`, `expert_diversity/expert_set_spec.json`).
그 기회를 배포 가능한 함수로 바꾸는 데 실패한 것이다.

UCI -193.9% 는 원고 본문에 전체 규모로 남아 있다. 축소·각주 처리하지 않았다.
`multi_benchmark` 의 FreshRetailNet-LT / UCI 는 `AVAILABILITY_UNKNOWN` 상태로
`benchmark_spec_v1.json` 에 기록되어 있고, 원고에서는 core validation data 가 아니라
stress test 로만 쓴다.

### 2.4 절대 성능 (범위 정직성)

M5 mean rank: `SBA 3.152 < Croston 3.260 < TSB 3.411 < SES 3.483 < direct 4.202 <
factorized 4.220`. **두 신경망 arm 모두 고전 추정기에 뒤진다.** Abstract·5.4·6.5 세 곳에
명시했다.

---

## 3. 문헌 경계 (08-12 확정)

12개 레코드 전부 Crossref 검증(11 peer-reviewed + 1 preprint). 핵심은 **무엇이 우리 것이
아닌지**를 먼저 못 박은 것이다.

```
이미 prior   ADI/CV² 분류 [SBC05][KH06] · temporal dependence 조작 [ALR12] ·
             occurrence/magnitude 분해 [Cro72][SB05][TSB11] · occurrence-probability
             형태 [TSB11] · neural direct vs decomposed [Kou13] ·
             direct vs product form [NAR26]
주장 안 함   marginal 통제(= 실험 통제이지 발견 아님) · matched 비교 단독
남는 것      occ/mag 별도 축 x representation 교차 · matched-budget finite-sample
             inductive bias · synthetic -> real transfer boundary
```

**영구 guard 3개** (원고에서 절대 어기면 안 되는 표현):

```
LIT-W-KOU13   Kou13 NN-Dual 은 size/interval RATIO.  우리 p x mu PRODUCT 와 동일시 금지
LIT-W-NAR26   NAR26 은 LightGBM(비신경망).  same features != matched capacity.
              capacity/training match 는 NOT STATED = UNKNOWN 이지 "not matched" 아님
LIT-W3        ALR12 전문 미확보(6경로 실패).  무엇을 고정했다는 서술 금지. OPEN
```

---

## 4. 원고 상태

```
현재      manuscript_v2.md  9,727 words  (results/paper_manuscript_verified/)
판정      MANUSCRIPT_V2_HUMAN_REVIEW_PACKAGE_READY
구성      Abstract 254 / 1 Intro 881 / 2 Related Work 1,412 / 3+4.1-4.5 1,969 /
          4.6-4.8 979 / 5.1-5.3 665 / 5.4-5.9 952 / 6 Discussion 1,365 /
          7 Conclusion 277 / captions 973
감사      OVERCLAIM 0 · UNSUPPORTED 0 · UNMAPPED_NUMBER 0 · UNMAPPED_CITATION 0 ·
          NOTATION_CONFLICT 0
WORKING_TITLE  "When Does Factorized Forecasting Help for Intermittent Demand?
                Temporal Dependence, Finite-Sample Behaviour, and Empirical Boundaries"
                (미확정)
```

사용자 결정 대기 4건은 `paper_manuscript_verified/human_review_issue_package.md` B절:
design/result 분리 유지 여부 · 제목 · 6.6 병합 여부 · 5.4 위치.

---

## 5. 실데이터 회수가능성 분석 (08-13, 논문과 별개 축)

"조건을 알아내 모델을 고르면 실제로 이득인가"를 물었다. 재학습 0회, 캐시된 예측만 사용.

```
Gate 0  GREEN   33,294행 = 11,098 series x 3 origins, paired coverage 100%
                origin-oracle headroom  m5 2.80% [2.668,2.945] · fav 3.18% [3.034,3.349]
                winner share  point 33.85% / neutral 21.33% / hurdle 44.82%
Gate 1  YELLOW  component forecastability(A4) 가 shuffled control 과 분리 실패
                (0.218 vs control 0.176).  작동한 것(A5)은 "직전 origin 승자 지속성"
                D_gap 단독은 origin 수준 예측력 사실상 0 (rho ~ 0.01)
회수가능성 분해  headroom 의 약 60%가 series-고정, 40%가 origin-가변
                hard   m5 R_static 0.605 · fav 0.617
                convex m5 0.596 · fav 0.679
정책 (프로토콜 준수 하이퍼파라미터 선택)
                D3 exp-weighted eta=8, per_series_scale
                m5->fav +0.537% [+0.401,+0.677] capture 15.8%
                fav->m5 +0.558% [+0.467,+0.658] capture 18.9%
                50:50 은 안전하지 않음: m5 +0.561% / fav -0.476%, 둘 다 CI 0 제외, 방향 반대
판정    A — Diagnostic GREEN + dynamic method GREEN candidate (확정 불가)
```

---

## 6. 미해결 / 열린 항목

```
[열림]  origin instability vs seed noise 분리
        seed 분산은 0.10% 인데 practical winner 가 36.6% flip.  라벨이 ±2% 경계에
        몰려 취약한 것으로 보이나 [추정] 확증 못 함.  seed1/seed2 체크포인트로
        3-origin 재채점(추론만, 12 pass)하면 답이 나온다.  미실행, 승인 대기.
[열림]  LIT-W3  ALR12 전문 — 기관 구독/ILL.  제출 전 follow-up 이지 blocker 아님
[열림]  R1 single backbone — 최대 reviewer 노출.  second backbone 은 NICE_TO_HAVE
[열림]  venue 미정 -> word limit, BibTeX, figure template, anonymization 전부 대기
[불가]  12-origin full run — frozen split 상 test origin 은 정확히 3개.
        8~12개는 train_end 를 140~252 기간 뒤로 옮겨야 하므로 다른 실험이 된다
[불가]  4-dataset leave-one-dataset-out — FreshRetailNet/UCI 는 paired RMSE 스키마 없음
```

---

## 7. 이번 프로젝트에서 **내가 만들었다가 잡은 결함**

같은 실수를 반복하지 않기 위해 남긴다. 전부 내 산출물의 결함이며 외부 원인이 아니다.

```
UNIT-W1   C_neg / C_pos 를 G(pp) 옆에 인용.  실제로는 rho_M=0 slice 의 절대 delta.
          -> 본문 사용 금지로 고정
FLAG-W2   C_sign pooled significance flag 가 자기 CI 와 모순.  추론에 사용 금지
LIT-W6    NAR26 grade 가 matrix N3 / reference_metadata N2 로 불일치.
          -> EC6 검사 추가로 재발 방지
oracle    convex 정책을 hard oracle 로 나눔.  ORACLE convex 가 capture 105.7% 로
분모      나온 것이 신호였는데 지나침.  -> family 분리
4th       "4번째 origin 존재" 오류.  train_end 이후 112기간에서 validation 28기간을
origin    빼지 않음.  실제 test origin 은 정확히 3개
leakage   exp-weighted 의 eta/normalization 을 held-out 보고 선택.
          -> training dataset 에서만 선택하도록 교정 (교정 후가 오히려 더 좋았음)
절 순서   manuscript v1 이 4.5 -> 5.1 -> 5.3 -> 4.6 으로 역행.
          모든 문자열 게이트를 통과했는데도 존재.  -> v2 에서 교정
캡션      Figure 2 캡션이 blockquote 줄바꿈으로 "not a confidence > interval" 생성
수치      spearmanr 이 상수 예측의 1e-16 부동소수점 잡음 순위를 매겨 0.22 반환
          exp(-eta*L) 양쪽 언더플로 -> 0/0 NaN
결측      S_mag 28% 결측을 missing indicator 없이 중앙값 대치 -> shuffled control 에서도
          희소성 신호 밀반입 (A4_C0 0.2212 > A4 0.1993)
```

공통 교훈: **문자열 게이트는 구조·단위·분모 오류를 못 잡는다.** 잡아준 것은 (a) shuffled
control, (b) family 분리, (c) 실제 순서대로 읽기, (d) 재계산 대조였다.

---

## 8. 어디에 무엇이 있나

```
results/  200.6 MB, 22 디렉터리

  --- 합성 원본 검증 ---
  synthetic_source_recovery/         회수 패키지·전송 매니페스트
  synthetic_source_verification/     DGP·공정성·metric 부호·Stage1/2·H1H2H3 provenance
                                     stage1/2_verified_*.csv 가 모든 합성 수치의 원천

  --- 실데이터 ---
  external_validity_screen/          75 files.  prereg, Stage A, H1/H2/H3, seed robustness,
                                     Favorita transfer/independent, classical benchmark
                                     raw prediction parquet 3종에 origin·step·양 head 예측
  pointhurdle_condition_discovery/   Gate 0/1 (조건 발견 파일럿)
  pointhurdle_recoverability/        oracle family 분해·정책 벤치마크·무결성

  --- routing 축 (STOP) ---
  structure_gate/ gate_v3/ gate_safe_p0l1/ gate_p0l1_robustness/
  routing_information_ceiling/ temporal_routing_encoder/ expert_diversity/
  multi_benchmark/

  --- 논문 ---
  paper_synthesis/                   회수 이전 기록(보존, 미수정)
  paper_synthesis_verified/          claim_ledger_frozen.md  <- 표현의 최종 권위
  paper_outline_verified/            final_outline_freeze.md
  paper_rendering_verified/          figure/table draft + source_map.csv (88 quantities)
  literature_boundary_verified/      novelty_boundary_freeze.md, WARN_FAIL.md,
                                     verify_consistency.py (EC1~EC6 자동 검사)
  paper_methods_verified/            notation_registry.md  <- 기호의 최종 권위
  paper_writing_verified/            48 files.  intro v1~v6, related work v1~v3,
                                     methods/results/discussion/conclusion/abstract
  paper_manuscript_verified/         manuscript_v1/v2 + 맵 4종 + reviewer_attack_audit
                                     + human_review_issue_package.md  <- 사람이 먼저 볼 것

experiments/  12 디렉터리
  external_validity_screen(17 py) 가 실데이터 파이프라인,
  om_factorization_killtest 가 모델·trainer 원천(DGP 는 아님),
  pointhurdle_condition_discovery(9 py) 가 08-13 분석 스크립트
```

**권위 파일 4개** — 충돌 시 이쪽이 이긴다.

```
results/paper_synthesis_verified/claim_ledger_frozen.md        표현·상태
results/paper_methods_verified/notation_registry.md            기호
results/literature_boundary_verified/novelty_boundary_freeze.md 문헌 경계
results/paper_outline_verified/final_outline_freeze.md          구조
```

---

## 9. 지금 이어서 하려면

```
논문 축   human_review_issue_package.md B절 4건 결정 -> venue 선택 -> BibTeX/포맷
회수 축   seed1/seed2 3-origin 재채점 승인 (추론만) -> origin vs seed 분리
공통      commit/push 는 한 번도 하지 않았다.  전부 untracked 신규 디렉터리이며
          기존 추적 파일은 수정하지 않았다.
```
