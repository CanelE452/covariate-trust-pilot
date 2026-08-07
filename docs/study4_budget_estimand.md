# Study 4 estimand — 무엇을 재고 무엇을 재지 않는가

## 묻는 것

매일 07 UTC origin에서 4개 NYISO zone 중 **최대 K개**에만 premium forecast(M3)를
적용할 수 있을 때, 미래 부하를 보기 전에 대상 zone을 골라 portfolio forecast loss를
줄일 수 있는가.

Primary estimand는 K=1, K=2 각각에서

```
Δ(policy) = Loss_portfolio(NO_PREMIUM) − Loss_portfolio(policy)
```

이고, portfolio loss는 그날 4개 zone task loss의 평균, primary objective는 WQL이다.

## premium slot이 뜻하는 것 / 뜻하지 않는 것

**뜻하는 것**: 비싼 weather-conditioned 모델 실행 슬롯, premium forecasting pipeline
capacity, 제한된 expert forecast 처리 슬롯.

**뜻하지 않는 것**: 실제 ECMWF API 구매가격, GPU 과금액, 재고 구매 예산,
여러 기간에 걸친 재고상태 budget. slot cost는 **추상 단위 1**이다.

## 왜 공정한 평가가 가능한가

M1을 쓰든 M3를 쓰든 **실제 미래 NYISO load는 바뀌지 않는다**. 따라서 모든 정책을
동일한 realized future load에 대해 평가할 수 있고, 정책이 미래 target을 바꾸는
feedback이 없다. 이것이 이 pilot이 재고 시뮬레이션과 근본적으로 다른 점이다.

## 판정 순서 — 막히면 거기서 끝

```
BA1  budget headroom이 실제로 있는가        없으면 BUDGETED_ACQUISITION_NO_GO
BA2  단순 heuristic으로 충분한가            충분하면 NEW_VALUE_MODEL_NO_GO
BA3  ex ante로 value를 예측할 수 있는가      못하면 VALUE_NOT_PREDICTABLE
BA4  decision-specific value가 추가 가치가 있는가
BA5  소규모 fresh 방향 확인 (있으면)
```

value predictor를 성공시키는 것이 목적이 아니다. BA1이나 BA2에서 막히는 것도
정상적인 결론이다.

## 왜 Study 3와 다른 질문인가

Study 3(D7)은 **각 task에서 독립적으로** future weather를 쓸지 말지를 절대
reliability threshold로 판단했다. Study 4는 **여러 zone이 하나의 일일 슬롯 예산을
경쟁**하는 문제이고, 절대 threshold가 아니라 **M3가 M1보다 얼마나 좋아질지의 상대
순위**만 필요하다. 4개 중 최선 1개를 고르는 데는 threshold가 필요 없다.

Study 3의 Gate H/I 결과와 D7 threshold는 재평가도 재튜닝도 하지 않는다.
`reported_reliability_ratio`는 D7의 규칙이 아니라 **연속 feature 하나**로만 쓴다.

## Full-information offline setting (중요한 제한)

과거 train 기간에 대해 base와 premium을 **모두** 계산해 supervised value label을
만든다. 실제 배포에서는 선택하지 않은 zone의 premium 결과를 볼 수 없으므로
partial-feedback(bandit) 문제가 되지만, 그것은 이 pilot의 범위 밖이다.

따라서 P8(최근 28일 실제 V 평균)은 **deployable baseline이 아니다**. full-information
기준선이라고 명시해 부른다.

## 교란 요인과 처리

| 교란 | 처리 |
|---|---|
| selector가 현재 M3를 훔쳐볼 위험 | feature frame에서 **컬럼 존재 자체를 차단**하고 테스트로 검증 |
| oracle이 사실상 한 zone 고정일 위험 | BA1.4에서 zone 선택 집중도 80% 상한 |
| 단순 규칙으로 충분할 위험 | BA2를 BA3보다 **먼저** 판정 |
| test로 모델을 고르는 위험 | validation K=1 WQL 하나로만 선택하고 test에서 고정 |
| zone/day를 독립 표본으로 착각 | ISO calendar week cluster bootstrap |
| retrospective test가 새 데이터가 아님 | `RETROSPECTIVE_HELDOUT_PILOT`이라 부르고 fresh와 분리 보고 |

## 이 pilot이 답하지 않는 것

- 실제 금전적 절감 [미검증]
- 다기간 재고·capacity state 하의 가치 [미검증]
- partial-feedback / online exploration [미검증]
- 4개 zone·단일 정보원(future temperature)·단일 모델(Chronos-2) 밖으로의 일반화 [미검증]
- q90 pinball은 one-shot asymmetric reserve objective의 proxy일 뿐, 다기간 inventory
  decision의 증거가 아니다
