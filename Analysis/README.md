# Analysis 원고와 재현 자료

[상세 원고](03_Analysis_원고.md)는 3.1 ~ 3.4의 상세 본문·그림 5장·코드 설명을 담는다. 3.1은 생산량 전환과 전력 자체의 구간 전환을 연결한다. [3페이지용 압축본](03_Analysis_3페이지용_압축본.md)은 상세 3.1·3.2를 압축본 3.1로 합치고 날씨를 3.2, 최대전력을 3.3으로 배치했다. 압축본의 본문 그림은 2장이다. 백업 자료는 사용하지 않는다.

## 압축본의 근거와 그림

| 압축본 절 | 실행·검증 기록 | 결과표 폴더 | 본문 표현 |
|---|---|---|---|
| 3.1 생산량 전환과 사전 구분 | [A01 전력 변화](10.03_A01_생산량증감과전력변동_분석기록.md), [A02 앞선 전력 관계](10.03_A02_과거전력과다음시간변화_분석기록.md), [A02 전환 검증](10.03_A02_발전_전환사전구분과전력예측_분석기록.md) | `tables/a01_production_changes/`, `tables/a02_prior_power_signals/`, `tables/a02_transition_forecast/` | [전력 변화·적중/오탐 통합 그림](figures/a01_a02_transition_summary.png) |
| 3.2 날씨 정보 검토 | [A03 날씨 관계](10.03_A03_생산조건을고려한날씨전력관계_분석기록.md) | `tables/a03_conditioned_weather/` | 동일 기록의 4행 표. [날씨 그림](figures/a03_weather_conditions.png)은 보조 링크로 유지 |
| 3.3 다음 시간 최대전력 | [A04 최대전력과 선행정보](10.03_A04_다음시간최대전력과선행정보_분석기록.md) | `tables/a04_maximum_conditions/` | [네 구간 비교 그림](figures/a04_terminal_slots.png) |

현재 결과는 관계 분석과 입력 가치의 탐색적 검증이다. 전환 구분 성과를 최대전력 예측 개선으로 대신하지 않는다. A02의 전력 회귀·혼합 실험은 같은 실행 코드와 검증의 의존 자료이며, 실패한 구조를 반복하지 않도록 기록과 함께 남긴다.

전력 구간 전환의 근거는 [A06 집계](tables/a06_power_state_transitions/transition_summary.csv)와 [원자료 대조 결과](tables/a06_power_state_transitions/summary.json)다. EDA 상세 2.2에는 실제 시간순 전력·0 구간 확대, Analysis 상세 3.1에는 [월별 변화량](../Modeling/figures/observed_power_delta_monthly_2021_01_08.png)과 전환 비교를 배치했다. 기존 관측 그림 파일을 재사용하므로 이미지·생성 코드 링크는 `Modeling`을 가리키지만 모델 결과를 뜻하지 않는다.

## 전체 Python 코드와 재현 순서

프로젝트 루트에서 정규 CPython 3.13으로 아래 순서대로 실행한다. Windows 실행은 루트 [AGENTS.md](../AGENTS.md)의 `Start-Process`와 최초 실행 권한 규칙을 따른다. 전력 구간 보완에서는 A06만 새로 계산하고 기존 A01 ~ A04 결과와 관측 그림을 재사용했다. M02는 사용자 지시에 따라 보류 중이다.

1. A01: [a01_production_changes.py](scripts/a01_production_changes.py) → [a01_relative_changes.py](scripts/a01_relative_changes.py) → [plot_a01_production_changes.py](scripts/plot_a01_production_changes.py). 생산 조건별 변화와 상대 변화율을 계산하고 기존 단독 그림을 만든다.
2. A02 관측 구성: [a02_prior_power_signals.py](scripts/a02_prior_power_signals.py) → [a02_signal_diagnostics.py](scripts/a02_signal_diagnostics.py). 연속 세 시간 관측과 선행 관계·수준 조정 방식을 점검한다.
3. A02 시간순 검증: [a02_transition_forecast.py](scripts/a02_transition_forecast.py)에 `validation` → `mixture` → `final` 인수를 순서대로 적용한 뒤 [a02_transition_review.py](scripts/a02_transition_review.py)를 실행한다. 기존 통합 코드의 전력 실험과 시점 검증도 함께 재현한다.
4. A03: [a03_conditioned_weather.py](scripts/a03_conditioned_weather.py) → [a03_weather_diagnostics.py](scripts/a03_weather_diagnostics.py) → [plot_a03_weather.py](scripts/plot_a03_weather.py). 조건·날짜별 날씨 관계와 보조 그림을 만든다.
5. A04: [a04_maximum_conditions.py](scripts/a04_maximum_conditions.py) → [a04_maximum_diagnostics.py](scripts/a04_maximum_diagnostics.py) → [a04_terminal_unique.py](scripts/a04_terminal_unique.py) → [a04_support_diagnostics.py](scripts/a04_support_diagnostics.py) → [plot_a04_terminal_slots.py](scripts/plot_a04_terminal_slots.py). 최대값 조건, 구간별 관계, 평균·최대 중복 정보와 엄격 비교의 예외를 확인한다.
6. 압축 원고 그림: [plot_analysis_transition_summary.py](scripts/plot_analysis_transition_summary.py). 기존 A01·A02 결과표로 서로 다른 표본·지표의 두 패널을 구성한다. [검증 기록](tables/transition_summary_figure_verification.json)에 입력·그림 해시와 표본 수를 남긴다.
7. 원고 배치: [render_analysis_manuscript.py](scripts/render_analysis_manuscript.py). [3페이지용 압축본](03_Analysis_3페이지용_압축본.md)을 읽어 [A4 배치 확인 PDF](03_Analysis_압축_배치확인.pdf)를 만든다. 그래프를 먼저 생성해야 한다.
8. A06 관측 전환: [a06_power_state_transitions.py](scripts/a06_power_state_transitions.py). 원본 CSV에서 전체 정상 시간·구간·시차·지속시간을 계산하고 표준 CSV 순회로 독립 대조한다. 기존 그림과 같은 관측 자료인지 확인하기 위해 `Modeling/tables/m01/hourly_frame.csv`도 대조한다. 새 계산은 M01의 학습·검증 표본이나 모형에 의존하지 않는다. 현재 원고의 시계열 그림 재생성은 [plot_observed_power_timeline.py](../Modeling/scripts/plot_observed_power_timeline.py)의 `main()`에서 수행한다.

공통 원자료 구성은 `a01_production_changes.load()`, 순위 잔차·가중 관계 계산은 `a02_prior_power_signals.residuals()`와 `weighted_corr()`에 의존한다. 원본과 기존 분석 결과는 유지한다. 단독 그림은 통합 그림의 이전 표현과 보조 근거로 보존한다.

## 분량과 편집 판단

2026-10-04 사용자 요청에 따라 별도 압축본의 목차를4개에서3개로, 본문 그림을4장에서2장으로 줄였다. 기존 원고는 상세 내용 그대로 보존했다. 날씨 그림과 표의 중복은 표로 정리하고, 장황한 코드 설명은 이 재현 안내로 모았다. 분석 표본·비교 조건·핵심 수치·엄격 비교의 예외와 분석/예측의 구분은 유지했다. Markdown 전체 글자 수는 10,143자에서 4,466자로 약 56% 감소했다.

위 글자 수와 A4 배치 확인 PDF는 전력 구간 보완 전의 판본이다. 당시 맑은 고딕 11pt·줄 간격 17.6pt(160%)·좌우 20mm·상하 18mm에서 3페이지, 마지막 페이지 사용 높이 약72%였다. 2026-10-04 전력 구간 보완 후에는 Markdown 압축본을 갱신했으며 PDF는 재생성하지 않았다. 현재 압축본의 실제 제출 서식 페이지 수는 미확인이다. 기존 PDF를 최신 원고로 사용하지 않는다.

## 2026-10-04 전력 구간 보완의 근거와 판단

EDA의 평일 평균 20 ~ 26 구간과 실제 시간순 그림에서 출발해, 생산량 전환이 전력의 상승·하락을 얼마나 포괄하는지 질문했다. 구간 경계는 그대로 유지하고 전체 정상 5,784시간·정확한 시차 5,781개를 비교했다. 1차 집계에서는 20 ~ 26 이탈 70시간에 주말의 20 미만 값으로 내려간 6시간이 포함됐다. 이 결과에 따라 양수 20 미만을 분리했으며 관측값·기존 경계는 바꾸지 않았다.

최종 상승 전환 64시간 중 생산량 0→양수는 5시간이었다. 반대 전환도 64시간이고 최대값 변화 중앙값은 각각 +72·−68이다. 26 초과 유지 3,757시간은 전체 최대값 절대 변화량 합계의81.1%여서 전환만으로 오류 분석을 제한할 근거는 없다. 저전력 연속 구간은71개·중앙24시간·최장223시간이며, 전력0은17시간의 단일 사례다. 자세한 정의·예외·표와 해석은 별도 짧은 보고서 대신 상세 원고에 기록하고 두 압축본에 연결했다.

정규 Python 3.13 실행은 종료0, 원자료5,784행·정확한 시차5,781개·기존 그림 입력과의 일치를 독립 대조했다. 결과는7개CSV와해시를 포함한 `summary.json`에 저장한다. 시간·반복 배열의 의존성을 유지한 기술 집계이므로 인과효과·유의성·예측 정확도를 주장하지 않는다. 관측 전환 질문은 여기서 종료하며, 향후 예측이 생성되면 구간별 오류와 마지막값 추가 전후를 비교해 다음 순서를 갱신한다. M02는 시작하지 않았다.

## Modeling 연결과 파일 정리

[후속 질문 점검](10.03_A05_EDA_Analysis_후속질문점검.md)과 [Modeling 계획](../Modeling/README.md)에 예측 이후 확인할 질문을 정리했다. Modeling 계획은 기존 원고3.1~3.4를 참조하며 모델·평가 계획은 바꾸지 않았다. 압축본의 절 번호는 위 대응표로 구분한다.

2026-10-03 사용자 요청으로 철회한 대표 형태·평균 정보손실·실제 평균을 아는 최대값 복원 자료와 임시 이미지·캐시52파일을 삭제했다. 현재 원고의 근거·예외·재현 자료는 보존했다. 이번 압축 회차는 추가 삭제·새 관계 계산·학습 없이 본문과 그림의 표현을 정리했다.

