# Analysis 원고와 재현 자료

[기존 원고](03_Analysis_원고.md)는 3.1~3.4의 상세 본문·그림4장·코드 설명을 보존한다. [3페이지용 압축본](03_Analysis_3페이지용_압축본.md)은 별도 파일이며, 기존3.1·3.2를 압축본3.1로 합치고 날씨를3.2, 최대전력을3.3으로 배치했다. EDA처럼 기존 원고와 압축본을 함께 유지한다. 백업 자료는 사용하지 않는다.

## 압축본의 근거와 그림

| 압축본 절 | 실행·검증 기록 | 결과표 폴더 | 본문 표현 |
|---|---|---|---|
| 3.1 생산량 전환과 사전 구분 | [A01 전력 변화](10.03_A01_생산량증감과전력변동_분석기록.md), [A02 앞선 전력 관계](10.03_A02_과거전력과다음시간변화_분석기록.md), [A02 전환 검증](10.03_A02_발전_전환사전구분과전력예측_분석기록.md) | `tables/a01_production_changes/`, `tables/a02_prior_power_signals/`, `tables/a02_transition_forecast/` | [전력 변화·적중/오탐 통합 그림](figures/a01_a02_transition_summary.png) |
| 3.2 날씨 정보 검토 | [A03 날씨 관계](10.03_A03_생산조건을고려한날씨전력관계_분석기록.md) | `tables/a03_conditioned_weather/` | 동일 기록의 4행 표. [날씨 그림](figures/a03_weather_conditions.png)은 보조 링크로 유지 |
| 3.3 다음 시간 최대전력 | [A04 최대전력과 선행정보](10.03_A04_다음시간최대전력과선행정보_분석기록.md) | `tables/a04_maximum_conditions/` | [네 구간 비교 그림](figures/a04_terminal_slots.png) |

현재 결과는 관계 분석과 입력 가치의 탐색적 검증이다. 전환 구분 성과를 최대전력 예측 개선으로 대신하지 않는다. A02의 전력 회귀·혼합 실험은 같은 실행 코드와 검증의 의존 자료이며, 실패한 구조를 반복하지 않도록 기록과 함께 남긴다.

## 전체 Python 코드와 재현 순서

프로젝트 루트에서 정규 CPython 3.13으로 아래 순서대로 실행한다. Windows 실행은 루트 [AGENTS.md](../AGENTS.md)의 `Start-Process`와 최초 실행 권한 규칙을 따른다. 이번 원고 압축 작업은 기존 결과표를 사용했으며 분석·학습 전체를 재실행하지 않았다.

1. A01: [a01_production_changes.py](scripts/a01_production_changes.py) → [a01_relative_changes.py](scripts/a01_relative_changes.py) → [plot_a01_production_changes.py](scripts/plot_a01_production_changes.py). 생산 조건별 변화와 상대 변화율을 계산하고 기존 단독 그림을 만든다.
2. A02 관측 구성: [a02_prior_power_signals.py](scripts/a02_prior_power_signals.py) → [a02_signal_diagnostics.py](scripts/a02_signal_diagnostics.py). 연속 세 시간 관측과 선행 관계·수준 조정 방식을 점검한다.
3. A02 시간순 검증: [a02_transition_forecast.py](scripts/a02_transition_forecast.py)에 `validation` → `mixture` → `final` 인수를 순서대로 적용한 뒤 [a02_transition_review.py](scripts/a02_transition_review.py)를 실행한다. 기존 통합 코드의 전력 실험과 시점 검증도 함께 재현한다.
4. A03: [a03_conditioned_weather.py](scripts/a03_conditioned_weather.py) → [a03_weather_diagnostics.py](scripts/a03_weather_diagnostics.py) → [plot_a03_weather.py](scripts/plot_a03_weather.py). 조건·날짜별 날씨 관계와 보조 그림을 만든다.
5. A04: [a04_maximum_conditions.py](scripts/a04_maximum_conditions.py) → [a04_maximum_diagnostics.py](scripts/a04_maximum_diagnostics.py) → [a04_terminal_unique.py](scripts/a04_terminal_unique.py) → [a04_support_diagnostics.py](scripts/a04_support_diagnostics.py) → [plot_a04_terminal_slots.py](scripts/plot_a04_terminal_slots.py). 최대값 조건, 구간별 관계, 평균·최대 중복 정보와 엄격 비교의 예외를 확인한다.
6. 압축 원고 그림: [plot_analysis_transition_summary.py](scripts/plot_analysis_transition_summary.py). 기존 A01·A02 결과표로 서로 다른 표본·지표의 두 패널을 구성한다. [검증 기록](tables/transition_summary_figure_verification.json)에 입력·그림 해시와 표본 수를 남긴다.
7. 원고 배치: [render_analysis_manuscript.py](scripts/render_analysis_manuscript.py). [3페이지용 압축본](03_Analysis_3페이지용_압축본.md)을 읽어 [A4 배치 확인 PDF](03_Analysis_압축_배치확인.pdf)를 만든다. 그래프를 먼저 생성해야 한다.

공통 원자료 구성은 `a01_production_changes.load()`, 순위 잔차·가중 관계 계산은 `a02_prior_power_signals.residuals()`와 `weighted_corr()`에 의존한다. 원본과 기존 분석 결과는 유지한다. 단독 그림은 통합 그림의 이전 표현과 보조 근거로 보존한다.

## 분량과 편집 판단

2026-10-04 사용자 요청에 따라 별도 압축본의 목차를4개에서3개로, 본문 그림을4장에서2장으로 줄였다. 기존 원고는 상세 내용 그대로 보존했다. 날씨 그림과 표의 중복은 표로 정리하고, 장황한 코드 설명은 이 재현 안내로 모았다. 분석 표본·비교 조건·핵심 수치·엄격 비교의 예외와 분석/예측의 구분은 유지했다. Markdown 전체 글자 수는 10,143자에서 4,466자로 약 56% 감소했다.

A4, 맑은 고딕 11pt, 줄 간격 17.6pt(160%), 좌우 여백 20mm·상하 18mm에서 실제 3페이지로 배치했다. 마지막 페이지 사용 높이는 약72%로 전체 약2.7페이지 규모다. 본문·그림·표·코드와 근거 안내를 포함한 기준이며 세 페이지를 시각 확인했다. 최종 보고서 서식이 달라지면 페이지 수는 달라질 수 있다.

## Modeling 연결과 파일 정리

[후속 질문 점검](10.03_A05_EDA_Analysis_후속질문점검.md)과 [Modeling 계획](../Modeling/README.md)에 예측 이후 확인할 질문을 정리했다. Modeling 계획은 기존 원고3.1~3.4를 참조하며 모델·평가 계획은 바꾸지 않았다. 압축본의 절 번호는 위 대응표로 구분한다.

2026-10-03 사용자 요청으로 철회한 대표 형태·평균 정보손실·실제 평균을 아는 최대값 복원 자료와 임시 이미지·캐시52파일을 삭제했다. 현재 원고의 근거·예외·재현 자료는 보존했다. 이번 압축 회차는 추가 삭제·새 관계 계산·학습 없이 본문과 그림의 표현을 정리했다.

