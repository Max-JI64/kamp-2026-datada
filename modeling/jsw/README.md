# 모델링 — jsw

이 폴더에는 모델 학습·예측 평가·예측 오류 분석을 둔다. `EDA/jsw`에는 탐색적 데이터 분석을, `analysis/jsw`에는 과제 질문과 데이터 근거의 연결·해석 가능성 점검을 둔다. 이 폴더의 001~004는 2026-09-26에 처음 `analysis/jsw`에 작성했으나 모델링 작업임을 명확히 하려고 경로만 옮겼다. 모델 설정·결과·번호는 유지했다.

- `scripts/`: 재현 가능한 Python 분석 코드
- `tables/`: 모델 비교 및 오류 집계표, 실행 설정과 facts
- `predictions/`: 행별 개발·평가 예측값과 큰 오류 사례
- `reports/`: 결과·해석·한계·다음 질문
- `LOG.md`: 모델링 단계의 실행 및 변경 이력

이 폴더의 작업 번호는 EDA·과제 분석과 별도로 **001부터 시작**한다. 첫 작업은 [001 잠정 전력 모델 비교](reports/09.26_001_power_models.md)다. 원본은 `../../data/origin`, 이전 EDA의 기록 연결 점검표와 기준선 예측값은 `../../data/processed/jsw`에서 읽는다. 결과 CSV와 예측값은 이 폴더에 저장한다.

- [002 고전력 과소예측과 과대예측](reports/09.26_002_quantile_tradeoff.md)
- [003 직전 생산량·날씨 추가 가치](reports/09.26_003_past_context.md)
- [004 탐색적 고전력 시간 선별](reports/09.26_004_high_hour_selection.md)
