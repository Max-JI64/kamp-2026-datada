# 새 데이터·EDA 원고

2026-10-02 사용자 요청에 따라 두 원고와 본문에 필요한 코드·그림·결과표만 이전했다. 이전 분석은 루트 `백업`에 있으며, 이전 보고서 폴더 전체는 `백업/report`에 보관했다. 그 폴더의 피크·POT·군집 탐색과 미선정 그림은 현재 EDA에 포함하지 않는다. 이동 전 경로와 SHA-256은 [이동 목록](tables/migration_manifest.json)에 기록했다.

## 원고와 코드

| 원고 내용 | 계산 코드 | 그림 생성 코드 |
|---|---|---|
| 데이터 1.1~1.5 | [EDA/scripts/data_overview.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/data_overview.py>) | 본문 그림 없음 |
| EDA 2.1 생산량·전력 하루 패턴 | [EDA/scripts/data_overview.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/data_overview.py>), [EDA/scripts/eda_daily_pattern.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_daily_pattern.py>) | [EDA/scripts/eda_daily_pattern.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_daily_pattern.py>) |
| EDA 2.2 날짜별 전력 패턴 | [EDA/scripts/eda_daily_repetition.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_daily_repetition.py>) | [EDA/scripts/plot_daily_repetition_simple.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/plot_daily_repetition_simple.py>) |
| EDA 2.2 월별 평균 전력 | [EDA/scripts/verify_monthly_power_heatmap.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/verify_monthly_power_heatmap.py>) | [EDA/scripts/plot_monthly_power_heatmap.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/plot_monthly_power_heatmap.py>) |
| EDA 2.3 생산량·날씨 관계 | [EDA/scripts/eda_variable_relations.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_variable_relations.py>) | 같은 파일의 `figures()` |

- [데이터 원고](10.02_001_데이터_수록내용_선별.md)
- [EDA 원고](10.02_002_EDA_새원고.md)
- 코드 `scripts`, 그림 `figures`, 수치·검증 기록 `tables`.
- 공통 글꼴·색상·PNG 저장 설정은 [EDA/scripts/eda_figures.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_figures.py>)의 `setup()`·`finish()`.
- 입력은 `../data/origin/okm_augumented_2021.csv`. 원본은 변경하지 않는다.

## 실행 순서

프로젝트 루트에서 정규 CPython 3.13으로 아래 순서대로 실행한다. `pandas`, `numpy`, `scipy`, `matplotlib`, `Pillow`와 한글 글꼴 `Malgun Gothic`이 필요하다. Codex에서는 루트 `AGENTS.md`에 지정된 승격 `Start-Process` 경로를 사용한다.

```text
EDA/scripts/data_overview.py
EDA/scripts/eda_daily_pattern.py
EDA/scripts/eda_daily_repetition.py
EDA/scripts/plot_daily_repetition_simple.py
EDA/scripts/verify_monthly_power_heatmap.py
EDA/scripts/plot_monthly_power_heatmap.py --statistic mean
EDA/scripts/eda_variable_relations.py --diagnose --figures
```

모두 실행한 뒤 [EDA/scripts/verify_manuscript_assets.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/verify_manuscript_assets.py>)로 원고의 이미지·코드·근거 링크를 검사할 수 있다. 스크립트는 원고를 작성하거나 수정하지 않는다. 코드 경로는 각 `.py` 위치에서 계산하므로 다른 작업 디렉터리에서도 실행할 수 있다.

기본 실행은 본문 그림만 생성한다. 검수용 축소 시트가 필요한 경우에만 위 검증 코드에 `--contact-sheets`를 지정한다. 시트는 `tmp/eda_visual_review`에 생성하며 `EDA/figures`에는 추가하지 않는다. 이전 검수용 사본 5개는 `백업/EDA_검수/2026-10-02`에 보관했다.

## 범위와 해석

EDA는 정상 시간 5,784행·241일 전체를 사용하며, 평일 171일·주말 70일에 공휴일을 포함한다. EDA에 학습·검증 기간 분할을 적용하지 않는다. 2.2의 하락·상승은 같은 날짜의 시간별 평균 비교이며 피크 임계값이 아니다. 월별 히트맵은 생산량 0인 날짜도 포함한 산술평균이다. 원인이나 모델 성능 개선을 이 EDA의 결과로 주장하지 않는다.

그림의 전체 제목·하단 설명·화살표·해석 문구는 이미지 안에 넣지 않는다. 설명은 원고에 작성한다. 날짜별 모든 선·행을 표시하지 않고 질문에 맞게 요약한다.

데이터 원고의 기존 모델 입력·결측 처리 설명은 백업의 원본 모델 문서·코드로 연결했다. 이번에는 모델을 다시 학습하지 않았다. `data_overview.py`는 이전 일반 EDA 코드에서 원고에 필요한 계산만 남겨 재구성했다. 나머지 계산 코드도 새 경로를 사용하며, 이전 파일 위치의 복사본을 실행 기준으로 삼지 않는다.

## 이번 이동·반영 결과

이동 목록의 44개 파일은 이동 직후 원본 SHA-256과 대조했다. 원고 링크·코드 경로를 새 폴더로 갱신하고, 기존 2.2 생산량·날씨 관계를 2.3으로 옮겼다. 날짜별 하락·상승과 월별 평균 비교를 새 2.2에 반영했다. 각 절에는 담당 Python 코드와 그림 수정 위치를 적었다.

정규 CPython 3.13에서 선별한 계산·그림 코드 전체를 실행해 종료 코드 0을 확인했다. 원본 수치, 시간대별 평균 48개, 월별 집계 384칸, 원고 링크 39개를 검증했다. 선택된 다섯 그림은 각각 50% 크기의 접촉 시트로 확인했다. 결과는 `tables/manuscript_asset_verification.json`에 있다.

데이터 재현 코드의 첫 점검에서 공장인원 저장값을 고정 소수점 9자리로 가정한 검증이 실패했다. 기존 변수표의 최대 절대 오차 약 4.94×10⁻⁹를 확인해 이 가정을 수정했으며, 원본값·관측 대상·문서 수치를 바꾸지 않았다. 관측 관계의 원인과 모델 성능은 이번 이동·원고 반영의 확인 범위에 포함하지 않는다.
