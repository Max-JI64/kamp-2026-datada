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
| EDA 2.4 평일 전력 분포·네 시간 조합 | [EDA/scripts/eda_weekday_power_levels.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_weekday_power_levels.py>) | 같은 파일의 `plot_distribution()` |
| EDA 2.5 시간대별 15분 구간 패턴 | [EDA/scripts/eda_slot_time_patterns.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_slot_time_patterns.py>) | 같은 파일의 `plot_centered_heatmap()` |

- [데이터 원고](01_데이터_원고.md)
- [데이터 0.5페이지용 압축본](01_데이터_0.5페이지용_압축본.md): 원본 확인·전처리·EDA 입력 범위를 약 0.5페이지 분량으로 정리했다. 상세본과 처리 기준은 같으며 실제 제출 양식의 페이지 수는 아직 확인하지 않았다.
- [EDA 원고](02_EDA_원고.md)
- [EDA 3페이지용 압축본](02_EDA_3페이지용_압축본.md): 기존 원고를 유지한 별도 문서다. 다섯 절·그림 일곱 장을 세 묶음·그림 다섯 장으로 재구성했다. 통합 하루 패턴 그림은 [EDA/scripts/plot_report_daily_patterns.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/plot_report_daily_patterns.py>)로 재현한다. 약 3페이지를 목표로 하며 실제 제출 양식의 페이지 수는 아직 확인하지 않았다.
- 코드 `scripts`, 그림 `figures`, 수치·검증 기록 `tables`.
- 공통 글꼴·색상·PNG 저장 설정은 [EDA/scripts/eda_figures.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_figures.py>)의 `setup()`·`finish()`.
- 입력은 `../data/origin/okm_augumented_2021.csv`. 원본은 변경하지 않는다.

## 본문 이미지와 파일 위치

2.5에는 [전체 날짜의 시간대별 15분 패턴](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/tables/slot_time_patterns/README.md>)을 반영했다. [EDA/scripts/eda_slot_time_patterns.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_slot_time_patterns.py>)는 모든 정상 날짜를 모아 시간대별 네 구간 차이·반복 날짜 비율을 계산한다. 본문에는 `EDA/figures/slot_time_patterns.png`의 평일·주말 히트맵 한 장을 사용한다. 96구간 선그래프는 `tmp/eda_slot_time_patterns`에 검토용으로 남긴다. 아래 평균·최대 비교와 세 사례 비교는 앞선 검토 과정이며 이번 반복 패턴의 본문 근거로 사용하지 않는다.

앞선 검토 자료는 [시간 평균·최대 비교 결과](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/tables/mean_max_power/README.md>)에 있다. [EDA/scripts/eda_mean_max_power.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_mean_max_power.py>)로 재현하며, 검토용 그림은 `tmp/eda_mean_max_power`에 저장한다.

연속된 네 전력값의 재검토는 [15분 배열·선그래프 결과](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/tables/quarter_hour_patterns/README.md>)에 있다. [EDA/scripts/eda_quarter_hour_patterns.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/eda_quarter_hour_patterns.py>)는 앞의 검증 결과를 재사용하며 원본과 다시 대조한다. 검토용 선그래프·변동 폭 분포는 `tmp/eda_quarter_hour_patterns`에 생성한다.

그림의 축·색·비교 대상과 핵심 해석은 EDA 원고에서 각 이미지와 함께 설명한다. 이미지 아래에는 파일 위치를 표시하며, 각 절의 관련 Python 코드 링크에서 수정할 코드를 확인할 수 있다. 데이터 원고 1.1~1.5는 플롯 이미지를 사용하지 않는다.

| 원고 절 | 이미지 파일 | 생성 코드와 함수 |
|---|---|---|
| 2.1 하루 패턴 | [EDA/figures/restart_01_daily_pattern_clean.png](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/figures/restart_01_daily_pattern_clean.png>) | `EDA/scripts/eda_daily_pattern.py`: `main()` |
| 2.2 날짜별 분포 | [EDA/figures/daily_repetition/04_daily_pattern_simple.png](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/figures/daily_repetition/04_daily_pattern_simple.png>) | `EDA/scripts/plot_daily_repetition_simple.py`: `main()` |
| 2.2 월별 평균 | [EDA/figures/daily_repetition/07_monthly_power_mean_heatmap.png](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/figures/daily_repetition/07_monthly_power_mean_heatmap.png>) | `EDA/scripts/plot_monthly_power_heatmap.py`: `main()`, `--statistic mean` |
| 2.3 전체 상관 | [EDA/figures/restart_02_variable_relations.png](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/figures/restart_02_variable_relations.png>) | `EDA/scripts/eda_variable_relations.py`: `figures()` |
| 2.3 월별 상관 | [EDA/figures/restart_02_monthly_relations.png](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/figures/restart_02_monthly_relations.png>) | `EDA/scripts/eda_variable_relations.py`: `figures()` |
| 2.4 평일 분포 | [EDA/figures/weekday_power_levels.png](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/figures/weekday_power_levels.png>) | `EDA/scripts/eda_weekday_power_levels.py`: `plot_distribution()` |
| 2.5 시간대별 구간 차이 | [EDA/figures/slot_time_patterns.png](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/figures/slot_time_patterns.png>) | `EDA/scripts/eda_slot_time_patterns.py`: `plot_centered_heatmap()` |

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
EDA/scripts/eda_weekday_power_levels.py
EDA/scripts/eda_slot_time_patterns.py
```

모두 실행한 뒤 [EDA/scripts/verify_manuscript_assets.py](<D:/대회/제6회 K-인공지능 제조데이터 분석 경진대회/EDA/scripts/verify_manuscript_assets.py>)로 원고의 이미지·코드·근거 링크를 검사할 수 있다. 스크립트는 원고를 작성하거나 수정하지 않는다. 코드 경로는 각 `.py` 위치에서 계산하므로 다른 작업 디렉터리에서도 실행할 수 있다.

본문 그림은 `EDA/figures`에 생성한다. 2.5 코드는 검토용 96구간 선그래프도 `tmp`에 생성한다. 2.5의 히트맵 표시만 수정했다면 `--figures-only`로 검증된 결과표를 재사용할 수 있다. 검수용 축소 시트가 필요한 경우에만 위 검증 코드에 `--contact-sheets`를 지정한다. 시트는 `tmp/eda_visual_review`에 생성하며 `EDA/figures`에는 추가하지 않는다. 이전 검수용 사본 5개는 `백업/EDA_검수/2026-10-02`에 보관했다.

## 범위와 해석

EDA는 정상 시간 5,784행·241일 전체를 사용하며, 평일 171일·주말 70일에 공휴일을 포함한다. EDA에 학습·검증 기간 분할을 적용하지 않는다. 2.2의 하락·상승은 같은 날짜의 시간별 평균 비교이며 피크 임계값이 아니다. 월별 히트맵은 생산량 0인 날짜도 포함한 산술평균이다. 원인이나 모델 성능 개선을 이 EDA의 결과로 주장하지 않는다.

2.4는 사용자 요청에 따라 평일 171일·4,104시간을 대상으로 한다. 정수값별 빈도에서 27~48의 빈 구간을 확인해 20~26을 낮은 전력 구간으로 구분하고, 날짜마다 해당 시각을 모두 추출해 네 조합을 집계했다. 히스토그램 한 장과 표를 본문에 반영했다. 최종 수치와 검증은 `tables/weekday_power_levels`에 있다.

그림의 전체 제목·하단 설명·화살표·해석 문구는 이미지 안에 넣지 않는다. 설명은 원고에 작성한다. 날짜별 모든 선·행을 표시하지 않고 질문에 맞게 요약한다.

데이터 원고는 원본 확인 → 품질 점검과 처리 → EDA 입력 자료 구성 순서로 설명한다. 이전 모델의 입력·결측 처리와 5,520행 표본 설명은 현재 원고에 포함하지 않는다. `data_overview.py`의 원본 품질 점검 수치와 각 EDA 코드의 기간·시간 필터 및 변수별 결측 처리가 근거다. 별도의 전처리 완료 CSV를 공통 입력으로 저장하는 방식은 아니며 원본 CSV는 보존한다.

## 이번 이동·반영 결과

2026-10-03에는 2.5를 추가했다. 평일·주말의 24시간×네 구간 히트맵, 숫자의 계산 예시, 평균 형태가 실제로 나타난 날짜 비율을 반영했다. 본문 그림은 총 7개다. 이미지 파일 위치와 생성 Python 코드·함수, 수치 근거 링크를 함께 표시했다. 앞선 2.5 후보 그림은 본문에 추가하지 않았다.

2026-10-03 이미지 설명 점검에서는 본문 그림 6개를 각각 50% 크기로 확인하고, 원고의 축·범례·핵심 관찰을 대조했다. 2.2의 주말 평균·중앙값 차이, 2.3의 날씨·생산량 관계, 2.4 히스토그램의 낮은 구간과 빈 구간 해석을 보완했다. 수치와 그림은 기존 결과를 유지했다. 각 그림에 보이는 파일 경로와 생성 코드·함수를 표시했고, 데이터 원고에는 이미지 미사용을 명시했다. EDA의 Markdown 5개에서 링크 78개, 본문 그림의 표시 경로 6개와 Python 문법을 검증했다. 이 표기·해석 기준은 `EDA/AGENTS.md`에 반영했다. 다음 절의 날씨 비교는 아직 수행하지 않았다.

2026-10-03에는 최종 2.4를 추가했다. 미선정 날씨 비교·월별 차이·지속·전환 그림과 관련 후보 코드·결과 폴더는 사용자 요청에 따라 삭제했다. 삭제 목록은 `tables/weekday_power_levels/cleanup_manifest.json`에 남겼으며 후보 파일의 복사본을 별도로 만들지 않았다. 현재 본문 그림은 6개다. 아래 수치 39개 링크·5개 그림은 이전 이동 당시의 검증 기록이며 최신 검증은 `tables/manuscript_asset_verification.json`을 기준으로 한다.

이동 목록의 44개 파일은 이동 직후 원본 SHA-256과 대조했다. 원고 링크·코드 경로를 새 폴더로 갱신하고, 기존 2.2 생산량·날씨 관계를 2.3으로 옮겼다. 날짜별 하락·상승과 월별 평균 비교를 새 2.2에 반영했다. 각 절에는 담당 Python 코드와 그림 수정 위치를 적었다.

정규 CPython 3.13에서 선별한 계산·그림 코드 전체를 실행해 종료 코드 0을 확인했다. 원본 수치, 시간대별 평균 48개, 월별 집계 384칸, 원고 링크 39개를 검증했다. 선택된 다섯 그림은 각각 50% 크기의 접촉 시트로 확인했다. 결과는 `tables/manuscript_asset_verification.json`에 있다.

데이터 재현 코드의 첫 점검에서 공장인원 저장값을 고정 소수점 9자리로 가정한 검증이 실패했다. 기존 변수표의 최대 절대 오차 약 4.94×10⁻⁹를 확인해 이 가정을 수정했으며, 원본값·관측 대상·문서 수치를 바꾸지 않았다. 관측 관계의 원인과 모델 성능은 이번 이동·원고 반영의 확인 범위에 포함하지 않는다.
