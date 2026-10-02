# 보고서 원고와 재현 자료

- [데이터 장](10.02_001_데이터_수록내용_선별.md)
- [새 EDA 원고](10.02_002_EDA_새원고.md): 사용자 요청에 따라 처음부터 다시 구성. 2.1은 평일·주말의 하루 패턴, 2.2는 Spearman 관계, 2.3은 고전력 기준·시간대·생산량 및 기상 범위·시간 안의 형태를 작성했다.
- [이전 EDA 원고](10.02_002_EDA.md): 이전 구성·수치를 보존한 문서. 새 원고의 이후 절로 자동 채택하지 않는다.
- [EDA 보조 자료](10.02_002_EDA_보조자료.md): 공휴일 비교와 날짜별 저부하 수준, 그림 후보 4개.
- [7월 빈도·외부조건 후속 분석](10.02_002_EDA_외부조건_후속분석.md): 187 기준의 기상별 빈도·공통 표본·시간대 및 반복 배열 민감도. 조건 비교의 상세 근거.
- [Analysis 원고와 그림](10.02_003_analysis.md): 조건 비교·조정 범위·일별 및 월별 재배치 결과 5개 주제, 그림 후보 10개.

새 EDA 원고는 **하루 패턴 1개, 전체·월별 Spearman 관계 2개, 고전력 기준·빈도·조건·형태 3개**의 그림을 참조한다. 이전 EDA 본문·보조 자료의 16개와 analysis의 10개 그림도 보존한다. **최신 사용자 허용 후 새 EDA 2.3 그림 세 개를 50% 접촉 시트로 직접 확인했다. 나머지 그림의 이전 시각 미검수 상태는 유지한다.**

## 새 EDA 첫 절의 재현과 확인

[eda_daily_pattern.py](scripts/eda_daily_pattern.py)는 기존 시간대별 집계를 재사용하고, 평일·주말의 48개 시간대 평균·분모를 원본 CSV와 독립 대조한다. [채택 표](tables/eda/daily_pattern_restart.csv), [실행·검증 정보](tables/eda/daily_pattern_restart_manifest.json), `restart_01_daily_pattern_clean.png`을 생성한다. 문서의 이미지 링크도 이 새 파일명으로 교체했다.

```powershell
$p = Start-Process -FilePath 'C:\Program Files\Python313\python.exe' -ArgumentList 'report/scripts/eda_daily_pattern.py' -WorkingDirectory (Get-Location).Path -Wait -PassThru -NoNewWindow
exit $p.ExitCode
```

Codex에서는 정규 Python 실행의 최초 호출부터 `sandbox_permissions="require_escalated"`를 사용한다. 이번 실행은 자식 종료 코드 0, 수치 대조 48행 통과, 맑은 고딕 사용, 글꼴·레이아웃 경고 0이었다. PNG는 열지 않았다.

**이전 근거 → 구성 변경:** 기존 원고는 상관·조건 조정·반복 구조를 한꺼번에 설명해 읽기 어렵다는 사용자 지적이 있었다. 사용자가 지정한 첫 주제인 평일·주말의 생산량·전력 하루 패턴만 새 문서에 작성했다. 시간대별 평균에서 확인되는 12시 하락·13시 회복, 생산량과 전력의 최고 시각 차이, 주말 낮 시간의 낮은 생산량과 남는 전력 기록을 설명한다. 이 기록은 첫 절 작성 당시의 범위다. 이후 2.2를 추가했으며 전체 EDA 완료로 기록하지 않는다.

**공통 플롯 표현 수정:** 사용자 요청에 따라 모든 보고서 플롯에서 맨 위 전체 제목, 아래 작은 설명글, 축의 ‘원자료 기록값’·‘원자료 단위’ 표현을 제거했다. 패널 제목·범례·데이터 표시는 유지하며 전체 제목과 설명은 본문 및 생성 메타데이터에 둔다. 이 기준을 공통 저장 함수와 `report/AGENTS.md`에 적용했다.

[refresh_report_plots.py](scripts/refresh_report_plots.py)로 기존 EDA·analysis PNG 40개를 모두 재생성했다. 자식 종료 코드 0, 제목·하단 설명·축 표현의 비시각 점검 통과, 글꼴·레이아웃 경고 0이었다. 원본과 수치 CSV는 해시가 동일하며 이미지는 열지 않았다. [표현 수정 실행 정보](figures/plot_style_manifest.json)에 각 파일의 해시와 점검 범위를 남겼다.

## 새 EDA 2.2의 방법과 재현

**이전 근거 → 질문:** 첫 절에서 생산량과 전력의 하루 패턴이 달랐고, 사용자는 기상과 전력뿐 아니라 독립변수끼리의 관계도 요구했다. 분포를 확인하기 전에 두 상관계수를 나란히 제시하거나, 수치가 덜 변한다는 이유로 방법을 정한 설명을 수정했다.

**선택 근거:** 정상 5,784시간에서 생산량 43.34%, 강수량 75.06%가 0이며 두 변수는 오른쪽으로 크게 치우쳐 있었다. 이 자료의 분포를 고려하고, 값의 상대적 크기가 함께 움직이는 관계를 설명하려는 목적에 따라 Spearman을 본문의 주된 상관으로 사용한다. 수치형 값에 순위를 부여하며 동률은 평균 순위를 쓴다. 정규성 진단 및 상위 1% 민감도는 보조 점검으로 보존하지만, p값 탈락 또는 계수 안정성을 자동 선택 기준으로 삼지 않는다. 상위 1%를 제외한 민감도 표본은 본문 상관이나 그림의 입력으로 사용하지 않는다.

**결과 → 후속 확인:** 전체 및 월별 Spearman과 표본 수를 계산했다. 기온 관계의 월별 부호가 달라 기존 E007의 8월 생산 여부 비교를 같은 Spearman으로 재확인했다. 전체 8월 -0.076, 양수 생산 383시간 0.599였고, 8월 2~8일의 생산량 0·고기온·낮은 전력 기록을 기존 표 및 원본으로 대조한다. 인과효과나 예측 성능으로 해석하지 않는다.

[eda_variable_relations.py](scripts/eda_variable_relations.py)를 '--diagnose'로 실행하면 분포·정규성·민감도·구간별 수치표를 만든다. '--figures'는 전체 및 월별 Spearman, 8월 후속 표와 그림 두 개를 생성하고 기존 79행을 대조한다. 현재 표와 그림은 [실행·검증 정보](tables/eda/restart_relations_manifest.json)에 기록한다. 정규성 진단의 p값은 독립 표본을 가정한 참고 값으로, 시간적 의존성이 있는 자료에서 정규성을 확정하거나 유의성을 주장하는 근거로 쓰지 않는다.

~~~powershell
$p = Start-Process -FilePath 'C:Program FilesPython313python.exe' -ArgumentList 'report/scripts/eda_variable_relations.py','--diagnose','--figures' -WorkingDirectory (Get-Location).Path -Wait -PassThru -NoNewWindow
exit $p.ExitCode
~~~

Codex에서는 최초 호출부터 escalation을 적용한다. 이미지에는 전체 제목·하단 설명·‘원자료 기록값’ 표현을 넣지 않는다. 이 절 그림은 생성 단계에서 시각 확인하지 않았다.
## 새 EDA 2.3의 기준·기상 범위·그림

**이전 근거 → 질문:** 사용자가 187의 근거와 기상 구간의 생성 방법을 명확히 쓰고, 심사위원이 그림에서 빠르게 비교할 수 있도록 요구했다. 새 원고에 2.3을 추가했다. 전체 제목·하단 설명·‘원자료 기록값’ 표현을 이미지에 넣지 않는 기존 규칙을 유지했다.

**기준:** 기존 E004의 1~8월 5,832행(시간 오류 48행 포함) 행별 최대 95백분위수 187을 재사용한다. 정상 시간만의 95백분위수는 186이다. 원고에 산정 자료·대상·선형 보간·이후 정상 5,784행 적용을 구분했다. 95백분위수는 높은 꼬리의 탐색적 정의이며 최적·공식·미래 예측용 기준이 아니다. 90·95·99백분위수와 186·187의 월별 민감도를 계산해결과표와 그림에 남겼다.

**기상 범위:** 기존 외부조건 결과의 6~8월 양수 생산 평일 통합 사분위 경계를 그대로 사용했다. 군집이나 기상 상태 분류가 아니다. 강수량은 0/양수이며 다섯 변수는 각각 별도로 비교한다. 경계·범위·월별 실제 분모를 표와 그림에 명시했다. 상세 조건 표준화는 기존 후속 분석 문서로 연결했다.

**재현·검증:** [eda_peak_conditions.py](scripts/eda_peak_conditions.py)는 기존 표를 재사용하면서 원본으로 559행의 분모·백분위수·범위·고전력 배열을 확인한다. [실행 정보](tables/eda/restart_peak_manifest.json)에 범위·기준·방법·그림 해시를 남겼다. 정규 CPython 3.13.1 실제 실행은 종료 코드 0, 글꼴·레이아웃 경고 0이었다. 초기 호출은 실행 경로의 문자열 오류로 Python이 시작되지 않았다. 경로를 수정한 뒤 정상 실행이 통과했으며 코드 오류로 분류하지 않았다.

~~~powershell
$ErrorActionPreference = 'Stop'
$p = Start-Process -FilePath 'C:\Program Files\Python313\python.exe' -ArgumentList 'report/scripts/eda_peak_conditions.py' -WorkingDirectory (Get-Location).Path -Wait -PassThru -NoNewWindow
exit $p.ExitCode
~~~

Codex에서는 위 실행도 최초부터 escalation을 사용한다. 세 그림은 기준 위치·시간 빈도, 조건 범위별 색과 비율·분모, 같은 시간의 네 값과 평균·최대의 비교다. 문서에 새 파일명을 직접 연결했다. 최신 사용자 허용 후 세 그림을 50% 접촉 시트로 직접 확인했다. 마지막 그림의 범례가 구분선과 겹친 문제를 해결하고, 187 이상 표식의 뜻을 패널 제목에 표시했다. 수정된 접촉 시트를 다시 확인했으며 숫자·기준·색 범위는 유지했다. [시각 확인 기록](tables/eda/restart_peak_visual_review.json)에 실제 확인한 세 그림과 해시를 기록했다. 2.4와 전체 EDA의 완료를 선언하지 않는다.
## 이전 EDA 구성의 실행 순서

| 순서 | 코드 | 역할 |
|---|---|---|
| 1 | [eda_analysis.py](scripts/eda_analysis.py) | 원본 CSV에서 시간·달력·상관·시차·저부하·네 구간 배열을 재현 |
| 2 | [eda_story_figures.py](scripts/eda_story_figures.py) | 시간 곡선·분포·히트맵·상관·실제 배열 비교 그림 16개 생성 |
| 3 | [verify_eda.py](scripts/verify_eda.py) | 기존 결과표와 원본의 수치·분모 대조. PNG는 존재·파일 크기만 확인 |

그림 코드는 같은 폴더의 [eda_figures.py](scripts/eda_figures.py)에 있는 글꼴·저장 함수와 저부하 그림 함수를 재사용한다. 이 파일은 analysis 그림도 공유하므로 함께 보관한다. `eda_figures.py`를 직접 실행하면 예전 후보 구성을 생성하므로 현재 EDA에는 `eda_story_figures.py`를 실행한다.

분석 코드는 기존 `EDA/jsw/scripts`를 실행하거나 그 결과표를 원자료로 읽지 않는다. 원본부터 계산하는 보고서용 구현이며 `# %%`로 단계를 구분했다. 과거 결과표는 3단계의 대조에서만 사용한다.

원본은 `data/origin/okm_augumented_2021.csv`다. 해시 `8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830`을 확인하며 원본을 복사하거나 수정하지 않는다. 스크립트 위치에서 경로를 찾고 `--source`로 입력 경로를 지정할 수 있다.

정규 CPython 3.13.1, pandas 3.0.1, NumPy 2.2.6, SciPy 1.17.1, Matplotlib 3.10.0 환경을 사용했다. [requirements-eda.txt](requirements-eda.txt)에 버전을 기록했다. 글꼴은 `Malgun Gothic`, `Noto Sans CJK KR`, `NanumGothic` 순서로 탐색하며 이번 생성에는 맑은 고딕을 사용했다. 패키지나 글꼴을 새로 설치하지 않았다.

프로젝트 루트에서 다음 PowerShell 명령으로 전체 재현한다. **Codex 안에서는 모든 정규 Python 실행을 최초 호출부터 `sandbox_permissions="require_escalated"`로 요청한다.**

```powershell
$projectRoot = (Get-Location).Path
foreach ($edaStep in @('report/scripts/eda_analysis.py', 'report/scripts/eda_story_figures.py', 'report/scripts/verify_eda.py')) {
    $p = Start-Process -FilePath 'C:\Program Files\Python313\python.exe' -ArgumentList $edaStep -WorkingDirectory $projectRoot -Wait -PassThru -NoNewWindow
    if ($p.ExitCode -ne 0) { exit $p.ExitCode }
}
```

이번 개편에서는 달력 검정·시차·저부하의 기존 재현표를 재사용했다. 1단계에 `--refresh-visuals`를 붙여 하루 패턴·전체 두 상관·조건별 상관용 점·고전력 배열·외부조건 표를 갱신했다. 부분 실행은 기존 `calendar_summary`, `calendar_tests`, `monthly_maxima`, `lag_correlations`, `low_load_monthly` 등의 표가 이미 있을 때 사용한다. 깨끗한 제출 환경에서는 옵션 없이 전체 실행한다.

각 코드는 `report/tables/eda`, `report/figures/eda`의 해당 산출물만 갱신한다. 기존 분석 자료가 없는 환경에서도 1·2단계는 재현할 수 있으며, 3단계의 과거 결과 대조에는 기존 자료가 필요하다.

## 본문과 코드 연결

| 원고 | 분석 함수 | 주요 표 | 현재 그림 |
|---|---|---|---|
| 2.1 하루 패턴 | `narrative_tables` | `hourly_overview`, `monthly_hour_overview` | `01A_daily_rhythm`, `01B_month_hour_rhythm` |
| 2.2 평일·주말 | `calendar`, `prepare` | `calendar_summary`, `calendar_tests`, `daily_records` | `02A_calendar_joint`, `02B_calendar_ecdf` |
| 2.3 고전력 빈도·기상 관계 | `narrative_tables`, `hour_patterns`, `eda_external_conditions.analyze` | `monthly_hour_overview`, `external_*`, `monthly_maxima` | `03A_month_hour_peaks`, `03B_weather_peak_frequency` |
| 2.4 생산·기상 관계 | `narrative_tables`, `conditional_relations` | `overall_correlations`, `conditional_correlations`, `conditional_points` | `04A_correlation_context`, `04B_conditional_density` |
| 2.5 시간적 반복성 | `lags`, `prepare` | `lag_correlations`, `hourly_records` | `05A_monthly_lag_matrix`, `05B_weekly_change_trace` |
| 2.6 시간 안의 고전력 | `narrative_tables`, `mean_peak` | `high_slot_summary`, `high_slot_patterns`, `high_slot_records`, `same_mean_examples` | `06A_high_slot_map`, `06B_high_slot_profiles` |
| 공휴일 보조 | `calendar` | 달력 요약·검정·일별 기록 | `07A_calendar_joint`, `07B_calendar_ecdf` |
| 저부하 보조 | `low_load` | `low_load_monthly`, `daily_records` | `06A_low_load_monthly`, `06B_low_load_weighting` |

수치 표는 UTF-8 BOM CSV, 소스·JSON·마크다운은 UTF-8이다. 본문 이미지 경로는 앱 표시용 절대경로이므로 폴더 이동 시 마크다운의 프로젝트 경로를 갱신한다. 분석 코드·표의 상대 위치가 같으면 실행 경로를 따로 수정할 필요가 없다.

## 기준과 해석 범위

- EDA의 현재 고전력 기준 **187**은 E004의 1~8월 전체 5,832행에서 산정한 행별 최대 95백분위수를 재사용한다. 시간 오류 48행을 제외한 실제 시각별 분석은 5,784행이다. 정상 행만의 95백분위수는 **186**이므로 두 산정 범위를 혼동하지 않는다.
- 이전 `mean_peak_visibility`, `hour_patterns`는 **182** 기준의 기존 재현표로 보존한다. 현재 본문은 `hour_frequency_187`과 네 구간 배열 표를 사용한다. Analysis·modeling의 기존 기준과 결과는 변경하지 않았으며, 두 기준의 발생 건수를 같은 비교로 합치지 않는다.
- 전체 상관 행렬은 정상 기록에서 변수 쌍별 결측만 제외한다. 조건별 비교는 양수 생산 기록의 월×시각×평일/주말 집단에서 5개 이상 있는 조건만 사용한다. 조정 전후는 같은 표본이며, 전체 행렬의 모집단과 다르다.
- EDA에는 관측 패턴·상관·다음 질문까지만 수록한다. 조건별 심화 해석은 analysis, 실제 입력 추가에 따른 예측 성능 비교는 modeling의 역할이다. M003은 이번에 기존 결과를 읽어 범위를 확인했으며 재학습·재평가는 하지 않았다.
- 반복 날짜·전력 0·생산량 극단값을 보존한다. 그림에 표본 추출·상위값 절단을 적용하지 않는다. 구간 수를 실제 초과 지속시간으로 환산하지 않는다.

## 실행·검증 및 개편 기록

2026-10-02 개편의 부분 집계·그림 생성·수치 검증은 정규 CPython 3.13.1에서 모두 최종 자식 종료 코드 **0**이었다. 기존 표 대조 및 원본 점검 **19항목·1,255행**을 확인했다. 그림 16개를 생성했고 글꼴·레이아웃 경고는 없었다. PNG 내용은 열지 않았으므로 시각적 적합성은 검증하지 않았다.

- [기존 분석 실행 정보](tables/eda/run_manifest.json): 앞선 전체 재현 당시의 원본·코드 해시와 환경. 코드가 개편되어 이 파일의 코드 해시는 현재 코드 해시와 다를 수 있다.
- [개편 집계 실행 정보](tables/eda/narrative_manifest.json): 현재 코드 해시, 원본 해시, 187 산정 범위, 분석 범위와 구간 수.
- [수치 대조 결과](tables/eda/verification.json): 19개 점검 항목·1,255행의 결과.
- [현재 그림 목록](figures/eda/story_manifest.json): 현재 원고의 16개 파일·크기·해시·글꼴·경고·시각 미검수 상태.

**이전 근거 → 이번 변경:** E025의 평균·최대 비교에 모델 평가용 기간 구분이 들어가 EDA 질문이 흐려졌다는 사용자 지적을 반영했다. E026의 시간대 생산·전력 결과와 E015의 네 구간 배열을 채택하고 E007의 기상 네 변수를 포함한 상관을 확장 제시했다. 2.1의 기간별 기준 분류표를 하루 패턴으로 교체했으며 평균·최대 차이는 2.6의 실제 배열과 함께 설명한다. 저부하 요약은 보조 자료로 옮겼다.

**수정·재실행:** 최초 부분 집계에서 187을 정상 5,784행의 95백분위수라고 가정해 assertion에서 종료 1이었다. E004 원문·코드·원본을 대조해 산정 대상이 5,832행임을 확인하고 출처를 수정했다. 기준값이나 원본 관측값을 결과에 맞춰 바꾸지 않았다. 이후 집계·그림 생성·검증이 통과했다.

**종료:** 승인한 EDA 구성·상관·비교형 그림 생성을 완료했다. 새로운 예측 실험이나 운영효과 검증은 포함하지 않는다. 사용자가 그림과 지면을 선택할 수 있도록 원고에 모두 연결했으며, 이미지 분석은 수행하지 않았다.

**용어 정리:** 사용자 지적에 따라 보고서의 ‘저장 평균’ 표현을 ‘평균 전력’으로 통일했다. 2.1에서 CSV `생산량`·`평균` 열과 네 구간의 평균·반올림, 시간대별 241일 평균을 정의했다. 숫자나 표본은 바꾸지 않고 용어가 포함된 3개 주제의 그림 6개만 `eda_story_figures.py --topics rhythm relations slots`로 재생성했다. 최종 자식 종료 코드 0, 글꼴·레이아웃 경고 0이며 이미지는 열지 않았다. 다른 주제의 현재 그림은 기존 목록과 파일을 유지한다.
## 7월 빈도·외부조건·두 상관의 보완

**이전 근거 → 질문:** 사용자 지적에 따라 2.3의 발견을 ‘7월 월최대 한 건의 위치’에서 ‘7월 여러 시간대의 높은 발생 빈도’로 바로잡았다. 기존 A039는 평균·최대 수준의 조건 관계이므로, 현재 187 기준의 기상별 사건 빈도를 대신하지 않는다. E007에 있던 Spearman을 생략할 근거도 없었다.

**최소 분석 → 결과:** [eda_external_conditions.py](scripts/eda_external_conditions.py)에서 월별 달력·생산 분리, 양수 생산 평일의 시간대 표준화, 고정 기상 구간별 빈도와 공통 셀 비교를 계산했다. 6월·7월의 시간대 표준화 비율은 7.76%·24.44%였다. 차이가 남아 한 주씩 제외한 9회도 원래 가중치를 유지해 확인했고 차이는 +12.81~+21.37%p였다. 기상·생산량 구간의 공통 표본에서도 양의 차이가 남지만, 특히 기온은 공통 구간의 범위가 좁아 날씨의 영향을 분리한 결과로 확대할 수 없다.

**해석 변경 → 배치:** EDA 2.3에는 빈도의 발견과 기상 구간별 분포를 넣고, 공통 표본·조정 비교의 상세 해석은 [외부조건 후속 분석](10.02_002_EDA_외부조건_후속분석.md)에 분리했다. 2.4와 행렬에는 생산량·기상 네 변수 사이의 모든 쌍 및 전력과의 Pearson·Spearman을 넣었다. 기상 네 변수의 원값·조건 내 편차 분포도 모두 생성했다. 예측 성능은 EDA에 넣지 않았다.

**재현:** 외부조건 계산은 1단계 전체 실행 및 '--refresh-visuals'에서 함께 호출한다. 기존 표가 있을 때 주차 확인만 다시 실행하려면 다음 명령을 사용한다.

~~~powershell
$p = Start-Process -FilePath 'C:Program FilesPython313python.exe' -ArgumentList 'report/scripts/eda_external_conditions.py','--week-sensitivity' -WorkingDirectory (Get-Location).Path -Wait -PassThru -NoNewWindow
exit $p.ExitCode
~~~

Codex에서는 위 실행도 최초부터 escalation을 사용한다. [외부조건 실행 정의](tables/eda/external_manifest.json)에 경계·표본·공통 셀 기준·가중치·코드 해시와 주차 민감도를 남겼다. 3단계는 [verify_external_conditions.py](scripts/verify_external_conditions.py)를 호출해 원본으로 분모와 가중치를 독립 재구성한다.

**검증 수정:** 최초 확장 검증은 CSV의 기본 소수점 읽기가 편차값의 동률을 미세하게 바꿔 Spearman에서 약 0.000001 차이로 종료 1이었다. 숫자·허용오차·조건을 바꾸지 않고 'float_precision="round_trip"'으로 원래 float를 정확히 복원했다. 그림의 해당 표 읽기에도 같은 옵션을 적용했다.

**종료 범위:** 이번 보완의 수치·분모·지원 표본을 확인했다. 넓은 7월 증가와 기상 구간별 차이는 관측 사실이며, 운영 원인·날씨의 인과효과·예측 입력의 개선 효과는 이 결과로 식별하지 않는다. 그림 후보 4개를 교체했고 나머지는 유지했다. PNG는 생성만 했으며 열거나 분석하지 않았다.

**최종 실행:** 정규 CPython 3.13.1의 부분 집계·영향받는 그림 생성·수치 검증 모두 자식 종료 코드 0이었다. 수치 검증은 19항목·1,255행, 현재 그림 목록은 16개이며 글꼴·레이아웃 경고는 0이었다. Spearman의 숫자 복원 수정 후 상관 주제 두 그림만 다시 생성했다. 이미지 시각 검수는 수행하지 않았다.
## Analysis 실행 순서와 근거

| 순서 | 코드 | 역할 |
|---|---|---|
| 1 | [analysis_reproduce.py](scripts/analysis_reproduce.py) | 제공 CSV부터 A003·A037·A044·A045의 채택 결과를 재현. 수치 표 15개 생성 |
| 2 | [analysis_figures.py](scripts/analysis_figures.py) | 본문 5개 주제의 선택용 그림 10개 생성 |
| 3 | [verify_analysis.py](scripts/verify_analysis.py) | 기존 11개 표·2,912행 대조, 1,446개 시나리오 제약과 최대값 확인, 파일·마크다운 링크 확인 |

1단계는 기존 `analysis/jsw/scripts`나 기존 결과표를 불러오지 않는다. 원본 CSV에서 날짜별 96값 프로필과 기간별 반복 가중치를 다시 구성한다. `--source` 옵션과 입력 해시는 EDA와 같다. 2단계는 `report/scripts/eda_figures.py`의 글꼴·저장 함수만 공통으로 사용하며 EDA 계산이나 EDA 그림 생성은 실행하지 않는다. 제출 시 이 공통 파일도 함께 보관한다. 3단계만 기존 결과표를 읽으므로 제출 환경에서 기존 폴더가 없으면 1·2단계로 결과를 재현할 수 있다.

```powershell
$projectRoot = (Get-Location).Path
foreach ($analysisStep in @('report/scripts/analysis_reproduce.py', 'report/scripts/analysis_figures.py', 'report/scripts/verify_analysis.py')) {
    $p = Start-Process -FilePath 'C:\Program Files\Python313\python.exe' -ArgumentList $analysisStep -WorkingDirectory $projectRoot -Wait -PassThru -NoNewWindow
    if ($p.ExitCode -ne 0) { exit $p.ExitCode }
}
```

Codex 안에서는 이 명령도 최초 호출부터 `sandbox_permissions="require_escalated"`를 사용한다. 분석과 그림은 각각 `tables/analysis`와 `figures/analysis`의 같은 이름 산출물만 갱신한다. 패키지는 [requirements-analysis.txt](requirements-analysis.txt)에 기록했다. 이번 작업에서 설치한 패키지는 없다.

| 원고 | 재현 함수 | 주요 표 | 그림 | 기존 근거 |
|---|---|---|---|---|
| 3.1 조건별 분포 | `distribution` | `distribution`, `distribution_bins`, `distribution_sensitivity` | `01A`, `01B` | [A037 코드](../analysis/jsw/scripts/09.27_037_distribution_followup.py) |
| 3.2 생산 전환 | `transitions` | `transition_summary`, `transition_strata` | `02A`, `02B` | [A003 코드](../analysis/jsw/scripts/09.27_003_transition_context.py) |
| 3.3 상한별 조정 범위 | `redistribution` | `direct_caps`, `daily_concentration`, `daily_slots` | `03A`, `03B` | [A044 보고서](../analysis/jsw/reports/09.28_044_demand_management_scope.md) |
| 3.4 일별 재배치 | `lowest_cap`, `redistribution` | `redistribution`, `group_summary` | `04A`, `04B` | A044 및 [A045 보고서](../analysis/jsw/reports/09.28_045_peak_target_reaggregation.md) |
| 3.5 월최대 재집계 | `redistribution` | `monthly_max`, `july_sensitivity` | `05A`, `05B` | A045 |

추가로 `distribution_reference`에 시간대 가중치, `distribution_exclusions`에 모든 날짜·주 제외 결과, `transition_records`에 전환 비교의 시간별 분모를 저장했다. 그림의 수치는 모두 이 폴더의 표에서 가져온다. 그림 10개를 모두 사용해야 한다는 의미는 아니며, 3.2는 생산 시작을 설명할 때 선택하는 후보로 남겼다. 관측 빈도·조건 비교, 사후 최적화 가정, 미래 예측 평가의 범위를 구분하는 결과를 우선했다.

2026-10-02 분석·그림 생성·최종 검증은 정규 CPython 3.13.1에서 각각 자식 종료 코드 0으로 완료했다. 최초 그림 생성에서 맑은 고딕의 특수 빼기 기호 누락 경고가 발생해 해당 축 문구의 기호만 일반 `-`로 변경했다. 수치·조건은 그대로 두고 그림을 재생성했으며 최종 실행에는 글리프 경고가 없었다. **그림을 열어 확인하지 않았다.**

- [분석 실행 정보](tables/analysis/run_manifest.json): 원본·코드·표 해시, 라이브러리 버전, 임계값·가중 조건·예산.
- [수치 검증 기록](tables/analysis/verification.json): 기존 11개 표·2,912행의 채택 열 일치. 재배치 1,446개에 대해 원본 96값으로 합 보존·비음수·이동 예산·수신 조건을 만족하는 배열을 독립 구성하고, 최대값을 더 낮추면 예산 또는 수신 여력이 부족함을 확인. 이 배열을 현장 운영 일정으로 제시하지 않음.
- [그림 목록](figures/analysis/manifest.json): 10개 PNG의 제목·파일 크기·맑은 고딕 사용, 시각 미검수 상태.

원래 0인 칸의 수신 정책 두 조건, 0·5·10% 예산, 월별 최대 동률을 모두 보존했다. 7월 오류 날짜의 원값 유지 가정도 별도 표로 남겼다. 사후 재배치 시나리오의 재현이며 신규 예산 탐색·모델 학습·현장 효과 검증은 포함하지 않는다. 선택된 질문은 기존 결과와 수치 대조로 답했으므로 분석 범위를 추가로 넓히지 않았다.
