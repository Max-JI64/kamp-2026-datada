"""Generate detailed and structurally condensed Markdown manuscripts only."""
import argparse
import json
from pathlib import Path
from regime_forecast import sha, save_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Modeling/tables/modeling_close'
MAN=ROOT/'Modeling/04_Modeling_원고.md'
COMPACT=ROOT/'Modeling/04_Modeling_3페이지용_압축본.md'

def compact():
    return """# 4. Modeling

## 4.1 시간순 평가와 기본 모델 선택

Analysis에서 확인한 시간 내부 전력 차이를 반영해 **다음 한 시간의 네 15분 값 중 최대값**을 예측했다. 직전 관측까지의 전력·생산량과 달력을 사용하고, 시간 오류·공백은 정확한 시차 연결로 처리했다. 무작위 분할 대신 1~3월→4월, 1~4월→5월, 1~5월→6월로 평가했으며 Optuna 튜닝도 각 평가 월 이전의 내부 시간순 검증에서 수행했다.

![모델 개발과 매시간 적용 흐름](figures/modeling_close/pipeline.png)

*그림 1. 위는 개발, 아래는 관측 시점에 맞춘 적용 흐름이다. OOF는 해당 평가 월 이전 자료로 학습한 모델의 예측이다.*

Ridge·Elastic Net·SVR, HGB·XGBoost·LightGBM·CatBoost·ExtraTrees와 앙상블을 비교했다. 같은 기본 입력·2,184시간에서 대표 결과는 다음과 같다.

| 모델 | 전체 MAE | 일별 최대 시간 MAE |
|---|---:|---:|
| 직전 값 기준 | 13.793 | 36.589 |
| HGB | **6.738** | 11.751 |
| ExtraTrees | 6.825 | **11.540** |
| XGBoost | 7.131 | 12.864 |

**전체·최대 시간 오차를 함께 보는 사전 선택 규칙에 따라 HGB를 유지했다.** 마지막 15분 값 추가는 개발에서 유리했지만 후반기 최대 시간까지 개선이 유지되지는 않았다. 이에 전력 수준이 크게 올라 유지되는 시작을 따로 예측하고, 그때만 HGB를 보완했다.

관련 Python: [시차 구성](scripts/m01_prepare.py) · [시간순 모델 비교·튜닝](scripts/m02_rerun.py) · [입력 효과](scripts/m03_ablation.py). 그림 1 생성: [modeling_close_figures.py](scripts/modeling_close_figures.py)의 `pipeline()`.

## 4.2 지속 상승 예측과 선택적 HGB 보완

시간 산술평균이 직전 값과 과거 6시간 중앙값보다 각각 59.5 이상 높고, 중앙값 대비 상승폭이 과거 IQR의 1.5배 이상인 시작을 후보로 삼았다. 이후 3시간 동안 기존 기준선보다 29.75 이상 높은 상태가 이어지면 **지속 상승**으로 정의했다. 1월 자료에서 정한 분석 기준이며, 미래 3시간은 정답 확인에만 사용했다.

Logistic이 과거 정보로 상승을 경고할 때만 상태 입력을 보완한 HGB B1을 적용하고 나머지는 기본 HGB B0를 유지했다. 초기 오경보 38건은 모두 상태 경과시간이 학습 최대 64시간을 넘어선 180~227시간에서 발생했다. **분류기의 경과시간 입력 하나를 제거**하고 과거 OOF에서 경계 0.4555를 정하자 개발 적중 12건·오경보 0건을 유지했다.

![성능·오경보와 개선·미탐지 사례 통합](figures/modeling_close/compact_evidence.png)

*그림 2. 7~8월 탐색적 재평가. 위: 동일 1,344시간의 오차와 정답이 확인된 1,340시간의 경고 수. 일별 최대는 완전한 56일을 같은 가중치로 평가했다. 아래: 개선·미탐지 사례; 음영은 지속 정답 3시간, 점선은 직전 관측 완료 시점이다.*

상승 시작 MAE는 **20.037→18.205(9.1% 감소)**, 평균 과소예측은 **19.419→16.342(15.8% 감소)**, 일별 최대 시간 MAE는 13.179→12.574(4.6% 감소)였다. 전체 MAE는 6.335→6.317로 0.28% 줄었다. 보완은 7시간에만 적용했고 적중 7건을 유지하면서 오경보를 38→0건으로 줄였다.

분류는 1~6월 4,336시간, 회귀는 2~6월 3,598시간으로 한 번 학습해 7~8월에 고정했다. 매시간 입력만 갱신했으며 7월 정답으로 재학습하지 않았다. 이미 확인한 후반기 결과를 바탕으로 수정했으므로 독립 최종시험이 아닌 탐색적 결과다.

관련 Python: [사건 정의](scripts/regime_diagnose.py) · [분류](scripts/regime_forecast.py) · [보완 HGB](scripts/regime_integrate.py) · [선택 결합](scripts/regime_route.py) · [고정 평가](scripts/regime_followup.py) · [경과시간 제거](scripts/regime_age_ablation.py). 그림 2 생성: [modeling_close_figures.py](scripts/modeling_close_figures.py)의 `compact_evidence()`.

## 4.3 오류 해석·현장 활용과 재현성

공통 상승 13건 중 7건을 탐지해 **정밀도 100%(7/7), 재현율 53.8%(7/13)**였다. 경고 7건 중 값의 오차는 4건에서 줄고 3건에서 늘었다. 그림 2의 7/12는 실제 최대 198에 대해 181.13→192.52로 개선됐지만, 8/13은 경고가 없어 실제 207을 182.40으로 낮게 예측했다. 각 사례는 개선폭·실제 최대값의 중앙 순위로 선정했다.

표준화 입력×계수로 Logistic의 로짓을 분해하면 적중·미탐지의 과거 전력 평균 기여는 13.86·4.16이었다. 과거 전력도 판단에 기여했지만 경고는 모두 월요일 08시에 집중됐다. 사후 단순 주간 규칙은 8건 적중·오경보 1건이고 일별 최대 오차는 같아 추가 이득은 작았다. 성과는 **지속 상승 일부에서 오경보를 억제하고 과소예측을 줄인 선택적 보완**이며, 양방향 전환 전체의 탐지로 확대하지 않는다.

현장에서는 직전 관측 완료 후 최대 전력·상승 경고를 확인하고, 담당자가 예정된 가동·동시 사용 조건과 함께 보완 예측을 검토한다. 무경고를 안전 판정으로 쓰지 않으며 필수 시차가 없으면 보완을 보류한다. 가동 조정과 전력·비용 절감은 실측 성과가 아닌 활용 제안이다.

원자료부터 현재 고정 방법을 재실행해 분류 1,428개·회귀 4,032개 예측의 일치를 확인했다. 관련 Python: [원자료 재현·변수 기여·사례 선정](scripts/modeling_close_analysis.py) · [주간 규칙 대조](scripts/regime_age_calendar_control.py) · [한 번에 재현](scripts/modeling_reproduce_selected.py) · [수치·그림·링크 검증](scripts/modeling_close_verify.py). 정규 CPython 3.13에서 재현 스크립트에 `--output-dir Modeling/reproduction_new`로 새 폴더를 지정한다.

전체 실행 명령·단계별 검증 코드·근거표는 [상세 원고 4.5·4.12·4.13](04_Modeling_원고.md), 전체 파일 목록은 [제출 보존 목록](submission_manifest.json), 환경은 [requirements.txt](requirements.txt)에 연결했다. 새 환경 설치와 과거 Optuna 전체 재탐색을 검증한 것은 아니다.
"""


def update():
    for path,name in [(MAN,'manuscript_before_markdown_correction.md'),
                      (ROOT/'Modeling/04_Modeling_4페이지용_압축본.md','compact_before_3page.md')]:
        backup=OUT/name
        if path.exists() and not backup.exists():backup.write_bytes(path.read_bytes())
    text=MAN.read_text(encoding='utf-8')
    assert text.count('## 4.13 ')==1
    MAN.write_text(text.split('## 4.13 ')[0].rstrip()+'\n\n'+detailed().strip()+'\n',encoding='utf-8')
    COMPACT.write_text(compact(),encoding='utf-8')
    readme=ROOT/'Modeling/README.md'
    text=readme.read_text(encoding='utf-8')
    start=text.index('사용자 승인으로 추가 성능 탐색을 마치고')
    end=text.index('\n\n',start)
    note='사용자 승인으로 추가 성능 탐색을 마치고 [상세 원고 4.13](04_Modeling_원고.md#413-보고서-시각화영향요인-해석과-모델링-종료)에 사례·변수 기여·활용·재현을 연결했다. [3페이지용 압축본](04_Modeling_3페이지용_압축본.md)은 평가·모델 선택, 지속 상승 보완, 오류·활용·재현성의 세 절로 재구성했다. 흐름도와 성능·경고·사례 통합 그림 두 개를 사용하고, 절마다 Python 전체 소스 링크를 연결한다. 마크다운만 작성하며 3페이지는 편집 목표다. 전체 실험·그림·코드와 근거는 상세 원고에 보존한다.'
    readme.write_text(text[:start]+note+text[end:],encoding='utf-8')
    record={'status':'written','script_sha256':sha(__file__),'format':'markdown_only',
            'target_pages':3,'pagination_verified':False,'compact_sections':3,'compact_figures':2,
            'structural_merges':['evaluation_and_model_selection','state_forecast_and_performance','errors_use_and_reproduction'],
            'figure_merge':'metrics_alarms_improved_and_missed_cases',
            'outputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [MAN,readme,COMPACT]},
            'original_manuscript_sha256':sha(OUT/'manuscript_before.md')}
    save_json(OUT/'documents.json',record)
    print(json.dumps({'status':'written','format':'markdown_only','target_pages':3,'sections':3,'figures':2}))


def detailed():
    return '''## 4.13 보고서 시각화·영향요인 해석과 모델링 종료

최종 설명의 중심은 4.12의 지속 상승 경고와 선택적 HGB 보완이다. 추가 모델 탐색은 종료하고, 보고서용 그림·해석·활용과 현재 고정 방법의 재현 점검을 마쳤다. 전체 분석 흐름도는 4.1 앞에 두고, 성능·경고 비교는 4.12에 연결했다. 아래 그림은 제3장의 사례·영향요인 근거다.

![지속 상승의 개선·미탐지 사례](figures/modeling_close/transition_cases.png)

개선 사례는 공통 상승 중 경고가 있고 절대오차가 줄어든 4건을 개선폭 순으로 정렬한 중앙 순위의 7월 12일 08시다. 실제 최대값 198에 대해 기본 예측 181.129가 보완 후 192.522로 가까워졌다. 미탐지 사례는 놓친 6건의 실제 최대값 중앙 순위인 8월 13일 08시다. 실제 207에 비해 두 예측 모두 182.404로 낮았다. 시작 전 4시간부터 이후 5시간까지 공통 예측이 완전한 사례만 후보로 삼았고, 두 후보군 모두 해당 조건을 충족했다. [선정 규칙·결과](tables/modeling_close/case_selection.csv)와 [전체 사례 창](tables/modeling_close/case_windows.csv)을 보존했다. 경고 7건 중 개별 절대오차가 줄어든 것은 4건이고 3건은 늘었다. 사례 그림의 개선폭을 전체 평균 성과로 일반화하지 않는다.

![Logistic의 변수군별 기여](figures/modeling_close/logit_contributions.png)

추가 해석의 질문은 “경고가 월요일 08시에 집중되는 이유가 달력뿐인가, 과거 관측 상태도 판단에 기여하는가”였다. 현재 분류기는 표준화 뒤 선형 Logistic이므로, 입력별 표준화값에 계수를 곱한 항과 절편을 직접 합산해 모델의 로짓을 정확하게 복원했다. SHAP을 추가 실행하지 않았으며 위 값은 확률 백분율이나 인과효과가 아니다. 변수 간 상관 때문에 각 항을 독립적인 공정 영향력으로 해석하지 않는다.

| 공통 평가 조건 | 건수 | 과거 전력 평균 기여 | 달력 평균 기여 | 평균 로짓 |
|---|---:|---:|---:|---:|
| 상승 적중 | 7 | 13.857 | 0.300 | 4.008 |
| 상승 미탐지 | 6 | 4.159 | -1.220 | -7.339 |
| 월요일 08시 비사건 | 1 | 2.719 | 0.150 | -6.813 |

공통 절편은 -10.104이며 생산량·추적 상태 항도 합산에 포함했다. 과거 전력 입력도 점수에 기여했지만, 이 분해가 달력보다 독립적인 예측 가치가 크다는 실험은 아니다. 경고는 여전히 월요일 08시에 한정되고 단순 주간 규칙과의 성능 차이는 작다. 한 건뿐인 월요일 비사건의 해석도 해당 사례 범위로 제한한다. [전체 입력 계수](tables/modeling_close/coefficient_table.csv), [1,428시간 기여](tables/modeling_close/logit_contributions.csv), [조건별 요약](tables/modeling_close/contribution_summary.csv), [상승 13건](tables/modeling_close/event_explanations.csv)을 보존한다.

### 현장 활용과 종료 범위

제4장의 제안은 직전 관측 완료 → 다음 최대 전력·지속 상승 경고 확인 → 담당자가 계획된 가동·동시 사용 조건과 함께 보완 예측을 검토하는 절차다. 무경고를 안전으로 판단하지 않으며, 필수 시차가 없는 경우 보완 예측을 보류하고 자료 상태를 표시한다. 가동 조정·자동 제어·절감량은 실측하지 않았다. 제5장에는 전력 수준의 지속 전환을 별도 정의하고, 오류 진단에서 찾은 입력 문제를 수정해 불필요한 보완 적용을 줄인 기여를 연결한다.

현재 방법은 제한된 탐색 성과를 보고하는 후보로 고정한다. 원자료 진단·시간순 모델 비교·입력 효과·상승 보완·오류 해석·시각화·현재 환경 재현 점검이 끝났으므로 모델링의 추가 성능 탐색을 종료한다. 남은 작업은 최종 한글 양식에 각 장을 배치하고 EDA·Analysis와 분량을 조정하는 제출 편집이다. 전체 급등·급락 탐지가 완성되었다는 의미로 종료를 표현하지 않는다.

### 원자료부터 현재 선택 방법까지 재현

[modeling_close_analysis.py](scripts/modeling_close_analysis.py)는 기존 조건을 읽어 원자료에서 M01 시차와 지속 사건을 다시 만들고, 분류 학습 4,336시간·회귀 학습 3,598시간을 재구성했다. 4~6월 OOF 분류 3개와 6월 말 고정 분류기·B0·B1을 다시 학습한 뒤, 과거 경계 0.4554612321·분류 1,428개·회귀 4,032개 예측을 대조했다. 모두 일치했다. 새 후보나 튜닝을 추가한 재학습은 아니다.

최초 재현에서는 원자료에서 만든 실수값과 기존 CSV를 읽은 값 사이의 미세한 수치 차이로 HGB 예측이 일치하지 않았다. 기존 입력 파일과 합친 학습표의 CSV 저장·읽기 단계를 동일하게 적용하자 학습 입력이 정확히 일치했고 예측 대조도 통과했다. 실행 조건과 결과를 바꾸지 않고 수치 저장 경계를 코드에 명시했으며, 수정 전 코드·계약도 보존했다. 이 점은 다른 파서·환경에서 재실행할 때 확인해야 할 재현 조건이다.

평가자용 [modeling_reproduce_selected.py](scripts/modeling_reproduce_selected.py)는 새 출력 폴더를 지정해 원자료부터 고정 방법의 학습·추론·기존 결과 대조까지 한 번에 수행한다. 실제로 `Modeling/tables/modeling_replay`에서 실행했고 [실행 기록](tables/modeling_replay/run.json)의 분류·회귀 대조가 통과했다. 이미 사용한 폴더는 덮어쓰지 않으므로 재실행 때는 새로운 폴더명을 지정한다. 현재 방법의 재현에 필요한 기존 정의·설정·참조 결과를 함께 제출한다.

다음 명령을 프로젝트 루트의 정규 CPython 3.13·승격 Start-Process 경로에서 실행한다. 첫 줄은 평가자용 재현 경로이며, 뒤의 명령은 최초 분석·그림·문서 생성 이력이다. 현재 자료의 무결성은 verify 명령으로 확인한다.

~~~text
Modeling/scripts/modeling_reproduce_selected.py --output-dir Modeling/reproduction_new
Modeling/scripts/modeling_close_analysis.py freeze
Modeling/scripts/modeling_close_analysis.py run
Modeling/scripts/modeling_close_figures.py
Modeling/scripts/modeling_close_documents.py --update
Modeling/scripts/modeling_close_verify.py
Modeling/scripts/modeling_submission_audit.py --verify
~~~

[modeling_close_figures.py](scripts/modeling_close_figures.py)는 저장 결과에서 PNG·SVG 그림을 만든다. [modeling_close_documents.py](scripts/modeling_close_documents.py)는 상세 원고를 보존하고 [3페이지용 압축본](04_Modeling_3페이지용_압축본.md)을 마크다운으로 생성한다. 압축본은 평가·모델 선택, 지속 상승 보완, 오류·활용·재현성의 세 절로 재구성했다. 성능·경고 수·개선 및 미탐지 사례를 [통합 그림](figures/modeling_close/compact_evidence.png) 하나로 묶어 흐름도와 함께 두 그림만 싣는다. 변수 기여의 전체 그림·표와 과거 실험별 결과·전체 코드 연결은 이 상세 원고에 남겼다. [modeling_close_verify.py](scripts/modeling_close_verify.py)는 기여 합·예측 오차·그림 근거·문서 보존·링크를 검증한다. 3페이지는 편집 목표이며 마크다운 자체의 실제 페이지 수를 검증했다는 뜻은 아니다.

원본 상세 원고는 [시각화 추가 전 사본](tables/modeling_close/manuscript_before.md)에 보존했다. [실행 기록](tables/modeling_close/run.json)과 [원자료부터 재현한 예측](tables/modeling_close/reproduced_predictions.csv)을 제출 보존 목록에 추가한다. 현재 환경에서 선택 방법을 재실행한 결과이며, 깨끗한 새 환경의 설치 검증이나 기존 Optuna 전체 탐색을 다시 수행한 것은 아니다.
'''



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--update',action='store_true',required=True)
    parser.parse_args()
    update()
