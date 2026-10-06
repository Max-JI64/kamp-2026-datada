"""Integrate verified expanded M02 into the detailed manuscript and READMEs."""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from m03_verify import ROOT, rows, sha

OUT = ROOT / 'Modeling/tables/m02_rerun'


def main():
    verification = json.loads((OUT / 'independent_verification.json').read_text(encoding='utf-8'))
    assert verification['status'] == 'passed' and verification['run_sha256'] == sha(OUT / 'run.json')
    run = json.loads((OUT / 'run.json').read_text(encoding='utf-8'))
    for name, digest in run['outputs_sha256'].items():
        assert sha(OUT / name) == digest
    selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
    assert selection['selected']['model'] == 'HGB_initial' and not selection['M03_repeat_required']
    values = {r['model']: r for r in rows(OUT / 'selection_metrics.csv')}
    comparison = {(r['model'], r['split']): r for r in rows(OUT / 'comparison.csv')}
    conditions = {(r['model'], r['condition'], r['value']): r for r in rows(OUT / 'condition_errors.csv')}
    order = ['previous_hour', 'previous_day', 'previous_week', 'Ridge_initial', 'ElasticNet_initial', 'SVR_initial',
             'HGB_initial', 'HGB', 'XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'mean5', 'weighted_top3']
    labels = {'previous_hour': '직전 값', 'previous_day': '전날 같은 시각', 'previous_week': '전주 같은 시각',
              'Ridge_initial': '기존 Ridge', 'ElasticNet_initial': '기존 Elastic Net', 'SVR_initial': '기존 RBF SVR',
              'HGB_initial': '기존 HGB', 'HGB': 'Optuna HGB', 'mean5': '5모델 동일 평균', 'weighted_top3': '내부 상위 3모델 가중 평균'}
    lines = ['| 방법 | 전체 MAE | 실제 일별 최대 시간 MAE | 선택 점수 |', '|---|---:|---:|---:|']
    for model in order:
        r = values[model]
        lines.append(f"| {labels.get(model, model)} | {float(r['overall_MAE']):.3f} | {float(r['daily_maximum_MAE']):.3f} | {float(r['balanced_score']):.4f} |")
    table = '\n'.join(lines)
    monthly = ['| 방법 | 4월 MAE | 5월 MAE | 6월 MAE | 학습에 없는 배열 MAE |', '|---|---:|---:|---:|---:|']
    for model in ['HGB_initial', 'ExtraTrees', 'weighted_top3']:
        numbers = [float(comparison[(model, split)]['MAE']) for split in ['dev_apr', 'dev_may', 'dev_jun']]
        numbers.append(float(conditions[(model, 'train_profile_overlap', 'False')]['MAE']))
        monthly.append('| '+labels.get(model, model)+' | '+' | '.join(f'{x:.3f}' for x in numbers)+' |')
    monthly_table = '\n'.join(monthly)
    errors = ['| 조건 | 평가 시간 | 기존 HGB MAE | ExtraTrees MAE |', '|---|---:|---:|---:|']
    for condition, label in [('low->above26', '낮은 구간→26 초과'), ('above26->low', '26 초과→낮은 구간'),
                              ('low->low', '낮은 구간 유지'), ('above26->above26', '26 초과 유지')]:
        a = conditions[('HGB_initial', 'power_transition', condition)]
        b = conditions[('ExtraTrees', 'power_transition', condition)]
        errors.append(f"| {label} | {a['hours']} | {float(a['MAE']):.3f} | {float(b['MAE']):.3f} |")
    error_table = '\n'.join(errors)
    section = f'''

## 4.4 M02 확장 재실행: Optuna·시간순 검증·앙상블 비교 후의 판단

### 기존 HGB를 다른 모델과 다시 비교한 이유

4.2의 M02에서는 HGB가 기본 A 입력에서 가장 좋았고, 4.3의 M03에서는 마지막 값을 추가한 B의 개선을 확인했다. 그러나 이 비교만으로 XGBoost·LightGBM·CatBoost나 앙상블을 포함한 후보 중에서도 HGB를 유지할 수 있는지는 확인하지 못했다. 또한 상승 전환과 낮은 구간 유지의 오류가 남아 있었으므로, 모델을 바꾸면 그 오류가 줄어드는지도 함께 확인할 필요가 있었다.

사용자 요청에 따라 M02 자체를 확장 재실행했다. 모델 차이와 입력 효과를 분리하기 위해 **마지막 값이 없는 기본 A 입력에서** 다시 비교했고, 선택 모델이나 설정이 바뀌면 그 설정으로 M03을 반복하도록 정했다. 이전 M02·M03은 당시의 결과와 판단을 보여주는 근거로 유지한다. 이미 개발 결과를 보고 정한 확장 설계이므로 이번 점수를 새로운 독립 시험의 성적으로 표현하지 않는다.

### 학습과 예측 구간: 모든 적격 시간의 순차 비교

| 외부 개발 평가 | 학습 기간 | 예측·평가 기간 | 학습 시간 | 평가 시간 |
|---|---|---|---:|---:|
| 1차 | 1~3월 | 4월 전체 | 1,992 | 720 |
| 2차 | 1~4월 | 5월 전체 | 2,712 | 744 |
| 3차 | 1~5월 | 6월 전체 | 3,456 | 720 |

이 표의 시간 수는 정확한 1·2·24·168시간 시차가 모두 있는 common 조건의 수다. 평가월의 유리한 날만 고른 것이 아니라 조건을 충족하는 모든 시각 2,184개를 같은 순서로 비교했다. 1~3월은 학습과 내부 검증에 쓰이고, 4~6월은 모델·설정의 개발 선택에 쓰인다. 후속 M05에는 1~6월 학습→7~8월 평가를 남겼다. 7~8월은 이번 실행의 튜닝·선택에 쓰지 않았지만 과거 관찰 이력이 있어 완전히 보지 않은 독립 시험으로 부르지 않는다. 9월은 기존 분석 범위 밖이므로 편입하지 않는다.

각 외부 평가월의 설정은 그 이전 자료 안에서만 선택했다. 예를 들어 4월 평가의 Optuna는 `1월 학습→2월 검증`과 `1~2월 학습→3월 검증`을 사용한다. 5월 평가에는 4월 내부 검증이, 6월 평가에는 5월 내부 검증이 추가된다. 각 내부 학습 시각은 검증 시각보다 이르고 내부 검증의 마지막 시각은 외부 평가보다 이르다. 시각 오류로 생긴 공백 때문에 단순한 행 개수 분할 대신 실제 달력 경계를 사용했다.

외부 월이 시작되면 선택한 설정으로 이전 자료 전체를 학습하고 그 월 동안 모델을 고정한다. 관측이 끝난 입력은 매시간 갱신한다. 따라서 이 결과는 **매시간 다음 한 시간을 예측한 결과**이며, 월 전체를 월초에 한 번에 예측하거나 하루 최대 시각을 미리 지정한 결과가 아니다.

### 모델과 튜닝 범위

핵심 튜닝 대상은 HGB·XGBoost·LightGBM·CatBoost·ExtraTrees의 5계열이다. 부스팅의 서로 다른 학습 방식과 무작위 분할 트리 평균을 비교한다. 기존 Ridge·Elastic Net·RBF SVR·HGB는 초기 결과의 참조로 유지하며 직전·전날·전주 단순 기준을 포함한다. 선형·커널 참조 모델을 이번 Optuna에서 다시 튜닝하지는 않았다. 따라서 모든 계열에 동일한 탐색량을 부여한 대규모 최적화라는 주장은 하지 않는다.

각 계열과 외부 월 조합에 기본 30회, 내부 선택 점수가 최초 20회의 최선보다 1% 이상 개선된 경우에만 50회 탐색했다. 학습률·트리 수·깊이 또는 잎 수·규제 범위는 [실행 전 조건](config/m02_rerun_contract.json)에 고정했다. 외부 결과를 보고 탐색 범위나 횟수를 늘리지 않았다. 각 시도는 모든 내부 월을 검증했고, 외부 평가값으로 조기 종료하거나 예측값을 사후 보정하지 않았다.

앙상블은 튜닝한 5계열의 동일 가중 평균과 내부 검증 점수 상위 3계열의 비음수 가중 평균을 비교했다. 가중 합은 1이고 구성원·가중치는 외부 평가 전에 내부 예측만으로 선택했다. 내부 예측은 설정·가중치 선택에 사용했으므로 독립적인 성능 추정치로 해석하지 않는다. 복잡한 스태킹은 이번 비교에 포함하지 않았다.

실제 실행은 Optuna 총 {run['trials_including_ensemble']:,}회, 내부 모델 학습 {run['tuning_fits']:,}회, 외부 모델 학습 {run['outer_model_fits']}회였다. 모든 시도는 정규 CPython 3.13에서 수행됐고 총 실행 시간은 약 {run['seconds']/60:.1f}분이었다. 기존 참조 예측은 해시를 대조한 후 재사용했다.

### 전체 정확도와 최대 시간 정확도의 공동 평가

전체 시간 MAE와 **실제 일별 최대 시간 MAE**를 두 주지표로 사용했다. 후자는 예측이 끝난 뒤 실제 일별 최대가 나타난 시각의 수치 오차를 평가하며, 같은 날 최대가 여러 번이면 그 날의 합계 가중치를 1로 나눠 91일을 같은 비중으로 비교한다. 내부 검증에서는 적격 24시간이 모두 있는 날짜만 이 지표에 포함한다. 실제 최대 시각은 사후 평가용이며 입력에 들어가지 않는다.

두 목적값을 Optuna에 각각 전달하고 어느 한 지표를 개선하려면 다른 지표를 양보해야 하는 파레토 후보를 보존했다. 최종 한 후보를 정할 때는 사전에 다음 규칙을 기록했다.

`선택 점수 = max(전체 MAE / 직전 기준 전체 MAE, 최대 시간 MAE / 직전 기준 최대 시간 MAE)`

이 규칙은 두 상대오차 중 더 큰 쪽을 줄이는 절충이다. 분모는 항상 같은 기록의 직전 값 기준이다. 현장 손실비용을 알 수 없는 상태에서 임의의 비용 가중치를 만들지 않고, 두 원래 지표도 각각 공개했다. 최종 개발 선택에서는 다른 방법에 두 지표 모두 뒤지는 후보를 제외하고, 최선 점수의 1% 이내이면 기존 방법과 단일 모델을 우선한다. 실측한 운영 손실함수나 최적의 경제적 판단을 뜻하지는 않는다.

주 목표는 전력 크기의 연속값이므로 MAE와 RMSE로 크기 오차를 평가한다. AUROC·AUPRC는 고전력 사건의 정의와 예측 점수가 있는 분류·순위 문제에서 따로 평가할 수 있다. 이번에는 분류 기준을 새로 만들지 않았으며, 최대 시간 MAE가 낮다는 사실을 최대 시각 탐지 성과로 바꾸어 해석하지 않는다.

### 전체 비교 결과와 HGB 유지 판단

{table}

*학습 모델은 기본 A 입력을 사용하며, 단순 기준을 포함한 모든 방법은 같은 2,184시간을 평가한다. 최대 시간 지표는 동률을 나눈 91일 가중 평균이다. 기존 모델 행은 과거 개발에서 선택한 설정의 보존 참조이며, 새 5계열은 각 외부 월 이전 내부 검증으로 설정을 선택했다.*

정한 규칙의 최선은 **기존 HGB**였다. ExtraTrees는 전체 MAE가 6.825로 기존 6.738보다 약 1.3% 컸지만 최대 시간 MAE는 11.540으로 기존 11.751보다 약 1.8% 작았다. 두 방법만 서로 한쪽 지표에서 앞서는 파레토 후보로 남았다. 기존 HGB의 선택 점수 0.4885가 ExtraTrees의 0.4948보다 작아 기존 모델과 설정을 유지했다. 이 자료에서는 두 방법 모두 최대 시간의 상대오차보다 전체 상대오차가 커서, 최종 절충 점수의 결정 요인은 전체 MAE였다. **최대 시간만을 가장 우선한다면 ExtraTrees를 선호할 수 있으므로 목적에 따른 차이를 함께 밝힌다.**

Optuna HGB·XGBoost·LightGBM·CatBoost와 두 앙상블은 이번 범위에서 기존 HGB보다 두 주지표가 모두 컸다. 따라서 알고리즘의 이름이나 복잡성을 이유로 교체하지 않았다. 이는 이번 입력·기간·탐색 범위의 결론이며, 해당 모델의 모든 설정이나 입력에서 기존 HGB가 더 좋다는 뜻은 아니다.

### 월별 변화와 남은 실패조건

{monthly_table}

가중 앙상블은 5·6월 전체 MAE를 기존 HGB보다 낮췄지만 4월에는 8.974로 기존 8.308보다 컸다. 학습에 없는 하루 배열 1,152시간에서도 7.957로 기존 7.558보다 컸다. 4월은 모든 평가 날짜가 학습에 없는 배열이라는 M01 결과와 함께 보면, 후반 개발월의 개선만으로 전체 안정성이 좋아졌다고 주장하기 어렵다. 이 차이를 배열 반복이 오류의 원인이라는 인과 설명으로 단정하지는 않는다.

{error_table}

ExtraTrees는 26 초과 유지에서 MAE를 낮췄으나 상승 전환 20시간에서는 60.550으로 기존 47.866보다 컸고 낮은 구간 유지 625시간에서도 5.162로 기존 3.766보다 컸다. 최대 시간에서 평균 과소예측은 기존 11.260→10.897로 줄었지만 평균 과대예측은 0.491→0.643으로 늘었다. 최대 시간 MAE의 작은 개선만으로 급상승·낮은 유지 문제까지 해결했다고 판단할 수 없다.

### 기존 M03과의 연결 및 다음 질문

모델과 설정을 바꾸는 조건이 충족되지 않아 **M03을 다시 실행하지 않았다.** 기존 HGB의 A→B·C 비교는 같은 모델·설정·표본의 검증으로 계속 유효하다. 현재 후보는 그 비교에서 선택한 B이며, 최대전력 MAE 6.191과 최대 시간 MAE 7.589라는 기존 결과를 유지한다. 이번 확장 M02는 A 입력의 모델 비교이므로 모든 모델에 마지막 값을 추가했을 때의 순위를 검증한 결과는 아니다. 특히 ExtraTrees B가 HGB B보다 나쁘다는 결론은 낼 수 없다.

후속 우선순위는 기존 M03의 남은 질문으로 돌아간다. 직전 낮은 상태는 예측 시점에 알 수 있고, B에서는 해당 650시간의 MAE가 5.466으로 A 5.117보다 컸다. 다음 최소 M04는 그 조건에서 직전 값 기준을 사용하는 규칙이 낮은 유지 오차를 줄이는 대신 상승 과소예측을 얼마나 늘리는지 비교하는 것이다. 규칙과 허용 범위를 먼저 정하고 같은 시간의 전체·최대 시간·상승 오류로 채택 또는 기각한다. 이번 회차에서는 규칙과 M05를 실행하지 않았다. 새로운 근거 없이 모델 탐색만 계속 늘리지 않는다.

### 실행 코드와 검증 근거

- [M02 확장 실행 기록](10.04_M02_확장모델비교_실행기록.md), [실행 전 조건](config/m02_rerun_contract.json)
- [학습·시간순 내부 검증·두 목적 튜닝·앙상블 실행 코드](scripts/m02_rerun.py), [모델 생성과 탐색 보조 코드](scripts/m022_compare.py)
- [전체 비교표](tables/m02_rerun/comparison.csv), [두 주지표와 선택 점수](tables/m02_rerun/selection_metrics.csv), [조건별 오류](tables/m02_rerun/condition_errors.csv)
- [월별 설정과 앙상블 가중치](tables/m02_rerun/selected_configurations.json), [Optuna 모든 시도](tables/m02_rerun/trials.csv), [시간순 분할 감사](tables/m02_rerun/fold_audit.csv)
- [시각별 예측](tables/m02_rerun/predictions.csv), [개발 선택 판단](tables/m02_rerun/selection.json), [실행 환경·해시](tables/m02_rerun/run.json)
- [독립 검증 코드](scripts/m02_rerun_verify.py), [독립 검증 결과](tables/m02_rerun/independent_verification.json)

독립 검증은 표준 라이브러리로 원본 시차값 33,408개, 예측 30,576행, 지표 1,988행, 튜닝 700회, 설정·앙상블 21개를 대조했다. 시간순 학습 경계, 동일 2,184시각, 91일 최대 가중치, 저장 모델 해시·앙상블 계산·최종 선택의 일치가 확인됐고 종료 코드는 0이었다. 원고 압축과 후반기 평가는 별도 단계로 남긴다.
'''
    manuscript = ROOT / 'Modeling/04_Modeling_원고.md'
    text = manuscript.read_text(encoding='utf-8')
    assert '## 4.4 M02 확장 재실행:' not in text
    text = text.replace('이번 원고는 **M01의 예측 설계·자료 검증, M02의 기본 모델 비교, M03의 마지막 값·계절 시차 입력 비교**를 다룬다.',
                        '이번 원고는 **M01의 예측 설계·자료 검증, M02의 기본 및 확장 모델 비교, M03의 마지막 값·계절 시차 입력 비교**를 다룬다.')
    manuscript.write_text(text + section, encoding='utf-8')
    record = ROOT / 'Modeling/10.04_M02_확장모델비교_실행기록.md'
    appendix = f'''\n\n## 실행 결과와 판단 갱신\n\n{table}\n\n기존 HGB를 유지했다. ExtraTrees의 최대 시간 MAE는 약1.8% 작지만 전체 MAE는 약1.3% 크며, 정한 선택 점수는 기존 HGB가 더 낮다. 상승20시간·낮은 유지625시간에서도 ExtraTrees의 MAE가 더 컸다. 가중 앙상블의5·6월 개선은4월·새 배열 악화와 함께 제시한다. 두 지표가 모두 더 좋은 새 후보는 없었다.\n\n기존 HGB의 설정과 입력A가 그대로이므로 M03 재실행 조건이 충족되지 않았다. 기존 M03 결과와 B 개발 후보를 유지하며 ExtraTrees B의 우열은 미검증이다. 다음 최소 질문은 예측 시점에 알려진 직전 낮은 상태에서 직전 기준을 쓰는 규칙의 이익과 상승 과소예측 손실이다. M04 규칙·M05는 이번에 실행하지 않았다.\n\nOptuna700회·내부학습1630회·외부학습15회·약5.6분,예측30576행·지표1988행·원본시차33408값·700시도·21설정 독립검증종료0. 원본과 기존M01/M02/M03해시를 보존했다. 모든결과는 개발선택이며 후반기 독립시험 성과가 아니다.\n\n[전체 비교](tables/m02_rerun/comparison.csv) · [공동 지표·선택](tables/m02_rerun/selection_metrics.csv) · [오류](tables/m02_rerun/condition_errors.csv) · [설정](tables/m02_rerun/selected_configurations.json) · [선택 판단](tables/m02_rerun/selection.json) · [실행 기록](tables/m02_rerun/run.json) · [독립 검증](tables/m02_rerun/independent_verification.json) · [상세 원고4.4](04_Modeling_원고.md)\n'''
    record.write_text(record.read_text(encoding='utf-8') + appendix, encoding='utf-8')
    modeling = ROOT / 'Modeling/README.md'
    text = modeling.read_text(encoding='utf-8')
    text = text.replace('2026-10-04 M03 실행 갱신.', '2026-10-04 M02 확장 재실행 갱신.')
    text = text.replace('M01 검증, M02 네 계열 비교, M03 A/B·C 입력 비교와 기본 M04 진단 완료',
                        'M01 검증, M02 기본·확장 모델 비교, M03 A/B·C 입력 비교와 기본 M04 진단 완료')
    marker = '### 참고: 실제 시간순 전력과 변화량'
    update = '''### M02 확장 재실행과 현재 선택

[확장 실행 기록](10.04_M02_확장모델비교_실행기록.md) · [상세 원고4.4](04_Modeling_원고.md) · [사전 조건](config/m02_rerun_contract.json) · [두 주지표](tables/m02_rerun/selection_metrics.csv) · [독립 검증](tables/m02_rerun/independent_verification.json).

사용자 요청으로 A입력에서 Optuna·시간순 내부 검증을 적용해 HGB/XGBoost/LightGBM/CatBoost/ExtraTrees와 평균·가중 앙상블을 확장 비교했다. 기존 세 선형·커널 모형과 HGB, 단순기준3개를 포함한14방법이다. 전체 MAE·실제 일별 최대 시간 MAE를 각각 공개하고 직전 기준 대비 상대오차 중 더 큰 값을 줄이는 규칙을 실행 전에 고정했다.

**기존 HGB를 유지한다.** 전체/최대시간MAE는기존6.738/11.751,ExtraTrees6.825/11.540,가중앙상블6.872/13.035였다. ExtraTrees의최대시간개선과전체·상승·낮은유지악화를구분한다. 새5계열은정한범위에서두지표모두개선한후보가없었다. 기존HGB는초기개발에서설정을선택한참조이고새5계열은평가월이전내부검증으로선택했다는차이를밝힌다.

M02의외부예측은1~3월→4월720시간,1~4월→5월744시간,1~5월→6월720시간으로모든적격시각을평가한다. 내부에서는1월→2월,1~2월→3월등으로확장한다. 평가월모델은고정하고관측입력만매시간갱신한다. 7~8월은후속M05이며이번튜닝·선택에쓰지않았으나과거관찰이력이있어완전한독립시험으로부르지않는다.

Optuna700회·내부학습1630회·외부학습15회,30576예측행·1988지표행독립검증완료다. 모델/설정이그대로이므로M03재실행조건은충족되지않았다. 기존M03의B후보와결과를유지하며ExtraTrees B의우열은미검증이다. 다음최소질문은기존M04의직전낮은상태규칙의상충이며새모델탐색·규칙·M05를실행한것으로확대하지않는다.

'''
    assert marker in text
    text = text.replace(marker, update + marker)
    text = text.replace('4.3은 마지막 값과 계절 시차의 추가 효과·상충과 다음 질문을 설명한다.',
                        '4.3은 마지막 값과 계절 시차의 추가 효과·상충을, 4.4는 Optuna·시간순 검증·확장 모델과 HGB 유지 판단을 설명한다.')
    modeling.write_text(text, encoding='utf-8')
    rootreadme = ROOT / 'README.md'
    text = rootreadme.read_text(encoding='utf-8')
    marker = '같은 수의 관리시간으로 높은 최대전력 시간을 더 잘 찾는 성과는 별도 검증이 필요하다.'
    update = '''[M02 확장 재실행](Modeling/10.04_M02_확장모델비교_실행기록.md): 기본A입력에서HGB/XGBoost/LightGBM/CatBoost/ExtraTrees의Optuna시간순튜닝과두앙상블을비교했다. 전체·실제일별최대시간MAE를함께평가한규칙에서기존HGB6.738/11.751을유지했고,ExtraTrees6.825/11.540의상충을공개했다. 모델·설정이같아M03재실행조건은충족되지않았으며기존B후보를유지한다. Optuna700회·독립검증완료,상세원고4.4에결과를연결했다. M02는1~3월→4월,1~4월→5월,1~5월→6월의모든적격2184시간이며내부검증은1월→2월부터순차확장한다. 7~8월은후속M05구간으로이번선택에미사용이며과거관찰이력은유지한다.

'''
    assert marker in text
    text = text.replace(marker, update + marker)
    text = text.replace('다음 최소 질문은 직전 낮은 상태에서 단순 기준을 쓰는 규칙이',
                        '확장 M02에서도 기존 HGB를 유지했고, 다음 최소 질문은 직전 낮은 상태에서 단순 기준을 쓰는 규칙이')
    rootreadme.write_text(text, encoding='utf-8')
    # Fresh-read immediately before upserting durable project memory.
    memory_path = ROOT / 'AGENT_MEMORY.md'
    text = memory_path.read_text(encoding='utf-8')
    minute = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='minutes')
    import re
    text = re.sub(r'(?m)^- Last updated: .*$', f'- Last updated: {minute}', text, count=1)
    start = text.index('- `decision:modeling-m02-rerun-metric-first`')
    end = text.index('\n- `decision:', start+1)
    old = text[start:end]
    created = re.search(r'  - Created: (.*)', old).group(1)
    new = f'''- `decision:modeling-m02-rerun-metric-first`
  - Created: {created}
  - Updated: {minute}
  - Status: active
  - Content: 사용자 승인으로 A입력14방법의 확장M02를 실행했다. 핵심5계열은 Optuna 두MAE·시간순중첩검증, 직전 기준 상대오차의 최댓값 선택 규칙을 사전 기록했다. 700시도·내부1630/외부15학습·30576예측/1988지표 독립검증통과. 기존HGB6.738/11.751유지, ExtraTrees6.825/11.540의상충공개; 모델/설정불변으로M03재실행조건미충족·기존B유지·ExtraTrees B미검증. 외부1~3월→4월/1~4월→5월/1~5월→6월모든적격2184시간, 내부1월→2월부터확장. 7~8월이번선택미사용·과거관찰이력유지. 중단m022부분결과는선택근거제외.
  - Evidence: Modeling/config/m02_rerun_contract.json, Modeling/tables/m02_rerun/selection.json, Modeling/tables/m02_rerun/independent_verification.json, Modeling/04_Modeling_원고.md4.4.
'''
    text = text[:start] + new + text[end:]
    start = text.index('- `handoff:current`')
    end = text.index('\n## Session log', start)
    text = text[:start] + f'''- `handoff:current`
  - Updated: {minute}
  - Current state: M01·초기M02·M03와 확장M02 완료. 확장A14방법·700시도·내부1630/외부15학습·30576예측/1988지표독립검증종료0. 기존HGB6.738/11.751유지·ExtraTrees6.825/11.540상충, HGB설정불변으로M03재실행불필요·기존B최대MAE6.191/최대시간MAE7.589유지. 원고4.4·확장기록·두README에평가구간/선택이력/결과연결반영. 새압축본/PDF/M05미실행.
  - Next step: 기존M03에서직전낮은상태650시간B MAE5.466>A5.117이었다. 그조건에서직전기준을쓰는최소M04규칙의낮은유지개선과상승과소예측악화를같은시각에서비교하고채택/기각한다. 새로운실행조건과허용범위는실행전기록한다. 후반기M05는1~6월→7~8월이며과거관찰이력상완전한독립시험으로부르지않는다.
  - Blockers: 없음. 원본/기존M01~M03보존, 추가자료대기없음. 중단m022는미완료참고기록이고현재결과아님.
''' + text[end:]
    session = text.index('- `session:20261004-1056`')
    session_end = text.index('\n- `session:', session+1)
    block = text[session:session_end]
    block = re.sub(r'  - Last activity: .*', f'  - Last activity: {minute}', block, count=1)
    block += f'\n  - Expanded M02 execution update: 사용자최신승인으로A14방법·두목적Optuna700시도·시간순중첩비교를실행해기존HGB유지를판단했다. 30576예측/1988지표·원본33408시차값·21설정/앙상블·시간순경계·최종선택 독립검증종료0. ExtraTrees최대시간개선/전체·상승·낮은유지악화를공개하고M03재실행조건미충족을기록했다. 원고4.4·실행기록·두README를갱신했으며사용자질문에학습/내부검증/외부평가기간과모든적격시간예측을구분해설명했다. 규칙M04/M05/압축/PDF미실행.\n'
    text = text[:session] + block + text[session_end:]
    memory_path.write_text(text, encoding='utf-8')
    print(json.dumps({'documents_written': 5, 'selected': selection['selected']['model'], 'updated_at': minute}), flush=True)


if __name__ == '__main__':
    main()
