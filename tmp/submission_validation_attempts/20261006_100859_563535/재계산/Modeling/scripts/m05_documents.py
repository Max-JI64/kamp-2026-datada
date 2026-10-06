"""Write and verify the M05 detailed manuscript from independently checked results."""
import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from m03_verify import rows, sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Modeling/tables/m05'
MD=ROOT/'Modeling/04_Modeling_원고.md'
LOG=ROOT/'Modeling/10.05_M05_고정후반기평가_실행기록.md'
TITLE='## 4.9 M05: 고정 후반기 평가와 성과 범위의 갱신'


def metric(group, condition='all', period='pooled', pool='common'):
    return next(r for r in rows(OUT/'metrics.csv') if r['pool']==pool and r['group']==group and r['period']==period and r['condition']==condition and r['value']=='all')


def comparison():
    text='| 방법 | 전체 MAE | 실제 일별 최대 시간 MAE | 최대 시간 평균 과소 | 최대 시간 평균 과대 |\n|---|---:|---:|---:|---:|\n'
    for name in ['lag1','lag24','lag168','A','B','G_aux']:
        overall,peak=metric(name),metric(name,'daily_maximum')
        text+=f"| {name} | {float(overall['MAE']):.3f} | {float(peak['MAE']):.3f} | {float(peak['mean_under']):.3f} | {float(peak['mean_over']):.3f} |\n"
    return text


def months():
    text='| 月 | A 전체 MAE | B 전체 MAE | A 최대 시간 MAE | B 최대 시간 MAE |\n|---|---:|---:|---:|---:|\n'.replace('月','월')
    for month in ['7','8']:
        values=[metric(name,cond,month) for cond in ['all','daily_maximum'] for name in ['A','B']]
        text+=f'| {month}월 | '+' | '.join(f"{float(r['MAE']):.3f}" for r in values)+' |\n'
    return text


def alarms():
    records=rows(OUT/'risk_metrics.csv')
    text='| 기간 | 위험 점수 | 급등 수 | TP / FP / FN / TN | 재현율 | 정밀도 | 실제 오경보율 | AP |\n|---|---|---:|---|---:|---:|---:|---:|\n'
    for period in ['pooled','7','8']:
        for group in ['B_score','G_HGB_score']:
            r=next(r for r in records if r['group']==group and r['period']==period and r['cap']=='0.01' and r['boundary']=='93.0')
            text+=f"| {'7~8월' if period=='pooled' else period+'월'} | {group} | {r['events']} | "+' / '.join(r[key] for key in ['TP','FP','FN','TN'])+' | '+' | '.join(f'{100*float(r[key]):.1f}%' for key in ['recall','precision'])+f" | {100*float(r['false_positive_fraction']):.3f}% | {float(r['AP']):.3f} |\n"
    return text


def section():
    diagnostics=json.loads((OUT/'confirmed_diagnostics.json').read_text(encoding='utf-8'))
    c=json.loads((ROOT/'Modeling/config/m05_contract.json').read_text(encoding='utf-8'))
    base_a,base_b=metric('A'),metric('B')
    core_a,core_b=metric('A',pool='core'),metric('B',pool='core')
    core_peak_a,core_peak_b=metric('A','daily_maximum',pool='core'),metric('B','daily_maximum',pool='core')
    limits={r['group']:r['threshold'] for r in c['alarms']['thresholds'] if r['cap']==.01}
    gain=100*(1-float(metric('G_aux','surge_q95')['MAE'])/float(metric('B','surge_q95')['MAE']))
    return f'''{TITLE}

M03에서 마지막 값 추가 B는 개발 전체 MAE와 실제 일별 최대 시간 과소예측을 줄였지만, M04에서는 상승 전환·낮은 유지의 상충이 남았다. 일반 급등에서 기존 결합의 크기 오차가 줄고 위험 점수도 일부 탐지를 늘려 후속 후보로 연결했다. M05는 이 두 효과가 후반기에도 유지되는지 확인하기 위해 실행했다. [실행 조건](config/m05_contract.json)은 {c['recorded_at']}에 기록했으며, 그 뒤 모델을 학습하고 후반기 예측을 생성했다.

### 학습·평가와 고정한 정보

1~6월 공통 4,176시간으로 A·B HGB, 상승 가중 전문가, 직전 낮은 상태 HGB 분류기를 각각 한 번 학습했다. 분류기의 실제 학습 대상은 직전 낮은 상태 1,305시간이며 가장 늦은 대상은 6월28일07시다. 모델4개를 저장·재로딩했다. 평가 중에는 매시간 t-1 관측을 입력에 반영하되 모델·결합 규칙·경보 경계를 고정했다. 7월 정답을 추가한 8월 재학습이나 후반기 Optuna 탐색은 없었다.

주 비교는 단순 기준3개와 A/B를 같은 공통 1,344시간·56일에서 평가했다. 7월600시간·8월744시간이며 실제 일별 최대 시간은77행·날짜 가중치 합56이다. 기본 시차만 필요한1,436시간은 같은 A/B 모델의 추론 범위를 확장한 보조 평가다. 보조 표본에서 다시 학습하지 않았고 두 표본 점수를 섞지 않았다. 후반기는 이전 EDA 관찰 이력이 있어 완전히 처음 보는 독립 시험으로 표현하지 않는다.

A는17입력, B는 직전 마지막 값을 더한18입력이고 기존 HGB 잎15·반복150·학습률0.05·최소 잎 표본20·L2 1·조기종료 안 함·시드42를 고정했다. 위험 후보는 마지막 개발월6월의 설정을 선택한 뒤 1~6월로 재적합했다. 상승 학습 가중치1.730905, 분류기 잎15·최소 잎 표본20·L2 0.155280, 결합 확률 경계0.373766·비율0.822355를 사용했다. 이는 후반기 점수가 가장 좋은 설정을 고른 결과가 아니다.

급등 정답 경계는1월 양의 증가95분위수93을 유지했다. 경보 수치 경계는 기존4~6월의 시간순 외부 예측에서 비급등 점수의 음성 꼬리1%로 산정했다. B 위험 점수 경계는 {limits['B_score']:.6f}, 기존 결합 위험 점수는 {limits['G_HGB_score']:.6f}이며 두 평가 월에 같은 값을 썼다. 급등 정답 경계93, 결합 확률 경계, 경보 점수 경계는 각각 다른 역할이다. 학습 자료에 재대입한 예측이나7~8월 정답으로 경계를 고르지 않았다. [경보 산정표](tables/m05/alarm_calibration.csv)를 보존한다.

### 전체 및 실제 최대 시간의 비교

{comparison()}
G_aux는 위험 점수를 만들기 위한 보조 결합값이며 정상 전력 출력은 B다. 해당 행의 회귀 오차가 조금 낮더라도 이를 최종 전력 모델 교체로 해석하지 않는다.

B의 전체 MAE는 직전 기준12.678보다 {diagnostics['baseline_reduction_percent']:.1f}% 작았다. 실제 최대 시간 MAE도19.830→13.803으로 감소했다. A 역시 단순 기준보다 낮은 오차를 보였다. 따라서 과거 정보와 달력을 사용하는 HGB의 기본 예측 성과는 후반기에서도 확인됐다. 이는 마지막 값 한 변수의 추가 효과와 구분되는 모델 성과다.

A→B 전체 MAE는6.683→6.621로 {diagnostics['overall_reduction_percent']:.2f}% 감소했으나, 실제 최대 시간 MAE는13.376→13.803으로 {diagnostics['peak_MAE_change_percent']:.2f}% 증가했다. 최대 시간 평균 과소도12.936→13.341로 증가했다. 개발에서 나타난 전체8.1%·최대 시간 과소37.9% 감소가 후반기에서 같은 방향과 크기로 유지된 것은 아니다.

실행 전 확인 기준은 A 대비 전체·최대 시간 MAE 각각1% 이상 감소, 최대 시간 평균 과소 감소, 직전 기준보다 낮은 전체·최대 시간 MAE, 각 월의 A 대비 MAE 악화1% 이내, 배열 재가중 전체 MAE 비악화였다. 1%는 기존 수치상 동률 규칙을 이어받은 것이며 현장 비용의 최적 허용값이 아니다. [판단 파일](tables/m05/conclusion.json)에서 전체·최대 시간·과소·월별 요건이 실패했다. 단순 기준 개선과 배열 재가중 요건은 통과했지만, 마지막 값의 광범위한 효과는 확인되지 않았다.

### 월별·전력 상태·적용 범위의 차이

{months()}
7월은 전체·최대 시간 모두 B가 A보다 나빠졌고, 8월은 전체는 개선했지만 최대 시간 MAE가 소폭 증가했다. 후반기 공통 표본의 학습과 동일한 하루 배열은0일이었다. 전체 점수 하나만으로 개발의 효과가 새로운 배열에서도 안정적으로 유지됐다고 주장하지 않는다.

26초과 유지795시간의 MAE는 A8.041→B7.852, 26초과에서 낮은 상태로 내려간10시간은14.012→9.125로 감소했다. 반면 낮은 유지508시간은3.418→3.610, 상승 전환10시간은28.400→29.482로 증가했다. 전력 크기나 변화 방향에 따라 입력 추가의 이득이 달랐다. [조건별 결과](tables/m05/metrics.csv)와 [같은 시간의 A/B 오차 차이](tables/m05/paired_predictions.csv)를 남겼다.

보조1,436시간에서도 전체 MAE는 {float(core_a['MAE']):.3f}→{float(core_b['MAE']):.3f}로 줄었지만 최대 시간은 {float(core_peak_a['MAE']):.3f}→{float(core_peak_b['MAE']):.3f}로 늘었다. 주 표본에서 제외된 시간으로 범위를 넓혀도 최대 시간의 상충은 남았다. 좋은 점수의 표본만 선택해 결론을 바꾸지 않았다.

8월28일18시~29일10시의0값17시간에서 B는 평균24.047을 과대예측했다. 그 뒤 첫 양수 시간의 실제 최대값41은23.559로 예측해17.441만큼 작게 예상했다. 이 사례의 증가41은 주 급등 경계93에 미달하므로 급등 미탐지6건과 같은 사건으로 세지 않는다. 학습에는 최대값0 기록이 없었고, 실제0을 입력으로 받았을 때의 성능은 별도 한계로 확인됐다. 원인이나 설비 휴무 여부는 현 자료에서 식별하지 못했다. [0값 전후 사례](tables/m05/zero_cases.csv)를 보존한다.

### 급등 위험 출력의 탐지와 오경보

후반기 주 급등은6시간·6일이며7·8월 각각3건이다. 이 중4건은 직전 낮은 상태,2건은26초과 상태였다. 경보1% 조건의 결과는 다음과 같다.

{alarms()}
월별 AP의 사건 수 가중 평균은 B0.328→기존 결합0.488로 증가했다. 급등6건 중 탐지는3→4건, 오경보는16→14건으로 개선됐으나 실제 오경보율은1.196%→1.046%로 후보도 사전1% 요건을 넘었다. 정밀도는15.8%→22.2%로, 후보 경보18건 중 실제 급등은4건이다. 재현율66.7%만 제시하면 경보 부담을 판단할 수 없다.

7월에는 B1건→후보3건을 찾았지만8월에는 B2건→후보1건으로 감소했다. 8월9일08시는 두 점수가 모두57.887이었다. 직전 상태가26초과여서 결합값은 B와 같았지만, B 경계56.267은 넘고 후보 경계58.080은 넘지 못했다. 따라서 이 한 건의 추가 미탐지는 분류기의 순위 악화가 아니라 방법별 고정 경보 경계 차이에서 생겼다. 8월16일07시는 두 점수가 모두 경계 아래여서 두 방법 모두 놓쳤다. [급등6건](tables/m05/surge_cases.csv)·[검증된 점수/경보 사례](tables/m05/confirmed_diagnostics.json)를 함께 확인했다. 이 사례를 본 뒤 경계를 다시 낮추지는 않았다.

위험 후보는 가중 AP5% 이상 개선·합계 탐지 비악화·각 월 탐지 비악화·실제 오경보율1% 이내·탐지 증가 또는 오경보 감소를 함께 만족해야 했다. 합계 순위·탐지·오경보 수는 개선했지만 월별 탐지와 실제 오경보율이 실패해 최종 위험 채널의 안정적 우위를 확인하지 못했다. 주 급등 크기 MAE 자체는 B38.417→보조 결합32.100으로 {gain:.1f}% 감소했다. 이는 보조 결합값의 부분 성과이며 최종 채택을 의미하지 않는다.

증가61.6 이상은25시간·17일이었다. 보조 경계에서는 순위·정밀도 개선도 있었으나 사전에 민감도 점검으로 정한 범위다. 또한5% 경보 허용에서는 B가6건 모두 찾는 대신 오경보47건을 냈다. 이 조건의 좋은 탐지율로 주1% 평가를 대체하거나 '급등을 모두 잘 찾았다'고 주장하지 않는다. [전체 위험 지표](tables/m05/risk_metrics.csv)에 모든 조건을 보존했다.

### 보고서와 다음 작업의 판단 갱신

확인된 성과는 후반기에서도 HGB가 직전·전날·전주 기준보다 전체·실제 최대 시간 오차를 줄였다는 점이다. 유지할 근거는 M02의 동일 조건 모델 비교와 M03의 개발 입력 기여 결과이며, 수정할 주장은 마지막 값과 급등 결합이 후반기까지 안정적으로 최대 시간의 과소예측을 줄였다는 표현이다. 현재 결과에서는 그 강한 주장을 입증하지 못했다.

최종 보고서 제2장은 기본/확장 모델 선택과 후반기 단순 기준 대비 HGB 성과를 중심으로 구성할 수 있다. 마지막 값의 개발 개선은 후반기 결과와 함께 설명해야 한다. 사용자의 성과 중심 선별 방향에 따라 미채택 급등 분류·결합의 상세 실험표는 최종 성과 본문에서 제외할 수 있으나, 제3장에는 선택 후보의 상승·0구간·최대 시간 오류를 짧게 남긴다. 제5장에서 '급등 탐지 필살기를 최종 입증했다'고 쓰지 않는다. 상세 원고·코드·모델·검증 근거는 보존한다.

이번 M05는 선언한 고정 확인 평가로 종료한다. B의 개발 후보와 보조 위험 후보를 후반기 결과로 재튜닝하지 않았고, B의 최종 채택이나 위험 후보의 운영 적용을 확정하지 않았다. 다음 판단은 기존 A/B의 상충을 반영해 보고서의 주 성과와 모델 선택 근거를 정리할지, 현재 자료에서 상승·0구간에 대한 구체적인 새 가설을 세워 별도 개발을 시작할지다. 새 개발을 선택하면7~8월도 선택에 이용한 기간임을 기록하며 재평가를 새로운 독립 확인으로 표현하지 않는다. 현장 활용 제안은 확인된 범위의 성능과 실패조건에 맞춰야 한다.

### 실행 코드와 검증

[m05_evaluate.py](scripts/m05_evaluate.py)의 `--stage freeze`→`--stage run`으로 조건·경보 경계를 먼저 기록하고 학습·추론·결과를 생성했다. [m05_verify.py](scripts/m05_verify.py)는 원본에서149,040입력값,10,936예측행,1,000지표행,24경보행을 대조하고4모델을 재로딩했다. [m05_documents.py](scripts/m05_documents.py)는 검증된 결과를 상세 원고와 연결 기록에 반영하고 표·문서 링크를 검증한다. 새 Optuna 탐색은0회, 학습은4회이며 원자료·기존M01~M04 결과를 보존했다.

[실행 기록](tables/m05/run.json)·[독립 검증](tables/m05/independent_verification.json)·[저장 모델 목록](tables/m05/model_manifest.csv)·[학습 범위](tables/m05/training_audit.csv)·[전체 예측](tables/m05/predictions.csv)·[별도 위험 출력](tables/m05/risk_predictions.csv)·[일별 오류](tables/m05/daily_errors.csv)·[큰 오류](tables/m05/largest_errors.csv)를 함께 보존한다. 원고4.5의 전체 Python35개 목록과 제출 보존 목록도 갱신한다. 압축본·최종 제출·파일 삭제·실제 절감 실측은 이번 실행에 포함되지 않았다.
'''


def write():
    source=MD.read_text(encoding='utf-8')
    assert TITLE not in source
    source=source.replace('사용한 32개 파일이다.','사용한 35개 파일이다.')
    marker='| [modeling_submission_audit.py]'
    code_rows='''| [m05_evaluate.py](scripts/m05_evaluate.py) | 4.9 개발 설정·경보 경계 고정 후 후반기 학습·추론·평가 | `--stage freeze` → `--stage run` |
| [m05_verify.py](scripts/m05_verify.py) | 4.9 원본 입력·시간 경계·모델 재로딩·전체 지표와 판단 독립 검증 | `main()`; 새 학습 없음 |
| [m05_documents.py](scripts/m05_documents.py) | 4.9 검증 결과의 상세 원고·기록·README 반영과 문서 검증 | `--write` 또는 `--verify` |
'''
    source=source.replace(marker,code_rows+marker,1)
    source=source.replace('[경보 보정](config/m04_surge_calibration_contract.json).','[경보 보정](config/m04_surge_calibration_contract.json), [M05](config/m05_contract.json).')
    source=source.replace('`m04_surge_calibration/`, `observed_timeline/`','`m04_surge_calibration/`, `m05/`, `observed_timeline/`')
    source=source.replace('`m04_surge_risk/`의 파일.','`m04_surge_risk/`, `m05/`의 파일.')
    marker='| 14 | `Modeling/scripts/m02_rerun_document_verify.py`'
    source=source.replace(marker,'| 14 | `Modeling/scripts/m05_evaluate.py --stage freeze` → `--stage run` → `Modeling/scripts/m05_verify.py` | 후반기 조건 고정·학습·추론·독립 검증 |\n| 15 | `Modeling/scripts/m02_rerun_document_verify.py`')
    source=source.replace('`Modeling/scripts/m04_surge_documents.py --verify` | 기존','`Modeling/scripts/m04_surge_documents.py --verify`, `Modeling/scripts/m05_documents.py --verify` | 기존')
    source=source.replace('| 15 | `Modeling/scripts/modeling_submission_audit.py','| 16 | `Modeling/scripts/modeling_submission_audit.py')
    MD.write_text(source.rstrip()+'\n\n'+section().rstrip()+'\n',encoding='utf-8')
    LOG.write_text('# M05 고정 후반기 평가 실행 기록\n\n'+section(),encoding='utf-8')
    paragraph='''M05는1~6월 공통4,176시간으로4개 모델을 한 번씩 학습하고 고정해7~8월 공통1,344시간을 평가했다. 정상B의 전체MAE6.621은 직전12.678보다47.8%작지만 A6.683 대비 개선은0.92%였고 최대시간MAE13.376→13.803은3.19%악화했다. 급등6건의 위험 후보는TP3→4·FP16→14였으나8월TP2→1과 실제오경보율1.046%로 사전 요건을 충족하지 못했다. 기본HGB 성과는 유지되지만 마지막 값·급등 위험의 강한 후반기 효과는 미확인이다. 원본149,040입력·10,936예측·4모델 재로딩 독립검증을 완료했고 후반기 결과로 재튜닝하지 않았다.'''
    for path,target in [(ROOT/'README.md','Modeling/10.05_M05_고정후반기평가_실행기록.md'),(ROOT/'Modeling/README.md','10.05_M05_고정후반기평가_실행기록.md')]:
        text=path.read_text(encoding='utf-8')
        assert target not in text
        position=text.index('\n\n',text.index('# '))
        text=text[:position]+f'\n\n[M05 현재 결과]({target}): {paragraph}'+text[position:]
        if path.parent.name=='Modeling':
            text=text.replace('4.5는 전체32코드 보존','4.5는 전체35코드 보존').replace('사용한 Python 코드 32개','사용한 Python 코드 35개')
            text=text.replace('4.8은 일반 급등 재평가와 별도 위험 출력의 탐색적 후보를 설명한다.','4.8은 일반 급등 재평가와 별도 위험 출력의 탐색적 후보, 4.9는 고정 후반기 성과와 주장 범위의 갱신을 설명한다.')
            text=text.replace('날씨 D는 보류이며 후반기 M05는 아직 실행하지 않았다.','날씨 D는 보류이며 후반기 M05의 최신 결과와 판단은 문서 상단 및 원고4.9를 따른다.')
        else:
            text=text.replace('마지막 값의 개발 개선은 확인했고 M05 후반기 안정성과 최종 필살기 채택은 아직 미확인이다.','마지막 값의 개발 개선은 확인했지만 M05에서는 전체 개선0.92%와 최대시간 악화가 나타나 안정적 기여를 확정하지 못했다. 최신 판단은 문서 상단 및 원고4.9를 따른다.')
        path.write_text(text,encoding='utf-8')


def verify():
    source=MD.read_text(encoding='utf-8')
    assert source.count(TITLE)==1 and source.split(TITLE,1)[1].split('\n## 4.10',1)[0].rstrip()==section().split(TITLE,1)[1].rstrip()
    assert LOG.read_text(encoding='utf-8').split(TITLE,1)[1].rstrip()==section().split(TITLE,1)[1].rstrip()
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',source):
        if not target.startswith(('http:','https:','#','app:')):
            assert (MD.parent/target.split('#')[0].strip('<>')).resolve().exists(),target
    result={'status':'passed','checked_at':datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='minutes'),
        'script_sha256':sha(Path(__file__)),'manuscript_sha256':sha(MD),'execution_record_sha256':sha(LOG),
        'comparison_rows':6,'monthly_rows':2,'alarm_rows':6,'no_fitting':True}
    (OUT/'document_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    assert args.write!=args.verify
    run=json.loads((OUT/'run.json').read_text(encoding='utf-8'))
    independent=json.loads((OUT/'independent_verification.json').read_text(encoding='utf-8'))
    assert independent['status']=='passed' and independent['run_sha256']==sha(OUT/'run.json')
    for name,digest in run['outputs_sha256'].items():
        assert sha(OUT/name)==digest
    if args.write:
        write()
    verify()


if __name__=='__main__':
    main()
