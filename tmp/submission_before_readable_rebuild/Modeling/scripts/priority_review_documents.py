"""Preserve and extend the detailed manuscript with bounded positive evidence."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/priority_review"
MANUSCRIPT = ROOT / "Modeling/04_Modeling_원고.md"
HEADING = "## 4.11 하루 2회 검토를 위한 과소예측 우선순위"
README_HEADING = "## 하루 2회 검토 우선순위: 개발 성과 반영"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def section():
    metrics = pd.read_csv(OUT/"metrics.csv",encoding="utf-8-sig").set_index(["method","period"])
    sensitivity = pd.read_csv(OUT/"sensitivity.csv",encoding="utf-8-sig").set_index("reference")
    labels = {"high_prediction":"예측값이 높은 시간", "prior_maximum":"직전 최대값이 높은 시간",
              "neighbor_std":"과거 유사 상태의 변동이 큰 시간", "selected_simple":"이전 월에서 선택한 단순 점수",
              "selected_ML":"과소예측 보조 모델"}
    table = ["| 검토 우선순위 점수 | 확인 시간 | 큰 과소예측 선별 | 확인 중 적중 비율 | 전체 사건 재현율 |",
             "|---|---:|---:|---:|---:|"]
    for method in labels:
        r=metrics.loc[(method,"pooled")]
        table.append(f"| {labels[method]} | {int(r.reviews)} | {int(r.TP)} | {r.precision*100:.1f}% | {r.recall*100:.1f}% |")
    monthly = ["| 월 | 높은 예측값 기준 | 유사 상태 변동 기준 | 이전 월 선택 기준 | 보조 모델 |",
               "|---|---:|---:|---:|---:|"]
    for month in [4,5,6]:
        monthly.append("| "+str(month)+"월 | "+" | ".join(str(int(metrics.loc[(m,str(month)),"TP"])) for m in
                    ["high_prediction","neighbor_std","selected_simple","selected_ML"])+" |")
    return HEADING+"""

### 예측오류 분석에서 검토 우선순위로 연결

기존 HGB B는 다음 시간의 최대값을 예측한다. 이후 오류분석에서는 이 예측이 실제보다 크게 낮아질 가능성을 보조 점수로 계산했다. 고정 경고 경계의 성과는 월별로 달랐지만, 사후에 같은 수의 시간을 골랐을 때 보조 점수의 선별력이 높아지는 단서가 있었다. 이에 실제 확인 횟수를 제한하고 미래 점수를 보지 않아도 개선이 남는지 비교했다.

이번 비교의 목표는 **하루 2회라는 동일한 확인량에서 큰 과소예측을 더 많이 선별하는 것**이다. 점 예측값 자체를 수정하거나 급등의 발생 여부를 다시 정의하지 않았다. 정상 예측과 검토 우선순위를 함께 제시하는 보완 방식으로 설계했다.

### 사건 정의와 시간순 보조 점수

큰 과소예측은 실제 다음 시간 최대값이 B 예측보다 9.738515 이상 큰 경우다. 이 경계는 2월의 학습 밖 시간순 예측 오차에서 정한 과소예측량 90분위수로 고정했다. 현장의 위험 한계나 비용 기준은 아니며, 이전의 시간당 증가량 93 이상인 급등과 구분한다. 4 ~ 6월 2,184시간 중 해당 사건은 184건이었다.

보조 모델의 입력은 기존 B의 18개 변수, B 예측값, 과거 유사 상태의 다음 값 표준편차·범위·20번째 이웃 거리·이웃 평균과 B 예측의 차이로 총 23개다. 유사 상태는 해당 평가월보다 앞선 자료에서만 표준화하고 가까운 20개를 선택했다. 보조 모델의 정답도 점 예측 모델의 학습 밖 오차를 사용했다.

4월 평가는 2월 오차로 보조 모델을 학습하고 3월 점수로 모델과 기준을 정했다. 5월은 2 ~ 3월 학습·4월 선택, 6월은 2 ~ 4월 학습·5월 선택으로 확장했다. Logistic과 HGB 중 이전 월 AP가 높은 모델을 선택한 결과 세 달 모두 HGB였다. 높은 B 예측값·직전 최대값·유사 상태 변동의 세 단순 점수와, 이들 중 이전 월 AP로 고른 기준도 함께 비교했다. 이번 선택 규칙 평가에서는 기존 점수와 모델을 그대로 사용했고 신규 학습·튜닝은 하지 않았다.

### 미래 점수를 보지 않는 하루 2회 선택

매시간 다음 시간의 점수가 생성되는 시점에 바로 검토 여부를 정했다. 하루의 24개 예측 시점을 예정된 확인 기회로 두고, 남은 확인 횟수를 b, 현재를 포함한 남은 시간을 r로 표시했다. 이전 달 점수 분포의 1−b/r 분위수를 현재 점수가 넘으면 한 번을 배정했다. 이미 두 번 확인했으면 더 선택하지 않았고, 남은 시간과 확인 횟수가 같으면 남은 시간을 선택했다. 이 규칙으로 모든 방법이 매일 정확히 2회, 91일간 182시간을 선택했다.

예측 시점에는 현재까지 관측한 입력·이전 달 점수·그날 남은 확인 횟수만 사용했다. 아직 오지 않은 시간의 점수나 실제 전력은 사용하지 않았다. 하루 2회와 24시간 확인 가능 조건은 비교를 위한 운영 가정이며, 실제 사업장의 인력·응답 시간으로 검증한 값은 아니다.

결과를 보기 전에 보고서에 남길 개발 성과의 기준을 정했다. 전체 선별 건수가 가장 좋은 단순 기준보다 최소 5건·20% 이상 많고, 이전 월에서 선택한 단순 기준보다 세 달 중 두 달 이상에서 많으면 제한된 기여를 인정하기로 했다. 모든 달에서 이겨야 한다는 조건은 두지 않았으며, 월별 차이와 과소예측량은 함께 제시한다. 이는 자동 경고 후보의 이전 채택 기준을 변경한 것이 아니라 동일 확인량이라는 별도 활용 목적의 비교다.

### 동일 확인량에서 확인한 개선

"""+"\n".join(table)+"""

가장 좋은 단순 기준인 유사 상태 변동 점수는 20건, 보조 모델은 27건을 선별했다. 같은 182시간에서 **7건, 35.0% 증가**했으며 확인 중 적중 비율은 11.0%에서 14.8%로 높아졌다. 높은 예측값만 보는 기준의 12건과 이전 월에서 고른 단순 기준의 14건보다도 많았다. 단순히 예상 전력이 큰 시간을 확인하는 것에 더해, 예측이 크게 낮아질 가능성을 별도 점수로 제공한 기여다. 개별 이웃 변수의 기여를 분리한 실험은 아니므로 특정 입력 하나의 효과로 해석하지 않는다.

월별 확인량은 4월 60회, 5월 62회, 6월 60회로 각 방법에 동일했다.

"""+"\n".join(monthly)+"""

보조 모델은 이전 월 선택 기준보다 세 달 모두 많이 선별했다. 유사 상태 변동 기준과 비교하면 4월 6→7건, 5월 8→6건, 6월 6→14건으로 월별 차이가 남았다. 선택된 시간의 과소예측량 합은 유사 상태 변동 기준 586.723에서 보조 모델 667.693으로 13.8% 증가했다. 이는 확인 대상으로 포함한 예측오차의 합이며 실제로 보정하거나 절감한 전력량이 아니다.

### 성과의 적용 범위

이 결과는 **동일한 하루 2회 확인 조건에서 선별 효율을 높인 탐색적 개발 성과**다. 전체 큰 과소예측 184건 중 157건은 선택하지 않았으므로 자동 경고나 전체 오류 방지 성능으로 확대하지 않는다. 보조 모델의 선택 182회 중 33회는 남은 확인 횟수를 소진하는 규칙에 따라 배정됐고 이 중 큰 과소예측은 없었다. 이 비효율도 동일 확인량 규칙의 적용 결과로 남긴다.

날짜별 결과를 달력 주 단위로 묶어 동일한 주를 함께 재표집한 2,000회 민감도 분석에서, 유사 상태 변동 기준 대비 선별 증가 7건의 95% 백분위 범위는 −6 ~ 19건이었다. 표본 구성에 따라 우위가 바뀔 여지가 있다. 4 ~ 6월은 여러 개발 실험에서 사용한 기간이고 반복 배열도 있어 이 범위를 독립 시험의 통계적 입증으로 해석하지 않는다. 이번 35%는 해당 개발 조건에서 관측한 개선률이다.

현장 활용안은 다음 시간의 최대값 예측과 함께 검토 필요 표시를 제시하고, 선택된 시간에 담당자가 이미 확인 가능한 가동 계획과 관측 상태를 추가 점검하는 것이다. 실제 조치의 효과는 측정하지 않았다. 결과보고서에서는 제3장의 오류분석에 선별 성과를, 제4장에 하루 확인량을 제한한 활용 절차를 연결하고, 제5장에는 예측값과 과소예측 검토 우선순위를 함께 제공한 기여를 기술할 수 있다.

### 전체 Python 코드와 재현 근거

| 코드 | 역할 |
|---|---|
| [error_warning.py](scripts/error_warning.py) | diagnose: B·6시간 입력의 동일 상태 진단, 과거 이웃과 보조 입력 구성 |
| [error_warning_fit.py](scripts/error_warning_fit.py) | freeze·run: 사건 경계·시간순 분할 고정, 보조 모델 6개 학습, 월별 점수·이전 월 선택 저장 |
| [error_warning_verify.py](scripts/error_warning_verify.py) | 기존 입력·이웃·모델·점수·사건 정의와 HGB 독립 재학습 검증 |
| [priority_review.py](scripts/priority_review.py) | freeze·run: 하루 2회 순차 선택, 동일 확인량 비교, 일별·월별 지표와 주별 민감도 |
| [priority_review_verify.py](scripts/priority_review_verify.py) | 별도 순차 계산으로 선택·분위수·지표 검증, 미래 점수 변조로 과거 선택 불변 확인 |
| [priority_review_documents.py](scripts/priority_review_documents.py) | 기존 원고 보존, 결과표 기반 상세 절 생성, 새 절·README·코드 링크 검증 |

부모 입력의 공통 구성은 [profile_history.py](scripts/profile_history.py)와 그 [검증 코드](scripts/profile_history_verify.py)를 따른다. 계약과 점수의 해시로 기존 [보조 모델 결과](tables/error_warning/run.json), [모델 목록](tables/error_warning/model_manifest.csv), [부모 검증](tables/error_warning/independent_verification.json)을 연결했다. 첫 진단 파일의 초기 freeze·run 초안은 학습 진입점으로 사용하지 않으며 실제 학습은 별도 fit 파일을 사용한다.

이번 [고정 계약](tables/priority_review/contract.json), [전체 비교표](tables/priority_review/metrics.csv), [매시간 선택](tables/priority_review/predictions.csv), [날짜별 결과](tables/priority_review/daily.csv), [주별 민감도](tables/priority_review/sensitivity.csv), [판정](tables/priority_review/decision.json), [실행 기록](tables/priority_review/run.json), [독립 검증](tables/priority_review/independent_verification.json)을 함께 보존한다.

최초 실행 순서는 프로젝트 루트에서 정규 CPython 3.13으로 다음과 같다. Windows에서는 루트 AGENTS.md의 Start-Process 경로를 사용한다. 부모 보조 점수는 위 error_warning 코드로 먼저 재현한다.

~~~text
Modeling/scripts/priority_review.py freeze
Modeling/scripts/priority_review.py run
Modeling/scripts/priority_review_verify.py
Modeling/scripts/priority_review_documents.py --write
Modeling/scripts/priority_review_documents.py --verify
~~~

완료된 계약·결과의 덮어쓰기는 막으며 현재 저장 결과는 두 검증 명령으로 확인한다. 독립 검증은 10,920개 시간별 선택, 455개 방법·날짜별 할당, 지표 20행과 2,000회 주별 재표집을 대조했다. 미래 점수만 변경한 45개 시험에서 앞선 선택이 유지됐고, 기존 모델 6개와 부모 결과의 해시도 확인했다. 이번 비교의 신규 학습은 0회다. 상세 원고만 추가하며 압축본과 기존 M05 결과는 변경하지 않았다.
"""

def write():
    decision=json.loads((OUT/"decision.json").read_text(encoding="utf-8"))
    verified=json.loads((OUT/"independent_verification.json").read_text(encoding="utf-8"))
    assert decision["reportworthy_development_result"] and verified["status"]=="passed"
    original=MANUSCRIPT.read_text(encoding="utf-8")
    assert HEADING not in original and not (OUT/"manuscript_before.md").exists()
    backup=OUT/"manuscript_before.md"
    backup.write_bytes(MANUSCRIPT.read_bytes())
    assert sha(backup)==sha(MANUSCRIPT)
    intro="이번 원고는 M01의 자료·평가 설계에서 M02 모델 비교, M03 입력 비교, M04 오류분석, M05 고정 후반기 평가와 후속 개발까지 이어진다. 마지막 값의 개발 개선과 후반기 한계를 구분하고, 4.11에서는 하루 2회 확인 조건의 과소예측 검토 우선순위 성과를 다룬다. 각 절의 당시 판단을 보존하며 현재 성과 범위는 후속 평가에 따라 갱신한다. 수치는 원자료의 전력 척도로 제시하고 실제 전력·비용 절감량으로 환산하지 않는다."
    updated=re.sub(r"(?m)^이번 원고는 .*?$",lambda _:intro,original,count=1)
    MANUSCRIPT.write_text(updated.rstrip()+"\n\n"+section().strip()+"\n",encoding="utf-8")
    readme=ROOT/"Modeling/README.md"
    text=readme.read_text(encoding="utf-8")
    assert README_HEADING not in text
    addition=README_HEADING+"""

사용자가 완벽한 월별 성과보다 보고서에 쓸 수 있는 실용적 기여를 검토하도록 요청해, 자동 경고와 별도로 하루 2회 검토 우선순위를 평가했다. 매시간 이전 월 점수·남은 횟수·시간만 사용하는 규칙을 먼저 고정했다. 4 ~ 6월 91일간 각 방법이 똑같이 182시간을 선택했으며 보조 모델은 큰 과소예측 27건, 가장 좋은 단순 기준은 20건을 선별했다. 7건·35.0% 증가로 사전의 최소 5건·20% 개선과 2개월 이상 개선 조건을 통과했다.

제한된 개발 성과로 [상세 원고 4.11](04_Modeling_원고.md#411-하루-2회-검토를-위한-과소예측-우선순위)에 반영했다. 보조 모델의 월별 적중은 7·6·14건이며 가장 좋은 단순 점수는 6·8·6건이다. 5월 열세, 전체 184건 중 157건 미선택, 주별 재표집의 증가 범위 −6 ~ 19건도 함께 남겼다. 35%는 이 개발 조건의 관측 개선률이고 새 독립 시험이나 자동 경고 채택을 뜻하지 않는다. 이 절이 아래 과거 실험의 다음 작업 안내에 우선한다.

| 실제 사용 코드 | 결과·재현 |
|---|---|
| [priority_review.py](scripts/priority_review.py) | freeze·run, [계약](tables/priority_review/contract.json), [전체 지표](tables/priority_review/metrics.csv), [매시간 선택](tables/priority_review/predictions.csv), [판정](tables/priority_review/decision.json) |
| [priority_review_verify.py](scripts/priority_review_verify.py) | [독립 검증](tables/priority_review/independent_verification.json): 10,920개 선택·455개 일별 할당·45개 미래 점수 변경 시험 통과 |
| [priority_review_documents.py](scripts/priority_review_documents.py) | --write·--verify, 원고 원본 보존·추가·결과표와 링크 대조 |

부모 error_warning 코드·모델·점수는 아래 절과 원고 코드표에 모두 연결했다. 새 학습·튜닝은 0회이며 반복 실행으로 유리한 규칙을 고르지 않았다. 기존 자동 경고 후보의 미채택 판정과 M05를 유지하고 압축본은 변경하지 않았다. 보고서에서는 예측오류 점수와 제한된 확인 횟수를 연결한 보조 활용으로 기술하며 일반적인 성능 우위나 실제 절감 효과를 주장하지 않는다. 이 비교는 종료하고, 동일 개발 기간에서 수치를 더 높이기 위한 규칙 탐색은 확대하지 않는다.

"""
    readme.write_text(text.replace("## 과소예측 위험 경고 비교 결과",addition+"## 과소예측 위험 경고 비교 결과",1),encoding="utf-8")
    root=ROOT/"README.md"
    text=root.read_text(encoding="utf-8")
    notice="**현재 상태 — 2026-10-05 하루 2회 검토 우선순위 개발 성과 반영:** 미래 점수를 보지 않는 순차 선택으로 4 ~ 6월 182시간씩 확인했을 때, 큰 과소예측 선별은 가장 좋은 단순 기준 20건에서 보조 모델 27건으로 35.0% 늘었다. 월별 열세와 낮은 전체 재현율·표본 불확실성은 함께 명시해 [상세 원고 4.11](Modeling/04_Modeling_원고.md#411-하루-2회-검토를-위한-과소예측-우선순위)에 제한된 개발 성과로 반영했다. [규칙·코드·검증](Modeling/README.md#하루-2회-검토-우선순위-개발-성과-반영)을 보존했다. 이번 기여는 같은 확인량에서의 선별 효율이며 자동 경고 채택·새 독립 시험·실제 절감 입증은 아니다."
    text=re.sub(r"(?m)^[*][*]현재 상태[^\r\n]*",lambda _:notice,text,count=1)
    assert notice in text
    root.write_text(text,encoding="utf-8")
    analysis=ROOT/"Analysis/README.md"
    text=analysis.read_text(encoding="utf-8")
    note="""**후속 활용 비교:** 기존 위험 점수를 이용한 하루 2회 순차 검토에서 동일 182시간의 큰 과소예측 선별 20→27건을 확인했다. [Modeling 상세 원고 4.11](../Modeling/04_Modeling_원고.md#411-하루-2회-검토를-위한-과소예측-우선순위)에 개발 성과와 월별 한계를 함께 기록했다. EDA·Analysis 본문과 압축본은 변경하지 않았다.

"""
    analysis.write_text(text.replace("**동일·유사 입력과 과소예측 위험 진단 완료:**",note+"**동일·유사 입력과 과소예측 위험 진단 완료:**",1),encoding="utf-8")
    print("Preserved original and added detailed section 4.11.",flush=True)

def verify():
    manuscript=MANUSCRIPT.read_text(encoding="utf-8")
    assert manuscript.count(HEADING)==1
    # Later manuscript sections are outside the generated 4.11 scope.
    actual_section = re.split(r"(?m)^## ", manuscript.split(HEADING)[1], maxsplit=1)[0]
    assert actual_section.strip()==section().split(HEADING)[1].strip()
    assert (OUT/"manuscript_before.md").exists()
    paths=[MANUSCRIPT,ROOT/"Modeling/README.md",ROOT/"README.md",ROOT/"Analysis/README.md"]
    count=0
    for path in paths:
        text=path.read_text(encoding="utf-8")
        if path==ROOT/"README.md":
            text="\n".join(text.splitlines()[:4])
        elif path==ROOT/"Modeling/README.md":
            text=text.split("## 과소예측 위험 경고 비교 결과")[0]
        elif path==ROOT/"Analysis/README.md":
            text=text.split("**동일·유사 입력과 과소예측 위험 진단 완료:**")[0]
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):
            if target.startswith(("http:","https:","#","app:")):
                continue
            assert (path.parent/target.split("#")[0].strip("<>")).exists(),target
            count+=1
    # Use the simple literal delimiter pattern to check all links as well.
    for target in re.findall(r"\]\(([^)]+)\)",manuscript):
        assert (MANUSCRIPT.parent/target.split("#")[0]).exists(),target
    record={"status":"passed","documents_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in paths},
            "original_manuscript_sha256":sha(OUT/"manuscript_before.md"),
            "run_sha256":sha(OUT/"run.json"),"verifier_sha256":sha(OUT/"independent_verification.json"),
            "script_sha256":sha(__file__),"links_checked":count,"generated_section_matches":True}
    (OUT/"document_verification.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(record,ensure_ascii=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--write",action="store_true")
    g.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if args.write:
        write()
    else:
        verify()
