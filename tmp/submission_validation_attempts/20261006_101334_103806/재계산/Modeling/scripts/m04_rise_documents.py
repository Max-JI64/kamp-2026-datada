"""Write or verify the detailed M04 rise manuscript from independently checked results."""
import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from m03_verify import rows, sha

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "Modeling/04_Modeling_원고.md"
LOG = ROOT / "Modeling/10.05_M04_상승예측결합_실행기록.md"
PARENT = ROOT / "Modeling/tables/m04_rise"
OUT = ROOT / "Modeling/tables/m04_rise_gate"
TITLE = "## 4.7 M04 추가 개발: 상승 예측 결합의 개선과 채택 한계"
ORDER = ["B", "D_dynamic", "W_rise_weighted", "G_Logistic", "G_HGB", "G_Logistic_rise_focus", "G_HGB_rise_focus"]
LABELS = {"B": "기존 B", "D_dynamic": "변화 입력 HGB", "W_rise_weighted": "상승 가중 HGB",
    "G_Logistic": "Logistic 분류 결합", "G_HGB": "HGB 분류 결합",
    "G_Logistic_rise_focus": "Logistic 결합·상승 중심 선택", "G_HGB_rise_focus": "HGB 결합·상승 중심 선택"}


def comparison():
    indexed = {r["group"]: r for r in rows(OUT / "selection_metrics.csv")}
    text = "| 방법 | 전체 MAE | 일별 최대 시간 MAE | 상승 MAE | 상승 평균 과소 | 낮은 유지 MAE | 낮은 유지 평균 과대 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for name in ORDER:
        r = indexed[name]
        text += "| " + LABELS[name] + " | " + " | ".join(f"{float(r[k]):.3f}" for k in ["overall_MAE", "daily_maximum_MAE", "rise_MAE", "rise_under", "low_stay_MAE", "low_stay_over"]) + " |\n"
    return text


def monthly():
    metrics = rows(PARENT / "metrics.csv")
    alarms = rows(PARENT / "classification.csv")
    configs = json.loads((PARENT / "selected_configurations.json").read_text(encoding="utf-8"))
    text = "| 개발 월 | 상승 수 | B 상승 MAE | HGB 결합 상승 MAE | AP | TP / FP / FN / TN | 임계값 / 결합 비율 |\n|---|---:|---:|---:|---:|---|---|\n"
    for split, label in [("dev_apr", "4월"), ("dev_may", "5월"), ("dev_jun", "6월")]:
        alarm = next(r for r in alarms if r["split"] == split and r["role"] == "outer" and r["model"] == "G_HGB")
        errors = []
        for name in ["B", "G_HGB"]:
            pool = [r for r in rows(PARENT / "predictions.csv") if r["split"] == split and r["group"] == name and r["rise_event"] == "True"]
            errors.append(sum(float(r["absolute_error"]) for r in pool) / len(pool))
        gate = next(r for r in configs if r["split"] == split)["classifiers"]["HGB"]["gate"]
        confusion = " / ".join(alarm[k] for k in ["TP", "FP", "FN", "TN"])
        text += f"| {label} | {alarm['positive_hours']} | {errors[0]:.3f} | {errors[1]:.3f} | {float(alarm['AP']):.3f} | {confusion} | {gate['threshold']:.4f} / {gate['blend']:.4f} |\n"
    return text


def section():
    table = {r["group"]: r for r in rows(OUT / "selection_metrics.csv")}
    rise_gain = 100 * (1 - float(table["G_HGB"]["rise_MAE_ratio"]))
    under_gain = 100 * (1 - float(table["G_HGB"]["rise_under_ratio"]))
    low_cost = 100 * (float(table["G_HGB"]["low_stay_MAE_ratio"]) - 1)
    return f"""{TITLE}

M03에서 마지막 값 추가는 전체·일별 최대 시간 오차를 줄였지만, 낮은 상태에서 26초과 상태로 바뀐 20시간을 모두 과소예측했다. 4.6의 직전 값 대체도 그 과소예측을 48.772에서 75.400으로 늘려 기각했다. 이 결과에서 '과거 변화 정보나 상승 사례의 학습 비중을 늘리면 상승 오차를 줄일 수 있는가, 그리고 상승 확률로 적용 대상을 고르면 낮은 유지의 오경보 부담을 줄일 수 있는가'를 후속 질문으로 정했다. [추가 실행 기록](10.05_M04_상승예측결합_실행기록.md)과 [첫 비교 조건](config/m04_rise_contract.json)에 질문·비교 범위·판단 기준을 기록했다.

### 상승의 정의와 시간순 비교

이번 상승은 기존 low→above26 전환이다. 직전 최대값이 양수이고 직전 정확한 평균을 `floor(mean+0.5)`로 반올림한 값이 20~26이며, 다음 시간의 제공 평균값이 26을 넘는 경우를 뜻한다. 모든 큰 증가나 0 이후 재가동을 포괄하는 정의가 아니다. 그 범위의 성과는 별도로 확인해야 한다. 목표는 다음 시간 네 15분 값의 최댓값이며 단위가 확인되지 않은 값에 임의로 kW를 붙이지 않는다.

외부 개발 비교는 1~3월 학습→4월, 1~4월→5월, 1~5월→6월의 같은 2,184시간·91일이다. 학습 상승 사례는 각각 31·39·47시간, 평가 상승은 8·8·4시간이다. 내부 검증은 1월→2월부터 과거 월만 순차 확장해 외부 평가 월 이전에 끝낸다. 직전 상태는 예측 시점에 관측할 수 있고, 다음 상태·일별 최대 시간·다음 생산량은 입력에 넣지 않는다. 7~8월은 이번 튜닝과 비교에 사용하지 않았다. 다만 후반기 자료의 과거 관찰 이력은 있으므로 완전히 처음 보는 독립 시험으로 표현하지 않는다.

전체 MAE와 일별 최대 시간 MAE, 낮은 유지 MAE는 B 대비 악화를 1% 이내로 제한하고, 낮은 유지 평균 과대예측은 5% 이내로 제한했다. 상승 MAE와 평균 과소예측은 각각 10% 이상 줄고 각 월의 상승 MAE도 악화되지 않아야 한다. 분류 결합의 오경보율은 직전 낮은 상태의 비상승 사례 중 5% 이내로 정했다. 이는 실행 전에 기록한 개발 허용 범위이며 현장 비용에서 추정한 최적값이 아니다. 외부 개발 결과로 채택 여부를 확인하되 그 결과에 맞춰 임계값을 직접 조정하지 않았다.

### 변화 입력·상승 가중 학습·분류 결합의 구분

B의 18개 입력에 과거 변화 입력 9개를 추가해 총 27개를 비교했다. 추가 입력은 직전 네 구간의 마지막값-첫값, 범위, 선형 기울기, 직전과 그 이전 시간의 마지막값·평균·최대·생산량 차이, 최대 6시간의 낮은 상태 연속 길이와 그 조회 중 시간 누락 여부다. 모든 값은 t-1 이전 관측으로 계산하고 누락 시각을 건너뛰어 연결하지 않는다. 기존 날씨 후보 D와 혼동하지 않도록 이번 `D_dynamic`은 변화 입력 HGB로 표기한다.

변화 입력 HGB는 기존 B의 설정을 고정한다. 상승 가중 HGB는 같은 변화 입력에서 학습 자료의 상승 정답에만 가중치 1~12를 부여하고 Optuna로 선택한다. 미래 상승 정답은 학습 가중치에 쓰며 추론 때 주어지는 정보로 사용하지 않는다. HGB 설정은 잎 15, 반복 150, 학습률 0.05, 최소 잎 표본 20, L2 규제 1, 조기 종료 사용 안 함, 난수 시드 42다. 변화 입력만 추가한 효과와 상승 학습 비중의 효과를 구분할 수 있도록 순차 비교했다.

분류기는 직전 상태가 낮은 학습 행만 이용해 다음 시간 상승 확률을 추정한다. Logistic은 학습 구간에서만 표준화하며 C를 0.01~100에서 찾고, HGB 분류기는 잎 3·7·15, 최소 잎 표본 10·20·40, L2 0.1~10을 비교했다. 희소 상승을 상위로 구분하는 능력은 내부 AP로 선택하고 Brier를 보조 확인했다. 여기서 AP는 비보간 평균 정밀도이며 사다리꼴 PR 면적과 같은 수치로 표기하지 않는다. AUROC·정밀도·재현율·오경보도 보존했다. 분류 지표는 상승 여부, 회귀 MAE는 전력 크기를 평가하므로 서로 대체하지 않는다.

결합은 `B + I(직전 낮은 상태이고 상승확률≥임계값) × 결합비율 × (상승 가중 HGB-B)`다. 조건 밖에서는 B를 그대로 사용한다. 확률 임계값 0.01~0.8과 결합 비율 0~1은 과거 내부 검증의 예측만으로 선택했다. 첫 회차는 전체·최대 시간·상승의 B 대비 상대 MAE 중 최댓값과 허용 범위 초과 벌점을 최소화했다. 가중치·두 분류기·두 결합을 외부 월마다 각 30회 탐색해 총 Optuna 450회, 학습 840회를 실행했다. 이 중 외부 최종 학습은 12회이며 저장 모델 12개를 재로딩해 예측을 대조했다.

### 같은 개발 시간에서의 결과

{comparison()}
상승 가중 HGB는 상승 MAE를 35.682로 줄였지만 전체·최대 시간·낮은 유지 오차를 모두 키웠다. 변화 입력만 추가한 모형도 기존 B보다 좋아지지 않았다. Logistic 결합의 첫 내부 선택은 결합 비율을 0으로 정해 회귀 예측이 B와 같았다. 분류기를 붙였다는 사실만으로 상승 예측이 개선되는 것은 아니었다.

HGB 분류 결합은 상승 MAE를 {rise_gain:.1f}%, 평균 과소예측을 {under_gain:.1f}% 줄였고 전체 MAE도 6.191에서 6.146으로 약 0.7% 줄였다. 그러나 낮은 유지 MAE는 4.110에서 4.228로 {low_cost:.1f}% 늘어 사전 허용 범위 1%를 넘었다. 다른 채택 요건은 통과했지만 이 요건을 충족하지 못해 최종 교체 후보로 채택하지 않았다. 일별 최대 시간 MAE가 같은 이유는 이 결합이 바꾼 예측이 해당 최대 시간과 겹치지 않았기 때문이다. 이번 상승 개선을 일별 최대 시간 개선으로 표현하지 않는다.

### 어느 월에서 상승을 찾았고 어디서 오경보가 남았는가

다음 표는 첫 HGB 결합의 월별 결과다. AP의 분모는 직전 낮은 상태의 200·262·188시간이고 상승은 8·8·4시간이다. TP는 상승에서 결합 조건을 충족한 수, FP는 비상승에서 그 조건을 충족한 수다. 각 월의 확률·비율은 해당 월 이전 내부 검증으로 정했다.

{monthly()}
세 달을 합치면 상승 20시간 중 9시간에서 조건을 충족해 재현율 45.0%였고, 총 15번 중 9번이 상승이므로 정밀도는 60.0%였다. 오경보 6시간은 비상승 630시간의 약 0.95%다. 비상승에는 낮은 유지 625시간과 낮은 상태에서 20미만으로 내려간 5시간이 포함된다. 오경보율이 작아도 예측값을 크게 높인 오경보 한 번이 회귀 오차를 늘릴 수 있다. 이 수치는 현장 경보를 실제 운영한 성과가 아닌 개발 자료의 결합 조건 평가다.

예를 들어 6월 6일 08시는 실제 최대값 22를 90.295로, 6월 27일 07시는 실제 23을 87.341로 예측했다. 이처럼 낮은 유지에서 상승 전문가가 과도하게 반영된 사례가 남았다. 4월에는 상승 8시간을 모두 놓쳤고 결합 비율도 약 0.0013이어서 수정 효과가 거의 없었다. 반면 5·6월은 일부 상승을 찾아 그 크기 오차를 줄였다. 월별 AP의 0.188·0.796·1.000 차이와 6월 상승 4시간의 작은 분모를 함께 제시해야 하며 후반기에도 같은 성능이 나온다고 단정하지 않는다.

### 결과에 따른 한 번의 선택 점수 수정

첫 내부 점수는 최대 시간 오차가 B와 같으면 최댓값 항이 1에 머물러, 다른 조건이 허용 범위 안에 있는 상승 개선을 충분히 구분하지 못할 수 있었다. 이 문제만 확인하기 위해 [후속 조건](config/m04_rise_gate_contract.json)을 기록하고 최댓값 항을 상승 상대 MAE로 바꿨다. 나머지 벌점·채택 허용 범위·분류기·회귀모델·입력·분할은 고정했다. 저장된 과거 내부 예측으로 결합 임계값과 비율만 두 계열·세 외부 월에 각 50회, 총 300회 탐색했으며 새 모델 학습은 없었다.

HGB의 상승 중심 선택은 상승 MAE 40.471, 낮은 유지 MAE 4.226으로 첫 결합보다 충분한 개선을 보이지 않았고 낮은 유지 요건도 계속 실패했다. Logistic의 상승 중심 선택은 낮은 유지 오차를 줄였으나 상승 MAE 감소가 약 0.5%에 그쳐 10% 요건을 충족하지 못했다. 합계 750회의 개발 탐색 이후 모든 요건을 통과한 추가 후보는 없었고 기존 B를 유지했다. 이번에 정한 비교와 한 번의 점수 수정은 여기서 종료하며, 외부 결과에 맞춰 허용 범위를 늘리거나 선택 경계를 계속 바꾸지 않는다.

### 보고서 연결과 다음 판단

확인된 사실은 마지막 값 추가의 기존 효과와 별개로, 상승 분류와 가중 회귀의 결합이 일부 상승의 과소예측을 줄일 수 있었다는 점이다. 유지할 판단은 기존 B를 현재 모델 선택의 개발 후보로 유지한다는 점이며, 수정할 주장은 '급상승을 전반적으로 잘 예측한다'는 표현이다. 낮은 유지의 과대예측과 4월 상승 미탐지가 남았고, 운영 원인과 실제 절감 효과는 현 자료에서 식별하거나 실측하지 못했다.

제2장에는 동일 조건의 추가 비교와 B 유지 이유, 제3장에는 상승 개선·오경보·월별 실패를 연결한다. 제5장의 주된 차별성은 M03에서 검증한 마지막 값의 기여로 작성한다. 이번 17.5% 상승 오차 감소는 허용 요건을 통과하지 못한 보완 실험의 부분 성과로 설명하며 최종 모델 성과에 합산하지 않는다. 짧은 최종 보고서에서는 전체 탐색표를 재현 자료로 두고 핵심 상충만 남길 수 있다. 다음 질문은 고정 B의 A 대비 개선과 이 실패조건이 후반기에도 유지되는가다. M05 전에 방법·판단 기준을 고정한 뒤 1~6월 학습→7~8월 확인으로 연결하며, 이번에는 M05를 실행하지 않았다.

### 전체 코드와 독립 검증

[학습·분류·가중 회귀·결합 코드](scripts/m04_rise_compare.py), [원본 입력·모델 재로딩·독립 검증 코드](scripts/m04_rise_verify.py), [저장 내부 예측의 선택 점수 수정 코드](scripts/m04_rise_gate.py), [300회 선택·결합 독립 검증 코드](scripts/m04_rise_gate_verify.py), [원고 생성·검증 코드](scripts/m04_rise_documents.py)를 4.5의 전체 코드 보존 목록에 포함했다. 공통 모듈도 해당 목록과 함께 보존한다.

[첫 전체 지표](tables/m04_rise/metrics.csv), [첫 월별 설정](tables/m04_rise/selected_configurations.json), [분류 지표](tables/m04_rise/classification.csv), [개별 예측](tables/m04_rise/predictions.csv), [첫 독립 검증](tables/m04_rise/independent_verification.json), [최종 비교표](tables/m04_rise_gate/selection_metrics.csv), [최종 선택](tables/m04_rise_gate/selection.json), [후속 독립 검증](tables/m04_rise_gate/independent_verification.json)에 근거를 남겼다. 첫 회차는 과거 입력 112,752값·10,920예측행·260지표행·450시도와 12개 모델을, 후속은 15,288예측행·364지표행·300시도의 점수·AP·혼동행렬과 변경 없는 채택 요건을 대조했다. 두 검증은 정규 CPython 3.13에서 종료 코드 0으로 통과했다. 원자료와 기존 M01~M04 결과를 유지했고 압축본·PDF·새 그림·파일 삭제는 수행하지 않았다.
"""


def write():
    text = MD.read_text(encoding="utf-8")
    assert TITLE not in text, "Preserve existing section; use --verify after writing."
    text = text.replace("사용한 20개 파일이다.", "사용한 25개 파일이다. 상승 추가 개발의 학습·검증·문서 코드도 포함한다.")
    added = "\n".join([
        "| [m04_rise_compare.py](scripts/m04_rise_compare.py) | 4.7 변화 입력·상승 가중·두 분류 결합의 시간순 튜닝과 예측 | `build_frame()`, `stats()`, `score()`, `main()` |",
        "| [m04_rise_verify.py](scripts/m04_rise_verify.py) | 4.7 원본 27입력·AP·모든 지표·튜닝 선택·저장 모델 검증 | `classification()`, `stats()`, `main()` |",
        "| [m04_rise_gate.py](scripts/m04_rise_gate.py) | 4.7 과거 내부 예측에서 상승 중심 결합 점수만 변경 | `main()`; 새 학습 없음 |",
        "| [m04_rise_gate_verify.py](scripts/m04_rise_gate_verify.py) | 4.7 결합식·300회 점수·지표·변경 없는 채택 요건 독립 검증 | `gate_rows()`, `objective()`, `main()` |",
        "| [m04_rise_documents.py](scripts/m04_rise_documents.py) | 4.7 검증된 표의 상세 원고·기록·README 통합 및 수치·링크 검증 | `--write` 최초 작성, 이후 `--verify` |",
    ])
    text = text.replace("| [modeling_submission_audit.py]", added + "\n| [modeling_submission_audit.py]", 1)
    text = text.replace("[M04](config/m04_contract.json).", "[M04](config/m04_contract.json), [상승 비교](config/m04_rise_contract.json), [결합 점수 수정](config/m04_rise_gate_contract.json).")
    text = text.replace("`m04/`, `observed_timeline/`", "`m04/`, `m04_rise/`, `m04_rise_gate/`, `observed_timeline/`")
    text = text.replace("`m02_rerun/`의 파일. 모델 목록", "`m02_rerun/`, `m04_rise/`의 파일. 모델 목록")
    old = "| 9 | `Modeling/scripts/m02_rerun_document_verify.py`, `Modeling/scripts/m04_verify.py --documents` | 원고 수치·링크·기존 M03 문서 및 M04 문서 검증 |\n| 10 |"
    new = "| 9 | `Modeling/scripts/m04_rise_compare.py` → `Modeling/scripts/m04_rise_verify.py` | 변화 입력·상승 가중·분류 결합의 추가 학습과 독립 검증 |\n| 10 | `Modeling/scripts/m04_rise_gate.py` → `Modeling/scripts/m04_rise_gate_verify.py` | 검증된 과거 내부 예측의 결합 점수 수정·독립 검증; 새 학습 없음 |\n| 11 | `Modeling/scripts/m02_rerun_document_verify.py`, `Modeling/scripts/m04_verify.py --documents`, `Modeling/scripts/m04_rise_documents.py --verify` | 기존·추가 원고 수치와 링크 검증; 문서 작성 코드는 학습 재현과 별도로 사용 |\n| 12 |"
    assert old in text
    text = text.replace(old, new)
    MD.write_text(text.rstrip() + "\n\n" + section(), encoding="utf-8")
    LOG.write_text("# M04 상승 예측 결합 실행 기록\n\n" + section().replace(TITLE, "## 질문·방법·결과와 판단") +
        "\n## 재현 실행 기록\n\n첫 실행은 `Modeling/scripts/m04_rise_compare.py`, 검증은 `m04_rise_verify.py`로 수행했다. 첫 run.json의 학습 수는 내부 회귀288·내부 분류540·외부12이며 Optuna450회다. 후속 `m04_rise_gate.py`는 검증된 첫 실행의 모델과 내부 예측을 재사용해 Optuna300회만 수행했고 `m04_rise_gate_verify.py`로 검증했다. 완료 폴더를 덮어쓰지 않는 보호 조건을 유지한다.\n\n[첫 run.json](tables/m04_rise/run.json), [첫 시간순 분할](tables/m04_rise/fold_audit.csv), [첫 모든 시도](tables/m04_rise/trials.csv), [첫 모델 목록](tables/m04_rise/model_manifest.csv), [후속 run.json](tables/m04_rise_gate/run.json), [후속 모든 시도](tables/m04_rise_gate/trials.csv), [후속 설정](tables/m04_rise_gate/selected_configurations.json)을 함께 보존한다. 문서 작성과 검증은 `m04_rise_documents.py --write`·`--verify`이며 원자료를 수정하지 않는다.\n", encoding="utf-8")
    root_readme = ROOT / "README.md"
    text = root_readme.read_text(encoding="utf-8")
    text = text.replace("M04 조건별 오류·낮은 상태 규칙 비교까지 완료했다.", "M04 조건별 오류·낮은 상태 규칙·상승 결합 비교까지 완료했다.")
    anchor = "같은 수의 관리시간으로 높은 최대전력 시간을 더 잘 찾는 성과는 별도 검증이 필요하다."
    summary = "[M04 상승 추가 개발](Modeling/10.05_M04_상승예측결합_실행기록.md): 변화 입력·상승 가중 HGB·Logistic/HGB 분류 결합과 한 번의 결합 점수 수정을 시간순으로 비교했다. HGB 결합은 상승 MAE48.772→40.217(17.5% 감소), 전체6.191→6.146이지만 낮은 유지 MAE4.110→4.228(2.9% 증가)로 사전1% 허용을 넘었다. 상승20시간 중9시간을 찾아 재현율45%·정밀도60%, 비상승630시간 중오경보6시간이었다. Optuna750회·새학습840회와 독립 검증을 마쳤고 모든 교체 요건을 통과한 후보가 없어 B를 유지했다. 원고4.7·전체25코드·제출 보존 목록에 연결했으며 M05는 미실행이다.\n\n"
    assert anchor in text
    text = text.replace(anchor, summary + anchor, 1)
    root_readme.write_text(text, encoding="utf-8")
    readme = ROOT / "Modeling/README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("M04 조건별 오류·낮은 상태 규칙 비교 완료", "M04 조건별 오류·낮은 상태 규칙·상승 결합 비교 완료", 1)
    proposal = next(line for line in text.splitlines() if line.startswith("직전 low 일괄 대체 질문은 기각으로 종료한다."))
    replacement = "직전 low 일괄 대체 질문은 기각으로 종료했다. 사용자 요청으로 이어간 [상승 추가 개발](10.05_M04_상승예측결합_실행기록.md)은 변화 입력·상승 가중 HGB·Logistic/HGB 분류 결합을 비교했다. 첫 HGB 결합은 상승 MAE48.772→40.217(17.5% 감소), 전체6.191→6.146이나 낮은 유지4.110→4.228(2.9% 악화)로 사전1% 허용을 넘었다. 재현율45%·정밀도60%·오경보6/630과 4월 미탐지를 확인했다. 이 결과에서 결합 내부 선택 점수만 상승 중심으로 바꿔 한 번 더 비교했지만 추가 후보도 교체 요건을 통과하지 못했다. [첫 조건](config/m04_rise_contract.json), [후속 조건](config/m04_rise_gate_contract.json), [최종 비교표](tables/m04_rise_gate/selection_metrics.csv), [독립 검증](tables/m04_rise_gate/independent_verification.json)에 근거를 보존한다. 총 Optuna750회·학습840회 후 선언한 비교는 종료하고 B를 유지한다. 원고4.7·4.5의 전체25코드와 재현 파일을 연결했다. 다음 질문은 고정 A/B의 효과와 실패조건이 7~8월에서도 유지되는가이며 M05는 아직 실행하지 않았다."
    text = text.replace(proposal, replacement)
    text = text.replace("4.4는 Optuna·시간순 검증·확장 모델과 HGB 유지 판단을 설명한다.", "4.4는 Optuna·시간순 검증·확장 모델과 HGB 유지 판단, 4.5는 전체25코드 보존, 4.6은 낮은 상태 규칙 기각, 4.7은 상승 결합의 부분 개선·오경보·B 유지 판단을 설명한다.")
    readme.write_text(text, encoding="utf-8")


def verify():
    for directory, verifier in [(PARENT, "m04_rise_verify.py"), (OUT, "m04_rise_gate_verify.py")]:
        result = json.loads((directory / "independent_verification.json").read_text(encoding="utf-8"))
        assert result["status"] == "passed" and result["run_sha256"] == sha(directory / "run.json")
        assert result["script_sha256"] == sha(ROOT / "Modeling/scripts" / verifier)
    manuscript = MD.read_text(encoding="utf-8")
    assert manuscript.count(TITLE) == 1
    assert manuscript.split(TITLE)[1].split("\n## 4.8")[0].rstrip() == section().split(TITLE)[1].rstrip()
    first_predictions = rows(PARENT / "predictions.csv")
    baseline = {r["timestamp"]: r for r in first_predictions if r["group"] == "B"}
    hgb = [r for r in first_predictions if r["group"] == "G_HGB"]
    assert not any(float(r["daily_maximum_weight"]) > 0 and
                   abs(float(r["prediction"]) - float(baseline[r["timestamp"]]["prediction"])) > 1e-9 for r in hgb)
    assert sum(r["trigger"] == "True" and r["rise_event"] == "True" for r in hgb) == 9
    assert sum(r["trigger"] == "True" and r["rise_event"] == "False" for r in hgb) == 6
    for timestamp, actual, prediction in [("2021-06-06 08:00:00", 22, 90.295), ("2021-06-27 07:00:00", 23, 87.341)]:
        case = next(r for r in hgb if r["timestamp"] == timestamp)
        assert float(case["actual"]) == actual and f"{float(case['prediction']):.3f}" == f"{prediction:.3f}"
    first_run = json.loads((PARENT / "run.json").read_text(encoding="utf-8"))
    follow_run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert sum(first_run["fits"].values()) == 840 and first_run["trials"] + follow_run["trials"] == 750
    linked = 0
    for path in [MD, LOG]:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#")):
                continue
            assert (path.parent / target.split("#")[0].strip("<>")).resolve().exists(), target
            linked += 1
        assert comparison() in text and monthly() in text
    result = {"status": "passed", "checked_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "manuscript_sha256": sha(MD), "execution_record_sha256": sha(LOG), "script_sha256": sha(Path(__file__)),
        "comparison_rows": 7, "monthly_rows": 3, "local_links_checked": linked,
        "selection": "B", "no_training_or_deletion": True}
    (OUT / "document_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    if args.write:
        write()
    verify()


if __name__ == "__main__":
    main()
