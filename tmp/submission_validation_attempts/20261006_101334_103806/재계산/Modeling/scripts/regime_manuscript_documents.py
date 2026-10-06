"""Preserve the detailed manuscript and append verified S01-S06 section 4.12.

No training, compression, root README changes, or historical-result overwrite.
"""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "Modeling/04_Modeling_원고.md"
README = ROOT / "Modeling/README.md"
OUT = ROOT / "Modeling/tables/regime_manuscript"
RESULT = ROOT / "Modeling/tables/regime_age_ablation"
HEADING = "## 4.12 지속 상승 예측과 선택적 HGB 보완"
INTRO = "이번 원고는 M01의 자료·평가 설계에서 M02 모델 비교, M03 입력 비교, M04 오류분석, M05 고정 후반기 평가와 후속 개발까지 이어진다. 4.9에는 기존 M05의 고정 후반기 결과를 보존하고, 4.11에는 과소예측 검토 우선순위의 보조 활용을 기록했다. 현재 차별화 성과는 4.12의 지속 상승 예측과 선택적 HGB 보완에 정리한다. 이 절은 개발 성과와 이미 관측한 7~8월에서의 탐색적 개선을 연결하며, 각 과거 절의 당시 판단과 구분한다. 수치는 원자료의 전력 척도로 제시하고 실제 전력·비용 절감량으로 환산하지 않는다."


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(name):
    with (RESULT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def revise_history(text):
    replacements = {
        "이 원고에 사용한 Python 파일의 전체 소스는": "이 원고에 사용한 Python 파일의 전체 소스는 아래 링크의 실제 `.py` 파일에 보존한다. 최종 제출에는 공통 모듈·원본 조회·검증 절차까지 같은 폴더 구조로 포함한다. 아래 표는 M01~M05와 초기 재개발의 코드이며, 후속 분석의 추가 코드는 각 절과 4.12의 코드표에 연결했다. 현재 전체 보존 범위와 의존 관계는 `submission_manifest.json`을 기준으로 확인한다.",
        "M02 시점에는 전날·전주 입력의 필요성과": "M02 시점에는 전날·전주 입력의 필요성과 날씨 비교의 순서를 남은 오류에 따라 갱신하기로 했다. 이어진 4.3에서는 입력 비교를, 4.9에서는 기존 후보의 고정 후반기 평가를 수행했다. 현재의 지속 상승 보완 성과는 4.12에서 별도로 설명한다.",
        "현재 개발 후보는 **B:": "M03에서 선택한 개발 후보는 **B: 기본 입력에 마지막 15분 값만 추가한 HGB**였다. 같은 설정에서 전체·월별·최대 시간의 개선이 확인됐고 C_B보다 입력 수가 적으며 전체와 최대 시간의 오차도 작았다. 계절 시차는 6월에 유리했지만 4월과 학습에 없는 배열에서 B보다 나빠 기본 구성에 추가하지 않았다. 이는 당시 입력·설정에서의 선택이며, 4.9의 후반기 결과와 4.12의 새 비교 조건까지 포함한 최종 우위를 뜻하지 않는다.",
        "필살기의 설명도 결과에 맞춰 좁힌다.": "M03에서 확인한 성과는 **직전 마지막 값을 추가했을 때 다음 최대전력의 전체 오차와 실제 일별 최대 시간의 과소예측이 줄어든 개발 결과**다. 이후 4.9에서는 이 효과가 후반기 최대 시간까지 유지되지 않았음을 확인했다. 따라서 마지막 값 하나의 효과를 현재 보고서의 주된 차별성으로 제시하지 않고, 기본 입력을 선택한 근거와 남은 오류를 설명하는 결과로 사용한다.",
        "따라서 필살기 후보의 핵심 방법과 개발 성과는 M03에 있다.": "M03은 전력 크기의 기본 예측을 구성하는 단계다. M04에서는 상승 전환과 낮은 상태에서 남은 오류를 분석하고 보완 규칙을 비교했으며, M05에서는 당시 후보를 고정해 후반기를 확인했다. 이 과정에서 전력 크기의 오차 감소와 상승 시작을 미리 찾는 능력을 구분할 필요가 드러났다. 그 질문을 이어받은 지속 상승 정의·분류·선택적 HGB 보완은 4.12에 정리한다.",
        "직전 상태와 최대 시간의 겹침 질문은 위 최소 진단으로 답했다.": "직전 상태와 최대 시간의 겹침 질문은 위 최소 진단으로 확인했다. 이어진 M04의 질문은 **직전 낮은 구간에서만 직전 기준을 쓰는 규칙이 유지 오차를 줄이면서 상승 과소예측을 얼마나 늘리는가**였다. 4.6에서 이 규칙을 실제 비교하고 상충 때문에 기각했다. 이후에도 가동 원인을 추정으로 채우지 않고 관측 가능한 전력 전환을 정의하는 방향으로 후속 분석을 진행했다.",
        "제출 보고서에서는 제2장에 동일 조건의 입력 비교와 B 선택 이유를": "제출 보고서 제2장에는 동일 조건의 입력 비교와 당시 B 선택 이유를, 제3장에는 상승·낮은 유지에서 남은 오류와 최대 시간 개선의 조건을 연결한다. 제5장의 현재 차별성은 4.12의 지속 상승 경고와 선택적 보완을 중심으로 설명한다. M03의 개발 개선률을 후속 방법의 개선률과 합산하지 않는다.",
        "제3장은 개선된 조건과 남은 실패·규칙 기각 근거를 연결한다.": "제3장에는 개선 조건과 규칙 기각의 근거를 연결한다. 낮은 상태에서 직전 값을 사용하는 규칙은 상승 과소예측을 악화해 채택하지 않았다. 4.9의 고정 후반기 평가에서도 마지막 값의 최대 시간 개선은 안정적으로 유지되지 않았다. 이 결과가 4.12에서 현재 상태만으로 일괄 보정하지 않고 다음 시간의 지속 상승을 별도로 예측해 보완 여부를 정한 배경이다.",
        "제2장에는 동일 조건의 추가 비교와 B 유지 이유, 제3장에는": "제2장에는 동일 조건의 추가 비교와 당시 B 유지 이유, 제3장에는 상승 개선·오경보·월별 실패를 연결한다. 이번 17.5% 상승 오차 감소는 당시 허용 요건을 통과하지 못한 보완 실험의 부분 성과로 보존한다. 이후 4.9에서 후반기를 확인했고, 현재 보고서의 차별화 결과는 지속 상승을 새로 정의한 4.12에 제시한다. 두 실험의 상승 정의와 평가 조건이 다르므로 개선률을 같은 성과로 합산하지 않는다.",
        "최종 보고서 제2장은 기본/확장 모델 선택과 후반기 단순 기준 대비": "최종 보고서 제2장에는 기본·확장 모델 선택과 이 절의 단순 기준 대비 HGB 성과를 제시하고, 이후 4.12의 동일 조건 지속 상승 보완 비교를 연결한다. 제3장에는 선택 후보의 상승·낮은 구간·최대 시간 오류와 후속 오경보 억제를 설명한다. 제5장의 차별성은 4.12에서 확인한 탐색적 선택 보완 효과를 중심으로 제시한다. 이 M05의 미채택 후보를 그 성과로 바꾸어 설명하지 않는다.",
        "이번 M05는 선언한 고정 확인 평가로 종료한다.": "이 M05는 선언한 고정 확인 평가로 종료했다. 당시 후보의 고정 결과를 보존하고, 이후 별도 개발에서는 이미 확인한 7~8월을 탐색에 이용한 기간으로 명시했다. 4.10의 재개발과 4.11의 검토 우선순위를 거쳐, 4.12에서는 지속 상승 시작을 예측하고 보완 HGB 적용을 제한하는 방법을 평가했다. 후속 성과가 나왔더라도 이 절의 기존 결과와 학습 조건은 변경하지 않는다.",
    }
    for prefix, replacement in replacements.items():
        pattern = r"(?m)^" + re.escape(prefix) + r"[^\n]*"
        text, count = re.subn(pattern, lambda _: replacement, text, count=1)
        assert count == 1, prefix
    return text


def section():
    table = rows("followup_metrics.csv")
    def value(variant, condition, metric="mae"):
        found = [r for r in table if r["variant"] == variant and r["period"] == "Jul-Aug" and r["condition"] == condition]
        assert len(found) == 1
        return float(found[0][metric])
    performance = []
    for variant, label in [("B0", "기본 HGB B0"), ("G1_noage", "지속 상승 경고 때 보완 HGB 적용"),
                           ("hour08", "모든 08시에 보완 HGB 적용")]:
        performance.append(f"| {label} | {value(variant, 'all'):.3f} | {value(variant, 'up_start'):.3f} | {value(variant, 'up_start', 'under'):.3f} | {value(variant, 'daily_peak_day_equal'):.3f} |")
    gains = [(1 - value("G1_noage", cond, metric) / value("B0", cond, metric)) * 100
             for cond, metric in [("up_start", "mae"), ("up_start", "under"), ("daily_peak_day_equal", "mae"), ("all", "mae")]]
    return f'''{HEADING}

### 중요한 상승 구간의 오차를 줄이기 위한 예측 설계

기본 HGB는 전체 전력 예측에서 단순 시차 기준보다 낮은 오차를 보였지만, 큰 변화가 시작되는 시간의 과소예측은 남았다. 따라서 전력 크기 예측과 함께 **평소 수준보다 높은 상태가 여러 시간 이어지는 시작을 한 시간 전에 예측하고, 그 경고가 나온 시간에만 보완 HGB를 적용하는 방법**을 검토했다. 이는 4.11의 예측오류 검토 우선순위와는 별도로 실제 전력 수준의 상승 시작을 정답으로 학습하는 방법이다.

관측 기간의 적용 결과, 수정한 경고는 공통 평가의 지속 상승 13건 중 7건을 선별했고 오경보는 0건이었다. 이 시간에 보완 HGB를 사용하면 기본 HGB 대비 상승 시작 MAE는 **{gains[0]:.1f}%**, 상승 과소예측은 **{gains[1]:.1f}%**, 실제 일별 최대 시간 MAE는 **{gains[2]:.1f}%** 감소했다. 전체 MAE도 **{gains[3]:.2f}%** 줄었다. 아래에서는 이 성과가 나온 사건 정의·평가 조건과 남은 탐지 범위를 설명한다.

이 절은 기존 M05를 새 수치로 바꾼 결과가 아니다. 4.9 M05 이후 전력 수준의 지속 전환을 별도 대상으로 정의한 S01~S06의 후속 연구다. 회귀 학습 구간도 달라졌으므로 4.9의 B 점수를 이 절의 B0 점수와 직접 비교하지 않는다. 특히 7~8월은 이미 확인한 자료이며 경과시간 입력 수정의 가설도 그 오류에서 나왔으므로, **후속 수치는 탐색적 재평가이고 새 독립 시험이 아니다.**

### 한 시간의 높은 값과 지속 상승의 시작을 구분했다

전력 수준은 한 시간 안의 네 15분 전력값을 산술평균해 정의했다. 네 값 중 최대값은 기존 HGB의 예측 목표로 유지했다. 따라서 지속 상승 분류는 평균 수준이 바뀌는 시점을 찾고, 회귀 모델은 그 시간의 최대 전력 크기를 예측한다. 한 구간의 일시적인 높은 값과 여러 시간 유지되는 높은 수준을 같은 사건으로 취급하지 않았다.

| 항목 | 고정한 정의 |
|---|---|
| 예측 시점 | t-1 시간의 관측을 마친 직후 다음 t 시간 예측 |
| 상승 크기 기준 | 1월의 양의 시간 평균 증가량 90분위수인 59.5 |
| 상승 시작 조건 | 직전 시간 대비 증가량과 과거 6시간 중앙값 대비 상승폭이 모두 59.5 이상이고, 중앙값 대비 상승폭이 과거 6시간 IQR의 1.5배 이상 |
| 지속 조건 | t·t+1·t+2의 평균이 시작 직전의 과거 중앙값보다 기준의 절반 이상 높은 상태 유지 |
| 중복 시작 방지 | 시작 당시 기준값을 고정해 같은 방향의 진행 중인 상태에서 시작을 반복 집계하지 않음 |
| 사용 가능한 입력 | 예측 전에 확인된 과거 전력·생산량·상태와 달력 정보 |

지속 여부는 미래 두 시간까지 관측된 뒤 정답으로 확정한다. 이를 과거 학습의 정답으로 사용하되, t 시점의 평균·최대값이나 미래 지속시간은 예측 입력에 넣지 않았다. 월별 학습에서도 다음 월 시작 전에 t+2 정답까지 확인된 기록만 사용했다. 59.5는 자료에서 정한 분석 기준이며 현장의 물리적 위험 한계값은 아니다.

1~6월 진단에서는 상승 시작 34건 중 28건이 3시간 지속 조건을 충족했다. 하락은 시작 112건 중 43건이 해당 방향으로 지속됐다. 이 진단을 바탕으로 상승·하락을 별도로 예측했으며, 아래 결합 성과는 개발 근거가 남은 **지속 상승**에 해당한다. 현재의 지속 정의는 시작 기준선으로부터 높은 수준에 머무는 조건이며, 일정한 폭의 박스권이나 일정한 전력값을 보장하지 않는다. [사건 진단](tables/regime_diagnosis/event_summary.csv), [개별 사건](tables/regime_diagnosis/events.csv), [정답·시점 조건](tables/regime_diagnosis/contract.json)을 보존한다.

### 상승 경고에 따른 두 HGB의 선택

기본 HGB B0는 기존 B 구성의 18개 입력을 사용한다. 보완 HGB B1은 여기에 과거 6시간의 중앙값·변동 폭·변화 방향 등 상태 정보를 더한 31개 입력을 사용한다. 두 모델은 같은 행에서 학습하고 HGB 설정도 같게 유지했다. 잎 수 15, 반복 150, 학습률 0.05, 최소 잎 표본 20, L2 1, 조기 종료 미사용, 난수 시드 42다.

초기 S04에서는 전환 확률을 모든 시간의 HGB 입력에 추가했으나 전체·최대 시간 오차가 악화했다. 이후 보완 HGB가 필요한 시간을 분류 경고로 제한하는 방식을 검토했다. 적용 규칙은 다음과 같다.

~~~text
지속 상승 경고가 있으면: 보완 HGB B1의 다음 시간 최대 전력 예측 사용
지속 상승 경고가 없으면: 기본 HGB B0의 다음 시간 최대 전력 예측 사용
~~~

상승 분류기는 과거 전력·생산량·상태와 달력을 사용하는 Logistic 모델이다. 표준화는 각 학습 자료에서만 계산했고 C=1, 최대 반복 3,000, 클래스 가중치 미사용, 난수 시드 42를 유지했다. 경고 경계는 이전 월의 모델이 낸 예측과 당시 확인된 정답으로 산정했다. 과거 음성 오경보 비율 1% 이내에서 적중 수가 가장 많은 경계를 고르고, 동률이면 오경보가 적고 경계가 높은 경우를 우선했다. 같은 학습 자료에 재대입한 점수로 경계를 정하지 않았다.

### 개발 평가와 고정 후속 평가의 연결

| 구분 | 학습·경계 산정 | 평가와 해석 |
|---|---|---|
| 4~6월 개발 | 각 월 이전의 확인된 정답으로 누적 학습. 경고 경계는 이전 월의 예측에서 산정 | 4월은 순위 평가, 5~6월은 자동 경고와 회귀 결합 평가 |
| 7~8월 후속 | 분류기는 1~6월 정답 확정 4,336시간, 회귀는 2~6월 공통 3,598시간으로 한 번씩 학습. 경계는 4~6월 OOF에서 산정 | 같은 모델·경계를 두 달 동안 유지. 매시간 입력은 갱신하고 7월 정답으로 8월을 다시 학습하지 않음 |

1월 자료 전체로 사건 기준을 정했으므로, 회귀 결합의 이전 월 분류 예측은 2월부터 사용했다. 이 조건에 맞춰 B0·B1도 동일한 2월 이후 행으로 학습했다. 초기 분류 표본이 부족한 시기의 대체와 분류기 가용 표시도 같은 조건으로 관리했다. 기존 M05처럼 회귀를 1월부터 학습한 점수를 이번 비교의 기준으로 재사용하지 않았다.

5~6월 개발 평가에서는 상승 12건을 모두 탐지하고 오경보 0건을 기록했다. 경고 때 B1을 적용한 결과 상승 시작 MAE는 10.046→8.661로 13.8%, 상승 과소예측은 8.002→7.137로 10.8% 줄었다. 경고 밖의 1,450시간은 기본 HGB의 예측을 유지했다. 전체 MAE 감소는 약 0.2%였다. [개발 탐지](tables/regime_age_ablation/development_detection.csv), [개발 회귀](tables/regime_age_ablation/development_metrics.csv)에 원래 분류기와 수정 분류기의 결과를 함께 남겼다.

### 후속 오류 진단에서 경과시간 입력을 수정했다

초기 S05 후속 결과에서는 상승 7건을 찾았지만 8월에 오경보 38건이 발생했다. 경고 때 보완 HGB를 사용하는 조합의 전체 MAE는 기본 HGB 6.335에서 6.680으로 증가했다. 개별 입력의 기여를 확인하니, 오경보 38건 모두 과거 상태의 경과시간이 180~227시간으로 학습 최대값 64시간을 넘었다. 이 변수는 38건 모두에서 상승 확률을 높이는 가장 큰 양의 선형 기여를 보였다.

이 근거로 S06에서는 상승 분류기에서 경과시간 입력 하나만 제거해 24개에서 23개 입력으로 줄였다. HGB B0·B1과 그 예측값은 그대로 사용했으므로, 경고 대상을 바꾼 효과를 분리할 수 있었다. B1의 경과시간 입력까지 제거한 실험은 아니다. 개발 평가에서 12건 적중·오경보 0건과 회귀 개선이 유지된 뒤에만 후속 기간을 재평가했다.

변경된 점수의 경고 경계 0.4554612321은 같은 과거 전용 알고리즘으로 4~6월 OOF 2,182시간에서 정했다. 이 자료의 상승 18건을 찾고 오경보는 0건이었다. 7~8월의 정답에 맞춰 경계를 다시 조정하거나 경과시간이 긴 평가 기록을 삭제하지 않았다. [오경보 조건](tables/regime_followup/verified_alarm_conditions.csv), [단일 입력 제거 조건](tables/regime_age_ablation/contract.json), [과거 경계 산정](tables/regime_age_ablation/calibration.csv)을 함께 보존한다.

### 중요한 시간의 오차 감소와 불필요한 보완 적용의 억제

7~8월 회귀 비교는 동일한 1,344시간을 사용했다. 그중 지속 정답까지 확인할 수 있는 1,340시간에서 분류 성능을 계산했다. 미래 정답이 없는 네 시간도 과거 입력과 현재 최대값이 있으므로 회귀 평가에는 유지했다. 일별 최대 시간 평가는 하루 24시간 공통 예측이 모두 있는 56일을 사용하고, 최대값 동률에는 하루 가중치 합이 1이 되도록 나눴다.

| 방법 | 전체 MAE | 상승 시작 MAE | 상승 평균 과소예측 | 일별 최대 시간 MAE |
|---|---:|---:|---:|---:|
{chr(10).join(performance)}

상승 평균 과소예측은 상승 시작에 속한 모든 평가 시간에서 `max(실제값-예측값, 0)`을 평균한 값이다. 과소예측한 사례만 따로 골라 평균한 수치가 아니다. 실제 상승이나 일별 최대 시간은 평가 조건을 구분하는 데 사용했으며, 예측 대상을 선택하는 입력은 관측 전의 분류 경고였다.

기본 HGB 대비 상승 시작 MAE는 20.037→18.205, 평균 과소예측은 19.419→16.342로 감소했다. 일별 최대 시간 MAE는 13.179→12.574로 감소했다. 전체 MAE는 6.335→6.317로 변화해 전체 평균에 대한 추가 이득은 작았지만, 중요한 구간의 개선을 얻으면서 전체 오차 악화를 피했다. 월별 전체 MAE도 7월 6.528→6.498, 8월 6.180→6.171로 감소했다.

| 지속 상승 탐지 | 경과시간 입력 포함 | 경과시간 입력 제거 |
|---|---:|---:|
| 공통 평가의 실제 상승 | 13 | 13 |
| 적중 / 미탐지 | 7 / 6 | 7 / 6 |
| 오경보 | 38 | 0 |
| 재현율 | 53.8% | 53.8% |
| 정밀도 | 15.6% | 100% (7/7건) |

경과시간 제거는 탐지한 상승 수를 늘리기보다 불필요한 경고를 억제했다. 보완 HGB 적용은 45회에서 7회로 줄었고, 나머지 1,337시간은 기본 HGB 예측과 같다. 오경보 38건을 없애면서 기존의 7개 적중을 유지한 것이 전체 MAE 악화를 해소한 직접적인 변화다. 이 표의 정밀도 100%는 해당 기간 경고 7건의 결과이며, 이후 기간의 무오경보를 보장하지 않는다. [후속 탐지](tables/regime_age_ablation/followup_detection.csv), [동일 시간 회귀 지표](tables/regime_age_ablation/followup_metrics.csv), [보완 적용 횟수](tables/regime_age_ablation/followup_burden.csv)를 근거로 한다.

### 탐지 범위와 단순 주간 규칙에 대한 해석

탐지한 7건은 모두 월요일 08시였다. 결과를 확인한 뒤 월요일 08시에 항상 보완 HGB를 쓰는 규칙도 대조했다. 이 사후 규칙은 상승 8건을 찾는 대신 오경보 1건이 발생했고, 수정 모델은 7건을 찾고 오경보 0건을 기록했다. 단순 규칙의 전체 MAE는 6.320, 상승 시작 MAE는 18.434로 모델의 6.317·18.205보다 조금 컸으며 일별 최대 시간 MAE 12.574는 같았다. 따라서 관측된 개선을 복잡한 분류기만의 큰 우위라고 설명하지 않는다. [주간 규칙의 사후 조건](tables/regime_age_ablation/calendar_control_contract.json), [탐지](tables/regime_age_ablation/calendar_control_detection.csv), [회귀](tables/regime_age_ablation/calendar_control_metrics.csv)를 남겼다.

공통 평가에서는 13건 중 6건을 놓쳤다. 과거 6시간 이력만 요구하는 전체 분류 범위에서는 상승 14건 중 7건을 찾아 재현율 50%였고, 상승 한 건은 회귀 공통 시차 조건을 충족하지 않아 공통 비교에서 제외됐다. 따라서 현재 모델은 모든 상승을 포괄하는 탐지기로 설명하기보다, 관측된 일부 상승에서 오경보를 억제하며 예측값을 보완한 후보로 해석한다.

사전에 정한 분류 기준은 재현율 80%·정밀도 80%·음성 오경보 비율 1% 이하였다. 수정 모델은 재현율 기준에 미달했다. 반면 상승 MAE·과소예측 각각 5% 이상 개선, 전체·일별 최대 MAE 악화 1% 이내라는 회귀 결합 기준은 통과했다. 두 판단을 유지하면서 보고서에서는 확인된 오경보 억제와 오차 감소를 주된 성과로 제시한다. 이는 다른 팀의 결과와 비교한 우위나 실제 운영에서 검증한 절감 효과는 아니다.

### 현장 활용과 보고서의 연결

활용 흐름은 직전 시간의 관측 완료 → 다음 시간의 지속 상승 경고와 최대 전력 예측 생성 → 경고가 있으면 보완 HGB 결과 확인 → 담당자가 예정된 가동과 동시 사용 조건을 점검하는 순서다. 경고가 없을 때도 기본 HGB의 예측은 계속 제공한다. 경고가 없는 시간을 안전하다고 판정하는 체계로 사용하면 현재 미탐지 범위를 설명할 수 없으므로, 담당자의 확인을 보조하는 용도로 제안한다. 실제 가동 조정이나 전력·비용 절감량은 측정하지 않았다.

최종 결과보고서에서는 제1장에 지속 전환의 정의와 데이터 진단 근거를 짧게 연결하고, 제2장에 동일 조건의 HGB 결합과 성능표를 제시한다. 제3장에는 경과시간 외삽에 따른 오경보·입력 제거 효과·미탐지와 주간 규칙 대조를 배치한다. 제4장은 예측값을 확인할 시점과 조치 검토 흐름, 제5장은 **데이터에서 찾은 실패조건을 반영해 필요한 시간에만 예측을 보완한 기여**를 요약한다. 제6장에는 아래 코드·조건·저장 결과의 재현 경로를 제시한다. S01~S06 번호는 내부 작업 이력이며 최종 보고서의 장 번호로 사용할 필요가 없다.

### 사용 코드와 검증·제출 보존

| 단계 | Python 코드와 역할 |
|---|---|
| S01~S02 | [regime_diagnose.py](scripts/regime_diagnose.py): 사건 기준·과거 상태·선행 조건 진단. [regime_diagnose_verify.py](scripts/regime_diagnose_verify.py): 원자료에서 시점·사건·조건 재구성 |
| S03 | [regime_forecast.py](scripts/regime_forecast.py): 방향별 시간순 분류·과거 경계·입력 구성. [regime_forecast_verify.py](scripts/regime_forecast_verify.py): AP·경계·시간순 조건·저장 모델 검증 |
| S04 | [regime_integrate.py](scripts/regime_integrate.py), [regime_integrate_verify.py](scripts/regime_integrate_verify.py): 동일 행 HGB 입력 비교와 독립 검증. [regime_route.py](scripts/regime_route.py), [regime_route_verify.py](scripts/regime_route_verify.py): 경고 시간에만 적용하는 결합·대조 |
| S05 | [regime_followup.py](scripts/regime_followup.py), [regime_followup_verify.py](scripts/regime_followup_verify.py): 6월 말 고정 모델의 후속 평가·원자료와 예측의 대조 |
| S06 | [regime_age_ablation.py](scripts/regime_age_ablation.py): 경과시간 입력 제거·개발 통과 뒤 조건부 후속 평가. [regime_age_ablation_verify.py](scripts/regime_age_ablation_verify.py): 독립 재학습·경계·오차 검증 |
| 사후 주간 대조 | [regime_age_calendar_control.py](scripts/regime_age_calendar_control.py): 월요일 08시 규칙을 새 학습 없이 대조 |
| 입력·문서·보존 | [m01_prepare.py](scripts/m01_prepare.py): 공통 시차와 HGB 입력. [regime_manuscript_documents.py](scripts/regime_manuscript_documents.py): 원고 보존·절 추가·수치와 문서 검증. [modeling_submission_audit.py](scripts/modeling_submission_audit.py): 의존 코드·모델·표·문서 보존 검사 |

이 코드들의 전체 `.py` 소스와 공통 의존 파일을 같은 폴더 구조로 제출 보존한다. 4.5의 기존 코드표에 더해 위 코드들이 이번 4.12의 실행 근거다. [S01~S02 결과](tables/regime_diagnosis), [S03 결과](tables/regime_forecast), [S04 입력 비교](tables/regime_integration), [S04 선택 결합](tables/regime_routing), [S05 결과](tables/regime_followup), [S06 결과](tables/regime_age_ablation)와 각 모델 목록도 함께 유지한다.

S06의 모델 4개를 저장 후 다시 불러오고 별도 코드로 모두 재학습했다. 학습 구간 안의 표준화·정답 확정 시점·과거 OOF 경계, 분류 지표 24개와 회귀 지표 150개를 검증했다. 원자료의 정상 시각 5,784시간에서 최대값을 대조하고, 기존 HGB 예측의 재사용·경고에 따른 선택·일별 최대와 시작 후 3시간 오차를 확인했다. 모든 검증이 통과했으며, S06의 새 회귀 학습은 0회다. [모델 목록](tables/regime_age_ablation/model_manifest.csv), [실행](tables/regime_age_ablation/run.json), [독립 검증](tables/regime_age_ablation/independent_verification.json), [판단](tables/regime_age_ablation/decision.json)을 보존한다.

아래는 최초 분석 실행 순서다. 정규 CPython 3.13에서 루트 AGENTS.md의 승격 Start-Process 경로를 사용한다. 완료된 분석은 기존 실행 파일의 덮어쓰기를 거부하므로 현재 제출 자료의 보존 검사는 마지막 명령으로 수행한다. S06 분석 당시의 원고 상태는 [추가 전 원고](tables/regime_manuscript/manuscript_before.md)에 보존했다. 분석 단계의 검증 기록과 이후 원고 편집의 검증 기록을 구분한다.

~~~text
Modeling/scripts/regime_diagnose.py probe
Modeling/scripts/regime_diagnose.py freeze
Modeling/scripts/regime_diagnose.py run
Modeling/scripts/regime_diagnose_verify.py
Modeling/scripts/regime_forecast.py freeze
Modeling/scripts/regime_forecast.py run
Modeling/scripts/regime_forecast_verify.py
Modeling/scripts/regime_integrate.py freeze
Modeling/scripts/regime_integrate.py run
Modeling/scripts/regime_integrate_verify.py
Modeling/scripts/regime_route.py freeze
Modeling/scripts/regime_route.py run
Modeling/scripts/regime_route_verify.py
Modeling/scripts/regime_followup.py freeze
Modeling/scripts/regime_followup.py run
Modeling/scripts/regime_followup_verify.py
Modeling/scripts/regime_age_ablation.py freeze
Modeling/scripts/regime_age_ablation.py run
Modeling/scripts/regime_age_ablation_verify.py
Modeling/scripts/regime_age_calendar_control.py
Modeling/scripts/regime_manuscript_documents.py --write
Modeling/scripts/regime_manuscript_documents.py --verify
Modeling/scripts/modeling_submission_audit.py --verify
~~~

이번 원고 편집에서는 새로운 학습·예측이나 압축본을 만들지 않았다. M03 이후의 잠정 결론과 후속 연결 문장을 현재 결과에 맞게 수정했고, 기존 M05를 포함한 성능표·수치·실험 조건은 유지했다. 편집 전 원고 전체를 별도로 보존했다. 재현 파일의 무결성 검증과 깨끗한 새 환경에서의 전체 재실행은 구분하며, 최종 제출 완료를 주장하지 않는다.
'''


def verify():
    text = MANUSCRIPT.read_text(encoding="utf-8")
    # Subsequent reporting additions have their own verifier; isolate this section.
    text = re.sub(r'<!-- MODELING_CLOSE_BEGIN -->.*?<!-- MODELING_CLOSE_END -->\s*', '', text, flags=re.S)
    text = text.split('## 4.13 ')[0]
    before = (OUT / "manuscript_before.md").read_text(encoding="utf-8")
    assert text.count(HEADING) == 1 and text.split(HEADING)[1].strip() == section().split(HEADING)[1].strip()
    start = "## 4.1 M01:"
    assert text.split(start, 1)[1].split(HEADING)[0].strip() == revise_history(before).split(start, 1)[1].strip()
    old_tables = [line for line in before.splitlines() if line.startswith("|")]
    historical = text.split(HEADING)[0]
    assert old_tables == [line for line in historical.splitlines() if line.startswith("|")]
    c = json.loads((RESULT / "contract.json").read_text(encoding="utf-8"))
    assert sha(OUT / "manuscript_before.md") == c["manuscript_sha256"]
    assert sha(ROOT / "README.md") == c["root_readme_sha256"]
    sources = {}
    for name in ["followup_metrics.csv", "followup_detection.csv", "development_detection.csv", "decision.json",
                 "run.json", "independent_verification.json", "calendar_control_metrics.csv", "calendar_control_detection.csv"]:
        sources["Modeling/tables/regime_age_ablation/" + name] = sha(RESULT / name)
    run = json.loads((RESULT / "run.json").read_text(encoding="utf-8"))
    verified = json.loads((RESULT / "independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"] == "passed" and verified["run_sha256"] == sha(RESULT / "run.json")
    for name, expected in run["outputs_sha256"].items():
        assert sha(RESULT / name) == expected
    for name, expected in c["inputs_sha256"].items():
        assert sha(ROOT / name) == expected
    det = [r for r in rows("followup_detection.csv") if r["scope"] == "common_regression" and r["period"] == "Jul-Aug" and r["method"] == "noage"][0]
    assert [int(det[n]) for n in ["hours", "events", "tp", "fp", "fn"]] == [1340, 13, 7, 0, 6]
    links = 0
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if not target.startswith(("http:", "https:", "#")):
            assert (MANUSCRIPT.parent / target.split("#")[0].strip("<>")).exists(), target
            links += 1
    record = {"status": "passed", "script_sha256": sha(__file__), "source_sha256": sources,
              "documents_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in [MANUSCRIPT, README, ROOT / "README.md"]},
              "original_manuscript_sha256": sha(OUT / "manuscript_before.md"), "historical_tables_preserved": True, "historical_narrative_updated_as_declared": True,
              "links_checked": links, "new_fits": 0}
    (OUT / "document_verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "links_checked": links, "historical_tables_preserved": True, "new_fits": 0}), flush=True)


def write():
    original = MANUSCRIPT.read_text(encoding="utf-8")
    assert HEADING not in original and not (OUT / "manuscript_before.md").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manuscript_before.md").write_bytes(MANUSCRIPT.read_bytes())
    assert sha(OUT / "manuscript_before.md") == sha(MANUSCRIPT)
    updated, count = re.subn(r"(?m)^이번 원고는 .*?$", lambda _: INTRO, revise_history(original), count=1)
    assert count == 1
    MANUSCRIPT.write_text(updated.rstrip() + "\n\n" + section().strip() + "\n", encoding="utf-8")
    text = README.read_text(encoding="utf-8")
    head = "## S06 실행 결과: 경과시간 입력 제거로 오경보 억제, 상승 탐지 범위의 한계"
    note = "\n\n**상세 원고 반영:** [4.12 지속 상승 예측과 선택적 HGB 보완](04_Modeling_원고.md#412-지속-상승-예측과-선택적-hgb-보완)에 S01~S06의 정의·학습·성능·오경보 개선·활용과 사용 코드를 연결했다. [regime_manuscript_documents.py](scripts/regime_manuscript_documents.py)로 추가 전 원고를 보존하고 기존 4.1~4.11 본문과 수치·링크를 검증했다. 최종 보고서는 방법·성능을 제2장, 오류 개선을 제3장, 차별성을 제5장에 배치한다."
    assert text.count(head) == 1
    README.write_text(text.replace(head, head + note, 1), encoding="utf-8")
    verify()


def refresh():
    # Rebuild only this authorized document from the preserved original and declared edits.
    assert '<!-- MODELING_CLOSE_BEGIN -->' not in MANUSCRIPT.read_text(encoding='utf-8'), 'Later reporting additions exist; use their document workflow.'
    original = (OUT / "manuscript_before.md").read_text(encoding="utf-8")
    updated, count = re.subn(r"(?m)^이번 원고는 .*?$", lambda _: INTRO, revise_history(original), count=1)
    assert count == 1 and MANUSCRIPT.read_text(encoding="utf-8").count(HEADING) == 1
    MANUSCRIPT.write_text(updated.rstrip() + "\n\n" + section().strip() + "\n", encoding="utf-8")
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true"); mode.add_argument("--verify", action="store_true")
    mode.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    write() if args.write else refresh() if args.refresh else verify()
