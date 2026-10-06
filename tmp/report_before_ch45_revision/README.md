# 결과보고서 제3~6장

[제3~6장 원고](03_06_보고서_원고.md)는 네 장을 합쳐 1~2페이지를 목표로 작성한 마크다운이다. 제1·2장의 설명·전체 성능표를 반복하지 않고 제3장에 사례 그림 하나, 제4장에 활용 절차, 제5장에 기여와 적용 범위, 제6장에 재현 경로를 남겼다. 실제 쪽수는 최종 양식의 글꼴·줄간격·그림 크기에 따라 달라진다. 이 README는 편집·실행 안내이며 본문 분량에 포함할 원고가 아니다.

근거는 [Modeling 상세 원고 4.12~4.13](../Modeling/04_Modeling_원고.md)과 저장 결과다. 기존 EDA·Analysis·Modeling 원고는 보존하며 본 폴더만 단독 제출하면 상대 링크와 재현 의존 파일이 누락되므로 프로젝트 구조를 함께 유지한다. PDF는 생성하지 않는다.

## 장별 코드·결과 연결

| 장 | 사용 코드 | 근거 파일 |
|---|---|---|
| 제3장 오경보·탐지 | [경과시간 제거 비교](../Modeling/scripts/regime_age_ablation.py), [독립 검증](../Modeling/scripts/regime_age_ablation_verify.py) | [탐지 결과](../Modeling/tables/regime_age_ablation/followup_detection.csv), [판단](../Modeling/tables/regime_age_ablation/decision.json) |
| 제3장 사례·변수 기여 | [분해·사례 선정](../Modeling/scripts/modeling_close_analysis.py), [그림 생성](../Modeling/scripts/modeling_close_figures.py)의 `cases()` | [선정 기준](../Modeling/tables/modeling_close/case_selection.csv), [사례 창](../Modeling/tables/modeling_close/case_windows.csv), [변수 기여](../Modeling/tables/modeling_close/contribution_summary.csv), [상승별 오차](../Modeling/tables/modeling_close/event_explanations.csv) |
| 제4장 활용 | [선택 결합](../Modeling/scripts/regime_route.py), [고정 평가](../Modeling/scripts/regime_followup.py) | [Modeling의 활용 제안](../Modeling/04_Modeling_원고.md); 현장 화면·보류 처리·가동 조정은 제안이며 구축·실증 완료가 아님 |
| 제5장 차별성 | [선택 보완 평가](../Modeling/scripts/regime_age_ablation.py), [주간 규칙 대조](../Modeling/scripts/regime_age_calendar_control.py) | [회귀 오차](../Modeling/tables/regime_age_ablation/followup_metrics.csv), [주간 탐지](../Modeling/tables/regime_age_ablation/calendar_control_detection.csv), [주간 오차](../Modeling/tables/regime_age_ablation/calendar_control_metrics.csv) |
| 제6장 재현성 | [한 번에 재현](../Modeling/scripts/modeling_reproduce_selected.py), [원자료 처리·학습·대조](../Modeling/scripts/modeling_close_analysis.py), [보존 검사](../Modeling/scripts/modeling_submission_audit.py) | [실제 재현 기록](../Modeling/tables/modeling_replay/run.json), [전체 파일 목록](../Modeling/submission_manifest.json), [패키지 버전](../Modeling/requirements.txt) |

## 평가 범위와 편집 기준

- 제3장 분류 결과는 회귀와 공통 조건을 갖춘 정답 확정 1,340시간·상승 13건이다. 제5장 회귀는 동일 1,344시간이며 일별 최대는 완전한 56일을 같은 가중치로 평가했다. 분류 전체 범위의 14건과 혼합하지 않는다.
- 평균 과소예측은 평가 대상의 `max(실제값 − 예측값, 0)` 평균이다. 과소예측한 행만을 대상으로 한 평균이 아니다.
- 최종 결합의 후반기 결과는 이미 관측한 7~8월의 탐색적 재평가다. 제2장의 독립 평가와 혼동하지 않도록 제3장의 해당 문장을 유지한다.
- 제2장에 동일 사례 그림을 이미 넣으면 제3장 그림을 중복 삽입하지 않고 해당 그림 번호를 참조한다. 현재 그림 3-1은 제3장 내 번호이며 최종 문서의 번호 체계에 맞춰 조정한다.
- 제5장은 제2장의 전체 성능표 대신 핵심 기여 두 수치만 인용한다. 모델 선택·학습 기간·하이퍼파라미터는 제2장에 배치한다.
- 제1·2장에 사용하는 전체 과거 분석 코드와 실행 순서는 [Modeling 상세 원고](../Modeling/04_Modeling_원고.md)의 4.5·4.12·4.13에 남아 있다. 아래 명령은 현재 선택 방법을 재현하는 경로다.

## 재현 실행

프로젝트 루트에서 정규 CPython 3.13과 `Modeling/requirements.txt`에 기록된 패키지를 사용한다. `data/`, `Modeling/scripts/`, `Modeling/config/`, `Modeling/tables/`, `Modeling/models/`와 보존 목록의 의존 파일을 유지한다. 원자료만 있으면 자동으로 전체 실험 이력을 복구하는 방식은 아니며, 고정 설정·정의와 대조용 결과도 필요하다.

Windows PowerShell 실행 예시는 다음과 같다. 출력 폴더는 새 이름이어야 하며, 이미 내용이 있는 폴더는 덮어쓰지 않는다. 이 프로젝트의 Codex 환경에서는 Python 실행 도구에 첫 시도부터 `require_escalated`를 지정한다.

```powershell
$p = Start-Process -FilePath 'C:\Program Files\Python313\python.exe' -ArgumentList 'Modeling/scripts/modeling_reproduce_selected.py','--output-dir','Modeling/reproduction_report_check' -WorkingDirectory (Get-Location).Path -Wait -PassThru -NoNewWindow
exit $p.ExitCode
```

정상 종료 시 새 출력 폴더에 `contract.json`, `run.json`, `reproduced_predictions.csv`, `logit_contributions.csv`, `coefficient_table.csv`, `contribution_summary.csv`, `event_explanations.csv`, `case_windows.csv`, `case_selection.csv`가 생성된다. `run.json`의 `status: passed`와 예측 대조 건수를 확인한다. 기존 [modeling_replay/run.json](../Modeling/tables/modeling_replay/run.json)은 실제 실행 완료 기록이고, 위 `reproduction_report_check`는 다음 실행용 예시 경로다.

기존 제출 자료의 코드·결과 파일 보존 확인은 별도 명령으로 수행한다.

```powershell
$p = Start-Process -FilePath 'C:\Program Files\Python313\python.exe' -ArgumentList 'Modeling/scripts/modeling_submission_audit.py','--verify' -WorkingDirectory (Get-Location).Path -Wait -PassThru -NoNewWindow
exit $p.ExitCode
```

현재 재현은 고정 방법의 전처리·학습·추론과 기존 예측 대조를 확인한다. 운영 시스템 배포, 새 환경 설치, 과거 Optuna 전체 재탐색은 별도 범위다.
