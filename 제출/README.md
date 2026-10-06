# 제출 파일 실행 안내

## 파일 구성

- `통합_분석.ipynb`: 데이터 점검·전처리·EDA·Analysis·Modeling 계산, 표와 그림, 해석.
- `data/raw/okm_augumented_2021.csv`: 수정하지 않은 제공 원자료.
- `requirements.txt`: 검증에 사용한 Python 3.13 패키지 버전.
- `reference/`: 현재 고정 예측 대조에 필요한 CSV. 현재 모델의 학습 입력은 원자료에서 구성한다.
- `테스트데이터_예측결과.csv`: 7~8월 공통 1,344시간의 최종 예측. 전체 실행 때 생성한다.
- `data/processed/`, `output/`: 노트북이 생성한 전처리 자료·결과표·그림.

제출 실행에 별도 `.py` 파일은 사용하지 않는다. 사용자 제공 예시 노트북은 제출 폴더 밖의 `tmp/submission_before_readable_rebuild/`에 보존했다.

## 가상환경과 커널 설정 (Windows PowerShell)

제출 폴더에서 Python 3.13으로 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name manufacturing-submit --display-name "Python (제조데이터 제출)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Jupyter에서 노트북을 열고 **Python (제조데이터 제출)** 커널을 선택한 뒤 **커널 재시작 및 전체 실행**을 수행한다. VS Code에서는 우측 상단 커널 선택에서 같은 환경을 선택한다.

`requirements.txt`는 패키지를 설치하며 노트북 커널을 자동 선택하지 않는다. 커널 등록은 한 번 수행하며 이후 같은 커널을 선택해 실행한다. 가상환경 자체를 제출할 필요는 없다.

등록 후 화면 없이 전체 실행하고 출력까지 저장하려면 다음 표준 명령을 사용한다. 별도 실행기 파일은 필요 없다.

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=manufacturing-submit --ExecutePreprocessor.timeout=1800 통합_분석.ipynb
```

## 분석 범위와 해석

원자료를 매번 읽고 1~8월·정상 0~23시를 필터링한다. 시간 오류 48행은 결과에 따로 기록하며 전력 0·생산량 0·큰 값은 유지한다. 이후 계산은 이 자료의 정확한 과거 시차를 사용한다. 기상 결측은 관련 계산에서만 제외한다.

과거 시간순 모델 비교는 당시 내부 검증에서 선택한 모델 설정을 고정해 재학습한다. Optuna 전체 재탐색은 수행하지 않는다. 초기 Ridge·Elastic Net·SVR도 당시 고정 설정으로 원자료 시차 입력에서 다시 학습한다. 최종 Logistic 및 HGB B0/B1은 원자료로 재학습하고 OOF 경계도 다시 계산한다.

7~8월은 이미 관찰한 자료의 탐색적 재평가이며 독립 최종시험으로 해석하지 않는다. 마지막 셀에서 고정 참조 예측과 수치 일치를 검증한다.

정규 CPython 3.13의 기존 설치 환경에서 전체 실행을 검증했다. 별도의 새 가상환경 설치 검증과 구분한다.

커널 설정 참고: [IPython 공식 문서](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).
