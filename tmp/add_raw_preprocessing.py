from pathlib import Path
import textwrap
import nbformat
root=Path(__file__).resolve().parents[1];path=root/'제출/통합_분석.ipynb'
nb=nbformat.read(path,as_version=4)
nb.cells=[c for c in nb.cells if c.metadata.get('role')!='executed_result']
adapter=r'''
PREPARED_FULL = None
PREPARED_READ_LOG = []
@contextlib.contextmanager
def prepared_input():
    """원본 해시·독립 원자료 검사는 유지하고, 분석의 pandas 입력은 이번 전처리 CSV로 연결한다."""
    read_original = pd.read_csv
    raw_locations = {(PACKAGE/'data/origin/okm_augumented_2021.csv').resolve(),
                     (WORK/'data/origin/okm_augumented_2021.csv').resolve()}
    def read_prepared(source, *args, **kwargs):
        if isinstance(source, (str, Path)) and Path(source).resolve() in raw_locations:
            assert PREPARED_FULL is not None and PREPARED_FULL.exists(), 'raw 전처리 셀을 먼저 실행하세요.'
            PREPARED_READ_LOG.append(str(source))
            kwargs.setdefault('float_precision', 'round_trip')
            return read_original(PREPARED_FULL, *args, **kwargs)
        return read_original(source, *args, **kwargs)
    pd.read_csv = read_prepared
    try:yield
    finally:pd.read_csv = read_original
'''
preprocess=r'''
RAW_SOURCE = PACKAGE/'data/origin/okm_augumented_2021.csv'
PROCESSED = RUN/'data/processed'
PROCESSED.mkdir(parents=True)
raw = pd.read_csv(RAW_SOURCE, encoding='utf-8-sig')
assert raw.shape == (6168, 18)
numeric = raw.copy()
for column in numeric.columns:
    numeric[column] = pd.to_numeric(numeric[column], errors='raise')
assert numeric['날짜'].notna().all() and numeric['시간'].notna().all()
dates = pd.to_datetime(numeric['날짜'].astype(str), format='%Y%m%d', errors='raise')
scoped = numeric.loc[numeric['날짜'].between(20210101, 20210831)].copy()
invalid = scoped.loc[~scoped['시간'].between(0, 23)].copy()
valid = scoped.loc[scoped['시간'].between(0, 23)].copy()
valid['timestamp'] = pd.to_datetime(valid['날짜'].astype(str), format='%Y%m%d') + pd.to_timedelta(valid['시간'], unit='h')
valid = valid.sort_values('timestamp').reset_index(drop=True)
assert len(scoped) == 5832 and len(invalid) == 48 and len(valid) == 5784
assert valid.timestamp.is_unique and valid.groupby('날짜').size().eq(24).all()
slots=['15분','30분','45분','60분']
np.testing.assert_array_equal(valid['평균'], np.floor(valid[slots].mean(axis=1)+.5))
PREPARED_FULL = PROCESSED/'정규화_전체자료.csv'
numeric.to_csv(PREPARED_FULL, index=False, encoding='utf-8-sig', float_format='%.17g')
valid.to_csv(PROCESSED/'정상시간_전처리자료.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
invalid.assign(exclusion_reason='invalid_hour').to_csv(PROCESSED/'시간오류_제외기록.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
restored = pd.read_csv(PREPARED_FULL, encoding='utf-8-sig', float_precision='round_trip')
pd.testing.assert_frame_equal(numeric, restored, check_exact=True)
preprocessing = {'status':'passed','raw_source':RAW_SOURCE.relative_to(PACKAGE).as_posix(),
                 'raw_sha256':digest(RAW_SOURCE),'raw_rows':len(raw),'columns':len(raw.columns),
                 'scope_rows':len(scoped),'invalid_hours':len(invalid),'valid_hours':len(valid),
                 'valid_days':valid['날짜'].nunique(),'exact_numeric_roundtrip':True,
                 'missing_imputed':False,'zero_or_extreme_values_deleted':False,
                 'analysis_input':PREPARED_FULL.relative_to(PACKAGE).as_posix(),
                 'valid_hour_input':(PROCESSED/'정상시간_전처리자료.csv').relative_to(PACKAGE).as_posix()}
(PROCESSED/'전처리_검증.json').write_text(json.dumps(preprocessing,ensure_ascii=False,indent=2),encoding='utf-8')
RECORDS.append({'step':'00_raw전처리','status':'passed','seconds':0.0})
display(Markdown('### raw 전처리 실행 결과\n\n'+pd.DataFrame([preprocessing]).to_markdown(index=False)))
'''
for c in nb.cells:
    if c.cell_type=='code' and 'def computation_paths()' in c.source:
        c.source=c.source.replace('    try:yield\n    finally:\n        for m,k,v in previous:', '    try:\n        with prepared_input():yield\n    finally:\n        for m,k,v in previous:')
        if 'def prepared_input()' not in c.source:c.source+='\n'+textwrap.dedent(adapter).strip()
    if c.cell_type=='code' and c.source.startswith("current=MODULES['modeling_close_analysis']"):
        c.source=c.source.replace("    execute_step('현재모델원자료재학습대조',current.run)","    with prepared_input():\n        execute_step('현재모델원자료재학습대조',current.run)")
    if c.cell_type=='code' and c.source.startswith("assert all(digest(PACKAGE/r['path'])"):
        c.source=c.source.replace("'historical_optuna_search_rerun':False,", "'historical_optuna_search_rerun':False,'raw_preprocessing':preprocessing,'preprocessed_reads':len(PREPARED_READ_LOG),")
        c.source="assert len(PREPARED_READ_LOG)>0, '전처리 입력 사용 기록 없음'\n"+c.source
index=next(i for i,c in enumerate(nb.cells) if c.cell_type=='code' and c.source.startswith('with computation_paths():'))
nb.cells[index:index]=[
    nbformat.v4.new_markdown_cell('''### 1.6 제출 raw 데이터의 공통 전처리

입력은 이 제출 폴더의 `data/origin/okm_augumented_2021.csv`다. 18개 열의 숫자 형식·날짜·시간·중복을 확인하고 1 ~ 8월 5832행에서 시간 오류 48행을 구분해 정상 5784시간을 구성한다. 결측·0·극단값은 원고의 기준에 따라 유지한다. 날씨 결측 처리와 학습용 스케일링은 필요한 모델의 학습 자료 안에서 수행한다.

이번 실행에서 만든 `정규화_전체자료.csv`를 뒤의 EDA·Analysis·현재 모델 입력에 연결한다. 정상 시간 자료와 제외 기록도 별도 CSV로 저장한다. 기존 코드의 원본 해시·독립 검사는 raw를 그대로 대조하며, 분석용 `pandas.read_csv`는 명시한 두 raw 경로에 대해서만 이번 전처리 CSV를 읽는다. 다른 파일과 CSV 수치 왕복 단계에는 적용하지 않는다. 전체자료를 유지해 데이터 감사의 원본 분모를 보존하고, 각 분석은 동일한 기간·정상시간 조건을 적용한다.''',metadata={'role':'raw_preprocessing_description'}),
    nbformat.v4.new_code_cell(textwrap.dedent(preprocess).strip(),metadata={'role':'raw_preprocessing'})]
nb.cells[0].source+='\n\nRaw 데이터는 `data/origin/okm_augumented_2021.csv`로 함께 제출한다. 1장의 공통 전처리 셀에서 이번 실행의 전처리 CSV를 만든 뒤 각 분석과 현재 모델이 이를 읽는다.'
nbformat.write(nb,path)
readme=root/'제출/README_실행안내.md'
s=readme.read_text(encoding='utf-8')
s+='''
Raw 데이터는 `data/origin/okm_augumented_2021.csv`에 포함했으며 원본과 SHA-256이 같다. 실행 입력 경로는 제출 폴더 내부에서 찾는다. 노트북 1장의 공통 전처리 셀이 숫자·날짜·시간·중복을 점검하고, 전체 정규화 자료·정상 시간 자료·시간 오류 제외 기록·전처리 검증 JSON을 새 실행 폴더에 저장한다. 이후 분석용 raw 읽기를 이번 전처리 CSV에 연결하므로 기존 프로젝트의 raw나 이전 전처리 파일을 찾아 읽지 않는다.
'''
readme.write_text(s,encoding='utf-8')
print('Raw preprocessing and downstream prepared-input routing added.')
