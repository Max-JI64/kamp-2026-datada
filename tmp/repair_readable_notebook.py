from pathlib import Path
import nbformat
root=Path(__file__).resolve().parents[1]
path=root/'제출/통합_분석.ipynb'
nb=nbformat.read(path,as_version=4)
for c in nb.cells:
    c.source=c.source.replace('.rename("결측 셀 수").reset_index(names="변수")','.rename("결측 셀 수").rename_axis("변수").reset_index()')
    c.source=c.source.replace('.rename("시간 수").reset_index(names="기준 이상 구간 수")','.rename("시간 수").rename_axis("기준 이상 구간 수").reset_index()')
    c.source=c.source.replace('"模型"','"모델"')
    if c.cell_type=='code' and c.source.startswith('from pathlib import Path'):
        c.source=c.source.replace('from sklearn.linear_model import LogisticRegression','from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet\nfrom sklearn.svm import SVR\nfrom sklearn.compose import TransformedTargetRegressor')
    if c.cell_type=='code' and 'dev_predictions={"previous_hour"' in c.source:
        c.source=c.source.replace('    for item in [c for c in SELECTED_CONFIGS if c["split"]==split and "parameters" in c]:','''    initial_models={
        "Ridge_initial":make_pipeline(StandardScaler(),Ridge(alpha=10.)),
        "ElasticNet_initial":TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000,tol=.0001,selection="cyclic")),transformer=StandardScaler()),
        "SVR_initial":TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),SVR(C=1.,gamma=.03,epsilon=.05,kernel="rbf",cache_size=512)),transformer=StandardScaler()),
        "HGB_initial":HistGradientBoostingRegressor(max_leaf_nodes=15,max_iter=150,learning_rate=.05,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=42)}
    for name,initial_model in initial_models.items():
        initial_model.fit(train_dev[basic_features],train_dev.target_maximum)
        dev_predictions[name]=initial_model.predict(valid_dev[basic_features])
    for item in [c for c in SELECTED_CONFIGS if c["split"]==split and "parameters" in c]:''')
        c.source=c.source[:c.source.index('initial_reference=pd.read_csv')].rstrip()
    if c.cell_type=='markdown':
        c.source=c.source.replace(' 과거 초기 선형·SVR 비교는 저장 예측을 대조 자료로 함께 제시한다。','')
        c.source=c.source.replace(' 과거 초기 선형·SVR 비교는 저장 예측을 대조 자료로 함께 제시한다.',' 초기 Ridge·Elastic Net·SVR도 당시 고정 설정으로 같은 시차 자료에서 다시 학습한다.')
        c.source=c.source.replace(' 초기 선형·SVR 대조 표는 당시 저장 예측의 재집계로 재학습 결과와 구분한다.',' 모든 모델은 위에서 구성한 원자료 시차 입력으로 다시 학습한다.')
    if c.cell_type=='code':c.execution_count=None;c.outputs=[]
nbformat.write(nb,path)
readme=root/'제출/README.md'
s=readme.read_text(encoding='utf-8').replace('과거 초기 모델 비교와 현재 고정 예측 대조에 필요한 CSV','현재 고정 예측 대조에 필요한 CSV').replace('초기 선형·SVR 비교는 당시 저장 예측의 재집계다.','초기 Ridge·Elastic Net·SVR도 당시 고정 설정으로 원자료 시차 입력에서 다시 학습한다.')
readme.write_text(s,encoding='utf-8')
backup=root/'tmp/submission_before_readable_rebuild'
for name in ['검증_요약.json','선별_파일목록.json','README_실행안내.md']:
    old=root/'제출'/name
    if old.exists():old.replace(backup/name)
ref=root/'제출/reference/초기모델_개발예측.csv'
if ref.exists():ref.replace(backup/'초기모델_개발예측.csv')
print('Repaired Series table conversion and replaced stored historical comparisons with raw-data model fits.')
