from pathlib import Path
import ast, json, pprint, shutil
import nbformat
import pandas as pd
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1]
path=root/'제출/통합_분석.ipynb'
nb=nbformat.read(path,as_version=4)
for c in nb.cells:
    if c.cell_type!='code':continue
    s=c.source
    if 'show_table(summary,"05_반복기록")' in s:
        s+='\nprocessed_data = data.copy()'
    if s.startswith('def build_pairs():'):
        start=s.index('    raw=raw_data.copy()');end=s.index('    assert len(data)',start)
        s=s[:start]+'    data=processed_data.copy()\n'+s[end:]
    if 'def build_model_frame(' in s:
        s=s.replace('valid = scoped.loc[scoped["시간"].between(*c["valid_hour"])].copy()','valid = processed_data.copy()')
        s=s.replace('model_frame=pd.read_csv(io.StringIO(model_frame.to_csv(index=False)),parse_dates=["timestamp"])','feature_csv=model_frame.to_csv(index=False)\ndevelopment_frame=pd.read_csv(io.StringIO(feature_csv),float_precision="round_trip",parse_dates=["timestamp","date"])\nmodel_frame=pd.read_csv(io.StringIO(feature_csv),parse_dates=["timestamp"])')
    if 'def build_state_features(' in s:
        s=s.replace('raw = raw_data.copy()','raw = processed_data.copy()')
    if 'common_frame=model_frame.loc[model_frame.eligible_common]' in s:
        s=s.replace('common_frame=model_frame.loc[model_frame.eligible_common]','common_frame=development_frame.loc[development_frame.eligible_common]')
        s += '\nnp.testing.assert_allclose(development_summary.set_index("모델").loc["HGB_initial",["전체 MAE","일별 최대 시간 MAE"]].to_numpy(float),[6.737973239313947,11.750641760429726],atol=1e-8)'
    if s.startswith('def detection(g):'):
        s=s.replace('det=detection(class_common);show_table(pd.DataFrame([det]),"20_공통범위분류성능")','''det=detection(class_common)
features_original=features.copy()
features_original.insert(features_original.index("prior_state_left_censored"),"prior_state_age")
original_clf=classification_model("logistic",MODEL_SETTINGS)
original_train=pd.read_csv(io.StringIO(ct.to_csv(index=False)))
original_clf.fit(original_train[features_original],original_train.sustained_onset.eq(1).astype(int))
original_score=original_clf.predict_proba(late[features_original])[:,1]
original_class=classification.assign(score=original_score,alarm=(original_score>=.2244084160368694).astype(int))
original_common=original_class.loc[original_class.timestamp.isin(test.timestamp)&original_class.persistence_known]
show_table(pd.DataFrame([{"방법":"경과시간 포함",**detection(original_common)},{"방법":"경과시간 제거",**det}]),"20_공통범위분류성능")
assert detection(original_common)["fp"]==38''')
    for prefix in ['FEATURE_CONTRACT = ','SELECTED_CONFIGS = ','MODEL_SETTINGS = ']:
        if s.startswith(prefix):
            first,rest=s.split('\n',1)
            s=prefix+pprint.pformat(ast.literal_eval(first[len(prefix):]),width=100,sort_dicts=False)+'\n'+rest
    c.source=s;c.execution_count=None;c.outputs=[]
idx=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## 4.2'))
ablation='''ablation_parts=[]
for month in [4,5,6,7]:
    start=pd.Timestamp(2021,month,1)
    train_ab=development_frame.loc[development_frame.eligible_common&development_frame.timestamp.lt(start)]
    valid_ab=development_frame.loc[development_frame.eligible_common&(development_frame.month.eq(month) if month<7 else development_frame.month.ge(7))]
    for variant,cols in [("A: 평균·최대",basic_features),("B: 마지막 값 추가",basic_features+["lag1_last"])]:
        reg_ab=HistGradientBoostingRegressor(max_leaf_nodes=15,max_iter=150,learning_rate=.05,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=42)
        reg_ab.fit(train_ab[cols],train_ab.target_maximum)
        part=valid_ab[["timestamp","date"]].copy();part["기간"]="개발 4~6월" if month<7 else "후반 7~8월"
        part["입력"]=variant;part["actual"]=valid_ab.target_maximum;part["prediction"]=reg_ab.predict(valid_ab[cols]);ablation_parts.append(part)
input_predictions=pd.concat(ablation_parts,ignore_index=True)
input_rows=[]
for (period,variant),g in input_predictions.groupby(["기간","입력"]):
    peaks=[metrics(day.loc[day.actual.eq(day.actual.max())])["mae"] for _,day in g.groupby("date") if len(day)==24]
    input_rows.append({"기간":period,"입력":variant,"전체 MAE":metrics(g)["mae"],"일별 최대 시간 MAE":np.mean(peaks),"평가 시간":len(g)})
input_summary=pd.DataFrame(input_rows);show_table(input_summary,"17b_마지막값추가효과")
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,(period,g) in zip(axes,input_summary.groupby("기간")):
    g.set_index("입력")[["전체 MAE","일별 최대 시간 MAE"]].plot.bar(ax=ax,color=COLORS[:2]);ax.set_title(period);ax.set_ylabel("MAE");ax.tick_params(axis="x",rotation=0)
show_figure("17b_입력추가효과")'''
nb.cells[idx:idx]=[nbformat.v4.new_markdown_cell('### 마지막 15분 값의 추가 효과\n\nAnalysis 3.4의 관측 관계가 실제 예측 이득으로 이어지는지 확인한다. 동일한 HGB 설정·동일 행으로 평균·최대 입력 A와 마지막 값 추가 B를 비교한다. 후반기는 6월 말 모델을 고정하며 두 기간의 오차를 구분한다.'),nbformat.v4.new_code_cell(ablation),nbformat.v4.new_markdown_cell('**결과 해석**\n\n마지막 값은 개발 기간의 오차를 줄이지만, 후반기 일별 최대 시간의 개선은 유지되지 않는다. 따라서 개발의 상관·평균오차 개선을 모든 높은 부하의 개선으로 확대하지 않는다. 다음 절에서는 전력이 크게 올라 유지되기 시작하는 시간에만 상태 정보를 보완하는 방법을 평가한다.')]
# Define explicit scope instead of including an entire separate example notebook.
backup=root/'tmp/submission_before_readable_rebuild'
example=root/'제출/ipynb 예시.ipynb'
if example.exists():shutil.copy2(example,backup/example.name);example.unlink()
readme=root/'제출/README.md';s=readme.read_text(encoding='utf-8')
s=s.replace('`ipynb 예시.ipynb`는 사용자 제공 구성 참고 자료다.','사용자 제공 예시 노트북은 제출 폴더 밖의 `tmp/submission_before_readable_rebuild/`에 보존했다.')
s=s.replace('정규 CPython 3.13의 기존 설치 환경에서 전체 실행을 검증한다.','정규 CPython 3.13의 기존 설치 환경에서 전체 실행을 검증했다.')
readme.write_text(s,encoding='utf-8')
nbformat.write(nb,path)
qa=root/'tmp/readable_visual_review';qa.mkdir(exist_ok=True)
# These previews are generated from already executed scientific plots.
files=sorted((root/'제출/output').glob('*.png'))
for start in range(0,len(files),4):
    items=files[start:start+4];thumbs=[]
    for file in items:
        im=Image.open(file).convert('RGB');im=im.resize((im.width//2,im.height//2))
        thumbs.append((file.name,im))
    width=max(im.width for _,im in thumbs);height=max(im.height for _,im in thumbs)+30
    sheet=Image.new('RGB',(width*2,height*2),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,im) in enumerate(thumbs):
        x=(i%2)*width;y=(i//2)*height;draw.text((x+5,y+5),name,fill='black');sheet.paste(im,(x,y+30))
    sheet.save(qa/f'sheet_{start//4+1}.png')
print('Final structure updated; common preprocessed data reused and historical CSV precision preserved.')