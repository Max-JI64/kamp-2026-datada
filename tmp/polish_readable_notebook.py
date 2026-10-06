from pathlib import Path
import ast,json,re
import nbformat
root=Path(__file__).resolve().parents[1]
path=root/'제출/통합_분석.ipynb';nb=nbformat.read(path,as_version=4)
for c in nb.cells:
    if c.cell_type!='code':continue
    s=c.source
    if s.startswith('from pathlib import Path'):
        s=s.replace('import matplotlib.pyplot as plt','import matplotlib.pyplot as plt\nimport matplotlib.dates as mdates')
        s=s.replace('cmap="coolwarm", annotate=True):','cmap="coolwarm", annotate=True, fmt=".1f"):')
        s=s.replace('f"{values[i,j]:.1f}"','f"{values[i,j]:{fmt}}"')
    if s.startswith('SELECTED_CONFIGS = '):
        tree=ast.parse(s);first=tree.body[0];configs=ast.literal_eval(first.value)
        configs=[{k:v for k,v in item.items() if k in ['split','model','parameters','members','weights']} for item in configs]
        rest='\n'.join(s.splitlines()[first.end_lineno:])
        s='SELECTED_CONFIGS = [\n'+',\n'.join('    '+repr(item) for item in configs)+'\n]\n'+rest
        s=s.replace('dev_predictions={"previous_hour":valid_dev.lag1_maximum.to_numpy()}','dev_predictions={"previous_hour":valid_dev.lag1_maximum.to_numpy(),"previous_day":valid_dev.lag24_maximum.to_numpy(),"previous_week":valid_dev.lag168_maximum.to_numpy()}')
        s=s.replace('fig,ax=plt.subplots(figsize=(10,4));development_summary.set_index("모델")','plot_models=development_summary.replace({"모델":{"HGB_initial":"HGB 기본","HGB":"HGB 튜닝","Ridge_initial":"Ridge","ElasticNet_initial":"Elastic Net","SVR_initial":"SVR","previous_hour":"직전값","previous_day":"전날값","previous_week":"전주값","mean5":"5개 평균","weighted_top3":"상위 3개 가중평균"}})\nfig,ax=plt.subplots(figsize=(13,4));plot_models.set_index("모델")')
        s=s.replace('ax.tick_params(axis="x",rotation=30)','ax.tick_params(axis="x",rotation=35,labelsize=8)')
    if s.startswith('grid=data.set_index'):
        s=s.replace('ax.set_title(f"{month}월");ax.set_ylim(0,225);ax.tick_params(axis="x",labelsize=8)','ax.set_title(f"{month}월");ax.set_ylim(0,225)\n    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7));ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"));ax.tick_params(axis="x",labelsize=8)')
    if s.startswith('variables=["생산량"]'):
        s=s.replace('variables,variables,-1,1)','variables,variables,-1,1,fmt=".2f")')
        s=s.replace('monthly.columns,monthly.index,-1,1)','monthly.columns,monthly.index,-1,1,fmt=".2f")')
    if s.startswith('def power_state'):
        s=s.replace('overlap.plot.bar(stacked=True','overlap.rename(columns={"positive_to_positive":"양수→양수","positive_to_zero":"양수→0","zero_to_positive":"0→양수","zero_to_zero":"0→0"}).plot.bar(stacked=True')
    if s.startswith('def weather_association'):
        s=s.replace('ax.axhline(0,color="gray",lw=.8);ax.set_ylabel("순위 관계")','ax.legend(title="");ax.set_xlabel("기상변수");ax.axhline(0,color="gray",lw=.8);ax.set_ylabel("순위 관계")')
    if s.startswith('def detection'):
        s=s.replace('regression_summary.pivot(index="조건",columns="방법",values="mae").plot.bar','regression_summary.pivot(index="조건",columns="방법",values="mae").rename(columns={"B0":"기본 B0","B1_all":"모든 시간 보완","G1_noage":"선택적 보완"}).plot.bar')
    c.source=s;c.outputs=[];c.execution_count=None
nbformat.write(nb,path)
# Keep docs precise about kernel selection and retain original facts.
readme=root/'제출/README.md';text=readme.read_text(encoding='utf-8')
text+='\n커널 설정 참고: [IPython 공식 문서](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).\n'
readme.write_text(text,encoding='utf-8')
print('Removed unused configuration fields and polished figure labels, correlation precision and date ticks.')