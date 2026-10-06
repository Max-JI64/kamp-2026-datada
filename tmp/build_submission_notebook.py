"""Build the manuscript-scoped, portable submission notebook without editing originals."""
from pathlib import Path
import ast, hashlib, importlib.metadata, json, re, shutil, textwrap
import nbformat

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'제출'
INFO=json.loads((ROOT/'tmp/submission_inspection.json').read_text(encoding='utf-8'))
SCRIPTS={x['path']:x for x in INFO['scripts']}
DEST.mkdir(exist_ok=True)
if (DEST/'통합_분석.ipynb').exists(): shutil.copy2(DEST/'통합_분석.ipynb', ROOT/'tmp/submission_notebook_before_rebuild.ipynb')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={}
def retain(p,reason):
    if not p.is_file():return
    rel=p.relative_to(ROOT).as_posix()
    if rel.startswith(('제출/','백업/')) or (rel.startswith('tmp/') and rel!='tmp/current_eda_input_audit.json') or '__pycache__' in rel:return
    if p.suffix=='.py' and rel not in SCRIPTS:return
    files.setdefault(rel,set()).add(reason)
for rel in SCRIPTS:retain(ROOT/rel,'manuscript code or direct import dependency')
for rel in INFO['docs']:retain(ROOT/rel,'source manuscript')
retain(ROOT/'README.md','fixed reproduction root README hash')
retain(ROOT/'Modeling/requirements.txt','verified modeling environment')
retain(ROOT/'data/origin/okm_augumented_2021.csv','sole computation input')
# Keep the evidence paths actually referenced by manuscripts and selected source.
dirs=set()
for rel in list(SCRIPTS)+INFO['docs']:
    p=ROOT/rel;s=p.read_text(encoding='utf-8-sig')
    for link in re.findall(r'\]\(<?([^\)<>]+)>?\)',s):
        if link.startswith(('http:','https:','#')):continue
        q=(p.parent/link.split('#')[0]).resolve()
        if q.is_relative_to(ROOT):
            if q.is_dir():dirs.add(q)
            else:retain(q,'manuscript link:'+rel)
    for raw in re.findall(r'''["']((?:EDA|Analysis|Modeling|data)/[^"'\n]+)["']''',s):
        if '{' in raw:continue
        if raw in ['Modeling/tables','Modeling/models','Modeling/scripts']:continue
        q=ROOT/raw
        if q.is_dir():dirs.add(q)
        elif q.exists():retain(q,'source path:'+rel)
# The three M03 stages and named modeling families share directory-valued constants.
for name in ['m01','m02','m022','m02_rerun','m03/ab','m03/c','m03/diagnostics','m04','m04_rise','m04_rise_gate','m04_surge','m04_surge_risk','m04_surge_calibration','m05','redevelopment','profile_history','error_warning','priority_review','regime_diagnosis','regime_forecast','regime_integration','regime_routing','regime_followup','regime_age_ablation','modeling_close','modeling_replay']:
    for kind in ['tables','models']:
        q=ROOT/'Modeling'/kind/name
        if q.exists():dirs.add(q)
for name in ['EDA/tables','Analysis/tables','Modeling/config']:
    dirs.add(ROOT/name)
for directory in sorted(dirs):
    for p in directory.rglob('*'):
        if p.is_file() and p.suffix.lower() not in ['.pyc','.sqlite3','.db']:
            retain(p,'required evidence directory:'+directory.relative_to(ROOT).as_posix())
# Follow stored hash dependencies, including models and fixed configurations.
changed=True
while changed:
    before=len(files)
    for rel in list(files):
        p=ROOT/rel
        if p.suffix not in ['.json','.csv']:continue
        if p.suffix=='.json':
            try:obj=json.loads(p.read_text(encoding='utf-8-sig'))
            except (ValueError,UnicodeError):continue
            def walk(x):
                if isinstance(x,dict):
                    for k,v in x.items():
                        if isinstance(k,str):
                            for q in [ROOT/k,p.parent/k]:
                                if q.is_file() and q.resolve().is_relative_to(ROOT):retain(q,'hash dependency:'+rel)
                        walk(v)
                elif isinstance(x,list):
                    for v in x:walk(v)
                elif isinstance(x,str) and len(x)<250 and '/' in x:
                    q=ROOT/x
                    if q.is_file():retain(q,'stored path:'+rel)
            walk(obj)
        elif p.name=='model_manifest.csv':
            import csv
            for r in csv.DictReader(p.read_text(encoding='utf-8-sig').splitlines()):
                for key in ['file','path','model_file']:
                    if key in r:retain(ROOT/r[key],'model manifest:'+rel)
    changed=len(files)>before
for rel in files:
    q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,q)
manifest={'scope':'Detailed and compact current manuscripts, direct Python imports, fixed reference evidence. Unreferenced research scripts excluded.',
          'source_manuscripts':INFO['docs'],'code_count':len(SCRIPTS),'files':[{'path':rel,'sha256':sha(DEST/rel),'bytes':(DEST/rel).stat().st_size,'reason':sorted(reasons)} for rel,reasons in sorted(files.items())]}
(DEST/'선별_파일목록.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

SETUP=r'''
from pathlib import Path
from datetime import datetime
import contextlib, hashlib, importlib, importlib.metadata, io, json, os, shutil, sys, time, types
import numpy as np
import pandas as pd
from IPython.core.magic import register_cell_magic
from IPython.display import display, Markdown, Image
assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled(), "정규 CPython 3.13 커널을 선택하세요."
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLBACKEND", "Agg")
PACKAGE = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "선별_파일목록.json").is_file())
inventory = json.loads((PACKAGE / "선별_파일목록.json").read_text(encoding="utf-8"))
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(digest(PACKAGE/r['path']) == r['sha256'] for r in inventory['files']), "제출 입력이 변경되었습니다."
for stage in ["EDA", "Analysis", "Modeling"]:
    sys.path.insert(0, str(PACKAGE / stage / "scripts"))
RUN = PACKAGE / "실행결과" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
WORK = RUN / "재계산"
WORK.mkdir(parents=True)
for stage in ["data", "EDA", "Analysis", "Modeling"]:
    shutil.copytree(PACKAGE/stage, WORK/stage, ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(PACKAGE/'README.md',WORK/'README.md')
MODULES = {}
RECORDS = []
@register_cell_magic
def submission_source(line, cell):
    """원문을 그대로 표시하고 모듈로 등록한다. CLI main은 뒤의 실행 셀에서 호출한다."""
    relative = line.strip()
    path = PACKAGE / relative
    assert cell.rstrip() == path.read_text(encoding='utf-8-sig').rstrip(), relative
    name = path.stem
    module = sys.modules.get(name) or types.ModuleType(name)
    sys.modules[name] = module
    module.__file__ = str(path)
    module.__name__ = name
    exec(compile(cell, str(path), 'exec'), module.__dict__)
    MODULES[name] = module
    display(Markdown(f"코드 등록 완료: `{relative}`"))

@contextlib.contextmanager
def computation_paths():
    """모든 모듈의 경로를 같은 실행 사본으로 옮겨 원본·대조 결과를 보존한다."""
    previous=[]
    for m in list(sys.modules.values()):
        path=getattr(m,'__file__',None)
        if not path or not str(path).startswith(str(PACKAGE)): continue
        for k,v in list(vars(m).items()):
            if isinstance(v,Path) and v.is_absolute() and v.is_relative_to(PACKAGE):
                previous.append((m,k,v));setattr(m,k,WORK/v.relative_to(PACKAGE))
    try:yield
    finally:
        for m,k,v in previous:setattr(m,k,v)

def execute_step(name, callback):
    started=time.monotonic();stream=io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):callback()
    except BaseException:
        (RUN/(name+'.log')).write_text(stream.getvalue(),encoding='utf-8')
        raise
    (RUN/(name+'.log')).write_text(stream.getvalue(),encoding='utf-8')
    row={'step':name,'status':'passed','seconds':round(time.monotonic()-started,3)}
    RECORDS.append(row)
    display(Markdown(f"### 실행 결과: {name}\n\n정상 종료. 소요 시간 {row['seconds']:.1f}초. 상세 출력: `{(RUN/(name+'.log')).relative_to(PACKAGE).as_posix()}`."))

def cli(name,*args):
    old=sys.argv
    try:sys.argv=[str(PACKAGE/'Modeling/scripts'/f'{name}.py'),*args];MODULES[name].main()
    finally:sys.argv=old

def show_table(relative, limit=16):
    p=WORK/relative
    data=pd.read_csv(p,encoding='utf-8-sig')
    view=data.head(limit)
    display(Markdown(f"### 실행 결과표: `{relative}`\n\n전체 {len(data):,}행. 아래는 {len(view):,}행이며, 전체 표는 실행결과 폴더에 저장합니다.\n\n"+view.to_markdown(index=False)))

def verify_step(name, callback):
    before={p:p.stat().st_mtime_ns for p in (WORK/'Modeling/tables').rglob('*') if p.is_file()}
    execute_step(name,callback)
    for p in (WORK/'Modeling/tables').rglob('*'):
        if not p.is_file() or before.get(p)==p.stat().st_mtime_ns:continue
        relative=p.relative_to(WORK)
        q=RUN/'검증보고서'/name/relative
        q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        reference=PACKAGE/relative
        if reference.exists():shutil.copy2(reference,p)

display(Markdown(f"정규 Python {sys.version.split()[0]}. 선별 코드 {inventory['code_count']}개와 입력 {len(inventory['files'])}개 해시 확인. 이번 출력 폴더: `{RUN.relative_to(PACKAGE).as_posix()}`."))
'''

cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s,**meta):cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip(),metadata=meta))
md('''# 제조데이터 분석 통합 제출 노트북

데이터 확인 → EDA → Analysis → 모델링과 오류 분석 → 현재 선택 방법 재현 순서로 실행한다. `제출` 폴더 전체를 유지한 채 정규 CPython 3.13 커널에서 **Restart Kernel and Run All Cells**를 선택한다. 네 상세 원고와 압축본에 사용된 Python 코드 및 직접 import 의존 코드만 포함했다.

원문 코드는 아래 셀에 모두 들어 있다. `%%submission_source`는 셀을 원래 모듈로 등록하여 같은 이름의 함수·변수 충돌과 `__file__` 경로 문제를 처리한다. 실행은 뒤의 대목차별 셀에서 순서대로 수행한다. 문서 생성·보존 검사 코드는 원고 작성 근거로 포함했으며 실행 중 원고를 다시 편집하지 않는다.

데이터·EDA·Analysis는 실행 사본에서 계산한다. 과거 모델 탐색은 저장된 고정 결과의 해시·모델·지표를 검증해 재사용하며, 현재 선택 방법은 원자료에서 6회 재학습해 예측을 대조한다. 과거 Optuna 재탐색이나 이미 본 7 ~ 8월을 새 독립 시험으로 해석하지 않는다. 원고의 결과 설명과 이번 실행에서 계산한 결과를 구분해 표시한다.

목차: 0. 실행 환경 / 1. 데이터 확인 / 2. EDA / 3. Analysis / 4. 모델링 / 5. 실행 확인. 결과와 로그는 매번 새로운 `실행결과/실행시각/` 폴더에 저장한다.''')
md('## 0. 실행 환경과 코드 등록\n\n`requirements.txt`의 패키지를 준비한다. 입력 해시를 먼저 확인하고, 원고·참조 결과의 복사본에서 계산한다. 오류가 나면 해당 셀에서 중단하며 완료로 표시하지 않는다.')
code(SETUP)

# Every selected .py appears verbatim in an inspectable code cell.
for stage in ['EDA','Analysis','Modeling']:
    md(f'## {stage} 원고에 사용한 전체 코드\n\n아래 셀은 코드 정의를 등록한다. 계산 순서와 이번 실행 결과는 뒤의 1 ~ 5장에 정리한다.')
    for rel,row in SCRIPTS.items():
        if not rel.startswith(stage+'/'):continue
        refs='; '.join(row['references'])
        md(f"### `{rel}`\n\n원고 연결: {refs}.\n\n코드 설명: {row['doc'] or '연결된 원고의 계산·그림·검증 함수.'}\n\n주요 함수: "+', '.join('`'+x+'()`' for x in row['functions'])+'.')
        cells.append(nbformat.v4.new_code_cell('%%submission_source '+rel+'\n'+(ROOT/rel).read_text(encoding='utf-8-sig'),metadata={'source_file':rel,'role':'manuscript_source'}))

def manuscript(rel):
    txt=(ROOT/rel).read_text(encoding='utf-8')
    txt=re.sub(r'\]\(<?([^\)<>]+)>?\)',lambda m: ']('+((Path(rel).parent/m.group(1)).as_posix() if not m.group(1).startswith(('http:','https:','#')) else m.group(1))+')',txt)
    md('以下' if False else '원고에 기록된 결과와 해석을 아래에 보존했다. 다음 실행 셀의 결과는 이번 실행에서 계산하거나 검증한 값이다.\n\n'+txt)

manuscript('EDA/01_데이터_원고.md')
md('### 1.6 데이터 확인 코드 실행\n\n6168행의 원본에서 1 ~ 8월 정상 5784시간을 확인하고, 평균 반올림·공장인원 파생 관계·결측·시간 오류·반복 배열을 검증한다. 이어 M01의 정확한 시차와 학습·평가 대상 행을 재구성한다.')
code(r'''with computation_paths():
    execute_step('01_데이터확인',MODULES['data_overview'].main)
    execute_step('02_시차자료구성',MODULES['m01_prepare'].main)
    execute_step('03_시차독립검증',MODULES['m01_verify'].main)
facts=json.loads((WORK/'EDA/tables/data_overview_manifest.json').read_text(encoding='utf-8'))
display(Markdown('### 데이터 실행 결과\n\n'+pd.DataFrame([facts]).to_markdown(index=False)))''')

manuscript('EDA/02_EDA_원고.md')
md('### 2.6 EDA 코드 실행\n\n평일·주말 하루 패턴, 날짜별 반복·월별 평균, 생산·날씨 관계, 평일 전력 분포, 15분 구간의 중심화 패턴을 순서대로 계산한다. 원고의 관측 타임라인은 같은 정상 시간으로 다시 생성한다. 이미 검증한 보조 요약표는 함께 대조한다.')
code(r'''with computation_paths():
    for name in ['eda_daily_pattern','eda_daily_repetition','plot_daily_repetition_simple','plot_monthly_power_heatmap','verify_monthly_power_heatmap']:
        execute_step(name,MODULES[name].main)
    relation=MODULES['eda_variable_relations']
    execute_step('eda_variable_relations_diagnose',lambda:relation.diagnose(relation.source_frame()))
    execute_step('eda_variable_relations_figures',lambda:relation.figures(relation.source_frame()))
    for name in ['eda_weekday_power_levels','eda_slot_time_patterns','plot_report_daily_patterns','plot_observed_power_timeline']:
        execute_step(name,MODULES[name].main)
for relative in ['EDA/tables/daily_repetition/group_summary.csv','EDA/tables/daily_repetition/month_hour_mean.csv']:
    if (WORK/relative).exists():show_table(relative)''')

manuscript('Analysis/03_Analysis_원고.md')
md('### 3.5 Analysis 코드 실행\n\n生産' if False else '### 3.5 Analysis 코드 실행\n\n생산량 전환과 전력 변화, 과거 전력의 전환 구분, 조건을 고려한 날씨 관계, 마지막 15분 값의 추가 관계를 다시 계산한다. A02의 고정 분류·회귀 설정과 혼합 진단은 같은 시간순 조건으로 실행한다. 전력 자체의 구간 전환을 확인한 뒤 P01의 시간 안 최대 형태를 연결한다.')
code(r'''with computation_paths():
    for name in ['a01_production_changes','a01_relative_changes','a02_prior_power_signals','a02_signal_diagnostics']:
        execute_step(name,MODULES[name].main)
    forecast_output=WORK/'Analysis/tables/a02_transition_forecast'
    preserved=WORK/'고정참조/a02_transition_forecast'
    preserved.parent.mkdir(exist_ok=True)
    shutil.move(str(forecast_output),str(preserved))
    for stage in ['validation','mixture','final']:
        execute_step('a02_transition_forecast_'+stage,lambda stage=stage:cli('a02_transition_forecast',stage))
    for name in ['a02_transition_review','a03_conditioned_weather','a03_weather_diagnostics','a04_maximum_conditions','a04_maximum_diagnostics','a04_terminal_unique','a04_support_diagnostics','a06_power_state_transitions','plot_a01_production_changes','plot_a03_weather','plot_a04_terminal_slots','plot_analysis_transition_summary']:
        execute_step(name,MODULES[name].main)
    peak=MODULES['p01_peak_shapes']
    peak.OUT=WORK/'EDA/tables/p01_recalculated'
    execute_step('p01_peak_shapes',peak.main)
show_table('Analysis/tables/a02_transition_forecast/classification_test.csv')
show_table('Analysis/tables/a04_maximum_diagnostics/slot_group_associations.csv') if (WORK/'Analysis/tables/a04_maximum_diagnostics/slot_group_associations.csv').exists() else None''')

manuscript('Modeling/04_Modeling_원고.md')
md('### 4.14 고정된 과거 결과의 재사용과 검증\n\n과거 모델 비교·입력 추가·상승 결합·재개발·오류 검토의 결과 파일을 해시로 확인한다. 저장 모델의 예측과 MAE·일별 최대 오차를 원고 검증 코드로 다시 대조한다. 새로운 모델 선택이나 Optuna 재탐색은 수행하지 않는다. 아래 원문 코드에는 과거 탐색 구현도 모두 보존되어 있다.')
code(r'''def verify_historical():
    rows=[]
    for p in sorted((PACKAGE/'Modeling/tables').rglob('run.json')):
        if 'implementation_attempt' in str(p):continue
        r=json.loads(p.read_text(encoding='utf-8'))
        checks=0
        for name,h in r.get('outputs_sha256',{}).items():
            q=p.parent/name
            if not q.exists():q=PACKAGE/name
            assert q.exists() and digest(q)==h,(p,name)
            checks+=1
        rows.append({'experiment':p.parent.relative_to(PACKAGE/'Modeling/tables').as_posix(),'stored_status':r.get('status','recorded'),'output_hashes_checked':checks})
    pd.DataFrame(rows).to_csv(RUN/'과거결과_해시검증.csv',index=False,encoding='utf-8-sig')
    display(Markdown('### 이번 실행의 과거 결과 해시 검증\n\n'+pd.DataFrame(rows).to_markdown(index=False)))
execute_step('과거결과해시검증',verify_historical)
# 원고 검증 함수가 쓰는 보고서는 실행 사본에만 생성한다.
with computation_paths():
    # EDA/Analysis 재계산과 독립적으로 고정된 모델 입력을 복원한다.
    shutil.copytree(PACKAGE/'Modeling/tables',WORK/'Modeling/tables',dirs_exist_ok=True)
    for name,args in [('m01_verify',()),('m02_verify',()),('m02_rerun_verify',()),('m03_verify',('--stage','ab')),('m03_verify',('--stage','c')),('m04_verify',()),('m04_rise_verify',()),('m04_rise_gate_verify',()),('m04_surge_verify',()),('m04_surge_risk_verify',()),('m04_surge_calibration_verify',()),('m05_verify',())]:
        verify_step(name+'_'+('_'.join(args) or 'results'),lambda name=name,args=args:cli(name,*args))''')
md('### 4.15 현재 선택 방법을 원자료에서 재현\n\n과거 OOF 분류 3개와 6월 말 고정 분류기·B0·B1을 다시 학습한다. 경계 0.4554612321, 분류 1428개·회귀 4032개 예측을 참조 결과와 대조한다. 로짓 기여의 합과 예측 오차, 선택한 사례도 다시 계산한다.')
code(r'''current=MODULES['modeling_close_analysis']
original_out=current.OUT
current.OUT=RUN/'현재모델'
try:
    execute_step('현재모델정의고정',current.freeze)
    execute_step('현재모델원자료재학습대조',current.run)
finally:current.OUT=original_out
replay=json.loads((RUN/'현재모델/run.json').read_text(encoding='utf-8'))
display(Markdown('### 현재 모델 재현 결과\n\n'+pd.DataFrame([{k:replay[k] for k in ['status','raw_valid_hours','class_train','regression_train','classification_predictions','regression_predictions_checked','refits','oof_threshold','exact_logit_reconstruction']}]).to_markdown(index=False)))
prediction=pd.read_csv(RUN/'현재모델/reproduced_predictions.csv',encoding='utf-8-sig')
result=[]
for variant,g in prediction.groupby('variant'):
    for condition,gg in [('all',g),('up_start',g.loc[g.sustained_onset.eq(1)])]:
        result.append({'variant':variant,'condition':condition,'n':len(gg),'MAE':(gg.prediction-gg.actual).abs().mean(),'underprediction':(gg.actual-gg.prediction).clip(lower=0).mean()})
display(Markdown('### 이번 실행에서 계산한 회귀 오차\n\n'+pd.DataFrame(result).to_markdown(index=False)))
pd.DataFrame(result).to_csv(RUN/'현재모델/재계산_회귀오차.csv',index=False,encoding='utf-8-sig')
events=pd.read_csv(RUN/'현재모델/event_explanations.csv',encoding='utf-8-sig')
assert len(events)==13 and events.alarm.sum()==7
display(Markdown(f'### 이번 실행의 상승 탐지\n\n공통 지속 상승 {len(events)}건 중 {int(events.alarm.sum())}건을 탐지했다. 적중 중 값오차 개선은 {int(((events.alarm==1)&(events.absolute_error_gain>0)).sum())}건, 악화는 {int(((events.alarm==1)&(events.absolute_error_gain<0)).sum())}건이다.'))''')

md('## 5. 전체 실행 확인\n\n입력·참조 파일 보존과 모든 실행 단계의 정상 종료를 확인한다. 실행 단계·소요 시간·현재 모델 재현 결과를 JSON으로 저장한다. 이 기록은 현재 환경에서의 실행 검증이며 운영 배포·절감 실측을 뜻하지 않는다.')
code(r'''assert all(digest(PACKAGE/r['path'])==r['sha256'] for r in inventory['files']), '참조 자료 보존 실패'
summary={'status':'passed','source_code_count':inventory['code_count'],'input_files_checked':len(inventory['files']),'steps':RECORDS,'selected_model_reproduction':replay,'historical_optuna_search_rerun':False,'outputs':RUN.relative_to(PACKAGE).as_posix()}
(RUN/'실행검증.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
display(Markdown('### 전체 실행 결과\n\n'+pd.DataFrame(RECORDS).to_markdown(index=False)+f"\n\n모든 {len(RECORDS)}단계가 정상 종료했다. 입력 {len(inventory['files'])}개의 해시를 다시 확인했다. 실행 기록: `{summary['outputs']}/실행검증.json`."))''')
notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3.13 (regular CPython)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13'},'submission_scope':manifest['scope']})
nbformat.write(notebook,DEST/'통합_분석.ipynb')
pkgs=['numpy','pandas','scipy','scikit-learn','joblib','optuna','xgboost','lightgbm','catboost','matplotlib','Pillow','nbformat','nbclient','ipykernel','tabulate']
(DEST/'requirements.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in pkgs)+'\n',encoding='utf-8')
readme=f'''# 통합 분석 제출 파일

`통합_분석.ipynb`를 정규 CPython 3.13 커널로 열어 Restart Kernel and Run All Cells를 실행한다. 노트북만 이동하지 말고 이 폴더 전체를 유지한다. `requirements.txt`는 이번 실행 환경의 직접 패키지 버전이다.

상세·압축 원고의 코드 {len(SCRIPTS)}개를 원래 폴더 구조로 모았으며, 노트북에 전체 소스를 코드 셀로 넣었다. `선별_파일목록.json`은 각 파일의 선별 근거와 SHA-256을 기록한다. 미사용 탐색 코드와 백업 폴더를 일괄 복사하지 않았다.

노트북은 데이터·EDA·Analysis를 실행 사본에서 재계산하고 과거 모델 비교의 고정 결과를 검증한다. 현재 선택한 분류·회귀 방법은 원자료에서 6회 재학습하여 기존 예측과 대조한다. 과거 Optuna 탐색을 전부 다시 돌리는 작업은 기본 실행에 포함하지 않는다. 문서 편집·보존 목록 생성 코드는 소스를 보존하며 실행 중 원고를 수정하지 않는다.

매 실행의 표·그림·로그·검증 JSON은 `실행결과/실행시각/`에 저장한다. 입력과 고정 참조 결과는 실행 전후 해시로 검사한다. `통합_분석.ipynb`에는 원고의 대목차·소목차·코드 설명과 기록된 결과, 이번 실행에서 확인한 결과 마크다운 출력이 포함된다.
'''
(DEST/'README_실행안내.md').write_text(readme,encoding='utf-8')
print(json.dumps({'scripts':len(SCRIPTS),'input_files':len(files),'bytes':sum(x['bytes'] for x in manifest['files']),'cells':len(cells),'notebook':str(DEST/'통합_분석.ipynb')},ensure_ascii=False))
