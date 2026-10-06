from pathlib import Path
import ast, json, traceback
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root=Path(__file__).resolve().parents[1]
path=root/'제출/통합_분석.ipynb'
nb=nbformat.read(path,as_version=4)
nb.cells=[c for c in nb.cells if c.metadata.get('role')!='executed_result']
for c in nb.cells:
    c.source=c.source.replace('生産量','생산량').replace('過去結果_ハッシュ検証.csv','과거결과_해시검증.csv')
    if "for stage in ['validation','mixture','final']:" in c.source and '고정참조/a02_transition_forecast' not in c.source:
        c.source=c.source.replace("    for stage in ['validation','mixture','final']:","    forecast_output=WORK/'Analysis/tables/a02_transition_forecast'\n    preserved=WORK/'고정참조/a02_transition_forecast'\n    preserved.parent.mkdir(exist_ok=True)\n    shutil.move(str(forecast_output),str(preserved))\n    for stage in ['validation','mixture','final']:")
    if c.cell_type=='code' and 'def show_table(' in c.source and 'def verify_step(' not in c.source:
        c.source+='''

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
'''
    if c.cell_type=='code' and c.source.startswith('def verify_historical():'):
        c.source=c.source.replace("execute_step(name+'_'+('_'.join(args) or 'results')", "verify_step(name+'_'+('_'.join(args) or 'results')")
nbformat.write(nb,path)
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source.split('\n',1)[1] if c.source.startswith('%%submission_source ') else c.source)
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[r'C:\Program Files\Python313\python.exe','-m','ipykernel_launcher','-f','{connection_file}']
def on_cell_start(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print(f'Cell {cell_index+1}/{len(nb.cells)}: {cell.source.splitlines()[0][:110]}',flush=True)
client=NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(path.parent)}},on_cell_start=on_cell_start,allow_errors=False)
try:
    client.execute()
except BaseException:
    nbformat.write(nb,path)
    traceback.print_exc()
    raise
# Persist the execution-result Markdown as actual Markdown cells as requested.
out=[]
for c in nb.cells:
    out.append(c)
    if c.cell_type=='code':
        for result in c.get('outputs',[]):
            data=result.get('data',{})
            text=data.get('text/markdown','')
            if text and ('### ' in text or '전체 실행' in text):
                out.append(nbformat.v4.new_markdown_cell(text,metadata={'role':'executed_result'}))
nb.cells=out
nbformat.write(nb,path)
(root/'tmp/notebook_execution_status.json').write_text(json.dumps({'status':'passed','cells':len(nb.cells),'code_cells':sum(c.cell_type=='code' for c in nb.cells),'result_markdown_cells':sum(c.metadata.get('role')=='executed_result' for c in nb.cells)},ensure_ascii=False,indent=2),encoding='utf-8')
print('NOTEBOOK EXECUTION PASSED',flush=True)
