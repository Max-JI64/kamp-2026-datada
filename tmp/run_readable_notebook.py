from pathlib import Path
import ast,json,sys,traceback,time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root=Path(__file__).resolve().parents[1]
path=root/'제출/통합_분석.ipynb'
nb=nbformat.read(path,as_version=4)
assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
for c in nb.cells:
    if c.cell_type=='code':
        ast.parse(c.source)
        assert '__name__' not in c.source and '%%' not in c.source and 'exec(' not in c.source
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
def progress(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print(f'CELL {cell_index+1}/{len(nb.cells)} {cell.source.splitlines()[0][:80]}',flush=True)
client=NotebookClient(nb,km=km,timeout=1800,allow_errors=False,
    resources={'metadata':{'path':str(path.parent)}},on_cell_start=progress)
start=time.monotonic()
try:client.execute()
except BaseException:
    nbformat.write(nb,path)
    traceback.print_exc()
    raise
nbformat.write(nb,path)
code_cells=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code_cells)
assert all(c.outputs for c in code_cells)
assert not list(path.parent.rglob('*.py'))
results={'status':'passed','cells':len(nb.cells),'code_cells':len(code_cells),
 'figures':sum('image/png' in o.get('data',{}) for c in code_cells for o in c.outputs),
 'executed_code_cells':len(code_cells),'elapsed_seconds':round(time.monotonic()-start,2),
 'runtime':sys.executable,'clean_virtualenv_tested':False}
(root/'tmp/readable_notebook_execution.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False),flush=True)
