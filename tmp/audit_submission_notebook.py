from pathlib import Path
import ast, collections, json, re
import nbformat
root=Path(__file__).resolve().parents[1];p=root/'제출'
n=nbformat.read(p/'통합_분석.ipynb',as_version=4)
inventory=json.loads((p/'선별_파일목록.json').read_text(encoding='utf-8'))
groups=collections.Counter()
for r in inventory['files']:
    parts=r['path'].split('/')
    groups['/'.join(parts[:3])]+=r['bytes']
print('Largest evidence groups:',groups.most_common(12))
sources=[c.metadata['source_file'] for c in n.cells if c.metadata.get('role')=='manuscript_source']
assert len(sources)==len(set(sources))==97
for c in n.cells:
    if c.cell_type=='code':ast.parse(c.source.split('\n',1)[1] if c.source.startswith('%%submission_source ') else c.source)
missing=[]
for c in n.cells:
    if c.cell_type!='markdown':continue
    for target in re.findall(r'\]\(<?([^\)<>]+)>?\)',c.source):
        if target.startswith(('http:','https:','#','app:')):continue
        target=target.split('#')[0]
        if not (p/target).exists():missing.append(target)
print('Missing markdown links:',sorted(set(missing)))
print('Code cells:',len([c for c in n.cells if c.cell_type=='code']))
print('Error outputs:',[(i+1,o.get('ename')) for i,c in enumerate(n.cells) for o in c.get('outputs',[]) if o.output_type=='error'])
nbformat.validate(n)
assert not missing
assert not [(i,o) for i,c in enumerate(n.cells) for o in c.get('outputs',[]) if o.output_type=='error']
assert all(c.execution_count is not None for c in n.cells if c.cell_type=='code')
run=max((p/'실행결과').iterdir(),key=lambda x:x.name)
summary=json.loads((run/'실행검증.json').read_text(encoding='utf-8'))
assert summary['status']=='passed' and summary['raw_preprocessing']['status']=='passed' and summary['preprocessed_reads']>0
assert summary['raw_preprocessing']['valid_hours']==5784
assert summary['selected_model_reproduction']['classification_predictions']==1428
assert summary['selected_model_reproduction']['regression_predictions_checked']==4032
assert summary['selected_model_reproduction']['refits']==6
import hashlib
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(root/'data/origin/okm_augumented_2021.csv')==digest(p/'data/origin/okm_augumented_2021.csv')
assert all(digest(p/r['path'])==r['sha256'] for r in inventory['files'])
source_paths={r['path'] for r in inventory['files'] if r['path'].endswith('.py')}
assert source_paths==set(sources)
result={'status':'passed','raw_matches_original':True,'raw_rows':6168,'scoped_rows':5832,'invalid_hours':48,'valid_hours':5784,'raw_preprocessed_input_reads':summary['preprocessed_reads'],
        'manuscript_source_files':len(sources),'code_cells':sum(c.cell_type=='code' for c in n.cells),'markdown_cells':sum(c.cell_type=='markdown' for c in n.cells),
        'executed_result_markdown_cells':sum(c.metadata.get('role')=='executed_result' for c in n.cells),'execution_steps':len(summary['steps']),
        'classification_predictions_checked':1428,'regression_predictions_checked':4032,'selected_model_refits':6,
        'notebook_sha256':digest(p/'통합_분석.ipynb'),'input_manifest_sha256':digest(p/'선별_파일목록.json'),'run':run.relative_to(p).as_posix(),
        'historical_optuna_search_rerun':False,'limitations':'Historical fixed outputs and models are reused and verified; selected current models are refitted. No deployment or new independent holdout.'}
(p/'검증_요약.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
