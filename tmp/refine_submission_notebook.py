from pathlib import Path
import hashlib, json, re, shutil
import nbformat
root=Path(__file__).resolve().parents[1];dest=root/'제출'
info=json.loads((root/'tmp/submission_inspection.json').read_text(encoding='utf-8'))
inventory=json.loads((dest/'선별_파일목록.json').read_text(encoding='utf-8'))
archive=root/'tmp/submission_validation_attempts';archive.mkdir(exist_ok=True)
for relative in ['Modeling/tables/training_design','Modeling/models/training_design']:
    p=(dest/relative).resolve()
    assert p.is_relative_to(dest.resolve()) and p.exists()
    target=archive/'unused_evidence'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists()
    shutil.move(str(p),str(target))
inventory['files']=[r for r in inventory['files'] if not r['path'].startswith(('Modeling/tables/training_design/','Modeling/models/training_design/'))]
relative='tmp/current_eda_input_audit.json';p=root/relative;q=dest/relative
q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
inventory['files'].append({'path':relative,'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size,'reason':['data manuscript input audit link']})
inventory['files']=sorted(inventory['files'],key=lambda r:r['path'])
(dest/'선별_파일목록.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
nb=nbformat.read(dest/'통합_분석.ipynb',as_version=4)
for row in info['scripts']:
    sentences=[]
    for relative in row['references']:
        if relative.startswith('dependency:'):continue
        p=root/relative
        for line in p.read_text(encoding='utf-8').splitlines():
            if Path(row['path']).name in line and len(re.findall('[가-힣]',line))>=8:
                clean=re.sub(r'\[([^\]]+)\]\(<?[^\)<>]+>?\)',r'\1',line).strip('-| ')
                clean=clean.replace('**관련 Python 코드:**','').replace('**','').strip()
                sentences.append(clean)
    if not sentences:continue
    explanation=max(sentences,key=lambda x:len(re.findall('[가-힣]',x)))
    for c in nb.cells:
        if c.cell_type=='markdown' and c.source.startswith('### `'+row['path']+'`'):
            a=c.source.index('코드 설명:');b=c.source.index('\n\n주요 함수:',a)
            c.source=c.source[:a]+'코드 설명: '+explanation+c.source[b:]
nbformat.write(nb,dest/'통합_분석.ipynb')
# Incomplete validation attempts are kept outside the deliverable, never deleted.
for p in (dest/'실행결과').iterdir():
    if p.is_dir():
        target=archive/p.name
        assert p.resolve().is_relative_to(dest.resolve()) and not target.exists()
        shutil.move(str(p),str(target))
print('Refined input files:',len(inventory['files']))
