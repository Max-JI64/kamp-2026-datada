from pathlib import Path
import re, ast, json
root=Path(__file__).resolve().parents[1]
docs=[p for d in ['EDA','Analysis','Modeling','Report'] for p in (root/d).glob('*.md') if '원고' in p.name or '압축본' in p.name]
selected={}
for p in docs:
    text=p.read_text(encoding='utf-8')
    for link in re.findall(r'\]\(<?([^\)<>]+\.py)>?\)',text):
        q=(p.parent/link).resolve()
        if q.exists():selected.setdefault(q,[]).append(str(p.relative_to(root)))
allpy={p.stem:p for d in ['EDA','Analysis','Modeling'] for p in (root/d/'scripts').glob('*.py')}
pending=list(selected)
while pending:
    p=pending.pop();tree=ast.parse(p.read_text(encoding='utf-8-sig'))
    for n in ast.walk(tree):
        names=[]
        if isinstance(n,ast.Import):names=[a.name.split('.')[0] for a in n.names]
        if isinstance(n,ast.ImportFrom) and n.module:names=[n.module.split('.')[0]]
        for name in names:
            q=allpy.get(name)
            if q and q not in selected:selected[q]=['dependency:'+str(p.relative_to(root))];pending.append(q)
rows=[]
for p,refs in sorted(selected.items()):
    s=p.read_text(encoding='utf-8-sig');tree=ast.parse(s)
    rows.append({'path':str(p.relative_to(root)).replace('\\','/'),'references':refs,'lines':len(s.splitlines()),'imports':[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)],'doc':ast.get_docstring(tree),'functions':[n.name for n in tree.body if isinstance(n,ast.FunctionDef)],'tail':'\n'.join(s.splitlines()[-35:]),'paths':[line for line in s.splitlines() if ('ROOT' in line or 'Path(__file__)' in line or 'read_csv' in line or 'read_json' in line)][:30]})
(root/'tmp/submission_inspection.json').write_text(json.dumps({'docs':[str(p.relative_to(root)) for p in docs],'scripts':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print('Selected',len(rows),'scripts',sum(x['lines'] for x in rows),'lines')
for x in rows:print(x['path'],x['lines'],','.join(x['functions'][-5:]))
