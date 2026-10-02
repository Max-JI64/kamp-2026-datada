"""Check saved manuscript links, numerics and hashes; never assert new visual review."""
from pathlib import Path
import re,json,hashlib
from urllib.parse import unquote
import pandas as pd
root=Path.cwd()
report=root/'report'
review=json.loads((report/'tables/eda/restart_peak_visual_review.json').read_text(encoding='utf-8'))
text=(report/'10.02_002_EDA_새원고.md').read_text(encoding='utf-8')
targets=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
for target in targets:
    target=unquote(target)
    path=Path(target) if re.match(r'^[A-Za-z]:/',target) else report/target
    assert path.is_file(),str(path)
assert len(re.findall(r'!\[',text))==6
manifest=json.loads((report/'tables/eda/restart_peak_manifest.json').read_text(encoding='utf-8'))
assert manifest['status']=='passed' and manifest['checked_rows']==559
for item in review['source_figures']:
    assert hashlib.sha256((report/'figures/eda'/item['file']).read_bytes()).hexdigest()==item['sha256']
    assert next(x for x in manifest['figures'] if x['file']==item['file'])['sha256']==item['sha256']
monthly=pd.read_csv(report/'tables/eda/restart_peak_month_sensitivity.csv',encoding='utf-8-sig')
assert all(monthly.loc[part.high_rate.idxmax(),'month']==7 for _,part in monthly.groupby('threshold'))
rates=pd.read_csv(report/'tables/eda/restart_peak_month_hour.csv',encoding='utf-8-sig')
assert [int(rates.loc[rates.month.eq(m),'high_rate'].ge(.2).sum()) for m in [6,7,8]]==[2,8,6]
audit=dict(status='passed',document_links=len(targets),document_figures=6,reviewed_figures=3,checked_rows=559)
(report/'tables/eda/restart_peak_document_verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit))