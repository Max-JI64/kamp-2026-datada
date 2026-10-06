"""Check separate manuscript evidence links, SHAP aggregates and preservation."""
import json
import re
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Modeling/tables/feature_weather_audit'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    names=['Modeling/보완_변수기여와날씨선택근거_삽입용.md','Modeling/10.06_변수기여와날씨비교_분석기록.md']
    links=0
    for name in names:
        path=ROOT/name; text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith('http'): continue
            assert (path.parent/target.strip('<>')).exists(),(name,target)
            links+=1
        assert '4.14' not in text or '새로운 4.14절을 추가하는 구성이 아니다' in text
    summary=pd.read_csv(OUT/'shap_feature_summary.csv',encoding='utf-8-sig')
    checked=0
    for period,folds in [('Apr-Jun',[4,5,6]),('Jul-Aug',[7])]:
        for v in ['B0','B1']:
            values=[]
            for fold in folds:
                phi=pd.read_csv(OUT/f'shap_{fold:02d}_{v}.csv',encoding='utf-8-sig',parse_dates=['timestamp'])
                values.append(phi)
            f=pd.concat(values,ignore_index=True)
            s=summary.loc[summary.period.eq(period)&summary.variant.eq(v)&summary.condition.eq('all')]
            for row in s.itertuples():
                np.testing.assert_allclose(abs(f[row.feature]).mean(),row.mean_abs_shap,atol=1e-10)
                checked+=1
    c=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
    protected={n:sha(ROOT/n)==h for n,h in c['protected_manuscripts_sha256'].items()}
    assert all(protected.values()),protected
    manifest={x.relative_to(ROOT).as_posix():sha(x) for x in OUT.iterdir() if x.is_file() and x.name!='documents_verification.json'}
    manifest.update({n:sha(ROOT/n) for n in names})
    result={'status':'passed','local_links_checked':links,'shap_aggregates_independently_checked':checked,'protected_manuscripts_unchanged':protected,'artifacts_sha256':manifest}
    (OUT/'documents_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'}),flush=True)
if __name__=='__main__': main()
