from pathlib import Path
import json,hashlib
import pandas as pd
root=Path(__file__).resolve().parents[3]
source=root/'data/origin/okm_augumented_2021.csv'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
raw=pd.read_csv(source,encoding='utf-8-sig')
valid=raw.loc[raw['날짜'].between(20210101,20210831)&raw['시간'].between(0,23)]
assert len(valid)==5784
p=valid[['15분','30분','45분','60분']].max(axis=1)
q1,q3=p.quantile([.25,.75])
iqr=q3-q1
upper=q3+1.5*iqr
result=dict(n=len(p),q1=q1,q3=q3,iqr=iqr,upper_fence=upper,maximum=float(p.max()),above_upper_fence=int(p.gt(upper).sum()))
out=root/'report/EDA/tables/restart_peak_iqr_check.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result))