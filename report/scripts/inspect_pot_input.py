from pathlib import Path
import json
import pandas as pd
import numpy as np
import scipy
from scipy.stats import genpareto
ROOT=Path(__file__).resolve().parents[2]
d=pd.read_csv(ROOT/'report/tables/eda/peak_definition_comparison/assignments.csv',encoding='utf-8-sig')
x=d.loc[d.month.le(6),'P'].to_numpy()
rows=[]
for u in [145,160,170,176,182,187,190,194,200,205]:
    y=x[x>u]-u
    shape,loc,scale=genpareto.fit(y,floc=0)
    rows.append(dict(u=u,n=len(y),unique=len(np.unique(y)),mean_excess=float(y.mean()),shape=shape,
                     scale=scale,modified_scale=scale-shape*u,
                     endpoint=u-scale/shape if shape<0 else None))
print(json.dumps(dict(scipy=scipy.__version__,numpy=np.__version__,train_n=len(x),candidates=rows),ensure_ascii=False))
