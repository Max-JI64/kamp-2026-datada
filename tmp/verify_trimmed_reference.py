from pathlib import Path
import pandas as pd,numpy as np
root=Path.cwd();package=root/'제출'
a=pd.read_csv(root/'Modeling/tables/regime_age_ablation/followup_classification.csv',encoding='utf-8-sig');a=a.loc[a.method.eq('noage')].set_index('timestamp')
b=pd.read_csv(package/'reference/고정_분류예측.csv',encoding='utf-8-sig').set_index('timestamp')
np.testing.assert_allclose(a.loc[b.index,'score'],b.score,atol=1e-12,rtol=0)
np.testing.assert_array_equal(a.loc[b.index,'alarm'],b.alarm)
print('Trimmed reference class predictions preserved: 1428 rows; alarm array unchanged.')