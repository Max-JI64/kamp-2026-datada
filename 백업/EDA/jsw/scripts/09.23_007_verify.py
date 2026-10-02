"""Validate saved 007 outputs and key joins without recomputing the analysis."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd
import numpy as np

BASE=Path(__file__).parent.parent
OUT=BASE/'tables'
P='09.23_007'
source=BASE/'../../data/origin/okm_augumented_2021.csv'
facts=json.loads((OUT/f'{P}_relations_facts.json').read_text(encoding='utf-8'))
assert hashlib.sha256(source.read_bytes()).hexdigest()==facts['source_sha256']
for path in (BASE/'scripts').glob(f'{P}_*.py'):
    compile(path.read_text(encoding='utf-8'),str(path),'exec')
def read(name):
    return pd.read_csv(OUT/f'{P}_{name}.csv',encoding='utf-8-sig')
raw=pd.read_csv(source,encoding='utf-8-sig')
df=raw.loc[raw['날짜'].lt(20210901)&raw['시간'].between(0,23)].copy()
df.index=pd.to_datetime(df['날짜'].astype(str),format='%Y%m%d')+pd.to_timedelta(df['시간'],unit='h')
assert df.index.is_unique and len(df)==5784
lags=read('lag_correlations')
for h,n in [(1,5781),(24,5712),(168,5568)]:
    pairs=[(df.at[t-pd.Timedelta(hours=h),'평균'],df.at[t,'평균'])
           for t in df.index if t-pd.Timedelta(hours=h) in df.index]
    assert len(pairs)==n
    actual=np.corrcoef(np.array(pairs).T)[0,1]
    saved=lags.loc[lags.scale.eq('raw')&lags.variable.eq('평균')&lags.lag_hours.eq(h)]
    assert len(saved)==1 and saved.n.iloc[0]==n and np.isclose(actual,saved.pearson.iloc[0])
profiles=read('profile_overlap')
assert profiles.days.sum()==241
assert profiles.train_days.sum()==181 and profiles.validation_days.sum()==60
assert not (profiles.train_days.gt(0)&profiles.validation_days.gt(0)).any()
for method in ['pearson','spearman']:
    matrix=read(method+'_matrix').set_index('variable')
    assert matrix.shape==(13,13) and np.allclose(matrix,matrix.T,equal_nan=True)
    assert np.allclose(np.diag(matrix),1)
counts=read('pair_counts').set_index('variable')
assert np.array_equal(counts.to_numpy(),df[counts.columns].notna().astype(int).T @ df[counts.columns].notna().astype(int))
for name in ['weather_bins','weather_coarse_bins']:
    tab=read(name)
    for (population,var),block in tab.groupby(['population','variable']):
        expected=df.loc[df['생산량'].gt(0),var].count() if population=='positive' else df[var].count()
        assert block.n.sum()==expected
assert read('august_state_weeks').n.sum()==744
assert (read('conditional_correlations').n+read('conditional_correlations').dropped_sparse_n==read('conditional_correlations').eligible_n).all()
report=BASE/'reports'/f'{P}_connected_eda.md'
for href in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
    assert (report.parent/href).exists(), href
assert len(list(OUT.glob(f'{P}_*.csv')))==19
print('PASS: source hash; 3 scripts compile; exact lag joins; profile split counts; matrices and pair counts; bin and August totals; report links; 19 CSV outputs.')
