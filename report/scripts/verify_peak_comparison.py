"""Independent membership audit and diagnosis of differences between peak methods."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'report/tables/eda/peak_definition_comparison'
def read(name):return pd.read_csv(OUT/(name+'.csv'),encoding='utf-8-sig')

def posterior(values,components):
    x=np.asarray(values)[:,None]
    means=components['mean'].to_numpy()[None,:]
    std=components['std'].to_numpy()[None,:]
    logp=np.log(components.weight.to_numpy())[None,:]-np.log(std)-.5*((x-means)/std)**2
    probabilities=np.exp(logp-logp.max(axis=1,keepdims=True))
    probabilities/=probabilities.sum(axis=1,keepdims=True)
    return probabilities

def main():
    labels=read('assignments');summary=read('comparison');components=read('components')
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    train=labels.month.le(6)
    for row in summary.itertuples():
        actual=labels[row.method].to_numpy(dtype=bool)
        assert actual.sum()==row.all_n
        assert actual[train].sum()==row.train_n and actual[~train].sum()==row.check_n
        if row.operator=='>=':assert np.array_equal(actual,labels.P.ge(row.threshold))
        if row.operator=='>':assert np.array_equal(actual,labels.P.gt(row.threshold))
    for name in ['GMM','KMeans']:
        c=components.loc[components.scope.eq('Jan-Jun')&components.method.eq(name)].sort_values('rank')
        if name=='GMM':expected=posterior(labels.P,c).argmax(axis=1)==len(c)-1
        else:expected=np.abs(labels.P.to_numpy()[:,None]-c['mean'].to_numpy()[None,:]).argmin(axis=1)==len(c)-1
        assert np.array_equal(expected,labels[name].to_numpy())
    for name,key in [('DailyMax','date'),('MonthlyMax','month')]:
        assert np.array_equal(labels[name],labels.P.eq(labels.groupby(key).P.transform('max')))
    c=components.loc[components.scope.eq('Jan-Aug')&components.method.eq('GMM')].sort_values('rank')
    grid=np.arange(0,260.01,.01)
    prob=posterior(grid,c);high=prob.argmax(axis=1)==len(c)-1
    indices=np.flatnonzero(high)
    spans=[[float(grid[p[0]]),float(grid[p[-1]])] for p in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)]
    observed=posterior(labels.P,c).argmax(axis=1)==len(c)-1
    mx=float(labels.P.max())
    diagnoses=dict(full_period_gmm_high_intervals_on_grid_0_260=spans,
        observed_max=mx,observed_max_assigned_to_high=bool(observed[labels.P.eq(mx)].all()),
        observed_high_min=float(labels.loc[observed,'P'].min()),observed_high_max=float(labels.loc[observed,'P'].max()),
        clarification='Upper reversal occurs beyond observed maximum; it is extrapolation behavior, not a missed observed maximum.')
    b=read('weekly_resampling')
    stability=[]
    for method,part in b.groupby('method'):
        stability.append(dict(method=method,n=len(part),minimum=float(part.threshold.min()),
            median=float(part.threshold.median()),maximum=float(part.threshold.max()),
            median_jaccard=float(part.jaccard.median()),selected_k_counts={str(k):int(v) for k,v in part.k.value_counts().items()}))
    diagnoses['stability']=stability
    diagnoses['gmm_boundary_by_selected_k']=b.loc[b.method.eq('GMM')].groupby('k').agg(n=('threshold','size'),minimum=('threshold','min'),maximum=('threshold','max')).reset_index().to_dict('records')
    m=read('monthly_rates')
    diagnoses['month_with_highest_rate']={name:int(part.loc[part.rate.idxmax(),'month']) for name,part in m.groupby('method') if part.rate.max()>0}
    conditions=read('condition_rates')
    selected=conditions.loc[(conditions.variable.eq('기온')&conditions.band.eq(3))|
                            (conditions.variable.eq('습도')&conditions.band.eq(0))|
                            (conditions.variable.eq('생산량')&conditions.band.eq(2))]
    diagnoses['key_conditions']=selected.to_dict('records')
    diagnoses['status']='passed'
    diagnoses['audited_assignment_cells']=len(labels)*len(summary)
    (OUT/'verification.json').write_text(json.dumps(diagnoses,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in diagnoses.items() if k not in ['key_conditions']},ensure_ascii=False))

if __name__=='__main__':main()
