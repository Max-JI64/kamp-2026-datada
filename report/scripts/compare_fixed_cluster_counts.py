"""Fixed-k follow-up: global scores alone do not choose a peak definition.

Compare k=3..8 for K-means and Gaussian mixture on the SAME Jan-Jun input.
Apply unchanged models to Jan-Aug. Refit fixed k on 40 calendar-week resamples.
Inspect upper-group separation, selected records, initialization effects and
the ability of the highest-mean component to include observed maxima.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from compare_peak_definitions import upper_group,boundary,exact_silhouette_1d,SEED,SHA,SOURCE

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'report/tables/eda/peak_definition_comparison'
OUT=BASE/'fixed_k'
OUT.mkdir(exist_ok=True)
KS=list(range(3,9));N_BOOT=40

def save(frame,name):frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')

def model(method,k,values,seed=SEED,n_init=10):
    x=np.asarray(values).reshape(-1,1)
    if method=='KMeans':return KMeans(n_clusters=k,n_init=n_init,random_state=seed).fit(x)
    m=GaussianMixture(n_components=k,n_init=n_init,random_state=seed,reg_covar=.01,max_iter=500).fit(x)
    assert m.converged_
    return m

def silhouette_by_cluster(values,model):
    u,w=np.unique(values,return_counts=True);labels=model.predict(u[:,None]);groups=np.unique(labels)
    D=np.abs(u[:,None]-u[None,:]);totals=np.array([w[labels==g].sum() for g in groups])
    sums=np.stack([D@(w*(labels==g)) for g in groups],axis=1)
    own=np.searchsorted(groups,labels)
    a=sums[np.arange(len(u)),own]/np.maximum(totals[own]-1,1)
    means=sums/totals;means[np.arange(len(u)),own]=np.inf;b=means.min(axis=1)
    score=np.divide(b-a,np.maximum(a,b),out=np.zeros_like(a),where=np.maximum(a,b)>0)
    score[totals[own]<=1]=0
    return float(np.average(score,weights=w)),{int(g):float(np.average(score[labels==g],weights=w[labels==g])) for g in groups}

def model_info(method,k,m,values,all_values):
    centers=m.means_.ravel() if method=='GMM' else m.cluster_centers_.ravel()
    order=np.argsort(centers);top=int(order[-1]);neighbor=int(order[-2])
    high=upper_group(m,all_values);selected=all_values[high]
    grid=np.arange(0,222.001,.05);gmask=upper_group(m,grid)
    indices=np.flatnonzero(gmask)
    monotone=bool(len(indices) and gmask[-1] and np.diff(indices).max(initial=1)==1)
    score,by_cluster=silhouette_by_cluster(values,m)
    lower,_,spans=boundary(m)
    info=dict(method=method,k=k,silhouette=score,highest_group_silhouette=by_cluster.get(top,np.nan),
              boundary=lower,min_selected=float(selected.min()),all_n=int(high.sum()),all_rate=float(high.mean()),
              median_selected=float(np.median(selected)),mean_selected=float(np.mean(selected)),
              mean_highest=float(centers[top]),mean_neighbor=float(centers[neighbor]),
              observed_monotone=monotone,includes_observed_max=bool(high[all_values==all_values.max()].all()),
              bic=float(m.bic(values[:,None])) if method=='GMM' else np.nan,
              overlap_high_neighbor=np.nan,mean_posterior_of_selected=np.nan)
    parameters=[]
    for rank,c in enumerate(order,1):
        parameters.append(dict(method=method,k=k,rank=rank,component=int(c),mean=float(centers[c]),
             std=float(np.sqrt(m.covariances_[c,0,0])) if method=='GMM' else np.nan,
             weight=float(m.weights_[c]) if method=='GMM' else np.nan))
    if method=='GMM':
        x=np.linspace(-100,400,10001)
        def pdf(c):
            sd=np.sqrt(m.covariances_[c,0,0]);return np.exp(-.5*((x-centers[c])/sd)**2)/(sd*np.sqrt(2*np.pi))
        # Normalized pairwise density overlap; no arbitrary ambiguity cutoff.
        info['overlap_high_neighbor']=float(np.trapezoid(np.minimum(pdf(top),pdf(neighbor)),x))
        info['mean_posterior_of_selected']=float(m.predict_proba(selected[:,None])[:,top].mean())
    return info,parameters

def run():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    d=pd.read_csv(BASE/'assignments.csv',encoding='utf-8-sig')
    train=d.month.le(6);values=d.loc[train,'P'].to_numpy();all_values=d.P.to_numpy()
    assert len(values)==4344 and len(d)==5784
    previous=pd.read_csv(BASE/'k_sensitivity.csv',encoding='utf-8-sig')
    rows=[];parameters=[];masks={};bank={};grids=[]
    grid=np.arange(0,230.001,.1)
    for method in ['KMeans','GMM']:
        for k in KS:
            m=model(method,k,values);bank[(method,k)]=m
            info,p=model_info(method,k,m,values,all_values);rows.append(info);parameters+=p
            mask=upper_group(m,all_values);masks[(method,k)]=mask
            old=previous.loc[previous.method.eq(method)&previous.k.eq(k)].iloc[0]
            assert int(mask.sum())==int(old.all_n)
            for value,selected in zip(grid,upper_group(m,grid)):
                row=dict(method=method,k=k,power=value,high=bool(selected))
                if method=='GMM':
                    centers=m.means_.ravel();top=int(centers.argmax())
                    row['high_probability']=float(m.predict_proba([[value]])[0,top])
                grids.append(row)
    summary=pd.DataFrame(rows)
    best_s=summary.loc[summary.method.eq('KMeans'),'silhouette'].max()
    best_bic=summary.loc[summary.method.eq('GMM'),'bic'].min()
    summary['silhouette_loss_from_k3']=best_s-summary.silhouette
    summary['bic_difference']=summary.bic-best_bic
    save(summary,'comparison');save(pd.DataFrame(parameters),'components');save(pd.DataFrame(grids),'decision_grid')
    # Adjacent k: quantify WHAT gets split, not only the score.
    transitions=[]
    for method in ['KMeans','GMM']:
        for k in KS[:-1]:
            before=masks[(method,k)];after=masks[(method,k+1)]
            removed=before&~after;added=after&~before
            transitions.append(dict(method=method,k_from=k,k_to=k+1,removed=int(removed.sum()),added=int(added.sum()),
                removed_min=float(all_values[removed].min()) if removed.any() else np.nan,
                removed_max=float(all_values[removed].max()) if removed.any() else np.nan,
                added_min=float(all_values[added].min()) if added.any() else np.nan,
                added_max=float(all_values[added].max()) if added.any() else np.nan))
    save(pd.DataFrame(transitions),'adjacent_k_changes')
    # Per-period and monthly memberships remain under each unchanged fixed k.
    assignment=d[['timestamp','date','month','P']].copy();monthly=[]
    for (method,k),mask in masks.items():
        assignment[f'{method}_{k}']=mask
        for month,p in d.groupby('month'):
            monthly.append(dict(method=method,k=k,month=int(month),n=len(p),selected=int(mask[p.index].sum()),rate=float(mask[p.index].mean())))
    save(assignment,'assignments');save(pd.DataFrame(monthly),'monthly_rates')
    tr=d.loc[train].copy();tr['week']=pd.to_datetime(tr.date).dt.to_period('W-SUN')
    blocks=[part.P.to_numpy() for _,part in tr.groupby('week')];rng=np.random.default_rng(SEED)
    repeats=[]
    for iteration in range(N_BOOT):
        boot=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        for method in ['KMeans','GMM']:
            for k in KS:
                m=model(method,k,boot)
                b,_,spans=boundary(m);mask=upper_group(m,all_values);old=masks[(method,k)]
                union=int((mask|old).sum())
                observed_grid=np.arange(0,222.001,.05);gm=upper_group(m,observed_grid);idx=np.flatnonzero(gm)
                mono=bool(len(idx) and gm[-1] and np.diff(idx).max(initial=1)==1)
                repeats.append(dict(method=method,k=k,iteration=iteration,boundary=b,
                    min_selected=float(all_values[mask].min()),all_n=int(mask.sum()),
                    jaccard=float((mask&old).sum()/union),observed_monotone=mono,
                    includes_observed_max=bool(mask[all_values==all_values.max()].all())))
        if (iteration+1)%5==0:print(json.dumps(dict(stage='fixed_k_week_resampling',completed=iteration+1,total=N_BOOT)),flush=True)
    repeats=pd.DataFrame(repeats);save(repeats,'weekly_resampling')
    stability=repeats.groupby(['method','k']).agg(boundary_min=('boundary','min'),boundary_median=('boundary','median'),
        boundary_max=('boundary','max'),boundary_p10=('boundary',lambda x:x.quantile(.1)),boundary_p90=('boundary',lambda x:x.quantile(.9)),
        median_jaccard=('jaccard','median'),minimum_jaccard=('jaccard','min'),
        monotone_runs=('observed_monotone','sum'),max_included_runs=('includes_observed_max','sum'),runs=('iteration','size')).reset_index()
    save(stability,'stability')
    # Check sensitivity to initialization and to repetition of identical daily profiles.
    seed_rows=[];scope_rows=[];raw=pd.read_csv(SOURCE,encoding='utf-8-sig')
    raw=raw.loc[raw['날짜'].between(20210101,20210630)&raw['시간'].between(0,23)].sort_values(['날짜','시간'])
    seen=set();unique=[]
    for date,p in raw.groupby('날짜'):
        profile=tuple(p[['15분','30분','45분','60분']].to_numpy().ravel())
        if profile not in seen:
            seen.add(profile);unique.extend(p[['15분','30분','45분','60분']].max(axis=1).to_numpy())
    for method in ['KMeans','GMM']:
        for k in KS:
            for seed in range(5):
                m=model(method,k,values,seed=seed)
                b,_,spans=boundary(m);mask=upper_group(m,all_values)
                seed_rows.append(dict(method=method,k=k,seed=seed,boundary=b,all_n=int(mask.sum())))
            for scope,x in [('Jan-Aug',all_values),('unique_Jan-Jun_daily_profiles',np.array(unique))]:
                m=model(method,k,x);b,_,spans=boundary(m);mask=upper_group(m,all_values)
                scope_rows.append(dict(method=method,k=k,scope=scope,fit_n=len(x),boundary=b,all_n=int(mask.sum())))
    save(pd.DataFrame(seed_rows),'initialization');save(pd.DataFrame(scope_rows),'scope_sensitivity')
    manifest=dict(status='passed',source_sha256=SHA,fit_n=4344,application_n=5784,
        k_candidates=KS,bootstrap_runs_per_fixed_k=N_BOOT,bootstrap_model_fits=N_BOOT*2*len(KS),
        purpose='Assess more clusters even with lower global score: highest group separation, subdivision, stability, and task relevance.',
        separation_metric='Exact row-weighted silhouette of the highest assigned group; for GMM also normalized top-neighbor Gaussian density overlap.',
        grid_boundary_resolution=.05,highest_group='largest centroid or component mean; all ties and original observations preserved',
        silhouette_is_not_peak_quality=True,actual_bill_threshold='unidentified',manuscript_edited=False,
        stability_caveat='Calendar-week resampling diagnostics, not confidence intervals; fixed k still an analyst choice',
        independent_period_caveat='Jul-Aug already explored; not an untouched test')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_json(orient='records'),flush=True)
    print(stability.to_json(orient='records'),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
