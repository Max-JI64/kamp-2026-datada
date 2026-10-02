"""Compare peak definitions without changing manuscripts or operational labels.

Primary fit: Jan-Jun valid hours. Fixed application: Jul-Aug (already explored).
Full-period and unique daily-profile fits are descriptive sensitivity analyses.
No electricity bill is reconstructed from the seasonal tariff field.
"""
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'report/tables/eda/peak_definition_comparison'
OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'data/origin/okm_augumented_2021.csv'
SHA='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
SLOTS=['15분','30분','45분','60분']
SEED=20261002
N_BOOT=40
GRID=np.arange(0.,230.001,.05)

def save(frame,name):
    frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')

def exact_silhouette_1d(values,labels,weights):
    distances=np.abs(values[:,None]-values[None,:])
    groups=np.unique(labels)
    totals=np.array([weights[labels==g].sum() for g in groups])
    sums=np.stack([distances@(weights*(labels==g)) for g in groups],axis=1)
    own=np.searchsorted(groups,labels)
    a=sums[np.arange(len(values)),own]/np.maximum(totals[own]-1,1)
    means=sums/totals
    means[np.arange(len(values)),own]=np.inf
    b=means.min(axis=1)
    scores=np.divide(b-a,np.maximum(a,b),out=np.zeros_like(a),where=np.maximum(a,b)>0)
    scores[totals[own]<=1]=0
    return float(np.average(scores,weights=weights))

def models(values,n_init=10):
    x=np.asarray(values,dtype=float).reshape(-1,1)
    u,w=np.unique(x[:,0],return_counts=True)
    rows=[]; gmms={}; kms={}
    for k in range(1,9):
        g=GaussianMixture(n_components=k,n_init=n_init,random_state=SEED,
                         reg_covar=.01,max_iter=500).fit(x)
        assert g.converged_
        gmms[k]=g
        rows.append(dict(method='GMM',k=k,bic=g.bic(x),silhouette=np.nan))
        if k>=2:
            m=KMeans(n_clusters=k,n_init=max(n_init,10),random_state=SEED).fit(x)
            kms[k]=m
            score=exact_silhouette_1d(u,m.predict(u[:,None]),w)
            rows.append(dict(method='KMeans',k=k,bic=np.nan,silhouette=score))
    table=pd.DataFrame(rows)
    gk=int(table.loc[table.method.eq('GMM')].sort_values('bic').iloc[0].k)
    kk=int(table.loc[table.method.eq('KMeans')].sort_values('silhouette',ascending=False).iloc[0].k)
    return gmms[gk],kms[kk],table,gmms,kms

def upper_group(model,values):
    centers=model.means_.ravel() if hasattr(model,'means_') else model.cluster_centers_.ravel()
    return model.predict(np.asarray(values).reshape(-1,1))==int(np.argmax(centers))

def boundary(model):
    mask=upper_group(model,GRID)
    indices=np.flatnonzero(mask)
    if len(indices)==0:
        return np.nan,False,[]
    spans=[]
    for block in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
        spans.append([float(GRID[block[0]]),float(GRID[block[-1]])])
    monotone=len(spans)==1 and bool(mask[-1])
    return float(GRID[indices[0]]),monotone,spans

def model_parameters(model,method,scope):
    centers=model.means_.ravel() if method=='GMM' else model.cluster_centers_.ravel()
    rows=[]
    for rank,idx in enumerate(np.argsort(centers),1):
        rows.append(dict(scope=scope,method=method,rank=rank,component=int(idx),mean=centers[idx],
                         std=float(np.sqrt(model.covariances_[idx,0,0])) if method=='GMM' else np.nan,
                         weight=float(model.weights_[idx]) if method=='GMM' else np.nan))
    return rows

def run():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    data=pd.read_csv(SOURCE,encoding='utf-8-sig')
    d=data.loc[data['날짜'].between(20210101,20210831)&data['시간'].between(0,23)].copy()
    d['date']=pd.to_datetime(d['날짜'].astype(str),format='%Y%m%d')
    d=d.sort_values(['date','시간']).reset_index(drop=True)
    d['P']=d[SLOTS].max(axis=1).astype(float)
    d['month']=d.date.dt.month
    d['timestamp']=d.date+pd.to_timedelta(d['시간'],unit='h')
    train=d.month.le(6)
    assert len(d)==5784 and train.sum()==4344 and (~train).sum()==1440
    assert d.groupby('date').size().eq(24).all()
    gm,km,selection,gmms,kms=models(d.loc[train,'P'])
    save(selection,'model_selection')
    # Independent exact silhouette check on a small full row-level distance matrix.
    small=d.loc[train,'P'].iloc[:480].to_numpy()
    u,w=np.unique(small,return_counts=True)
    assert np.isclose(exact_silhouette_1d(u,km.predict(u[:,None]),w),
                      silhouette_score(small[:,None],km.predict(small[:,None])))
    parameters=model_parameters(gm,'GMM','Jan-Jun')+model_parameters(km,'KMeans','Jan-Jun')
    gm_boundary,gm_monotone,gm_spans=boundary(gm)
    km_boundary,km_monotone,km_spans=boundary(km)
    print(json.dumps(dict(stage='primary_fit',gmm_k=gm.n_components,gmm_boundary=gm_boundary,
                          gmm_monotone=gm_monotone,gmm_spans=gm_spans,kmeans_k=km.n_clusters,kmeans_boundary=km_boundary)),flush=True)
    masks={}; rules={}
    for q in [.90,.95,.99]:
        name=f'Q{int(q*100)}'
        threshold=float(d.loc[train,'P'].quantile(q))
        masks[name]=d.P.ge(threshold).to_numpy()
        rules[name]=dict(threshold=threshold,operator='>=',kind='quantile')
    masks['Legacy187']=d.P.ge(187).to_numpy()
    rules['Legacy187']=dict(threshold=187.,operator='>=',kind='legacy_all_scope_reference')
    q1,q3=d.loc[train,'P'].quantile([.25,.75]); fence=float(q3+1.5*(q3-q1))
    masks['IQR']=d.P.gt(fence).to_numpy()
    rules['IQR']=dict(threshold=fence,operator='>',kind='outlier_fence')
    for name,model,b,mono,spans in [('GMM',gm,gm_boundary,gm_monotone,gm_spans),('KMeans',km,km_boundary,km_monotone,km_spans)]:
        masks[name]=upper_group(model,d.P)
        rules[name]=dict(threshold=b,operator='highest_component',kind='load_group',monotone=mono,intervals=spans)
    reference=float(d.loc[train,'P'].max())
    for name,mask,op in [('ReferenceMax',d.P.ge(reference),'>='),('NewRecord',d.P.gt(reference),'>')]:
        masks[name]=mask.to_numpy();rules[name]=dict(threshold=reference,operator=op,kind='observed_max_reference_not_bill')
    for name,group in [('DailyMax','date'),('MonthlyMax','month')]:
        masks[name]=d.P.eq(d.groupby(group).P.transform('max')).to_numpy()
        rules[name]=dict(threshold=None,operator='equals_block_max_all_ties',kind='retrospective_block_max_not_fixed_threshold')
    summary=[]; monthly=[]
    for name,mask in masks.items():
        selected=d.loc[mask]
        row=dict(method=name,threshold=rules[name]['threshold'],operator=rules[name]['operator'],
                 train_n=int(mask[train].sum()),train_rate=float(mask[train].mean()),
                 check_n=int(mask[~train].sum()),check_rate=float(mask[~train].mean()),
                 all_n=int(mask.sum()),all_rate=float(mask.mean()),days=selected.date.nunique(),
                 min_selected=float(selected.P.min()) if len(selected) else np.nan,
                 median_selected=float(selected.P.median()) if len(selected) else np.nan,
                 max_selected=float(selected.P.max()) if len(selected) else np.nan,
                 monthly_max_hits=int((mask&masks['MonthlyMax']).sum()),monthly_max_total=int(masks['MonthlyMax'].sum()))
        summary.append(row)
        for month,part in d.groupby('month'):
            mm=mask[part.index]
            monthly.append(dict(method=name,month=month,n=len(part),high_hours=int(mm.sum()),rate=float(mm.mean())))
    summary=pd.DataFrame(summary);save(summary,'comparison')
    save(pd.DataFrame(monthly),'monthly_rates')
    labels=d[['timestamp','date','month','시간','P','생산량','기온','풍속','습도','강수량','전기요금(계절)']].copy()
    for name,mask in masks.items():labels[name]=mask
    save(labels,'assignments')
    overlaps=[]
    for a,ma in masks.items():
        for b,mb in masks.items():
            union=int((ma|mb).sum())
            overlaps.append(dict(a=a,b=b,intersection=int((ma&mb).sum()),union=union,
                                 jaccard=float((ma&mb).sum()/union) if union else np.nan))
    save(pd.DataFrame(overlaps),'overlap')
    # Show how choosing k changes the highest group, rather than hiding alternatives.
    k_effect=[]
    for method,bank in [('GMM',gmms),('KMeans',kms)]:
        for k,m in bank.items():
            if k==1:continue
            b,mono,spans=boundary(m)
            mask=upper_group(m,d.P)
            k_effect.append(dict(method=method,k=k,threshold=b,monotone=mono,
                                 train_n=int(mask[train].sum()),check_n=int(mask[~train].sum()),all_n=int(mask.sum())))
    save(pd.DataFrame(k_effect),'k_sensitivity')
    grid=pd.DataFrame(dict(power=GRID,density=np.exp(gm.score_samples(GRID[:,None]))))
    for idx in range(gm.n_components):
        mu=gm.means_[idx,0];std=np.sqrt(gm.covariances_[idx,0,0])
        grid[f'component_{idx}']=gm.weights_[idx]/(std*np.sqrt(2*np.pi))*np.exp(-.5*((GRID-mu)/std)**2)
    grid['gmm_high']=upper_group(gm,GRID);grid['kmeans_high']=upper_group(km,GRID)
    save(grid,'fitted_distribution')
    # Weekly resampling preserves the hours within each sampled calendar week.
    tr=d.loc[train].copy();tr['week']=tr.date.dt.to_period('W-SUN')
    blocks=[p.P.to_numpy() for _,p in tr.groupby('week')]
    rng=np.random.default_rng(SEED)
    boot=[]
    for iteration in range(N_BOOT):
        values=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        bg,bk,_,_,_=models(values,n_init=3)
        for name,m in [('GMM',bg),('KMeans',bk)]:
            b,mono,spans=boundary(m)
            mask=upper_group(m,d.P);original=masks[name];union=(mask|original).sum()
            boot.append(dict(iteration=iteration,method=name,k=bg.n_components if name=='GMM' else bk.n_clusters,
                             threshold=b,monotone=mono,all_n=int(mask.sum()),jaccard=float((mask&original).sum()/union)))
        for q in [.9,.95,.99]:
            t=float(np.quantile(values,q));mask=d.P.ge(t).to_numpy();original=masks[f'Q{int(q*100)}']
            boot.append(dict(iteration=iteration,method=f'Q{int(q*100)}',k=np.nan,threshold=t,monotone=True,
                             all_n=int(mask.sum()),jaccard=float((mask&original).sum()/(mask|original).sum())))
        if (iteration+1)%5==0:print(json.dumps(dict(stage='weekly_resampling',completed=iteration+1,total=N_BOOT)),flush=True)
    boot=pd.DataFrame(boot);save(boot,'weekly_resampling')
    # A full-period fit and one vote per identical daily 96-value profile are diagnostics only.
    unique_indices=[];seen=set()
    for date,part in d.loc[train].groupby('date'):
        profile=tuple(part[SLOTS].to_numpy().ravel())
        if profile not in seen:unique_indices.extend(part.index);seen.add(profile)
    scopes={'Jan-Jun':d.loc[train,'P'].to_numpy(),'Jan-Aug':d.P.to_numpy(),
            'Jan-Jun_unique_daily_profiles':d.loc[unique_indices,'P'].to_numpy()}
    refits=[]
    for scope,values in scopes.items():
        if scope=='Jan-Jun':sg,sk=gm,km
        else:sg,sk,_,_,_=models(values)
        for name,m in [('GMM',sg),('KMeans',sk)]:
            b,mono,spans=boundary(m)
            refits.append(dict(scope=scope,method=name,fit_n=len(values),k=sg.n_components if name=='GMM' else sk.n_clusters,
                               threshold=b,monotone=mono,all_n=int(upper_group(m,d.P).sum())))
            if scope!='Jan-Jun':parameters+=model_parameters(m,name,scope)
        for q in [.9,.95,.99]:
            t=float(np.quantile(values,q))
            refits.append(dict(scope=scope,method=f'Q{int(q*100)}',fit_n=len(values),k=np.nan,threshold=t,
                               monotone=True,all_n=int(d.P.ge(t).sum())))
    save(pd.DataFrame(refits),'fit_scope_sensitivity');save(pd.DataFrame(parameters),'components')
    # Same existing pooled June-August condition bins, no redefinition from peak outcomes.
    facts=json.loads((ROOT/'report/tables/eda/external_manifest.json').read_text(encoding='utf-8'))
    base=d.loc[d.month.between(6,8)&d.date.dt.dayofweek.lt(5)&d['생산량'].gt(0)]
    assert len(base)==1244
    condition=[]
    for variable in ['생산량','기온','풍속','습도','강수량']:
        part=base.dropna(subset=[variable]).copy()
        edges=facts['bands'][variable]['edges']
        part['band']=part[variable].gt(0).astype(int) if variable=='강수량' else np.searchsorted(edges,part[variable],side='left')
        for (month,band),cell in part.groupby(['month','band']):
            for name,mask in masks.items():
                condition.append(dict(variable=variable,month=month,band=band,method=name,n=len(cell),
                                      high_hours=int(mask[cell.index].sum()),rate=float(mask[cell.index].mean())))
    save(pd.DataFrame(condition),'condition_rates')
    costs=d.groupby('month').agg(n=('P','size'),maximum=('P','max'),tariff_unique=('전기요금(계절)','nunique'),
                                tariff=('전기요금(계절)','first')).reset_index()
    assert costs.tariff_unique.eq(1).all();save(costs,'cost_evidence')
    manifest=dict(status='passed',source_sha256=SHA,fit_period='2021-01-01..2021-06-30',fit_n=int(train.sum()),
        fixed_application_period='2021-07-01..2021-08-31',application_n=int((~train).sum()),
        application_caveat='Previously explored period, not a pristine independent test',
        rules=rules,selected_gmm_k=gm.n_components,selected_kmeans_k=km.n_clusters,
        gmm_k_candidates=list(range(1,9)),kmeans_k_candidates=list(range(2,9)),
        bootstrap_kind='40 calendar-week block resamples, refit and reselect k; descriptive stability, not confidence interval',
        unique_train_daily_profiles=len(seen),train_days=181,
        actual_bill_threshold='not identified from provided data; seasonal tariff is constant within month',
        cost_proxy_caveat='reference maximum attainment/new records and monthly maxima are not actual bill increases',
        method_references=['https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html',
                           'https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html'],
        criteria='No preferred peak fraction. Compare separation, stability, scope sensitivity, and task meaning. No supervised truth labels.',
        manuscript_edited=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_json(orient='records'),flush=True)
    print(pd.DataFrame(refits).to_json(orient='records'),flush=True)
    print(boot.groupby('method').agg(minimum=('threshold','min'),median=('threshold','median'),maximum=('threshold','max'),
          median_overlap=('jaccard','median')).to_json(orient='index'),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
