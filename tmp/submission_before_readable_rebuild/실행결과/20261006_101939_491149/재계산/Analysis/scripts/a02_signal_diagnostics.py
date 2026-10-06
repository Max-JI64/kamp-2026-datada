"""Minimal follow-up: shared-level arithmetic and exact-level sign comparisons."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import pandas as pd
from a02_prior_power_signals import OUT, FEATURES, residuals, weighted_corr


def matched_sign(sample, feature, mode):
    # Fixed zero is a direction, not a searched event-size threshold.
    sample=sample.copy()
    sample['side']=np.sign(sample[feature]).astype(int)
    keys=['month','weekend','시간','past_level','known_production_positive']
    sample=sample.loc[sample.side.ne(0)]
    dates=sample.groupby(keys+['side']).date.nunique().unstack(fill_value=0)
    if -1 not in dates or 1 not in dates:
        return None,None
    support=dates.index[(dates[-1]>=2)&(dates[1]>=2)]
    if not len(support):
        return None,None
    ids={key:i for i,key in enumerate(support)}
    allkeys=list(sample[keys].itertuples(index=False,name=None))
    keep=np.array([key in ids for key in allkeys])
    sample=sample.loc[keep].copy()
    cell=np.array([ids[key] for key,k in zip(allkeys,keep) if k])
    side=sample.side.eq(1).to_numpy().astype(int)
    slot=2*cell+side
    w=np.ones(len(sample)) if mode=='date_hour' else sample.profile_weight.to_numpy()
    cell_weight=np.bincount(cell,weights=w,minlength=len(support))
    mass=np.bincount(slot,weights=w,minlength=2*len(support)).reshape(-1,2)
    result=dict(feature=feature,weighting=mode,cells=len(support),n_negative=int((side==0).sum()),
        n_positive=int((side==1).sum()),days=sample.date.nunique())
    for metric,value in [('next_delta',sample.power_delta.to_numpy()),
                         ('next_rise_rate',sample.power_delta.gt(0).to_numpy().astype(float)*100),
                         ('prior_feature',sample[feature].to_numpy()),
                         ('past_level',sample.past_level.to_numpy())]:
        total=np.bincount(slot,weights=w*value,minlength=2*len(support)).reshape(-1,2)
        means=np.average(total/mass,axis=0,weights=cell_weight)
        result[f'{metric}_negative']=means[0]
        result[f'{metric}_positive']=means[1]
    assert abs(result['past_level_negative']-result['past_level_positive'])<1e-10
    return result,sample


def main():
    prior=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    path=OUT/'observations.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==prior['outputs_sha256']['observations.csv']
    part=pd.read_csv(path,encoding='utf-8-sig',parse_dates=['timestamp','date'])
    contract=dict(previous='A02 adjusted rank correlation .305 for prior within-hour trend but only .013 within posthoc onset group; onset/zero-continuing prior means81.85 vs39.57.',
        reason='Recent differences share mean(t-1); prior-level group imbalance requires transparent conditioning, not an onset detection claim.',
        checks='Numeric-scale partial correlation with exact observed level polynomial; target-level and target-change residual identity; exact prior-level/calendar/known-production sign comparisons.',
        scope='Reuse immutable A02 observations only, no forecasting or threshold search.',
        exact_matching='month x upcoming hour x weekend x exact mean(t-1) x known production0/positive; >=2 dates in each sign; zero trend retained in source but outside opposite-sign comparison.',
        decision='If exact matching has little coverage report it; do not relax definitions for larger differences. If sign relationship reverses under repeat weighting do not promote it.',
        manuscript=False,backup=False,forecast=False)
    (OUT/'diagnostic_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    associations=[]
    form_sensitivity=[]
    for scope,sample in [('all',part),('known_production_zero',part.loc[part.past_production.eq(0)]),
                         ('zero_to_positive_posthoc',part.loc[part.transition.eq('zero_to_positive')])]:
        counts=sample.groupby(['month','weekend','시간']).date.transform('nunique')
        sample=sample.loc[counts.ge(3)].copy()
        cells=pd.factorize(pd.MultiIndex.from_frame(sample[['month','weekend','시간']]))[0]
        level=sample.past_level.to_numpy()
        level=(level-level.mean())/level.std()
        controls=np.column_stack([level,level**2,level**3,sample.known_production_positive])
        # Test whether the primary finding relies on the cubic adjustment.
        for form in ('linear_rank','cubic_rank','linear_numeric','cubic_numeric'):
            if 'rank' in form:
                from scipy.stats import rankdata
                lev=rankdata(sample.past_level.to_numpy())/len(sample)
            else:
                lev=level
            parts=[lev, sample.known_production_positive.to_numpy()]
            if 'cubic' in form:
                parts.extend([lev**2,lev**3])
            c=np.column_stack(parts)
            for feature in FEATURES:
                if 'rank' in form:
                    values=np.column_stack([rankdata(sample[feature]),rankdata(sample.power_delta)])/len(sample)
                else:
                    values=np.column_stack([sample[feature],sample.power_delta])
                res=residuals(values,cells,np.ones(len(sample)),c)
                form_sensitivity.append(dict(scope=scope,feature=feature,form=form,n=len(sample),
                    correlation=weighted_corr(res[:,0],res[:,1],np.ones(len(sample)))))
        for mode in ('date_hour','profile_balanced'):
            w=np.ones(len(sample)) if mode=='date_hour' else sample.profile_weight.to_numpy()
            for feature in FEATURES:
                for both in (False,True):
                    c=controls
                    if both:
                        other=FEATURES[1] if feature==FEATURES[0] else FEATURES[0]
                        c=np.column_stack([c,sample[other]])
                    values=np.column_stack([sample[feature],sample.power_delta,sample.level])
                    res=residuals(values,cells,w,c)
                    # mean(t)-mean(t-1) and mean(t) residuals must coincide when
                    # the exact observed mean(t-1) is included in the controls.
                    np.testing.assert_allclose(res[:,1],res[:,2],atol=1e-9,rtol=1e-9)
                    associations.append(dict(scope=scope,feature=feature,both_features=both,weighting=mode,
                        n=len(sample),correlation=weighted_corr(res[:,0],res[:,1],w),
                        next_level_correlation=weighted_corr(res[:,0],res[:,2],w)))
    pd.DataFrame(associations).to_csv(OUT/'numeric_associations.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(form_sensitivity).to_csv(OUT/'adjustment_form_sensitivity.csv',index=False,encoding='utf-8-sig')
    comparisons=[]
    for scope,sample in [('all',part),('known_production_zero',part.loc[part.past_production.eq(0)])]:
        for feature in FEATURES:
            for mode in ('date_hour','profile_balanced'):
                result,_=matched_sign(sample,feature,mode)
                if result:
                    comparisons.append(dict(scope=scope,**result))
    pd.DataFrame(comparisons).to_csv(OUT/'exact_level_sign_comparisons.csv',index=False,encoding='utf-8-sig')
    # Independent grouping arithmetic for the main exact-matching contrast.
    mainrow=next((row for row in comparisons if row['scope']=='all' and row['feature']==FEATURES[0] and row['weighting']=='date_hour'),None)
    groups={}
    with path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            value=float(row[FEATURES[0]])
            if not value:
                continue
            key=(row['month'],row['weekend'],row['시간'],row['past_level'],row['known_production_positive'])
            side=int(value>0)
            groups.setdefault(key,[[],[]])[side].append((row['date'],float(row['power_delta'])))
    numerator=np.zeros(2)
    denominator=0
    totals=np.zeros(2,dtype=int)
    for sides in groups.values():
        if not all(len({date for date,val in side})>=2 for side in sides):
            continue
        w=sum(len(side) for side in sides)
        numerator+=w*np.array([sum(v for d,v in side)/len(side) for side in sides])
        denominator+=w
        totals+=np.array([len(side) for side in sides])
    if mainrow:
        np.testing.assert_allclose(numerator/denominator,[mainrow['next_delta_negative'],mainrow['next_delta_positive']],atol=1e-10)
        assert list(totals)==[mainrow['n_negative'],mainrow['n_positive']]
    else:
        assert denominator==0 and totals.sum()==0
    support=[]
    for scope,sample in [('all',part),('known_production_zero',part.loc[part.past_production.eq(0)])]:
        for feature in FEATURES:
            sample=sample.copy()
            sample['side']=np.sign(sample[feature]).astype(int)
            keys=['month','weekend','시간','past_level','known_production_positive']
            dates=sample.loc[sample.side.ne(0)].groupby(keys+['side']).date.nunique().unstack(fill_value=0)
            neg=dates[-1] if -1 in dates else pd.Series(0,index=dates.index)
            pos=dates[1] if 1 in dates else pd.Series(0,index=dates.index)
            support.append(dict(scope=scope,feature=feature,opposite_sign_common_cells=int(((neg>0)&(pos>0)).sum()),
                minimum_two_dates_cells=int(((neg>=2)&(pos>=2)).sum()),direction_zero_hours=int(sample.side.eq(0).sum())))
    pd.DataFrame(support).to_csv(OUT/'exact_level_support.csv',index=False,encoding='utf-8-sig')
    verification=dict(status='passed',a02_observations_unchanged=True,
        exact_level_residual_identity_verified=True,exact_sign_comparison_independent=bool(mainrow),
        main_sign_stop_reason=None if mainrow else 'No exact-level common cells with at least2dates in each opposite-sign group; no threshold relaxation.',
        main_sign_n=int(totals.sum()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
            for name in ['numeric_associations.csv','exact_level_sign_comparisons.csv','exact_level_support.csv','adjustment_form_sensitivity.csv']})
    (OUT/'diagnostic_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(pd.DataFrame(associations).to_json(orient='records'))
    print(pd.DataFrame(comparisons).to_json(orient='records'))
    print(pd.DataFrame(support).to_json(orient='records'))
    print(pd.DataFrame(form_sensitivity).to_json(orient='records'))
    print(json.dumps(verification))


if __name__=='__main__':
    main()
