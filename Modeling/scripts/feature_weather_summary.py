"""Aggregate verified evidence for separate insertion manuscript."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Modeling/tables/feature_weather_audit'

def summarize():
    feature=pd.read_csv(OUT/'shap_features.csv',encoding='utf-8-sig')
    group=pd.read_csv(OUT/'shap_groups.csv',encoding='utf-8-sig')
    permutation=pd.read_csv(OUT/'permutation_repeats.csv',encoding='utf-8-sig')
    rankings=[]; group_rows=[]; perm_rows=[]
    for period,folds in [('Apr-Jun',[4,5,6]),('Jul-Aug',[7])]:
        for condition in ['all','up_start','fixed_alarm']:
            f=feature.loc[feature.fold.isin(folds)&feature.condition.eq(condition)]
            for (variant,name),g in f.groupby(['variant','feature']):
                rankings.append({'period':period,'condition':condition,'variant':variant,'feature':name,'hours':int(g.hours.sum()),'mean_abs_shap':float(np.average(g.mean_abs_shap,weights=g.hours)),'mean_signed_shap':float(np.average(g.mean_signed_shap,weights=g.hours))})
            f=group.loc[group.fold.isin(folds)&group.condition.eq(condition)]
            for (variant,name),g in f.groupby(['variant','group']):
                group_rows.append({'period':period,'condition':condition,'variant':variant,'group':name,'hours':int(g.hours.sum()),'mean_abs_group_shap':float(np.average(g.mean_abs_group_shap,weights=g.hours)),'mean_signed_group_shap':float(np.average(g.mean_signed_group_shap,weights=g.hours))})
            f=permutation.loc[permutation.fold.isin(folds)&permutation.condition.eq(condition)]
            for (variant,name),g in f.groupby(['variant','group']):
                # Same replicate number is shared across monthly blocks.
                repeats=np.array([np.average(q.mae_increment,weights=q.hours) for _,q in g.groupby('repeat')])
                perm_rows.append({'period':period,'condition':condition,'variant':variant,'group':name,'hours':int(g.loc[g.repeat.eq(0)].hours.sum()),'mean_mae_increment':float(repeats.mean()),'sd_mae_increment':float(repeats.std(ddof=1)),'min_increment':float(repeats.min()),'max_increment':float(repeats.max())})
    feature_summary=pd.DataFrame(rankings)
    feature_summary['rank']=feature_summary.groupby(['period','condition','variant']).mean_abs_shap.rank(method='min',ascending=False).astype(int)
    feature_summary.to_csv(OUT/'shap_feature_summary.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    pd.DataFrame(group_rows).to_csv(OUT/'shap_group_summary.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    pd.DataFrame(perm_rows).to_csv(OUT/'permutation_summary.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    # Fixed current routing is checked against the saved selected system.
    p=pd.read_csv(OUT/'predictions.csv',encoding='utf-8-sig',parse_dates=['timestamp'],float_precision='round_trip')
    old=pd.read_csv(ROOT/'Modeling/tables/regime_age_ablation/followup_predictions.csv',encoding='utf-8-sig',parse_dates=['timestamp'])
    a=p.loc[p.fold.eq(7)&p.variant.eq('G1')].set_index('timestamp')
    b=old.loc[old.variant.eq('G1_noage')].set_index('timestamp').loc[a.index]
    np.testing.assert_allclose(a.prediction,b.prediction,atol=1e-8,rtol=1e-9)
    daily=[]
    for v in ['B0','B1','G1']:
        aa=p.loc[p.variant.eq(v)].set_index('timestamp'); bb=p.loc[p.variant.eq(v+'_weather')].set_index('timestamp').loc[aa.index]
        gain=abs(aa.prediction-aa.target_maximum)-abs(bb.prediction-bb.target_maximum)
        f=pd.DataFrame({'gain':gain,'fold':aa.fold,'date':aa.index.normalize()})
        for period,folds in [('Apr-Jun',[4,5,6]),('Jul-Aug',[7])]:
            d=f.loc[f.fold.isin(folds)].groupby('date').gain.mean()
            daily.append({'variant':v,'period':period,'dates':len(d),'weather_better_dates':int(d.gt(1e-10).sum()),'weather_worse_dates':int(d.lt(-1e-10).sum()),'tied_dates':int(d.abs().le(1e-10).sum())})
    pd.DataFrame(daily).to_csv(OUT/'paired_day_counts.csv',index=False,encoding='utf-8-sig')
    payload={
        'top_features':feature_summary.loc[feature_summary.condition.eq('all')&feature_summary['rank'].le(6)].sort_values(['period','variant','rank']).to_dict('records'),
        'groups':group_rows,
        'permutation_all':[x for x in perm_rows if x['condition']=='all'],
        'permutation_alarm':[x for x in perm_rows if x['condition']=='fixed_alarm'],
        'current_routing_max_difference':float(np.max(abs(a.prediction-b.prediction))),
        'day_counts':daily,
        'monthly_weather':pd.read_csv(OUT/'paired_weather.csv',encoding='utf-8-sig').loc[lambda x:x.period.str.startswith('month_')&x.condition.eq('all')].to_dict('records')}
    (OUT/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k!='groups'},ensure_ascii=False),flush=True)

if __name__=='__main__': summarize()
