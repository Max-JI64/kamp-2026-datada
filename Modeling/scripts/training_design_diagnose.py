"""Diagnose new-profile loss before changing training weights. No new fit."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from m01_prepare import ROOT,sha
from redevelopment_r01 import dump,now
OUT=ROOT/'Modeling/tables/training_design/diagnosis'
def read(p):return pd.read_csv(ROOT/p,encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp','date'])
def save(x,n):x.to_csv(OUT/(n+'.csv'),index=False,encoding='utf-8-sig')
def main():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'run.json').exists()
    paths=['Modeling/tables/profile_history/predictions.csv','Modeling/tables/profile_history/run.json','Modeling/tables/profile_history/independent_verification.json','Modeling/tables/m01/hourly_frame.csv','Modeling/scripts/training_design_diagnose.py']
    ver=json.loads((ROOT/paths[2]).read_text(encoding='utf-8'));assert ver['status']=='passed' and ver['run_sha256']==sha(ROOT/paths[1])
    run=json.loads((ROOT/paths[1]).read_text(encoding='utf-8'));assert sha(ROOT/paths[0])==run['outputs_sha256']['Modeling\\tables\\profile_history\\predictions.csv']
    contract={'recorded_at':now(),'inputs_sha256':{p:sha(ROOT/p) for p in paths},'scope':'development April-June only; no fit',
       'strata':['month','hour','prior_level','month_hour','month_hour_prior_level'],
       'prior_level':'zero, positive<=26, then thirds of Jan-Mar common prior maximum >26; fixed edges',
       'standardization':'within cells containing seen and new profiles, min(group n) weights; robust cell requires >=5 hours and >=2 dates in each group',
       'residual_question':'Does history-minus-B absolute-error loss remain positive on new profiles in shared cells?',
       'weight_followup':'bounded B-input repetition-weight comparison justified if robust shared cells retain positive new loss and positive new-minus-seen gap; diagnostic association not proof of cause',
       'drift':'month pair hour+prior-level shared cells; descriptive, cannot isolate drift from changed state mix'}
    dump(contract,OUT/'contract.json')
    f=read('Modeling/tables/m01/hourly_frame.csv');f=f.loc[f.eligible_common]
    p=read(paths[0]);b=p.loc[p.method.eq('fixed_HGB_B')].copy();h=p.loc[p.method.eq('history_direct'),['timestamp','prediction']]
    b=b.merge(h,on='timestamp',suffixes=('_B','_H'),validate='one_to_one')
    assert len(b)==2184
    b['hour']=b.timestamp.dt.hour;b['new_profile']=~b.train_profile_overlap
    b['loss_B']=abs(b.prediction_B-b.actual);b['loss_H']=abs(b.prediction_H-b.actual);b['loss_change']=b.loss_H-b.loss_B
    b['bias_B']=b.prediction_B-b.actual;b['bias_H']=b.prediction_H-b.actual
    initial=f.loc[f.month.le(3)&f.lag1_maximum.gt(26),'lag1_maximum'];edges=np.quantile(initial,[1/3,2/3]).tolist()
    b['prior_level']=np.select([b.lag1_maximum.eq(0),b.lag1_maximum.le(26),b.lag1_maximum.le(edges[0]),b.lag1_maximum.le(edges[1])],['zero','low','lower','middle'],default='upper')
    save(b,'paired')
    definitions={'month':['month'],'hour':['hour'],'prior_level':['prior_level'],'month_hour':['month','hour'],'month_hour_prior_level':['month','hour','prior_level']}
    cells=[];std=[]
    for label,keys in definitions.items():
        for vals,g in b.groupby(keys):
            parts={int(k):v for k,v in g.groupby('new_profile')}
            row={'strata':label,'cell':str(vals)}
            for k in [0,1]:
                z=parts.get(k);name='new' if k else 'seen'
                row.update({name+'_hours':len(z) if z is not None else 0,name+'_dates':z.date.nunique() if z is not None else 0})
                for metric in ['loss_B','loss_H','loss_change','actual','bias_B','bias_H']:
                    row[name+'_'+metric]=float(z[metric].mean()) if z is not None else np.nan
            cells.append(row)
        cc=pd.DataFrame([x for x in cells if x['strata']==label])
        for robust in [False,True]:
            good=(cc.new_hours.ge(5)&cc.seen_hours.ge(5)&cc.new_dates.ge(2)&cc.seen_dates.ge(2)) if robust else (cc.new_hours.gt(0)&cc.seen_hours.gt(0))
            z=cc.loc[good];w=np.minimum(z.new_hours,z.seen_hours)
            row={'strata':label,'robust':robust,'cells':len(z),'new_hours':int(z.new_hours.sum()),'seen_hours':int(z.seen_hours.sum()),'new_coverage':float(z.new_hours.sum()/b.new_profile.sum()),'seen_coverage':float(z.seen_hours.sum()/(~b.new_profile).sum())}
            if len(z):
                for name in ['new','seen']:
                    for metric in ['loss_B','loss_H','loss_change','actual']:row[name+'_'+metric]=float(np.average(z[name+'_'+metric],weights=w))
                row['new_minus_seen_loss_change']=row['new_loss_change']-row['seen_loss_change']
            std.append(row)
    save(pd.DataFrame(cells),'strata_cells');save(pd.DataFrame(std),'standardized')
    monthly=[]
    for (month,new),z in b.groupby(['month','new_profile']):
        peak=z.daily_maximum_weight.sum()
        monthly.append({'month':month,'new':bool(new),'hours':len(z),'dates':z.date.nunique(),'MAE_B':z.loss_B.mean(),'MAE_H':z.loss_H.mean(),'change':z.loss_change.mean(),'peak_weight':peak,'peak_change':float(np.average(z.loss_change,weights=z.daily_maximum_weight)) if peak else None})
    save(pd.DataFrame(monthly),'month_overlap')
    drift=[]
    for left,right in [(4,5),(5,6),(4,6)]:
        aa=b.loc[b.month.eq(left)].groupby(['hour','prior_level']).agg(n=('actual','size'),actual=('actual','mean'),bias=('bias_B','mean'))
        zz=b.loc[b.month.eq(right)].groupby(['hour','prior_level']).agg(n=('actual','size'),actual=('actual','mean'),bias=('bias_B','mean'))
        j=aa.join(zz,how='inner',lsuffix='_a',rsuffix='_b');j=j.loc[j.n_a.ge(5)&j.n_b.ge(5)];w=np.minimum(j.n_a,j.n_b)
        drift.append({'left':left,'right':right,'cells':len(j),'hours_left':int(j.n_a.sum()),'hours_right':int(j.n_b.sum()),'target_mean_shift':float(np.average(j.actual_b-j.actual_a,weights=w)),'B_bias_shift':float(np.average(j.bias_b-j.bias_a,weights=w))})
    save(pd.DataFrame(drift),'month_state_shift')
    robust=next(x for x in std if x['strata']=='month_hour_prior_level' and x['robust'])
    proceed=bool(robust.get('new_loss_change',0)>0 and robust.get('new_minus_seen_loss_change',0)>0)
    result={'status':'completed','contract_sha256':sha(OUT/'contract.json'),'prior_edges':edges,'pooled':b.groupby('new_profile')[['loss_B','loss_H','loss_change']].mean().to_dict(),
        'robust_standardized':robust,'proceed_repetition_weights':proceed,'limitation':'shared-cell standardization covers only a subset; labels use full evaluation day posthoc; no causal attribution',
        'outputs_sha256':{p.name:sha(p) for p in OUT.glob('*.csv')},'new_fits':0}
    dump(result,OUT/'run.json');print(json.dumps(result,ensure_ascii=False),flush=True);print(pd.DataFrame(monthly).to_string(index=False));print(pd.DataFrame(drift).to_string(index=False))
if __name__=='__main__':main()