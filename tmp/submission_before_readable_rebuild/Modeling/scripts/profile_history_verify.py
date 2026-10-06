"""Independent raw-feature, model reload, selection and metric checks for P01-P04.
Also record fixed descriptive condition diagnostics. Does not fit models or edit manuscripts.
"""
import csv,json,hashlib,os,math
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Modeling/tables/profile_history';AO=ROOT/'Analysis/tables/p02_profile_history'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n,dates=None):return pd.read_csv(OUT/(n+'.csv'),encoding='utf-8-sig',float_precision='round_trip',parse_dates=dates)
def close(a,b):assert np.allclose(a,b,atol=1e-9,rtol=1e-11),(a,b)
def dump(x,p):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def summary(q):
    e=q.prediction.to_numpy()-q.actual.to_numpy();w=q.daily_maximum_weight.to_numpy()
    return [np.mean(abs(e)),np.average(abs(e),weights=w),np.average(np.maximum(-e,0),weights=w),np.average(np.maximum(e,0),weights=w)]
def main():
    run=json.loads((OUT/'run.json').read_text(encoding='utf-8'));c=json.loads((ROOT/'Modeling/config/profile_history_contract.json').read_text(encoding='utf-8'))
    assert run['contract_sha256']==sha(ROOT/'Modeling/config/profile_history_contract.json')
    for path,h in {**c['inputs_sha256'],**run['outputs_sha256']}.items():assert sha(ROOT/path)==h,path
    raw={}
    with (ROOT/'data/origin/okm_augumented_2021.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            date=int(r['날짜']);h=float(r['시간'])
            if not (20210101<=date<=20210831 and 0<=h<=23):continue
            ts=datetime.strptime(str(date),'%Y%m%d')+timedelta(hours=h)
            v=[float(r[x]) for x in ['15분','30분','45분','60분']];mean=sum(v)/4
            state='zero' if max(v)==0 else 'below20' if float(r['평균'])<20 else 'low' if float(r['평균'])<=26 else 'above26'
            raw[ts]={'values':v,'max':max(v),'mean':mean,'production':float(r['생산량']),'state':state}
    for ts,r in sorted(raw.items()):
        prev=raw.get(ts-timedelta(hours=1));r['age']=prev['age']+1 if prev and prev['state']==r['state'] else 1
        r['censored']=prev['censored'] if prev and prev['state']==r['state'] else int(prev is None)
    f=read('feature_audit',['timestamp']);checked=0
    for q in f.to_dict('records'):
        ts=q['timestamp'].to_pydatetime();r=raw[ts]
        exp={'month':ts.month,'weekend':int(ts.weekday()>=5),'hour_sin':math.sin(2*math.pi*ts.hour/24),'hour_cos':math.cos(2*math.pi*ts.hour/24),'target_maximum':r['max']}
        exp.update({f'dow_{i}':int(ts.weekday()==i) for i in range(7)})
        for lag in range(1,25):
            p=raw[ts-timedelta(hours=lag)]
            for s in range(1,5):exp[f'past_{lag}_slot_{s}']=p['values'][s-1]
            if lag<=2:exp[f'lag{lag}_mean']=p['mean'];exp[f'lag{lag}_maximum']=p['max']
            if lag==1:exp.update(lag1_last=p['values'][3],lag1_production=p['production'],lag1_production_zero=int(p['production']==0),prior_state_age=p['age'],prior_state_censored=p['censored'])
        exp.update({f'target_slot_{s}':r['values'][s-1] for s in range(1,5)})
        for key,value in q.items():
            if key!='timestamp':assert math.isclose(value,exp[key],abs_tol=1e-12), (ts,key,value,exp[key]);checked+=1
    p=read('predictions',['timestamp','date']);inner=read('inner_predictions',['timestamp','date']);sp=read('slot_predictions',['timestamp'])
    # Independent peak-date weights using raw values and complete evaluated dates.
    for table,methodcol in [(p,'method'),(inner,'group')]:
        for (_,day),g in table.groupby([methodcol,'date']):
            actual=[raw[t.to_pydatetime()]['max'] for t in g.timestamp];assert actual==g.actual.tolist()
            peak=max(actual);ties=actual.count(peak)
            expected=[1/ties if v==peak and len(g)==24 else 0 for v in actual];close(g.daily_maximum_weight,expected)
    manifest=read('model_manifest');models_checked=0;reload_values=0
    for q in manifest.to_dict('records'):
        path=ROOT/q['file'];assert sha(path)==q['sha256'];m=joblib.load(path)
        ev=f.loc[f.timestamp.dt.month.eq(q['eval_month'])];cols=c['features'][q['group']]
        assert list(m.feature_names_in_)==cols and all(m.get_params()[k]==v for k,v in c['parameters'].items())
        assert pd.Timestamp(q['train_end'])<ev.timestamp.min();assert q['train_hours']==int(f.timestamp.dt.month.lt(q['eval_month']).sum())
        pred=m.predict(ev[cols]);checks=[]
        if q['target']=='target_maximum':
            z=inner.loc[inner.month.eq(q['eval_month'])&inner.group.eq(q['group'])]
            if len(z):checks.append(z)
            z=p.loc[p.month.eq(q['eval_month'])&p.input.eq(q['group'])&p.variant.eq('direct')]
            checks.extend(g for _,g in z.groupby('method'))
        else:
            s=int(q['target'][-1]);methods=p.loc[p.month.eq(q['eval_month'])&p.input.eq(q['group'])&p.variant.eq('four_slots'),'method'].unique()
            checks.extend(sp.loc[sp.month.eq(q['eval_month'])&sp.method.eq(name)&sp.slot.eq(s)] for name in methods)
        assert checks
        for z in checks:
            assert z.timestamp.tolist()==ev.timestamp.tolist();close(pred,z.prediction);reload_values+=len(z)
        models_checked+=1
    for (name,month),g in sp.groupby(['method','month']):
        matrix=g.pivot(index='timestamp',columns='slot',values='prediction')
        q=p.loc[p.method.eq(name)&p.month.eq(month)].set_index('timestamp').loc[matrix.index]
        close(matrix.max(axis=1),q.prediction)
    sel=read('inner_selection');selected={}
    for month in [4,5,6]:
        z=inner.loc[inner.month.lt(month)];ref=z.loc[z.group.eq('B')];den=[np.mean(abs(ref.actual-ref.lag1_maximum)),np.average(abs(ref.actual-ref.lag1_maximum),weights=ref.daily_maximum_weight)]
        scores=[]
        for order,group in enumerate(['B','H2','H6','H24']):
            q=z.loc[z.group.eq(group)];s=summary(q);score=max(s[0]/den[0],s[1]/den[1]);scores.append((score,order,group))
            line=sel.loc[sel.outer_month.eq(month)&sel.group.eq(group)].iloc[0];close([line.MAE,line.peak_MAE,line.score],[s[0],s[1],score])
            assert set(q.month)==set(c['inner_months'][str(month)])
        selected[month]=min(scores)[2]
        assert sel.loc[sel.outer_month.eq(month)&sel.selected,'group'].tolist()==[selected[month]]
        assert set(p.loc[p.month.eq(month)&p.method.str.startswith('history_'),'input'])=={selected[month]}
    comp=read('comparison');sums={name:summary(q) for name,q in p.groupby('method')}
    decision=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    for row in decision['candidate_gates']:
        name=row['method'];q=p.loc[p.method.eq(name)];s=sums[name];flags={}
        close(s,comp.loc[comp.method.eq(name),['overall_MAE','peak_MAE','peak_under','peak_over']].to_numpy()[0])
        for ref in ['fixed_HGB_A','fixed_HGB_B']:
            r=sums[ref];flags.update({ref+'_overall':s[0]<=r[0]*1.01,ref+'_peak':s[1]<=r[1]*.95,ref+'_under':s[2]<=r[2]*.95,ref+'_over':s[3]-r[3]<=r[1]*.05})
            days=sorted(q.date.unique());rp=p.loc[p.method.eq(ref)];dif=[]
            for day in days:dif.append(summary(q.loc[q.date.ne(day)])[1]-summary(rp.loc[rp.date.ne(day)])[1])
            flags[ref+'_leave_one_date']=max(dif)<0
        flags['monthly']=all(summary(q.loc[q.month.eq(m)])[i]<=summary(p.loc[p.method.eq('fixed_HGB_B')&p.month.eq(m)])[i]*1.05 for m in [4,5,6] for i in [0,1])
        assert flags==row['gates'] and all(flags.values())==row['eligible']
    # P02 same hour, same rounded preceding mean, observed duration contrast.
    bh=p.loc[p.method.eq('fixed_HGB_B')].set_index('timestamp');hp=p.loc[p.method.eq('history_direct')].set_index('timestamp')
    a=pd.DataFrame({'timestamp':bh.index,'hour':bh.index.hour,'month':bh.month.to_numpy(),
        'prior_rounded_mean':[math.floor(raw[t.to_pydatetime()-timedelta(hours=1)]['mean']+.5) for t in bh.index],
        'prior_state':bh.prior_state.to_numpy(),'age':bh.prior_state_age.to_numpy(),'actual':bh.actual.to_numpy(),
        'error_B':abs(bh.prediction-bh.actual).to_numpy(),'error_history':abs(hp.prediction-hp.actual).to_numpy(),'daily_weight':bh.daily_maximum_weight.to_numpy()})
    a['age_bin']=pd.cut(a.age,[0,2,6,24,float('inf')],labels=['1-2','3-6','7-24','25+'])
    t=a.groupby(['hour','prior_rounded_mean','prior_state','age_bin'],observed=True).agg(hours=('actual','size'),actual_mean=('actual','mean'),actual_min=('actual','min'),actual_max=('actual','max'),MAE_B=('error_B','mean'),MAE_history=('error_history','mean')).reset_index()
    t.to_csv(AO/'matched_state_history.csv',index=False,encoding='utf-8-sig')
    paired=[]
    for label,mask in [('all',a.index==a.index),('non_daily_peak',a.daily_weight.eq(0)),('daily_peak_rows',a.daily_weight.gt(0))]:
        z=a.loc[mask];paired.append({'scope':label,'hours':len(z),'MAE_B':z.error_B.mean(),'MAE_history':z.error_history.mean()})
    pd.DataFrame(paired).to_csv(AO/'improvement_location.csv',index=False,encoding='utf-8-sig')
    assert not decision['selected'] # This run's observed outcome; no later evaluation should be dispatched.
    assert all(x<7 for x in p.month.unique()) and run['new_fits']==sum(~manifest.reused)
    v={'status':'passed','run_sha256':sha(OUT/'run.json'),'verifier_sha256':sha(Path(__file__)),
       'raw_feature_values_checked':checked,'model_reloads':models_checked,'prediction_values_rechecked':reload_values,
       'outer_prediction_rows':len(p),'slot_prediction_rows':len(sp),'inner_selections_checked':3,'gates_checked':len(decision['candidate_gates']),
       'analysis_outputs_sha256':{str(x.relative_to(ROOT)):sha(x) for x in AO.glob('*.csv')},'later_evaluation':'skipped: no eligible method','manuscripts_edited':False}
    dump(v,OUT/'independent_verification.json');print(json.dumps(v),flush=True)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
if __name__=='__main__':main()