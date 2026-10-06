"""P02/P03: bounded past-profile comparison and four-slot output ablation.
Freeze before fitting. Nested chronological selection; no later-period tuning.
"""
import argparse, json, os, sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(k,'4')
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from m01_prepare import ROOT, feature_columns, read_contract, sha
from redevelopment_r01 import dump, now
from redevelopment_compare import pw, assess, record
OUT=ROOT/'Modeling/tables/profile_history'
MODELS=ROOT/'Modeling/models/profile_history'
ANALYSIS=ROOT/'Analysis/tables/p02_profile_history'
CONTRACT=ROOT/'Modeling/config/profile_history_contract.json'
SLOTS=['15분','30분','45분','60분']
PARAM={'max_leaf_nodes':15,'max_iter':150,'learning_rate':.05,'min_samples_leaf':20,'l2_regularization':1.,'early_stopping':False,'random_state':42}
M1=read_contract();A=feature_columns('A',M1);B=feature_columns('B',M1)
FEATURES={'A':A,'B':B}
for n in [2,6,24]:
    # lag1 slot4 is exactly the existing lag1_last; avoid duplicate columns.
    FEATURES['H'+str(n)]=B+[f'past_{lag}_slot_{s}' for lag in range(1,n+1) for s in range(1,5) if (lag,s)!=(1,4)]+(['prior_state_age','prior_state_censored'] if n>2 else [])
def csvout(d,name):d.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def load():
    f=pd.read_csv(ROOT/'Modeling/tables/m01/hourly_frame.csv',encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp','date'])
    raw=pd.read_csv(ROOT/M1['source'],encoding='utf-8-sig')
    raw=raw.loc[raw['날짜'].between(20210101,20210831)&raw['시간'].between(0,23)].copy()
    raw['timestamp']=pd.to_datetime(raw['날짜'].astype(str),format='%Y%m%d')+pd.to_timedelta(raw['시간'],unit='h')
    raw=raw.sort_values('timestamp').set_index('timestamp')
    assert raw.index.tolist()==f.timestamp.tolist()
    features={}
    for lag in range(1,25):
        prev=raw.reindex(pd.DatetimeIndex(f.timestamp-pd.Timedelta(hours=lag)))
        for s,col in enumerate(SLOTS,1):features[f'past_{lag}_slot_{s}']=prev[col].to_numpy(float)
    states=np.select([raw[SLOTS].max(axis=1).eq(0),raw['평균'].lt(20),raw['평균'].between(20,26)],['zero','below20','low'],default='above26')
    age=[];censored=[];run=0;left=True
    for i,ts in enumerate(raw.index):
        contiguous=i>0 and ts-raw.index[i-1]==pd.Timedelta(hours=1)
        if not contiguous or states[i]!=states[i-1]:run=1;left=not contiguous
        else:run+=1
        age.append(run);censored.append(int(left))
    stateframe=pd.DataFrame({'state':states,'age':age,'censored':censored},index=raw.index)
    prev=stateframe.reindex(pd.DatetimeIndex(f.timestamp-pd.Timedelta(hours=1)))
    features['prior_state_age']=prev.age.to_numpy();features['prior_state_censored']=prev.censored.to_numpy()
    f=pd.concat([f,pd.DataFrame(features)],axis=1)
    f['prior_state']=prev.state.to_numpy()
    for s,col in enumerate(SLOTS,1): f['target_slot_'+str(s)]=raw[col].to_numpy(float)
    f['history_complete']=f[[f'past_{lag}_slot_1' for lag in range(1,25)]].notna().all(axis=1)
    f['eligible_history']=f.eligible_common & f.history_complete
    return f

def freeze():
    assert not CONTRACT.exists()
    proof=json.loads((ROOT/'EDA/tables/p01_peak_shapes/run.json').read_text(encoding='utf-8'))
    assert proof['verified']
    paths=['Modeling/scripts/profile_history.py','Modeling/scripts/redevelopment_compare.py','Modeling/scripts/redevelopment_r01.py','Modeling/scripts/m01_prepare.py',
           'Modeling/config/m01_contract.json','Modeling/tables/m01/hourly_frame.csv','Modeling/tables/m03/ab/model_manifest.csv','Modeling/tables/m03/ab/run.json',
           'EDA/tables/p01_peak_shapes/run.json','EDA/tables/p01_peak_shapes/verification.json',M1['source']]
    c={'version':'profile-history-v1','recorded_at':now(),'inputs_sha256':{p:sha(ROOT/p) for p in paths},'features':FEATURES,'parameters':PARAM,
        'reason':'P01 high360h: one high slot136 vs all four28; low duration median24h max223h. Test past shapes/duration before further tuning.',
        'eligibility':'M01 common plus all preceding 24 exact hours for every method; no imputation',
        'inner_months':{'4':[2,3],'5':[2,3,4],'6':[2,3,4,5]},
        'selection':'Pool inner predictions; minimize max(overall MAE/lag1 MAE, daily-maximum MAE/lag1 daily-maximum MAE); tie B,H2,H6,H24. B fallback allowed.',
        'history_for_slot_output':'Same inner direct-maximum selection used for both outputs to isolate output effect',
        'outer_months':[4,5,6],'no_optuna':True,'no_postprocessing':True,'maximum_new_fits':60,
        'models':['fixed_HGB_A','fixed_HGB_B','history_direct','B_four_slots','history_four_slots','lag1'],
        'slot_metrics':'Slot MAE; location credit=intersection of predicted/actual argmax sets divided by predicted ties; no operational timing claim',
        'gates':json.loads((ROOT/'Modeling/config/redevelopment_r02_contract.json').read_text(encoding='utf-8'))['gates'],
        'later':'Only eligible development policy; Jan-Jun fit once, July-Aug fixed, history from June outer selection; already observed, not independent',
        'no_manuscript_edit':True}
    dump(c,CONTRACT);print('Frozen bounded P02/P03 contract',flush=True)

def base(ev,tr,split):
    q=ev[['timestamp','date','month','target_maximum','lag1_maximum','lag1_last','diag_target_profile','prior_state','prior_state_age']].rename(columns={'target_maximum':'actual'}).copy()
    q['split']=split;q['daily_maximum_weight']=pw(ev)
    q['train_profile_overlap']=q.diag_target_profile.isin(tr.diag_target_profile)
    freq=q[['date','diag_target_profile']].drop_duplicates().diag_target_profile.value_counts()
    w=1/q.diag_target_profile.map(freq);q['profile_weight']=w/w.mean()
    return q

def execute():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    c=json.loads(CONTRACT.read_text(encoding='utf-8'))
    for p,h in c['inputs_sha256'].items(): assert sha(ROOT/p)==h,p
    for d in [OUT,MODELS,ANALYSIS]:d.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'run.json').exists()
    allf=load();f=allf.loc[allf.eligible_history].copy()
    csvout(allf[['timestamp','eligible_common','history_complete','eligible_history']],'coverage')
    csvout(f[['timestamp']+list(dict.fromkeys(A+FEATURES['H24']+['target_maximum','target_slot_1','target_slot_2','target_slot_3','target_slot_4']))],'feature_audit')
    manifests=[];fits=0;cache={};inner_records=[];selection=[];predictions=[];slot_records=[]
    parent=pd.read_csv(ROOT/'Modeling/tables/m03/ab/model_manifest.csv',encoding='utf-8-sig')
    def train_predict(tr,ev,group,target,tag,split=None):
        nonlocal fits
        key=(int(ev.month.iloc[0]),group,target)
        if key in cache:return cache[key]
        columns=FEATURES[group];assert tr.timestamp.max()<ev.timestamp.min()
        assert tr[columns].notna().all().all() and ev[columns].notna().all().all()
        reused=False
        oldtr=allf.loc[allf.eligible_common & allf.timestamp.lt(ev.timestamp.min())]
        if group in ['A','B'] and target=='target_maximum' and split and oldtr.timestamp.tolist()==tr.timestamp.tolist():
            row=parent.loc[parent.split.eq(split)&parent.group.eq(group)&parent.target.eq(target)].iloc[0]
            path=ROOT/row.file;assert sha(path)==row.sha256;m=joblib.load(path);reused=True
        else:
            m=HistGradientBoostingRegressor(**PARAM);m.fit(tr[columns],tr[target]);fits+=1
            path=MODELS/f'{tag}_{group}_{target}.joblib';joblib.dump(m,path,compress=3)
        p=m.predict(ev[columns]);assert np.isfinite(p).all()
        assert list(m.feature_names_in_)==columns
        assert all(m.get_params()[k]==v for k,v in PARAM.items())
        assert np.allclose(joblib.load(path).predict(ev[columns]),p,rtol=0,atol=1e-10)
        manifests.append({'tag':tag,'group':group,'target':target,'file':str(path.relative_to(ROOT)),'sha256':sha(path),'train_hours':len(tr),'eval_hours':len(ev),
            'train_end':str(tr.timestamp.max()),'eval_month':int(ev.month.iloc[0]),'reused':reused})
        cache[key]=p;return p
    # Each inner forecast is generated from strictly earlier months and cached.
    for month in range(2,6):
        tr=f.loc[f.month.lt(month)];ev=f.loc[f.month.eq(month)];q=base(ev,tr,f'inner_{month}')
        for group in ['B','H2','H6','H24']:
            p=train_predict(tr,ev,group,'target_maximum',f'inner_{month}',{4:'dev_apr',5:'dev_may'}.get(month))
            z=q.copy();z['group']=group;z['prediction']=p;inner_records.append(z)
        print(f'Inner month {month}: train={len(tr)} eval={len(ev)} fits={fits}',flush=True)
    inn=pd.concat(inner_records,ignore_index=True);csvout(inn,'inner_predictions')
    def choose(month):
        z=inn.loc[inn.month.lt(month)];lag=z.loc[z.group.eq('B')]
        den=[np.mean(abs(lag.actual-lag.lag1_maximum)),np.average(abs(lag.actual-lag.lag1_maximum),weights=lag.daily_maximum_weight)]
        rows=[]
        for i,group in enumerate(['B','H2','H6','H24']):
            p=z.loc[z.group.eq(group)];mae=float(np.mean(abs(p.actual-p.prediction)));peak=float(np.average(abs(p.actual-p.prediction),weights=p.daily_maximum_weight))
            rows.append({'outer_month':month,'group':group,'inner_months':','.join(str(v) for v in sorted(p.month.unique())),'MAE':mae,'peak_MAE':peak,'score':max(mae/den[0],peak/den[1]),'order':i})
        winner=min(rows,key=lambda x:(x['score'],x['order']))['group']
        for x in rows:x['selected']=x['group']==winner
        selection.extend(rows);return winner
    for month,split in [(4,'dev_apr'),(5,'dev_may'),(6,'dev_jun')]:
        tr=f.loc[f.month.lt(month)];ev=f.loc[f.month.eq(month)];q=base(ev,tr,split);history=choose(month)
        choices=[('fixed_HGB_A','A'),('fixed_HGB_B','B'),('history_direct',history)]
        for name,group in choices:
            p=train_predict(tr,ev,group,'target_maximum',split,split)
            predictions.append(record(q,name,'HGB',group,'direct',p))
        for name,group in [('B_four_slots','B'),('history_four_slots',history)]:
            ps=np.column_stack([train_predict(tr,ev,group,'target_slot_'+str(s),split,split) for s in range(1,5)])
            predictions.append(record(q,name,'HGB',group,'four_slots',ps.max(axis=1)))
            for s in range(1,5):
                z=ev[['timestamp','month']].copy();z['method']=name;z['slot']=s;z['actual']=ev['target_slot_'+str(s)].to_numpy();z['prediction']=ps[:,s-1];slot_records.append(z)
        predictions.append(record(q,'lag1','baseline','B','lag1',ev.lag1_maximum.to_numpy()))
        print(f'Outer {month}: history={history} train={len(tr)} eval={len(ev)} fits={fits}',flush=True)
    p=pd.concat(predictions,ignore_index=True);sp=pd.concat(slot_records,ignore_index=True)
    csvout(p,'predictions');csvout(sp,'slot_predictions');csvout(pd.DataFrame(selection),'inner_selection');csvout(pd.DataFrame(manifests),'model_manifest')
    tables,comparison,decision,leave=assess(p,c)
    csvout(tables,'metrics');csvout(comparison,'comparison');csvout(leave,'leave_one_date_out');dump(decision,OUT/'selection.json')
    # Conditional tables are explanatory only, never inputs or a new selection gate.
    conditions=[]
    for name,g in p.groupby('method'):
        g=g.copy();g['age_bin']=pd.cut(g.prior_state_age,[0,2,6,24,float('inf')],labels=['1-2','3-6','7-24','25+'])
        for (state,age),part in g.groupby(['prior_state','age_bin'],observed=True):
            conditions.append({'method':name,'state':state,'observed_age':str(age),'hours':len(part),'dates':part.date.nunique(),
                'actual_mean':part.actual.mean(),'next_low_fraction':float(part.actual.le(26).mean()),'MAE':float(abs(part.actual-part.prediction).mean()),'under':float((part.actual-part.prediction).clip(lower=0).mean())})
    pd.DataFrame(conditions).to_csv(ANALYSIS/'conditional_errors.csv',index=False,encoding='utf-8-sig')
    slot_metrics=[]
    for (name,month),g in sp.groupby(['method','month']):
        actual=g.pivot(index='timestamp',columns='slot',values='actual').to_numpy();pred=g.pivot(index='timestamp',columns='slot',values='prediction').to_numpy()
        ap=actual==actual.max(1)[:,None];pp=pred==pred.max(1)[:,None]
        slot_metrics.append({'method':name,'month':int(month),'hours':len(actual),'slot_MAE':float(abs(actual-pred).mean()),'location_credit':float(((ap&pp).sum(1)/pp.sum(1)).mean())})
    csvout(pd.DataFrame(slot_metrics),'slot_metrics')
    # Reused parent outputs are diagnostic references, not a later-period run.
    assert fits<=c['maximum_new_fits'] and p.timestamp.max()<pd.Timestamp('2021-07-01')
    result={'status':'completed','finished':now(),'contract_sha256':sha(CONTRACT),'new_fits':fits,'models':len(manifests),
        'full_common':int(allf.eligible_common.sum()),'history_common':len(f),'development_hours':int(p.loc[p.method.eq('fixed_HGB_B')].shape[0]),
        'inner_selection':[{k:v for k,v in r.items() if k in ['outer_month','group','score']} for r in selection if r['selected']],
        'selected':decision['selected'],'later_evaluated':False,'manuscript_edited':False,
        'outputs_sha256':{str(t.relative_to(ROOT)):sha(t) for t in list(OUT.glob('*.csv'))+[OUT/'selection.json',ANALYSIS/'conditional_errors.csv']}}
    dump(result,OUT/'run.json');print(comparison.to_string(index=False),flush=True);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    if args.run:execute()