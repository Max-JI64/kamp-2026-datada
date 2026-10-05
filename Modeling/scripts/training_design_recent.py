"""Recent-window stage; frozen separately after repetition weighting failed.
Preflight selects the same-month uniform parent when a window includes all rows.
The already executed weighting entry point remains unchanged.
"""
import argparse,json,os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from m01_prepare import ROOT,sha,feature_columns,read_contract
from redevelopment_r01 import now,dump
from redevelopment_compare import assess
M1=read_contract();B=feature_columns('B',M1)
PARAM={'max_leaf_nodes':15,'max_iter':150,'learning_rate':.05,'min_samples_leaf':20,'l2_regularization':1.,'early_stopping':False,'random_state':42}
BASE=ROOT/'Modeling/tables/training_design';MODELS=ROOT/'Modeling/models/training_design'
def read(path):return pd.read_csv(path,encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp','date'])
def metric(q):
    e=abs(q.prediction-q.actual);return float(e.mean()),float(np.average(e,weights=q.daily_maximum_weight))
def freeze(stage):
    out=BASE/stage;out.mkdir(parents=True,exist_ok=True);assert not (out/'contract.json').exists()
    diagnosis=json.loads((BASE/'diagnosis/run.json').read_text(encoding='utf-8'))
    assert diagnosis['proceed_repetition_weights'] if stage=='weights' else True
    paths=['Modeling/tables/training_design/diagnosis/run.json','Modeling/scripts/training_design_recent.py','Modeling/scripts/redevelopment_compare.py','Modeling/config/m01_contract.json',
       'Modeling/tables/m01/hourly_frame.csv','Modeling/tables/profile_history/predictions.csv','Modeling/tables/profile_history/inner_predictions.csv','Modeling/tables/profile_history/model_manifest.csv',M1['source']]
    if stage=='recent': paths+=['Modeling/tables/training_design/weights/run.json']
    candidates={'uniform':None,'sqrt_inverse':.5,'inverse':1.} if stage=='weights' else {'all_history':None,'recent_30d':30,'recent_60d':60,'recent_90d':90}
    c={'version':stage+'-v1','recorded_at':now(),'inputs_sha256':{p:sha(ROOT/p) for p in paths},'features':B,'parameters':PARAM,'candidates':candidates,
       'reason':'Matched diagnosis retains new loss; change repeated-day training contribution only' if stage=='weights' else 'Conditional month target means and B bias shifted; test limited training windows with uniform weights as separate factor',
       'weight':'n=number of distinct eligible training dates sharing exact 96-slot daily profile, normalized n**(-exponent) to mean1; all training dates fully observed; not an inference input',
       'window':'cutoff at evaluation month start; keep timestamps>=cutoff-days, lag history may precede window; compare on identical evaluation rows',
       'outer_months':[4,5,6],'inner_months':{'4':[2,3],'5':[2,3,4],'6':[2,3,4,5]},
       'selection':'min max(pooled inner MAE / lag1 MAE, pooled inner daily-peak MAE / lag1 daily-peak MAE); tie candidate declaration order; includes original training',
       'maximum_new_fits':10 if stage=='weights' else 15,'outer_policy':'selected internally per outer month; fixed candidate outer scores are diagnostics only; not posthoc reselected',
       'gates':json.loads((ROOT/'Modeling/config/redevelopment_r02_contract.json').read_text(encoding='utf-8'))['gates'],
       'no_optuna':True,'later':'only selected_policy eligible; Jan-Jun once then frozen July-Aug, no new independent-test claim','no_manuscript_edit':True}
    dump(c,out/'contract.json');print(f'{stage}: contract frozen',flush=True)
def run(stage):
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    out=BASE/stage;assert not (out/'run.json').exists();c=json.loads((out/'contract.json').read_text(encoding='utf-8'))
    for p,h in c['inputs_sha256'].items():assert sha(ROOT/p)==h,p
    modeldir=MODELS/stage;modeldir.mkdir(parents=True,exist_ok=True)
    f=read(ROOT/'Modeling/tables/m01/hourly_frame.csv');f=f.loc[f.eligible_common].copy()
    outer=read(ROOT/'Modeling/tables/profile_history/predictions.csv');oldinner=read(ROOT/'Modeling/tables/profile_history/inner_predictions.csv')
    manifests=pd.read_csv(ROOT/'Modeling/tables/profile_history/model_manifest.csv',encoding='utf-8-sig')
    predictions=[];audits=[];weights=[];models=[];fits=0
    for month in range(2,7):
        ev=f.loc[f.month.eq(month)];full=f.loc[f.month.lt(month)]
        if month<4:q=oldinner.loc[oldinner.month.eq(month)&oldinner.group.eq('B')].copy()
        else:q=outer.loc[outer.month.eq(month)&outer.method.eq('fixed_HGB_B')].copy()
        assert q.timestamp.tolist()==ev.timestamp.tolist()
        q=q.drop(columns=[x for x in ['method','group','family','input','variant','absolute_error','under_amount','over_amount','bias'] if x in q])
        default='uniform' if stage=='weights' else 'all_history'
        for name,value in c['candidates'].items():
            tr=full.copy()
            if stage=='recent' and value is not None:tr=tr.loc[tr.timestamp.ge(ev.timestamp.min()-pd.Timedelta(days=value))]
            if stage=='weights' and value is not None:
                frequencies=tr[['date','diag_target_profile']].drop_duplicates().diag_target_profile.value_counts()
                raww=tr.diag_target_profile.map(frequencies).to_numpy(dtype=float)**(-value);w=raww/raww.mean()
            else:w=np.ones(len(tr))
            assert tr.timestamp.max()<ev.timestamp.min() and tr.groupby('date').size().eq(24).all()
            if name==default:
                mr=manifests.loc[manifests.eval_month.eq(month)&manifests.group.eq('B')&manifests.target.eq('target_maximum')].iloc[0]
                path=ROOT/mr.file;assert sha(path)==mr.sha256;est=joblib.load(path);source='parent'
                assert len(tr)==mr.train_hours
            elif stage=='recent' and len(tr)==len(full):
                # Identical rows and uniform weights: no need to refit.
                path=ROOT/next(r['file'] for r in models if r['month']==month and r['candidate']==default);est=joblib.load(path);source='same_training_rows'
            else:
                est=HistGradientBoostingRegressor(**PARAM);est.fit(tr[B],tr.target_maximum,sample_weight=w);fits+=1
                path=modeldir/f'month{month}_{name}.joblib';joblib.dump(est,path,compress=3);source='new_fit'
            pred=est.predict(ev[B]);assert np.allclose(joblib.load(path).predict(ev[B]),pred,atol=1e-10,rtol=0)
            if name==default:assert np.allclose(pred,q.prediction,atol=1e-10,rtol=0)
            r=q.copy();r['candidate']=name;r['prediction']=pred;predictions.append(r)
            models.append({'month':month,'candidate':name,'file':str(path.relative_to(ROOT)),'sha256':sha(path),'source':source,'train_hours':len(tr),'eval_hours':len(ev)})
            weights.extend({'month':month,'candidate':name,'timestamp':ts,'date':date,'profile':profile,'weight':float(ww)} for ts,date,profile,ww in zip(tr.timestamp,tr.date,tr.diag_target_profile,w))
            audits.append({'month':month,'candidate':name,'train_hours':len(tr),'train_dates':tr.date.nunique(),'train_profiles':tr.diag_target_profile.nunique(),
                'train_start':str(tr.timestamp.min()),'train_end':str(tr.timestamp.max()),'eval_hours':len(ev),'weight_min':float(w.min()),'weight_max':float(w.max()),'effective_weighted_rows':float(w.sum()**2/(w*w).sum())})
        print(f'{stage}: month={month} new_fits={fits}',flush=True)
    pp=pd.concat(predictions,ignore_index=True);selection=[];selected=[]
    for month in [4,5,6]:
        inner=pp.loc[pp.month.lt(month)];ref=inner.loc[inner.candidate.eq(default)].copy();ref['prediction']=ref.lag1_maximum;den=metric(ref)
        scores=[]
        for order,name in enumerate(c['candidates']):
            mae,peak=metric(inner.loc[inner.candidate.eq(name)]);scores.append({'outer_month':month,'candidate':name,'MAE':mae,'peak_MAE':peak,'score':max(mae/den[0],peak/den[1]),'order':order})
        chosen=min(scores,key=lambda x:(x['score'],x['order']))['candidate']
        for row in scores:row['selected']=row['candidate']==chosen
        selection.extend(scores);x=pp.loc[pp.month.eq(month)&pp.candidate.eq(chosen)].copy();x['method']='selected_policy';selected.append(x)
    fixed=pp.loc[pp.month.ge(4)].copy();fixed['method']=fixed.candidate
    p=pd.concat([outer.loc[outer.method.isin(['fixed_HGB_A','fixed_HGB_B','lag1'])],fixed,*selected],ignore_index=True)
    e=p.prediction-p.actual;p['absolute_error']=abs(e);p['under_amount']=(-e).clip(lower=0);p['over_amount']=e.clip(lower=0);p['bias']=e
    tables,comparison,decision,leave=assess(p,c)
    policy=next(r for r in decision['candidate_gates'] if r['method']=='selected_policy')
    decision['selected']='selected_policy' if policy['eligible'] else None;decision['no_eligible_candidate']=decision['selected'] is None
    decision['fixed_candidate_scores_are_diagnostic']=True
    outputs={'all_month_predictions':pp,'predictions':p,'selection_scores':pd.DataFrame(selection),'model_manifest':pd.DataFrame(models),'training_weights':pd.DataFrame(weights),'training_audit':pd.DataFrame(audits),'metrics':tables,'comparison':comparison,'leave_one_date_out':leave}
    for name,d in outputs.items():d.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    dump(decision,out/'selection.json');assert fits<=c['maximum_new_fits']
    result={'status':'completed','finished':now(),'contract_sha256':sha(out/'contract.json'),'new_fits':fits,'models':len(models),'development_hours':2184,'selected':decision['selected'],
       'internal_choices':[{k:v for k,v in r.items() if k in ['outer_month','candidate']} for r in selection if r['selected']],
       'later_evaluated':False,'manuscripts_edited':False,'outputs_sha256':{p.name:sha(p) for p in list(out.glob('*.csv'))+[out/'selection.json']}}
    dump(result,out/'run.json');print(comparison.to_string(index=False));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['weights','recent']);p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.freeze:freeze(a.stage)
    if a.run:run(a.stage)