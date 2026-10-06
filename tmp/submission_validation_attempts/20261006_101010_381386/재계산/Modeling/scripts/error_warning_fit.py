"""Observed-input ambiguity and chronological large-underprediction warning study.
Normal HGB B forecasts remain fixed; risk scores use past OOF error labels only.
"""
import argparse,json,os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import numpy as np,pandas as pd,joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from m01_prepare import ROOT,sha
from profile_history import load,FEATURES
from redevelopment_r01 import now,dump
OUT=ROOT/'Modeling/tables/error_warning';MODELS=ROOT/'Modeling/models/error_warning'
RISK_FEATURES=FEATURES['B']+['prediction','neighbor_std','neighbor_range','neighbor_distance','neighbor_mean_minus_prediction']
def csvout(d,name):d.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def read(p):
    d=pd.read_csv(p,encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp','date'])
    # Diagnostic join stored both equal month columns. Normalize only in memory;
    # preserve frozen diagnostic artifact and verify its two sources agree.
    if 'month_x' in d and 'month_y' in d:
        assert d.month_x.equals(d.month_y) and np.array_equal(d.month_x,d.timestamp.dt.month)
        d=d.rename(columns={'month_x':'month'}).drop(columns='month_y')
    return d
def freeze():
    assert not (OUT/'contract.json').exists();diag=json.loads((OUT/'diagnosis.json').read_text(encoding='utf-8'));assert diag['eligible_for_warning_trial']
    d=read(OUT/'risk_frame.csv');feb=d.loc[d.month.eq(2)];threshold=float(np.quantile(feb.under_amount,.9));assert threshold>0
    c={'version':'error-warning-v1','recorded_at':now(),'entrypoint_sha256':sha(Path(__file__)),'diagnosis_sha256':sha(OUT/'diagnosis.json'),'frame_sha256':sha(OUT/'risk_frame.csv'),
        'event_threshold':threshold,'event_definition':'actual-fixed B prediction >= February OOF positive-or-zero underamount 90th percentile, ties included; diagnostic severity, not operational cost',
        'features':RISK_FEATURES,'point_model':'existing monthly fixed HGB B chronological predictions; no in-sample regression errors used',
        'methods':['high_prediction','prior_maximum','neighbor_std','Logistic','HGB'],
        'Logistic':{'C':1.0,'max_iter':3000,'random_state':42},'HGB':{'max_leaf_nodes':7,'max_iter':100,'learning_rate':.05,'min_samples_leaf':20,'l2_regularization':1.0,'early_stopping':False,'random_state':42},
        'splits':{'4':{'train':[2],'calibration':3},'5':{'train':[2,3],'calibration':4},'6':{'train':[2,3,4],'calibration':5}},
        'threshold':'calibration ascending score at index n-floor(.1*n)-1; strictly greater, hence <=floor(.1*n) alerts including ties; fixed for next month; no future-hour ranking',
        'ranking_budget':'retrospective top floor(.1*n) per month, stable timestamp tie break; diagnostics only, not an online selection policy',
        'selection':'calibration AP max among Logistic,HGB; tie Logistic; best simple reference chosen by calibration AP among high_prediction,prior_maximum,neighbor_std, tie declaration order',
        'gate':'selected ML must beat calibration-selected simple reference: event-weighted monthly AP >=1.05x; retrospective fixed-budget TP >=reference; online pooled TP>=reference+1; online warnings<=reference and <=10% of all eval rows; each month online TP>=reference; each month AP>=.95x; otherwise reject',
        'scope':'4-6 development already reused, no new independent test. Warning review alone does not correct forecast or prove savings.',
        'preflight_schema':'diagnostic month_x/month_y verified equal and normalized in memory before any warning fit; frozen diagnosis unchanged','no_tuning':True,'max_new_fits':6,'no_manuscript_edit':True,'later':'only if eligible, freeze latest policy for later already-observed evaluation; not run automatically within current dev code'}
    dump(c,OUT/'contract.json');print('Warning contract frozen; threshold='+str(threshold),flush=True)
def scores(m,x):return m.predict_proba(x)[:,1]
def metrics(g):
    y=g.event.to_numpy(bool);a=g.alarm.to_numpy(bool);t=g.top_budget.to_numpy(bool);e=g.under_amount.to_numpy()
    tp=int((y&a).sum());fp=int((~y&a).sum());n=int(y.sum())
    return {'hours':len(g),'events':n,'AP':float(average_precision_score(y,g.score)) if n else 0.,'warnings':int(a.sum()),'warning_fraction':float(a.mean()),'TP':tp,'FP':fp,'FN':n-tp,
        'precision':tp/int(a.sum()) if a.any() else 0.,'recall':tp/n if n else 0.,'top_budget_TP':int((y&t).sum()),'unwarned_events':int((y&~a).sum()),
        'unwarned_under_mean':float(e[~a].mean()) if (~a).any() else None,'warned_under_share':float(e[a].sum()/e.sum()) if e.sum() else 0.}
def evaluate(p):
    rows=[]
    for (method,month),g in p.groupby(['method','month']):rows.append({'method':method,'period':str(month),**metrics(g)})
    for method,g in p.groupby('method'):
        r=metrics(g);sub=[x for x in rows if x['method']==method];r['AP']=sum(x['AP']*x['events'] for x in sub)/sum(x['events'] for x in sub);rows.append({'method':method,'period':'pooled','AP_definition':'event-weighted monthly AP',**r})
    table=pd.DataFrame(rows);i=table.set_index(['method','period']);a=i.loc[('selected_ML','pooled')];b=i.loc[('selected_simple','pooled')]
    flags={'AP_5percent':bool(a.AP>=b.AP*1.05),'ranking_same_budget':bool(a.top_budget_TP>=b.top_budget_TP),'online_TP_gain':bool(a.TP>=b.TP+1),'no_more_warnings':bool(a.warnings<=b.warnings),'warning_cap':bool(a.warning_fraction<=.1),
        'monthly_TP':all(i.loc[('selected_ML',str(m)),'TP']>=i.loc[('selected_simple',str(m)),'TP'] for m in [4,5,6]),
        'monthly_AP':all(i.loc[('selected_ML',str(m)),'AP']>=i.loc[('selected_simple',str(m)),'AP']*.95 for m in [4,5,6])}
    return table,{'eligible':all(flags.values()),'gates':flags}
def run():
    assert not (OUT/'run.json').exists();c=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
    assert c['entrypoint_sha256']==sha(Path(__file__)) and c['frame_sha256']==sha(OUT/'risk_frame.csv')
    d=read(OUT/'risk_frame.csv');d['event']=d.under_amount.ge(c['event_threshold']);MODELS.mkdir(parents=True,exist_ok=True)
    ps=[];cs=[];models=[];selections=[];fits=0
    for month in [4,5,6]:
        spec=c['splits'][str(month)];tr=d.loc[d.month.isin(spec['train'])];cal=d.loc[d.month.eq(spec['calibration'])];ev=d.loc[d.month.eq(month)]
        assert tr.timestamp.max()<cal.timestamp.min()<ev.timestamp.min() and tr.event.nunique()==2
        candidates={};calibrations={}
        for name in c['methods']:
            if name in ['Logistic','HGB']:
                m=make_pipeline(StandardScaler(),LogisticRegression(**c['Logistic'])) if name=='Logistic' else HistGradientBoostingClassifier(**c['HGB'])
                m.fit(tr[c['features']],tr.event);fits+=1;path=MODELS/f'month{month}_{name}.joblib';joblib.dump(m,path,compress=3)
                cp=scores(m,cal[c['features']]);ep=scores(m,ev[c['features']]);assert np.allclose(scores(joblib.load(path),ev[c['features']]),ep,atol=1e-12)
                models.append({'month':month,'method':name,'file':str(path.relative_to(ROOT)),'sha256':sha(path),'train_rows':len(tr),'calibration_rows':len(cal),'eval_rows':len(ev),'train_end':str(tr.timestamp.max())})
            else:
                col={'high_prediction':'prediction','prior_maximum':'lag1_maximum','neighbor_std':'neighbor_std'}[name];cp=cal[col].to_numpy();ep=ev[col].to_numpy()
            boundary=float(np.sort(cp)[len(cp)-int(.1*len(cp))-1])
            def output(frame,score):
                z=frame[['timestamp','date','month','actual','prediction','under_amount','event','daily_maximum_weight','train_profile_overlap']].copy();z['method']=name;z['score']=score;z['threshold']=boundary;z['alarm']=z.score.gt(boundary)
                z['top_budget']=False;order=np.argsort(-np.asarray(score),kind='stable')[:int(.1*len(z))];z.iloc[order,z.columns.get_loc('top_budget')]=True;return z
            cc=output(cal,cp);cc['outer_month']=month;ee=output(ev,ep);cs.append(cc);ps.append(ee);candidates[name]=ee
            calibrations[name]=float(average_precision_score(cal.event,cp));assert cc.alarm.mean()<=.1+1e-12
        simple=max(c['methods'][:3],key=lambda n:calibrations[n]);ml=max(c['methods'][3:],key=lambda n:calibrations[n]);selections.append({'month':month,'simple':simple,'ML':ml,'calibration_AP':calibrations})
        for selected,label in [(simple,'selected_simple'),(ml,'selected_ML')]:z=candidates[selected].copy();z['method']=label;ps.append(z)
        print(f'Warning month={month}: reference={simple}, ML={ml}, train={len(tr)}, cal={len(cal)}, eval={len(ev)}',flush=True)
    p=pd.concat(ps,ignore_index=True);cp=pd.concat(cs,ignore_index=True);table,decision=evaluate(p)
    csvout(p,'predictions');csvout(cp,'calibration_predictions');csvout(table,'metrics');csvout(pd.DataFrame(models),'model_manifest');dump(selections,OUT/'selection.json');dump(decision,OUT/'decision.json')
    result={'status':'completed','contract_sha256':sha(OUT/'contract.json'),'new_fits':fits,'development_hours':2184,'event_threshold':c['event_threshold'],'events':int(p.loc[p.method.eq('selected_ML'),'event'].sum()),'decision':decision,
        'later_evaluated':False,'manuscripts_edited':False,'outputs_sha256':{x.name:sha(x) for x in [OUT/'predictions.csv',OUT/'calibration_predictions.csv',OUT/'metrics.csv',OUT/'model_manifest.csv',OUT/'selection.json',OUT/'decision.json']}}
    dump(result,OUT/'run.json');print(table.loc[table.period.eq('pooled')].to_string(index=False));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);a=p.parse_args();globals()[a.stage]()