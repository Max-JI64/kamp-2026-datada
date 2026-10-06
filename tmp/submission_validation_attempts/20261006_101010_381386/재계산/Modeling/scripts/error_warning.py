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
def read(p):return pd.read_csv(p,encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp','date'])
def diagnose():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'diagnosis.json').exists()
    paths=['Modeling/scripts/error_warning.py','Modeling/scripts/profile_history.py','Modeling/tables/profile_history/run.json','Modeling/tables/profile_history/independent_verification.json','Modeling/tables/profile_history/inner_predictions.csv','Modeling/tables/profile_history/predictions.csv','Modeling/tables/m01/hourly_frame.csv','Modeling/config/m01_contract.json','data/origin/okm_augumented_2021.csv']
    contract={'recorded_at':now(),'inputs_sha256':{p:sha(ROOT/p) for p in paths},'scope':'Jan-Jun only; full B and H6 exact groups including calendar, then past-only nearest20 diagnostic',
        'neighbors':'training StandardScaler; Euclidean; exactly20 prior-month eligible rows; repeated neighbors retained and distinct-date/profile support recorded; no future evaluation outcomes',
        'exact':'all input columns equal; report within-group target variation, empirical median absolute deviation; not a population irreducible-error bound',
        'followup':'if full-input conflicts exist OR historical neighbor outcomes vary, test warning model; neither implies inevitable failure or guarantees error predictability'}
    dump(contract,OUT/'diagnosis_contract.json')
    ver=json.loads((ROOT/'Modeling/tables/profile_history/independent_verification.json').read_text(encoding='utf-8'));assert ver['status']=='passed' and ver['run_sha256']==sha(ROOT/'Modeling/tables/profile_history/run.json')
    f=load();f=f.loc[f.eligible_history & f.month.le(6)].copy()
    exact=[];summaries=[]
    for name in ['B','H6']:
        selected=f[FEATURES[name]+['target_maximum','timestamp','date']].copy()
        count=0;conflict_rows=0;conflicts=0;mad_sum=0;groupid=0
        for key,g in selected.groupby(FEATURES[name],sort=False):
            if len(g)<2:continue
            target=g.target_maximum.to_numpy();spread=float(np.ptp(target));mad=float(abs(target-np.median(target)).mean())
            exact.append({'inputs':name,'group':groupid,'rows':len(g),'dates':g.date.nunique(),'first':str(g.timestamp.min()),'last':str(g.timestamp.max()),'target_min':float(target.min()),'target_max':float(target.max()),'range':spread,'MAD_to_group_median':mad})
            groupid+=1;count+=len(g);mad_sum+=mad*len(g)
            if spread>0:conflicts+=1;conflict_rows+=len(g)
        summaries.append({'inputs':name,'eligible_rows':len(f),'repeated_input_groups':groupid,'repeated_input_rows':count,'conflicting_groups':conflicts,'conflicting_rows':conflict_rows,'within_repeated_empirical_MAD':mad_sum/count if count else None})
    csvout(pd.DataFrame(exact),'exact_groups')
    old=read(ROOT/'Modeling/tables/profile_history/inner_predictions.csv');old=old.loc[old.group.eq('B')]
    outer=read(ROOT/'Modeling/tables/profile_history/predictions.csv');outer=outer.loc[outer.method.eq('fixed_HGB_B')&outer.month.eq(6)]
    point=pd.concat([old[['timestamp','date','month','actual','prediction','daily_maximum_weight','train_profile_overlap']],outer[['timestamp','date','month','actual','prediction','daily_maximum_weight','train_profile_overlap']]],ignore_index=True)
    records=[];neighbor_records=[];audits=[]
    for month in range(2,7):
        tr=f.loc[f.month.lt(month)];ev=f.loc[f.month.eq(month)];q=point.loc[point.month.eq(month)].copy()
        assert ev.timestamp.tolist()==q.timestamp.tolist()
        for name in ['B','H6']:
            scaler=StandardScaler().fit(tr[FEATURES[name]]);tx=scaler.transform(tr[FEATURES[name]]);ex=scaler.transform(ev[FEATURES[name]])
            nn=NearestNeighbors(n_neighbors=20,algorithm='brute',n_jobs=1).fit(tx);dist,idx=nn.kneighbors(ex)
            y=tr.target_maximum.to_numpy()[idx];dates=tr.date.to_numpy()[idx];profiles=tr.diag_target_profile.to_numpy()[idx]
            data={'neighbor_std':np.std(y,axis=1),'neighbor_range':np.ptp(y,axis=1),'neighbor_distance':dist[:,-1],
                'neighbor_mean_minus_prediction':y.mean(1)-q.prediction.to_numpy(),'neighbor_dates':np.array([len(set(a)) for a in dates]),'neighbor_profiles':np.array([len(set(a)) for a in profiles])}
            if name=='B':
                r=q.merge(ev[['timestamp']+FEATURES['B']],on='timestamp',validate='one_to_one')
                for col,val in data.items():r[col]=val
                records.append(r)
            audits.append({'month':month,'inputs':name,'train_hours':len(tr),'eval_hours':len(ev),'mean_neighbor_std':float(np.std(y,axis=1).mean()),'mean_distinct_neighbor_dates':float(data['neighbor_dates'].mean()),'mean_distinct_neighbor_profiles':float(data['neighbor_profiles'].mean())})
            for i,ts in enumerate(ev.timestamp):
                for rank,j in enumerate(idx[i]):neighbor_records.append({'inputs':name,'timestamp':ts,'rank':rank+1,'neighbor_timestamp':tr.timestamp.iloc[j],'distance':float(dist[i,rank]),'neighbor_actual':float(y[i,rank])})
    d=pd.concat(records,ignore_index=True);d['under_amount']=(d.actual-d.prediction).clip(lower=0)
    csvout(d,'risk_frame');csvout(pd.DataFrame(neighbor_records),'neighbors');csvout(pd.DataFrame(audits),'neighbor_summary')
    result={'status':'completed','contract_sha256':sha(OUT/'diagnosis_contract.json'),'exact_summaries':summaries,
        'eligible_for_warning_trial':any(r['conflicting_rows']>0 for r in summaries) or d.neighbor_std.gt(0).any().item(),
        'note':'Exact conflicts describe finite recorded cases; nearest-state variability is not proof of an irreducible limit or operational risk.',
        'outputs_sha256':{p.name:sha(p) for p in OUT.glob('*.csv')}}
    dump(result,OUT/'diagnosis.json');print(json.dumps(result,ensure_ascii=False),flush=True)
def freeze():
    assert not (OUT/'contract.json').exists();diag=json.loads((OUT/'diagnosis.json').read_text(encoding='utf-8'));assert diag['eligible_for_warning_trial']
    d=read(OUT/'risk_frame.csv');feb=d.loc[d.month.eq(2)];threshold=float(np.quantile(feb.under_amount,.9));assert threshold>0
    c={'version':'error-warning-v1','recorded_at':now(),'entrypoint_sha256':sha(Path(__file__)),'diagnosis_sha256':sha(OUT/'diagnosis.json'),'frame_sha256':sha(OUT/'risk_frame.csv'),
        'event_threshold':threshold,'event_definition':'actual-fixed B prediction >= February OOF positive-or-zero underamount 90th percentile, ties included; diagnostic severity, not operational cost',
        'features':RISK_FEATURES,'point_model':'existing monthly fixed HGB B chronological predictions; no in-sample regression errors used',
        'methods':['high_prediction','prior_maximum','neighbor_std','Logistic','HGB'],
        'Logistic':{'C':1.0,'max_iter':3000,'random_state':42},'HGB':{'max_leaf_nodes':7,'max_iter':100,'learning_rate':.05,'min_samples_leaf':20,'l2_regularization':1.0,'early_stopping':False,'random_state':42},
        'splits':{'4':{'train':[2],'calibration':3},'5':{'train':[2,3],'calibration':4},'6':{'train':[2,3,4],'calibration':5}},
        'threshold':'calibration score linear q90; strictly greater to limit calibration alarms to <=10%; fixed for next month; no ranking with future hours in deployed alarm',
        'ranking_budget':'retrospective top floor(.1*n) per month, stable timestamp tie break; diagnostics only, not an online selection policy',
        'selection':'calibration AP max among Logistic,HGB; tie Logistic; best simple reference chosen by calibration AP among high_prediction,prior_maximum,neighbor_std, tie declaration order',
        'gate':'selected ML must beat calibration-selected simple reference: event-weighted monthly AP >=1.05x; retrospective fixed-budget TP >=reference; online pooled TP>=reference+1; online warnings<=reference and <=10% of all eval rows; each month online TP>=reference; each month AP>=.95x; otherwise reject',
        'scope':'4-6 development already reused, no new independent test. Warning review alone does not correct forecast or prove savings.',
        'no_tuning':True,'max_new_fits':6,'no_manuscript_edit':True,'later':'only if eligible, freeze latest policy for later already-observed evaluation; not run automatically within current dev code'}
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
            boundary=float(np.quantile(cp,.9))
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
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnose','freeze','run']);a=p.parse_args();globals()[a.stage]()