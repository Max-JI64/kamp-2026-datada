"""Independent audit: exact groups, past-only neighbors, warnings, AP and refits."""
import os,json,hashlib,sys
from collections import defaultdict
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from profile_history import load,FEATURES
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Modeling/tables/error_warning'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(x,p):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def close(a,b):assert np.allclose(a,b,rtol=1e-9,atol=1e-8),(a,b)
def read(name):return pd.read_csv(OUT/(name+'.csv'),encoding='utf-8-sig',float_precision='round_trip')
def ap(y,s):
    pairs=sorted(zip(s,y),reverse=True);total=sum(y)
    if total==0:return 0.
    hits=0;count=0;result=0.;i=0
    while i<len(pairs):
        j=i
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        added=sum(int(v) for _,v in pairs[i:j]);hits+=added;count+=j-i;result+=added/total*hits/count;i=j
    return result

def main():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    run=json.loads((OUT/'run.json').read_text(encoding='utf-8'));c=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
    diag=json.loads((OUT/'diagnosis.json').read_text(encoding='utf-8'));dc=json.loads((OUT/'diagnosis_contract.json').read_text(encoding='utf-8'))
    assert run['contract_sha256']==sha(OUT/'contract.json') and c['entrypoint_sha256']==sha(ROOT/'Modeling/scripts/error_warning_fit.py')
    assert c['diagnosis_sha256']==sha(OUT/'diagnosis.json') and diag['contract_sha256']==sha(OUT/'diagnosis_contract.json')
    for p,h in dc['inputs_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in {**diag['outputs_sha256'],**run['outputs_sha256']}.items():assert sha(OUT/p)==h,p
    f=load();f=f.loc[f.eligible_history & f.month.le(6)].copy()
    # Cross-check reconstructed raw features against independently verified earlier audit.
    parent=ROOT/'Modeling/tables/profile_history';pv=json.loads((parent/'independent_verification.json').read_text(encoding='utf-8'));pr=json.loads((parent/'run.json').read_text(encoding='utf-8'))
    assert pv['status']=='passed' and pv['run_sha256']==sha(parent/'run.json')
    saved=pd.read_csv(parent/'feature_audit.csv',encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp']);saved=saved.loc[saved.timestamp.dt.month.le(6)]
    key=next(k for k in pr['outputs_sha256'] if k.endswith('feature_audit.csv'));assert sha(parent/'feature_audit.csv')==pr['outputs_sha256'][key]
    assert f.timestamp.tolist()==saved.timestamp.tolist();cols=list(dict.fromkeys(FEATURES['B']+FEATURES['H6']+['target_maximum']));close(f[cols],saved[cols])
    for name in ['B','H6']:
        groups=defaultdict(list)
        for row,y in zip(f[FEATURES[name]].to_numpy(),f.target_maximum):groups[tuple(row)].append(y)
        repeated=[v for v in groups.values() if len(v)>1];conflicts=[v for v in repeated if max(v)>min(v)];d=next(x for x in diag['exact_summaries'] if x['inputs']==name)
        assert d['repeated_input_groups']==len(repeated) and d['repeated_input_rows']==sum(map(len,repeated))
        assert d['conflicting_groups']==len(conflicts) and d['conflicting_rows']==sum(map(len,conflicts))
        close(d['within_repeated_empirical_MAD'],sum(sum(abs(np.array(v)-np.median(v))) for v in repeated)/sum(map(len,repeated)))
    risk=read('risk_frame');risk.timestamp=pd.to_datetime(risk.timestamp)
    assert risk.month_x.equals(risk.month_y) and np.array_equal(risk.month_x,risk.timestamp.dt.month)
    risk=risk.rename(columns={'month_x':'month'}).drop(columns='month_y')
    neighbors=read('neighbors');neighbors.timestamp=pd.to_datetime(neighbors.timestamp);neighbors.neighbor_timestamp=pd.to_datetime(neighbors.neighbor_timestamp)
    checked=0
    for month in range(2,7):
        tr=f.loc[f.month.lt(month)].copy();ev=f.loc[f.month.eq(month)].copy();rr=risk.loc[risk.month.eq(month)]
        assert rr.timestamp.tolist()==ev.timestamp.tolist()
        for name in ['B','H6']:
            tx=tr[FEATURES[name]].to_numpy();ex=ev[FEATURES[name]].to_numpy();mean=tx.mean(0);scale=tx.std(0);scale[scale==0]=1.;tx=(tx-mean)/scale;ex=(ex-mean)/scale
            ns=neighbors.loc[neighbors.inputs.eq(name)&neighbors.timestamp.dt.month.eq(month)].sort_values(['timestamp','rank'])
            assert len(ns)==len(ev)*20 and ns.groupby('timestamp')['rank'].apply(list).map(lambda x:x==list(range(1,21))).all()
            assert ns.groupby('timestamp').neighbor_timestamp.nunique().eq(20).all() and ns.neighbor_timestamp.max()<ev.timestamp.min()
            ix=pd.Index(tr.timestamp).get_indexer(ns.neighbor_timestamp);assert (ix>=0).all();ix=ix.reshape(len(ev),20)
            distance=np.linalg.norm(tx[ix]-ex[:,None,:],axis=2);close(distance,ns.distance.to_numpy().reshape(len(ev),20))
            # Check nearest membership while allowing exact tied distances.
            for start in range(0,len(ev),64):
                x=ex[start:start+64];sq=np.maximum((x*x).sum(1)[:,None]+(tx*tx).sum(1)[None,:]-2*x@tx.T,0)
                smallest=np.sqrt(np.partition(sq,19,axis=1)[:,:20].max(1));close(smallest,distance[start:start+64,-1])
            ys=tr.target_maximum.to_numpy()[ix];close(ys,ns.neighbor_actual.to_numpy().reshape(len(ev),20));checked+=len(ns)
            if name=='B':
                close(rr.neighbor_std,ys.std(1));close(rr.neighbor_range,np.ptp(ys,axis=1));close(rr.neighbor_distance,distance[:,-1]);close(rr.neighbor_mean_minus_prediction,ys.mean(1)-rr.prediction)
                close(rr[FEATURES['B']],ev[FEATURES['B']])
    # Normal predictions remain the verified chronological point forecasts.
    pi=pd.read_csv(parent/'inner_predictions.csv',encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp']);po=pd.read_csv(parent/'predictions.csv',encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp'])
    refs=pd.concat([pi.loc[pi.group.eq('B'),['timestamp','actual','prediction']],po.loc[po.method.eq('fixed_HGB_B')&po.month.eq(6),['timestamp','actual','prediction']]])
    assert refs.timestamp.tolist()==risk.timestamp.tolist();close(refs[['actual','prediction']],risk[['actual','prediction']])
    close(risk.under_amount,np.maximum(risk.actual-risk.prediction,0));close(np.quantile(risk.loc[risk.month.eq(2),'under_amount'],.9),c['event_threshold'])
    risk['event']=risk.under_amount.ge(c['event_threshold'])
    p=read('predictions');p.timestamp=pd.to_datetime(p.timestamp);cal=read('calibration_predictions');cal.timestamp=pd.to_datetime(cal.timestamp)
    manifest=read('model_manifest');models=0;refits=0;prediction_checks=0
    for mr in manifest.to_dict('records'):
        mth=mr['month'];spec=c['splits'][str(mth)];tr=risk.loc[risk.month.isin(spec['train'])];ca=risk.loc[risk.month.eq(spec['calibration'])];ev=risk.loc[risk.month.eq(mth)]
        assert tr.timestamp.max()<ca.timestamp.min()<ev.timestamp.min() and len(tr)==mr['train_rows']
        path=ROOT/mr['file'];assert sha(path)==mr['sha256'];m=joblib.load(path)
        for g,z in [(ca,cal.loc[cal.outer_month.eq(mth)&cal.method.eq(mr['method'])]),(ev,p.loc[p.month.eq(mth)&p.method.eq(mr['method'])])]:
            assert z.timestamp.tolist()==g.timestamp.tolist();close(m.predict_proba(g[c['features']])[:,1],z.score);prediction_checks+=len(g)
        if mr['method']=='HGB':
            fresh=HistGradientBoostingClassifier(**c['HGB']).fit(tr[c['features']],tr.event);close(fresh.predict_proba(ev[c['features']])[:,1],m.predict_proba(ev[c['features']])[:,1]);refits+=1
        models+=1
    selections=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));mt=read('metrics');computed={}
    for month in [4,5,6]:
        ca=cal.loc[cal.outer_month.eq(month)];aps={}
        for name in c['methods']:
            cc=ca.loc[ca.method.eq(name)];ee=p.loc[p.month.eq(month)&p.method.eq(name)];a=np.sort(cc.score.to_numpy());boundary=a[len(a)-int(.1*len(a))-1]
            close(cc.threshold,boundary);close(ee.threshold,boundary)
            for g in [cc,ee]:
                assert np.array_equal(g.event,g.under_amount.ge(c['event_threshold']));assert np.array_equal(g.alarm,g.score.gt(boundary))
                flag=np.zeros(len(g),bool);flag[np.argsort(-g.score.to_numpy(),kind='stable')[:int(.1*len(g))]]=True;assert np.array_equal(flag,g.top_budget)
            assert cc.alarm.sum()<=int(.1*len(cc));aps[name]=ap(cc.event.tolist(),cc.score.tolist())
            if name in c['methods'][:3]:
                col={'high_prediction':'prediction','prior_maximum':'lag1_maximum','neighbor_std':'neighbor_std'}[name]
                for g,mm in [(cc,int(cc.month.iloc[0])),(ee,month)]:close(g.score,risk.loc[risk.month.eq(mm),col])
        sr=next(x for x in selections if x['month']==month);assert sr['simple']==max(c['methods'][:3],key=lambda n:aps[n]);assert sr['ML']==max(c['methods'][3:],key=lambda n:aps[n])
        for label,name in [('selected_ML',sr['ML']),('selected_simple',sr['simple'])]:
            x=p.loc[p.month.eq(month)&p.method.eq(label)];y=p.loc[p.month.eq(month)&p.method.eq(name)];close(x.score,y.score);assert x.alarm.tolist()==y.alarm.tolist()
    for (method,month),g in p.groupby(['method','month']):
        ev=g.event.to_numpy(bool);alarm=g.alarm.to_numpy(bool);under=g.under_amount.to_numpy();row=mt.loc[mt.method.eq(method)&mt.period.eq(str(month))].iloc[0]
        vals={'events':int(ev.sum()),'warnings':int(alarm.sum()),'TP':int((ev&alarm).sum()),'FP':int((~ev&alarm).sum()),'FN':int((ev&~alarm).sum()),'top_budget_TP':int((ev&g.top_budget).sum()),'AP':ap(ev.tolist(),g.score.tolist())}
        for k,v in vals.items():close(row[k],v)
        computed[(method,month)]=vals
    pooled={}
    for method,g in p.groupby('method'):
        sub=[computed[(method,m)] for m in [4,5,6]];v={k:sum(r[k] for r in sub) for k in ['events','warnings','TP','FP','FN','top_budget_TP']};v['AP']=sum(r['AP']*r['events'] for r in sub)/v['events'];pooled[method]=v
        row=mt.loc[mt.method.eq(method)&mt.period.eq('pooled')].iloc[0]
        for k,val in v.items():close(row[k],val)
    a=pooled['selected_ML'];b=pooled['selected_simple'];flags={'AP_5percent':a['AP']>=1.05*b['AP'],'ranking_same_budget':a['top_budget_TP']>=b['top_budget_TP'],'online_TP_gain':a['TP']>=b['TP']+1,'no_more_warnings':a['warnings']<=b['warnings'],'warning_cap':a['warnings']/2184<=.1,
        'monthly_TP':all(computed[('selected_ML',m)]['TP']>=computed[('selected_simple',m)]['TP'] for m in [4,5,6]),'monthly_AP':all(computed[('selected_ML',m)]['AP']>=.95*computed[('selected_simple',m)]['AP'] for m in [4,5,6])}
    assert flags==run['decision']['gates'] and all(flags.values())==run['decision']['eligible']
    result={'status':'passed','run_sha256':sha(OUT/'run.json'),'diagnosis_sha256':sha(OUT/'diagnosis.json'),'verifier_sha256':sha(Path(__file__)),
        'exact_input_rows_checked':len(f),'neighbors_checked':checked,'point_forecasts_checked':len(risk),'warning_models_reloaded':models,'score_values_checked':prediction_checks,'independent_HGB_refits':refits,'evaluated_rows':len(p),'monthly_metric_rows':len(computed),'gate_decision':run['decision'],'later_evaluated':False}
    dump(result,OUT/'independent_verification.json');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()