"""Verify training design against raw day profiles, saved models and refits."""
import csv,hashlib,json,os,math,sys
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'Modeling/tables/training_design'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(x,p):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return pd.read_csv(p,encoding='utf-8-sig',float_precision='round_trip')
def close(a,b):assert np.allclose(a,b,atol=1e-9,rtol=1e-11),(a,b)
def met(q):
    e=q.prediction.to_numpy()-q.actual.to_numpy();w=q.daily_maximum_weight.to_numpy()
    return np.mean(abs(e)),np.average(abs(e),weights=w),np.average(np.maximum(-e,0),weights=w),np.average(np.maximum(e,0),weights=w)
def verify_diagnosis():
    d=BASE/'diagnosis';run=json.loads((d/'run.json').read_text(encoding='utf-8'));c=json.loads((d/'contract.json').read_text(encoding='utf-8'))
    assert run['contract_sha256']==sha(d/'contract.json')
    for path,h in c['inputs_sha256'].items():assert sha(ROOT/path)==h
    for name,h in run['outputs_sha256'].items():assert sha(d/name)==h
    q=read(d/'paired.csv');buckets=defaultdict(lambda:defaultdict(list))
    for row in q.to_dict('records'):
        close(row['loss_B'],abs(row['actual']-row['prediction_B']));close(row['loss_H'],abs(row['actual']-row['prediction_H']))
        close(row['loss_change'],row['loss_H']-row['loss_B'])
        buckets[(row['month'],row['hour'],row['prior_level'])][bool(row['new_profile'])].append(row)
    cells=[]
    for key,groups in buckets.items():
        if set(groups)!={True,False}:continue
        if any(len(g)<5 or len({x['date'] for x in g})<2 for g in groups.values()):continue
        cells.append((min(map(len,groups.values())),groups))
    expected=run['robust_standardized'];assert len(cells)==expected['cells']
    den=sum(n for n,g in cells)
    for new,label in [(True,'new'),(False,'seen')]:
        assert sum(len(g[new]) for n,g in cells)==expected[label+'_hours']
        for col in ['loss_B','loss_H','loss_change','actual']:
            value=sum(n*sum(x[col] for x in g[new])/len(g[new]) for n,g in cells)/den
            close(value,expected[label+'_'+col])
    assert expected['new_loss_change']>0 and expected['new_minus_seen_loss_change']>0 and run['proceed_repetition_weights']
    v={'status':'passed','run_sha256':sha(d/'run.json'),'verifier_sha256':sha(Path(__file__)),'paired_rows_checked':len(q),'robust_shared_cells_checked':len(cells)}
    dump(v,d/'independent_verification.json');return v

def main():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    diag=verify_diagnosis()
    f=read(ROOT/'Modeling/tables/m01/hourly_frame.csv');f.timestamp=pd.to_datetime(f.timestamp);f.date=pd.to_datetime(f.date);f=f.loc[f.eligible_common].copy()
    raw={};days=defaultdict(list)
    with (ROOT/'data/origin/okm_augumented_2021.csv').open(encoding='utf-8-sig',newline='') as fh:
        for r in csv.DictReader(fh):
            day=int(r['날짜']);hour=float(r['시간'])
            if not(20210101<=day<=20210831 and 0<=hour<=23):continue
            ts=datetime.strptime(str(day),'%Y%m%d')+timedelta(hours=hour);vals=[float(r[k]) for k in ['15분','30분','45분','60분']]
            raw[ts]=vals;days[ts.date()].append((hour,vals))
    profiles={day:hashlib.sha256(','.join(format(x,'.17g') for _,v in sorted(rows) for x in v).encode()).hexdigest() for day,rows in days.items()}
    for r in f.itertuples():assert profiles[r.timestamp.date()]==r.diag_target_profile
    reports={}
    for stage in ['weights','recent']:
        out=BASE/stage;c=json.loads((out/'contract.json').read_text(encoding='utf-8'));run=json.loads((out/'run.json').read_text(encoding='utf-8'))
        assert run['contract_sha256']==sha(out/'contract.json')
        for p,h in c['inputs_sha256'].items():assert sha(ROOT/p)==h,p
        for p,h in run['outputs_sha256'].items():assert sha(out/p)==h,p
        models=read(out/'model_manifest.csv');ws=read(out/'training_weights.csv');ws.timestamp=pd.to_datetime(ws.timestamp)
        pp=read(out/'all_month_predictions.csv');pp.timestamp=pd.to_datetime(pp.timestamp)
        preds=read(out/'predictions.csv');preds.timestamp=pd.to_datetime(preds.timestamp)
        choices=read(out/'selection_scores.csv');decision=json.loads((out/'selection.json').read_text(encoding='utf-8'))
        count_weights=0;count_preds=0;refits=0
        for mr in models.to_dict('records'):
            month=mr['month'];name=mr['candidate'];ev=f.loc[f.month.eq(month)];tr=f.loc[f.month.lt(month)]
            value=c['candidates'][name]
            if stage=='recent' and value is not None:tr=tr.loc[tr.timestamp.ge(ev.timestamp.min()-pd.Timedelta(days=value))]
            assert tr.timestamp.max()<ev.timestamp.min() and len(tr)==mr['train_hours'] and len(ev)==mr['eval_hours']
            counts=Counter(profiles[x.date()] for x in tr.date.drop_duplicates())
            expected=np.array([counts[profiles[x.date()]]**(-value) if stage=='weights' and value is not None else 1. for x in tr.timestamp]);expected/=expected.mean()
            w=ws.loc[ws.month.eq(month)&ws.candidate.eq(name)]
            assert w.timestamp.tolist()==tr.timestamp.tolist();close(w.weight,expected);count_weights+=len(w)
            # Every profile uses only already completed training dates; target-day label never enters input.
            assert not any(x.startswith(('diag_','target_')) for x in c['features'])
            m=joblib.load(ROOT/mr['file']);assert sha(ROOT/mr['file'])==mr['sha256'];assert list(m.feature_names_in_)==c['features']
            assert all(m.get_params()[k]==v for k,v in c['parameters'].items())
            p=m.predict(ev[c['features']]);z=pp.loc[pp.month.eq(month)&pp.candidate.eq(name)]
            assert z.timestamp.tolist()==ev.timestamp.tolist();close(p,z.prediction);count_preds+=len(z)
            close(z.actual,[max(raw[x.to_pydatetime()]) for x in z.timestamp])
            for day,g in z.groupby('date'):
                actual=g.actual.tolist();mx=max(actual);n=actual.count(mx);close(g.daily_maximum_weight,[1/n if y==mx and len(g)==24 else 0 for y in actual])
            # Refit one weighted and one recent-window model in each outer month.
            if month in [4,5,6] and name==('sqrt_inverse' if stage=='weights' else 'recent_30d'):
                independent=HistGradientBoostingRegressor(**c['parameters']).fit(tr[c['features']],tr.target_maximum,sample_weight=expected)
                close(independent.predict(ev[c['features']]),p);refits+=1
        for month in [4,5,6]:
            default=next(iter(c['candidates']));inner=pp.loc[pp.month.lt(month)];ref=inner.loc[inner.candidate.eq(default)].copy();ref.prediction=ref.lag1_maximum;den=met(ref)[:2]
            scores=[]
            for order,name in enumerate(c['candidates']):
                s=met(inner.loc[inner.candidate.eq(name)]);score=max(s[0]/den[0],s[1]/den[1]);scores.append((score,order,name))
                row=choices.loc[choices.outer_month.eq(month)&choices.candidate.eq(name)].iloc[0];close([row.MAE,row.peak_MAE,row.score],[s[0],s[1],score])
            chosen=min(scores)[2];assert choices.loc[choices.outer_month.eq(month)&choices.selected,'candidate'].tolist()==[chosen]
            policy=preds.loc[preds.month.eq(month)&preds.method.eq('selected_policy')];q=pp.loc[pp.month.eq(month)&pp.candidate.eq(chosen)]
            close(policy.prediction,q.prediction)
        metrics={name:met(g) for name,g in preds.groupby('method')}
        for row in decision['candidate_gates']:
            name=row['method'];q=preds.loc[preds.method.eq(name)];s=metrics[name];close(s,[row['overall_MAE'],row['peak_MAE'],row['peak_under'],row['peak_over']]);flags={}
            for reference in ['fixed_HGB_A','fixed_HGB_B']:
                r=metrics[reference];flags.update({reference+'_overall':s[0]<=r[0]*1.01,reference+'_peak':s[1]<=r[1]*.95,reference+'_under':s[2]<=r[2]*.95,reference+'_over':s[3]-r[3]<=r[1]*.05})
                other=preds.loc[preds.method.eq(reference)];ds=[]
                for day in q.date.unique():ds.append(met(q.loc[q.date.ne(day)])[1]-met(other.loc[other.date.ne(day)])[1])
                flags[reference+'_leave_one_date']=max(ds)<0
            flags['monthly']=all(met(q.loc[q.month.eq(month)])[i]<=met(preds.loc[preds.method.eq('fixed_HGB_B')&preds.month.eq(month)])[i]*1.05 for month in [4,5,6] for i in [0,1])
            assert flags==row['gates'] and all(flags.values())==row['eligible']
        assert run['new_fits']==int(models.source.eq('new_fit').sum()) and preds.month.max()==6
        policy=next(r for r in decision['candidate_gates'] if r['method']=='selected_policy')
        assert decision['selected']==('selected_policy' if policy['eligible'] else None)
        report={'status':'passed','run_sha256':sha(out/'run.json'),'verifier_sha256':sha(Path(__file__)),'raw_day_profiles_checked':len(profiles),
            'training_weights_checked':count_weights,'model_reloads':len(models),'prediction_values_checked':count_preds,'independent_refits':refits,'internal_selections_checked':3,'methods_checked':len(metrics),
            'selected':decision['selected'],'later_evaluation':'not eligible' if decision['selected'] is None else 'pending'}
        dump(report,out/'independent_verification.json');reports[stage]=report
    print(json.dumps({'diagnosis':diag,'stages':reports},ensure_ascii=False),flush=True)
if __name__=='__main__':main()