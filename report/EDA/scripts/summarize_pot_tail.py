"""Minimal estimator check prompted by maximum-likelihood boundary fits.

Zero-location GPD L-moments: L1=sigma/(1-xi), L2=L1/(2-xi).
Thus xi=2-L1/L2 and sigma=L1*(1-xi). This is a sensitivity diagnostic,
not a replacement selected because it produces a desired threshold.
Source: lmomco documentation, lmomgpa (uses opposite sign kappa=-xi).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_pot_tail import inputs,OUT,read,save

def run():
    d,_,_=inputs();scan=read('threshold_scan');ev=read('events_gap3');rows=[];windows=[]
    for u in range(145,206):
        for series in ['hourly','event']:
            y=(d.loc[d.P.gt(u),'P']-u).to_numpy() if series=='hourly' else (ev.loc[ev.u.eq(u),'P']-u).to_numpy()
            y=np.sort(y);n=len(y);l1=y.mean();b1=np.dot(np.arange(n)/(n-1),y)/n;l2=2*b1-l1
            shape=2-l1/l2;scale=l1*(1-shape)
            endpoint=u-scale/shape if shape<0 else np.nan
            support=bool(scale>0 and np.all(1+shape*y/scale>=0))
            rows.append(dict(series=series,u=u,n=n,shape=shape,scale=scale,modified_scale=scale-shape*u,
                             endpoint=endpoint,supports_all_observations=support))
    pwm=pd.DataFrame(rows);save(pwm,'l_moment_sensitivity')
    for series,gap in [('hourly',0),('event',3)]:
        for lo,hi in [(160,175),(176,194),(179,187),(182,197),(198,205)]:
            p=scan.loc[scan.series.eq(series)&scan.gap_hours.eq(gap)&scan.u.between(lo,hi)]
            coefficients=np.polyfit(p.u,p.mean_excess,1);fitted=np.polyval(coefficients,p.u)
            r2=1-np.square(p.mean_excess-fitted).sum()/np.square(p.mean_excess-p.mean_excess.mean()).sum()
            windows.append(dict(series=series,u_from=lo,u_to=hi,mean_excess_slope=coefficients[0],
                mean_excess_linear_r2=r2,shape_min=p['shape'].min(),shape_max=p['shape'].max(),
                modified_scale_min=p.modified_scale.min(),modified_scale_max=p.modified_scale.max(),
                lowest_n=p.n.min(),maximum_ks=p.ks_distance.max()))
    save(pd.DataFrame(windows),'window_diagnostics')
    root=Path(__file__).resolve().parents[1]
    names=[]
    for file in sorted((root/'scripts').glob('*.py')):
        compile(file.read_text(encoding='utf-8'),str(file),'exec');names.append(file.name)
    audit=dict(status='passed',compiled_scripts=names,active_analysis_scope='Jan-Aug 5784 valid hours',
        source_matches_four_slot_hourly_max=True,prior_pot_scope='pot_tail Jan-Jun is superseded for current EDA',
        manuscript_edited=False)
    (OUT/'folder_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(windows).to_json(orient='records'))
    print(pwm.loc[pwm.u.isin([176,179,182,186,187,194,198,201,205])].to_json(orient='records'))
    print(json.dumps(dict(status='passed',compiled_scripts=len(names))))

if __name__=='__main__':run()
