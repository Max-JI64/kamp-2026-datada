"""Verify closing analysis, selected forecasts, explanation identities and report scope."""
import argparse
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
from regime_forecast import sha, save_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Modeling/tables/modeling_close'


def main(visual=False):
    run=json.loads((OUT/'run.json').read_text(encoding='utf-8'))
    c=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
    assert run['status']=='passed' and run['script_sha256']==c['script_sha256']==sha(ROOT/'Modeling/scripts/modeling_close_analysis.py')
    assert run['contract_sha256']==sha(OUT/'contract.json')
    for p,h in run['inputs_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in run['outputs_sha256'].items():assert sha(OUT/p)==h,p
    x=pd.read_csv(OUT/'logit_contributions.csv',encoding='utf-8-sig',parse_dates=['timestamp'])
    all_terms=['calendar_contribution','power_contribution','production_contribution','state_contribution']
    for r in x.itertuples():
        summed=math.fsum([r.intercept]+[getattr(r,n) for n in all_terms])
        assert abs(summed-r.logit)<1e-9
        assert abs(1/(1+math.exp(-summed))-r.score)<1e-9
        assert r.alarm==int(r.score>=run['oof_threshold'])
    p=pd.read_csv(OUT/'reproduced_predictions.csv',encoding='utf-8-sig',parse_dates=['timestamp'])
    old=pd.read_csv(ROOT/'Modeling/tables/regime_age_ablation/followup_predictions.csv',encoding='utf-8-sig',parse_dates=['timestamp'])
    joined=p.merge(old,on=['timestamp','variant'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(joined)==4032
    np.testing.assert_allclose(joined.prediction_new,joined.prediction_old,atol=1e-8,rtol=1e-9)
    metrics=[]
    for (variant),g in p.groupby('variant'):
        for cond,gg in [('all',g),('up_start',g.loc[g.sustained_onset==1])]:
            errors=[r.prediction-r.actual for r in gg.itertuples()]
            metrics.append({'variant':variant,'condition':cond,'mae':math.fsum(abs(e) for e in errors)/len(errors)})
    refs=pd.read_csv(ROOT/'Modeling/tables/regime_age_ablation/followup_metrics.csv',encoding='utf-8-sig')
    for row in metrics:
        expected=float(refs.loc[(refs.variant==row['variant'])&(refs.condition==row['condition'])&(refs.period=='Jul-Aug'),'mae'].iloc[0])
        assert math.isclose(row['mae'],expected,abs_tol=1e-8)
    e=pd.read_csv(OUT/'event_explanations.csv',encoding='utf-8-sig')
    assert len(e)==13 and e.alarm.sum()==7
    assert ((e.alarm==1)&(e.absolute_error_gain>0)).sum()==4
    assert ((e.alarm==1)&(e.absolute_error_gain<0)).sum()==3
    fig=json.loads((OUT/'figures.json').read_text(encoding='utf-8'))
    assert fig['script_sha256']==sha(ROOT/'Modeling/scripts/modeling_close_figures.py')
    for n,h in fig['figures_sha256'].items():assert sha(ROOT/n)==h,n
    docs=json.loads((OUT/'documents.json').read_text(encoding='utf-8'))
    assert docs['script_sha256']==sha(ROOT/'Modeling/scripts/modeling_close_documents.py')
    for n,h in docs['outputs_sha256'].items():assert sha(ROOT/n)==h,n
    assert docs['original_manuscript_sha256']==sha(OUT/'manuscript_before.md')
    man=(ROOT/'Modeling/04_Modeling_원고.md').read_text(encoding='utf-8')
    before=(OUT/'manuscript_before.md').read_text(encoding='utf-8')
    clean=re.sub(r'<!-- MODELING_CLOSE_BEGIN -->.*?<!-- MODELING_CLOSE_END -->\s*','',man,flags=re.S).split('## 4.13 ')[0].strip()
    assert clean==before.strip()
    links=0
    for path in [ROOT/'Modeling/04_Modeling_원고.md',ROOT/'Modeling/04_Modeling_3페이지용_압축본.md',ROOT/'Modeling/README.md']:
        text=path.read_text(encoding='utf-8')
        if path.name=='README.md':text=text.split('## S06 실행 결과:')[0]
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#')):continue
            linked=(path.parent/target.split('#')[0].strip('<>')).resolve()
            if linked != (OUT/'verification.json').resolve():
                assert linked.exists(),target
            links+=1
    compact=(ROOT/'Modeling/04_Modeling_3페이지용_압축본.md').read_text(encoding='utf-8')
    assert len(re.findall(r'!\[',compact))==2
    assert len(re.findall(r'^## ',compact,flags=re.M))==3
    assert docs['format']=='markdown_only' and docs['target_pages']==3
    assert not docs['pagination_verified']
    for path in [ROOT/'Modeling/04_Modeling_원고.md',ROOT/'Modeling/04_Modeling_3페이지용_압축본.md']:
        assert '.pdf)' not in path.read_text(encoding='utf-8')
    assert sha(ROOT/'README.md')==c['root_readme_sha256']
    result={'status':'passed','script_sha256':sha(__file__),'run_sha256':sha(OUT/'run.json'),
            'documents_sha256':sha(OUT/'documents.json'),'figures_manifest_sha256':sha(OUT/'figures.json'),
            'classification_logit_identities':len(x),'regression_predictions_checked':len(joined),
            'scalar_regression_metrics_checked':len(metrics),'historical_manuscript_preserved':True,
            'compact_target_pages':3,'pagination_verified':False,'compact_sections':3,
            'compact_figures':2,'links_checked':links,
            'visual_review':'passed' if visual else 'pending','no_new_models_selected':True}
    save_json(OUT/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--visual-reviewed',action='store_true');main(p.parse_args().visual_reviewed)
