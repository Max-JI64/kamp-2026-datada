"""Post-evaluation sensitivity/diagnosis and a readable count plot; no refitting."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from a02_transition_forecast import ROOT,INPUT,OUT,load
from a01_production_changes import load as raw_load


def main():
    part=load(); _,raw=raw_load()
    # Daily profile is a retrospective repetition diagnostic only, never a feature.
    raw=raw.set_index('timestamp')
    profile_train=set(raw.loc[raw.month.le(6),'profile'])
    frame=pd.read_csv(OUT/'classification_test_predictions.csv',encoding='utf-8-sig',parse_dates=['timestamp'])
    risk=part.set_index('timestamp').loc[frame.timestamp]
    target_profile=raw.loc[frame.timestamp,'profile']
    unseen=~target_profile.isin(profile_train).to_numpy()
    rows=[]
    for group in ('calendar','level','shape'):
        for scope,mask in [('all',np.ones(len(frame),dtype=bool)),('unseen_target_daily_profile',unseen),('seen_target_daily_profile',~unseen)]:
            if mask.sum()==0 or frame.loc[mask,'onset'].nunique()<2:
                continue
            y=frame.loc[mask,'onset']; prob=frame.loc[mask,group]
            rows.append(dict(features=group,scope=scope,n=int(mask.sum()),events=int(y.sum()),
                ap=average_precision_score(y,prob),brier=brier_score_loss(y,prob),
                profile_weighted_ap=average_precision_score(y,prob,sample_weight=frame.loc[mask,'profile_weight']),
                profile_weighted_brier=brier_score_loss(y,prob,sample_weight=frame.loc[mask,'profile_weight'])))
    pd.DataFrame(rows).to_csv(OUT/'repetition_sensitivity.csv',index=False,encoding='utf-8-sig')
    # Diagnose failed experts via their training event counts and the validated
    # probabilities on actual onset vs no-onset records. No new models fitted.
    diag=[]
    for phase in ('validation','test'):
        prediction=pd.read_csv(OUT/f'power_{phase}_predictions.csv',encoding='utf-8-sig')
        for event in (0,1):
            sample=prediction.loc[prediction.known_zero.eq(1)&prediction.onset.eq(event)]
            diag.append(dict(phase=phase,event=event,n=len(sample),mean_probability=sample.onset_probability.mean(),
                mean_actual_power=sample.level.mean(),mean_mixture_prediction=sample.soft_mixture.mean(),
                mean_direct_prediction=sample.direct_shape.mean()))
    pd.DataFrame(diag).to_csv(OUT/'mixture_diagnosis.csv',index=False,encoding='utf-8-sig')
    splits=[]
    for scope,mask in [('class_train',part.month.le(4)),('class_valid',part.month.between(5,6)),('class_finaltrain',part.month.le(6)),
        ('power_validtrain',part.month.between(3,4)),('power_finaltrain',part.month.between(3,6)),('final_test',part.month.ge(7))]:
        subset=part.loc[mask]; risk=subset.loc[subset.known_zero.eq(1)]
        splits.append(dict(scope=scope,n=len(subset),risk_hours=len(risk),events=int(risk.onset.sum())))
    pd.DataFrame(splits).to_csv(OUT/'split_counts.csv',index=False,encoding='utf-8-sig')
    # One simple figure: detected true onsets and false detections, with total52
    # actual events explicitly shown. No accuracy or undefined peak criteria.
    scores=pd.read_csv(OUT/'classification_test.csv',encoding='utf-8-sig')
    scores=scores.loc[scores.scope.eq('all')].set_index('features').loc[['calendar','level','shape']]
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':12})
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    x=np.arange(3); width=.32
    a=ax.bar(x-width/2,scores.tp,width,label='실제 전환을 찾아낸 시간',color='#24788A')
    b=ax.bar(x+width/2,scores.fp,width,label='전환으로 잘못 예상한 시간',color='#D78947')
    ax.bar_label(a,padding=4); ax.bar_label(b,padding=4)
    ax.set_xticks(x,['달력만','달력 + 직전 평균 전력','달력 + 직전 평균 전력\n+ 시간 안의 변화'])
    ax.set_ylabel('시간 수'); ax.set_ylim(0,66)
    ax.axhline(52,color='#777777',linestyle=':',linewidth=1)
    ax.text(2.45,53,'실제 전환 52시간',ha='right',fontsize=10,color='#666666')
    ax.legend(loc='upper center',ncol=2,frameon=False)
    ax.spines[['top','right']].set_visible(False)
    figures=ROOT/'Analysis/figures'; figures.mkdir(exist_ok=True)
    path=figures/'a02_transition_detection.png'; fig.savefig(path,dpi=160); plt.close(fig)
    with Image.open(path) as img:
        img.resize((img.width//2,img.height//2),Image.Resampling.LANCZOS).save(figures/'a02_transition_detection_review.png')
    audit=pd.read_csv(OUT/'probability_time_audit.csv')
    assert (pd.to_datetime(audit.train_last_timestamp)<pd.to_datetime(audit.target_first_timestamp)).all()
    frozen_hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['frozen.json','power_frozen.json']}
    assert frozen_hashes['frozen.json']=='5cc22daeba82fc8f9c276c386215c8dc4021e17f0bfe0c0fd83340bb31c22b4e'
    assert frozen_hashes['power_frozen.json']=='a2f0d13883678e19d5b5625cb700cf265ca8c01ac00bde25ea8a66f2ef30229d'
    verification=dict(status='passed',frozen_conditions_unchanged=frozen_hashes,
        daily_profiles_used_only_for_posthoc_diagnostic=True,figure_path=str(path.relative_to(ROOT)),
        figure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),figure_reviewed=False,
        outputs_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['repetition_sensitivity.csv','mixture_diagnosis.csv','split_counts.csv']})
    (OUT/'review_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_json(orient='records'));print(pd.DataFrame(diag).to_json(orient='records'));print(pd.DataFrame(splits).to_json(orient='records'))


if __name__=='__main__':
    main()
