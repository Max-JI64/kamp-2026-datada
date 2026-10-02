"""EDA 2.3 review: day-by-day hourly mean power, no manuscript authoring.

Question: is the aggregate weekday/weekend daily shape present on actual dates?
Previous evidence: 2.1 shows 11->12 down and 12->13 up on weekday means.
Existing E018 concerns slot boundaries; its counts are not hourly-mean counts.
"""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'data/origin/okm_augumented_2021.csv'
SHA='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
TAB=ROOT/'report/EDA/tables/daily_repetition'
FIG=ROOT/'report/EDA/figures/daily_repetition'
MANUSCRIPT=ROOT/'report/10.02_002_EDA_새원고.md'
BLUE='#285B8C'; ORANGE='#D07834'; GRAY='#77818a'

def json_save(name,obj):
    (TAB/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def csv_save(name,frame):
    frame.to_csv(TAB/(name+'.csv'),index=False,encoding='utf-8-sig')

def main():
    TAB.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest()
    json_save('plan',{
        'previous_evidence':'2.1 weekday/weekend hourly means: weekday 11->12 decreases and 12->13 increases. E018 counts different, 15-minute boundary values.',
        'question':'How often does the same shape occur on individual dates, and how do level and spread change across dates?',
        'scope':'Jan-Aug 2021, valid hours; weekday includes public holidays. No training/test partition.',
        'target':'CSV 平均, hourly four-slot arithmetic mean rounded half up, same target as section 2.1.',
        'criteria':'For each date: M12<M11 and M13>M12. Strict comparisons; ties explicitly distinguished. No peak threshold or inference test.',
        'outputs':'Separate weekday/weekend heatmaps with shared absolute color scale; all-date trajectories and median/P10-P90/mean; paired change scatter.',
        'follow_up_rule':'If a shape is not present every day, inspect month/day-type composition and ties. If exact daily profiles repeat, compare one-vote-per-profile summaries descriptively.',
        'manuscript_edited':False})
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    cols=['15분','30분','45분','60분']
    raw=pd.read_csv(SOURCE,encoding='utf-8-sig')
    period=raw.loc[raw['날짜'].between(20210101,20210831)]
    v=period.loc[period['시간'].between(0,23)].copy()
    v['date']=pd.to_datetime(v['날짜'].astype(str),format='%Y%m%d')
    v=v.sort_values(['date','시간'])
    assert len(v)==5784 and v.date.nunique()==241
    assert v.groupby('date').size().eq(24).all() and not v.duplicated(['date','시간']).any()
    np.testing.assert_array_equal(v['평균'],np.floor(v[cols].mean(axis=1)+.5))
    matrix=v.pivot(index='date',columns='시간',values='평균').sort_index()
    assert list(matrix.columns)==list(range(24))
    days=pd.DataFrame(index=matrix.index)
    days['group']=np.where(days.index.dayofweek<5,'weekday','weekend')
    days['month']=days.index.month
    days['mean_power']=matrix.mean(axis=1)
    days['M11']=matrix[11];days['M12']=matrix[12];days['M13']=matrix[13]
    days['drop_change']=days.M12-days.M11;days['rise_change']=days.M13-days.M12
    days['dip_rebound']=days.drop_change.lt(0)&days.rise_change.gt(0)
    days['returned_to_M11']=days.M13.ge(days.M11)
    days['positive_production_hours']=v.groupby('date')['생산량'].apply(lambda x:x.gt(0).sum())
    # Exact 96-slot identities are auxiliary record structure, not process IDs.
    slot_matrix=v.set_index(['date','시간'])[cols].unstack('시간').sort_index()
    profile_ids=pd.factorize(pd.Series([tuple(x) for x in slot_matrix.to_numpy()]))[0]
    days['profile_id']=profile_ids
    csv_save('daily_values',days.reset_index())
    csv_save('hourly_matrix',matrix.reset_index())
    dist=[];group_summary=[];monthly=[];signs=[];profiles=[];production_context=[]
    old=pd.read_csv(ROOT/'report/EDA/tables/daily_pattern_restart.csv',encoding='utf-8-sig')
    for group in ['weekday','weekend']:
        subset=days.loc[days.group.eq(group)]
        a=matrix.loc[subset.index]
        expected=old.loc[old.population.eq(group)].sort_values('hour')
        np.testing.assert_allclose(a.mean(axis=0),expected.power_mean,rtol=1e-12,atol=1e-12)
        for h in range(24):
            dist.append({'group':group,'hour':h,'n':len(a),'mean':float(a[h].mean()),
                'median':float(a[h].median()),'p10':float(a[h].quantile(.1)),
                'p90':float(a[h].quantile(.9)),'min':int(a[h].min()),'max':int(a[h].max())})
        stat={'group':group,'n':len(subset),'drop':int(subset.drop_change.lt(0).sum()),
            'rise':int(subset.rise_change.gt(0).sum()),'both':int(subset.dip_rebound.sum()),
            'both_rate':float(subset.dip_rebound.mean()),'median_drop':float(subset.drop_change.median()),
            'median_rise':float(subset.rise_change.median()),
            'ties_at_one_or_both':int((subset.drop_change.eq(0)|subset.rise_change.eq(0)).sum()),
            'returned_among_both':int((subset.dip_rebound&subset.returned_to_M11).sum())}
        group_summary.append(stat)
        for positive in [False,True]:
            part=subset.loc[subset.positive_production_hours.gt(0).eq(positive)]
            production_context.append({'group':group,'any_positive_production':positive,'n':len(part),
                'both':int(part.dip_rebound.sum()),'both_rate':float(part.dip_rebound.mean()),
                'median_drop':float(part.drop_change.median()),'median_rise':float(part.rise_change.median())})
        for m,part in subset.groupby('month'):
            monthly.append({'group':group,'month':int(m),'n':len(part),'both':int(part.dip_rebound.sum()),
                'both_rate':float(part.dip_rebound.mean()),'drop_median':float(part.drop_change.median()),
                'rise_median':float(part.rise_change.median()),'day_mean_median':float(part.mean_power.median()),
                'zero_production_days':int(part.positive_production_hours.eq(0).sum())})
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                signs.append({'group':group,'drop_sign':x,'rise_sign':y,
                    'n':int((np.sign(subset.drop_change).eq(x)&np.sign(subset.rise_change).eq(y)).sum())})
        unique=subset.drop_duplicates('profile_id')
        profiles.append({'group':group,'all_days':len(subset),'unique_profiles':len(unique),
            'all_both_rate':float(subset.dip_rebound.mean()),'unique_both':int(unique.dip_rebound.sum()),
            'unique_both_rate':float(unique.dip_rebound.mean())})
    dist=pd.DataFrame(dist);monthly=pd.DataFrame(monthly)
    csv_save('hourly_distribution',dist);csv_save('group_summary',pd.DataFrame(group_summary))
    csv_save('monthly_summary',monthly);csv_save('sign_combinations',pd.DataFrame(signs))
    csv_save('profile_sensitivity',pd.DataFrame(profiles))
    csv_save('production_context',pd.DataFrame(production_context))
    # Independent direct per-date reconstruction of strict inequalities.
    counts={'weekday':0,'weekend':0}
    for date,part in v.groupby('date'):
        mm=part.set_index('시간')['평균']
        if mm[12]<mm[11] and mm[13]>mm[12]:
            counts['weekday' if date.dayofweek<5 else 'weekend']+=1
    assert all(counts[s['group']]==s['both'] for s in group_summary)
    assert sum(s['n'] for s in group_summary)==241
    assert sum(s['n'] for s in signs)==241
    # Calendar rows preserve invalid dates as gray, not compressed away.
    calendar=pd.date_range('2021-01-01','2021-08-31',freq='D')
    full=matrix.reindex(calendar)
    missing=full.index[full.isna().all(axis=1)]
    assert list(missing.strftime('%Y-%m-%d'))==['2021-07-13','2021-07-15']

    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
        'font.size':11,'axes.titlesize':14,'axes.labelsize':12,
        'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','savefig.facecolor':'white'})
    figures=[]
    def save(fig,name):
        assert fig._suptitle is None and not fig.texts
        for ax in fig.axes:
            assert all('원자료 기록값' not in t for t in [ax.get_title(),ax.get_xlabel(),ax.get_ylabel()])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
        bad=[str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
        assert not bad,bad
        plt.close(fig);figures.append(name+'.png')

    fig,axes=plt.subplots(1,2,figsize=(12.8,10),layout='constrained')
    cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#aeb8c2')
    norm=Normalize(vmin=0,vmax=float(matrix.max().max()))
    for ax,group,label,n in zip(axes,['weekday','weekend'],['평일','주말'],[171,70]):
        ix=calendar.dayofweek<5 if group=='weekday' else calendar.dayofweek>=5
        mat=full.loc[ix]
        im=ax.imshow(np.ma.masked_invalid(mat.to_numpy(dtype=float)),aspect='auto',interpolation='nearest',cmap=cmap,norm=norm)
        ticks=[];labels=[]
        for month in range(1,9):
            rows=np.flatnonzero(mat.index.month==month)
            ticks.append(int(rows[0]));labels.append(mat.index[rows[0]].strftime('%m/%d'))
            if month>1:ax.axhline(rows[0]-.5,color='white',lw=1.2)
        title=f'{label} · 정상 {n}일' + (' / 회색: 시간 오류' if group=='weekday' else '')
        ax.set(title=title,xlabel='시간대 (시)',ylabel='날짜 (위에서 아래로 시간순)',
            xticks=[0,3,6,9,12,15,18,21,23],yticks=ticks,yticklabels=labels)
    cb=fig.colorbar(im,ax=axes,shrink=.72,pad=.02);cb.set_label('시간 평균 전력')
    save(fig,'01_date_hour_heatmap')

    fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharex=True,sharey=True,layout='constrained')
    for ax,group,label,color in zip(axes,['weekday','weekend'],['평일','주말'],[BLUE,ORANGE]):
        a=matrix.loc[days.group.eq(group)]
        for j,row in enumerate(a.to_numpy()):
            ax.plot(range(24),row,c=color,alpha=.055,lw=.55,label='날짜별 선' if j==0 else None)
        q=dist.loc[dist.group.eq(group)]
        ax.fill_between(q.hour,q.p10,q.p90,color=color,alpha=.2,label='날짜별 값의 10~90백분위 범위')
        ax.plot(q.hour,q['median'],c=color,lw=2.4,label='중앙값')
        ax.plot(q.hour,q['mean'],c='#384450',ls='--',lw=1.5,label='평균 (2.1)')
        ax.axvspan(10.8,13.2,color='#c9d0d6',alpha=.12)
        ax.set(title=f'{label} · {len(a)}일',xlabel='시간대 (시)',xticks=np.arange(0,24,3),xlim=(0,23),ylim=(0,225))
        ax.grid(axis='y',alpha=.12)
        ax.legend(loc='upper left',fontsize=8.5,frameon=False)
    axes[0].set_ylabel('시간 평균 전력')
    save(fig,'02_daily_lines_and_spread')

    fig,axes=plt.subplots(1,2,figsize=(12.5,5.4),sharex=True,sharey=True,layout='constrained')
    extent=max(days.drop_change.abs().max(),days.rise_change.abs().max())+10
    for ax,group,label,color,s in zip(axes,['weekday','weekend'],['평일','주말'],[BLUE,ORANGE],group_summary):
        part=days.loc[days.group.eq(group)]
        points=part.groupby(['drop_change','rise_change']).size().reset_index(name='n')
        ax.axvspan(-extent,0,ymin=.5,ymax=1,color=color,alpha=.065)
        ax.scatter(points.drop_change,points.rise_change,s=24+15*np.sqrt(points.n),c=color,alpha=.72,edgecolors='white',linewidths=.5)
        ax.axhline(0,c=GRAY,lw=.8);ax.axvline(0,c=GRAY,lw=.8)
        ax.text(.025,.965,f'하락 후 상승: {s["both"]}/{s["n"]}일 ({s["both_rate"]:.1%})',transform=ax.transAxes,
            ha='left',va='top',fontsize=12,color=color,fontweight='bold')
        ax.set(title=label,xlabel='11시 -> 12시 전력 변화 (음수: 하락)',xlim=(-extent,extent),ylim=(-extent,extent))
        ax.grid(alpha=.1)
    axes[0].set_ylabel('12시 -> 13시 전력 변화 (양수: 상승)')
    save(fig,'03_paired_noon_changes')
    ims=[]
    for name in figures:
        with Image.open(FIG/name) as im:
            ims.append(im.convert('RGB').resize((im.width//2,im.height//2),Image.Resampling.LANCZOS))
    contact=Image.new('RGB',(max(im.width for im in ims),sum(im.height for im in ims)+32),'white')
    y=0
    for im in ims:contact.paste(im,(0,y));y+=im.height+16
    contact.save(FIG/'contact_50.png')
    summary={'source_sha256':SHA,'scope':{'hours':5784,'normal_dates':241,'weekday':171,'weekend':70,
        'calendar_dates':len(calendar),'invalid_dates':list(missing.strftime('%Y-%m-%d'))},
        'groups':group_summary,'monthly':monthly.to_dict('records'),'profile_sensitivity':profiles,
        'production_context':production_context,
        'unique_hourly_mean_profiles':int(matrix.drop_duplicates().shape[0]),'unique_96slot_profiles':int(days.profile_id.nunique()),
        'august_zero_production_dates':list(days.loc[days.month.eq(8)&days.positive_production_hours.eq(0)].index.strftime('%Y-%m-%d')),
        'figures':figures,'verification':'passed: original hash; 24 hours per valid date; exact rounding identity; all 48 group-hour means reconciled with 2.1; direct per-date sign count; missing dates preserved',
        'quantile_band':'Across observed dates, linear-interpolated P10/P90, not a confidence interval or forecast interval.',
        'manuscript_sha256_before':before,'manuscript_sha256_after':hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest(),
        'manuscript_edited':False,'visual_review':False}
    assert summary['manuscript_sha256_after']==before
    json_save('summary',summary)
    print(json.dumps({k:summary[k] for k in ['scope','groups','production_context','profile_sensitivity','unique_hourly_mean_profiles','unique_96slot_profiles','verification','manuscript_edited']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
