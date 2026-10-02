"""Plot aggregated EDA 2.3 curves with axes and legends only."""
from pathlib import Path
import hashlib
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
TAB=ROOT/'report/EDA/tables/daily_repetition'
OUT=ROOT/'report/EDA/figures/daily_repetition'
MANUSCRIPT=ROOT/'report/10.02_002_EDA_새원고.md'

def main():
    before=hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest()
    data=pd.read_csv(TAB/'hourly_distribution.csv',encoding='utf-8-sig')
    counts=pd.read_csv(TAB/'group_summary.csv',encoding='utf-8-sig').set_index('group')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
        'font.size':12,'axes.titlesize':17,'axes.titleweight':'bold','axes.labelsize':13,
        'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(14.6,6.5),sharey=True,layout='constrained')
    for ax,group,label,color in zip(axes,['weekday','weekend'],['평일','주말'],['#285B8C','#C46E2C']):
        q=data.loc[data.group.eq(group)].sort_values('hour').set_index('hour')
        stat=counts.loc[group]
        assert len(q)==24 and q['n'].eq(stat['n']).all()
        ax.fill_between(q.index,q.p10,q.p90,color=color,alpha=.15,
            label='10~90백분위 범위')
        ax.plot(q.index,q['mean'],c='#4c5963',ls='--',lw=2,label='평균',zorder=3)
        ax.plot(q.index,q['median'],c=color,lw=2.6,label='중앙값',zorder=4)
        ax.set_title(label,pad=12)
        ax.set(xlabel='시간대 (시)',xlim=(0,23),ylim=(0,225),xticks=[0,3,6,9,12,15,18,21,23],
            yticks=[0,50,100,150,200])
        ax.tick_params(axis='x',labelsize=10)
        ax.grid(axis='y',alpha=.13)
        ax.legend(loc='upper left',fontsize=10,frameon=False,handlelength=2.5)
        ax.set_ylabel('시간 평균 전력')
    assert fig._suptitle is None and not fig.texts and all(not ax.texts for ax in axes)
    target=OUT/'04_daily_pattern_simple.png'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');fig.savefig(target,dpi=180,bbox_inches='tight')
    bad=[str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not bad,bad
    plt.close(fig)
    with Image.open(target) as im:
        im.resize((im.width//2,im.height//2),Image.Resampling.LANCZOS).save(OUT/'simple_contact_50.png')
    # Month-specific curves reuse the saved per-date hourly matrix. All months
    # use the same axes; month panels expose level and composition differences.
    matrix=pd.read_csv(TAB/'hourly_matrix.csv',encoding='utf-8-sig',index_col='date')
    matrix.index=pd.to_datetime(matrix.index)
    matrix.columns=matrix.columns.astype(int)
    dates=pd.read_csv(TAB/'daily_values.csv',encoding='utf-8-sig',index_col='date')
    dates.index=pd.to_datetime(dates.index)
    fig,axes=plt.subplots(2,4,figsize=(15,8.6),sharex=True,sharey=True,layout='constrained')
    monthly=[]
    for month,ax in enumerate(axes.ravel(),start=1):
        sizes={}
        for group,label,color,ls in [('weekday','평일','#285B8C','-'),('weekend','주말','#C46E2C','--')]:
            ix=dates.index[dates.month.eq(month)&dates.group.eq(group)]
            values=matrix.loc[ix]
            sizes[group]=len(values)
            lo=values.quantile(.1);hi=values.quantile(.9);median=values.median()
            ax.plot(range(24),median,color=color,lw=2.2,ls=ls,label=f'{label} 중앙값')
            for hour in range(24):
                monthly.append({'month':month,'group':group,'hour':hour,'n':len(values),
                    'median':float(median[hour]),'p10':float(lo[hour]),'p90':float(hi[hour])})
        ax.set_title(f'{month}월',fontsize=14,pad=10)
        ax.set(xlim=(0,23),ylim=(0,230),xticks=[0,6,12,18,23],yticks=[0,50,100,150,200],
            xlabel='시간대 (시)')
        ax.grid(axis='y',alpha=.12)
        if month in [1,5]:ax.set_ylabel('시간 평균 전력')
        ax.legend(loc='upper left',fontsize=9,frameon=False)
    monthly=pd.DataFrame(monthly)
    # Both calendar groups in each month must account for every valid date.
    assert monthly.groupby(['month','group']).n.first().sum()==241
    monthly.to_csv(TAB/'monthly_hourly_distribution.csv',index=False,encoding='utf-8-sig')
    month_target=OUT/'05_monthly_daily_lines.png'
    assert fig._suptitle is None and not fig.texts and all(not ax.texts for ax in axes.ravel())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');fig.savefig(month_target,dpi=180,bbox_inches='tight')
    month_bad=[str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not month_bad,month_bad
    plt.close(fig)
    with Image.open(month_target) as im:
        im.resize((im.width//2,im.height//2),Image.Resampling.LANCZOS).save(OUT/'monthly_contact_50.png')
    after=hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest();assert before==after
    (TAB/'simple_figure_manifest.json').write_text(json.dumps({
        'figure':str(target),'monthly_figure':str(month_target),'change':'Removed all in-plot annotations, arrows, and outcome counts. Whole-period figure: mean, median and P10-P90 only. Monthly figure: two aggregated median lines per month, no band or per-date rows.',
        'period':'2021-01-01 through 2021-08-31 valid dates; holidays included by day of week',
        'criterion':'M12<M11 and M13>M12, same date; no minimum amplitude cutoff',
        'band':'P10-P90 across dates, not confidence or prediction interval',
        'manuscript_edited':False,'manuscript_sha256':after,'visual_review':False,
        'numeric_result_changed':False,'warnings':bad+month_bad},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'figure':str(target),'monthly_figure':str(month_target),'warnings':bad+month_bad,'manuscript_edited':False},ensure_ascii=False))

if __name__=='__main__':main()
