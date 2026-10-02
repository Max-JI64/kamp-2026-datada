"""Plot aggregated EDA 2.2 curves with axes and legends only."""
from pathlib import Path
import hashlib
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
TAB=ROOT/'EDA/tables/daily_repetition'
OUT=ROOT/'EDA/figures/daily_repetition'
MANUSCRIPT=ROOT/'EDA/10.02_002_EDA_새원고.md'

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
    # Preserve monthly descriptive values for the independent source audit.
    matrix=pd.read_csv(TAB/'hourly_matrix.csv',encoding='utf-8-sig',index_col='date')
    matrix.index=pd.to_datetime(matrix.index)
    matrix.columns=matrix.columns.astype(int)
    dates=pd.read_csv(TAB/'daily_values.csv',encoding='utf-8-sig',index_col='date')
    dates.index=pd.to_datetime(dates.index)
    monthly=[]
    for (month,group),part in dates.groupby(['month','group']):
        values=matrix.loc[part.index]
        for hour in range(24):
            monthly.append({'month':month,'group':group,'hour':hour,'n':len(values),
                'median':float(values[hour].median()),'p10':float(values[hour].quantile(.1)),
                'p90':float(values[hour].quantile(.9))})
    monthly=pd.DataFrame(monthly)
    assert monthly.groupby(['month','group']).n.first().sum()==241
    monthly.to_csv(TAB/'monthly_hourly_distribution.csv',index=False,encoding='utf-8-sig')
    after=hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest();assert before==after
    (TAB/'simple_figure_manifest.json').write_text(json.dumps({
        'figure':str(target),'change':'Removed all in-plot annotations, arrows, and outcome counts. Whole-period figure: mean, median and P10-P90 only. Monthly descriptive table retained for independent validation; the separate heatmap script generates the selected monthly plot.',
        'period':'2021-01-01 through 2021-08-31 valid dates; holidays included by day of week',
        'criterion':'M12<M11 and M13>M12, same date; no minimum amplitude cutoff',
        'band':'P10-P90 across dates, not confidence or prediction interval',
        'manuscript_edited':False,'manuscript_sha256':after,'visual_review':False,
        'numeric_result_changed':False,'warnings':bad},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'figure':str(target),'warnings':bad,'manuscript_edited':False},ensure_ascii=False))

if __name__=='__main__':main()
