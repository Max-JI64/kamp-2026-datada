"""Monthly weekday/weekend summary heatmap: eight rows per panel, no annotations."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[2]
TAB=ROOT/'EDA/tables/daily_repetition'
OUT=ROOT/'EDA/figures/daily_repetition'
MANUSCRIPT=ROOT/'EDA/02_EDA_원고.md'

def main(statistic='mean'):
    before=hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest()
    source=TAB/('independent_monthly_cells.csv' if statistic=='mean' else 'monthly_hourly_distribution.csv')
    d=pd.read_csv(source,encoding='utf-8-sig')
    if statistic=='mean':
        d=d.rename(columns={'n_raw':'n','mean_raw':'value'})
        reference=pd.read_csv(TAB/'hourly_distribution.csv',encoding='utf-8-sig').set_index(['group','hour'])
        for (group,hour),part in d.groupby(['group','hour']):
            combined=np.average(part['value'],weights=part['n'])
            assert np.isclose(combined,reference.loc[(group,hour),'mean'],rtol=1e-12,atol=1e-12)
    else:
        d=d.rename(columns={'median':'value'})
    assert len(d)==384 and not d.duplicated(['month','group','hour']).any()
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
        'font.size':12,'axes.titlesize':15,'axes.labelsize':12,
        'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(13.6,4.6),layout='constrained')
    norm=Normalize(vmin=0,vmax=210)
    for ax,group,label in zip(axes,['weekday','weekend'],['평일','주말']):
        p=d.loc[d.group.eq(group)].pivot(index='month',columns='hour',values='value')
        assert list(p.index)==list(range(1,9)) and list(p.columns)==list(range(24))
        assert not p.isna().any().any()
        assert p.to_numpy().min()>=norm.vmin and p.to_numpy().max()<=norm.vmax
        im=ax.imshow(p.to_numpy(),aspect='auto',interpolation='nearest',cmap='YlOrRd',norm=norm)
        ax.set(title=label,xlabel='시간대 (시)',ylabel='월',xticks=[0,3,6,9,12,15,18,21,23],
            yticks=np.arange(8),yticklabels=[f'{month}월' for month in range(1,9)])
        ax.set_yticks(np.arange(.5,7.5,1),minor=True)
        ax.grid(which='minor',axis='y',color='white',lw=1)
        ax.tick_params(which='minor',left=False)
        for spine in ax.spines.values():spine.set_visible(False)
    colorbar=fig.colorbar(im,ax=axes,shrink=.9,pad=.025,ticks=[0,50,100,150,200])
    colorbar.set_label('평균 전력' if statistic=='mean' else '시간 평균 전력의 중앙값')
    assert fig._suptitle is None and not fig.texts and all(not ax.texts for ax in axes)
    output=OUT/('07_monthly_power_mean_heatmap.png' if statistic=='mean' else '06_monthly_power_heatmap.png')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');fig.savefig(output,dpi=180,bbox_inches='tight')
    bad=[str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not bad,bad
    plt.close(fig)
    after=hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest();assert before==after
    manifest={
        'question':'Does month comparison add information beyond the full-period weekday/weekend curve?',
        'previous_evidence':'Monthly small-multiple curves show differences, but occupy eight panels. User requests assessment and an eight-row monthly heatmap if informative.',
        'decision':'Retain a compact monthly comparison candidate; replace eight-panel monthly line figure as preferred candidate with two eight-row summary heatmaps.',
        'reasons':['Weekday 08 medians: June 171.5, July 187.5.','Weekend 10 medians: January 85.5, March 91.5, February 24, July 23, August 21. Whole-period curves conceal this heterogeneity.'],
        'aggregation':'Arithmetic mean of CSV hourly 平均 over all dates in each month/day-type/hour, including zero-production dates.' if statistic=='mean' else 'Median over dates separately for each month, day type and hour; CSV hourly 平均 used.',
        'statistic':statistic,
        'rows_per_panel':8,'columns_per_panel':24,'checked_cells':384,'color_range':[0,210],
        'month_group_date_counts':d.groupby(['month','group']).n.first().reset_index().to_dict('records'),
        'limit':'Weekend monthly summaries contain only 8-10 dates and may shift with date composition. Descriptive period differences, not proof of a seasonal/weather cause or model gain.',
        'next_or_stop':'Current presentation question answered: preserve monthly differences compactly. Do not add further monthly plots without a new question. Production composition and other-variable explanations belong to separate analysis.',
        'source_file':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'figure':str(output),'warnings':bad,'visual_review':False,
        'manuscript_sha256':after,'manuscript_edited':False}
    if statistic=='mean':
        manifest['question']='How do monthly average levels and hourly shapes differ within weekdays and weekends?'
        manifest['previous_evidence']='User chose an arithmetic mean view after verifying that low July/August weekend values are genuine and discussing the mean-versus-median purpose.'
        manifest['decision']='Use a mean heatmap as the primary monthly comparison candidate; retain the median version as a supplementary calculation.'
        manifest['reasons']=['Consistent with section 2.1 arithmetic means.', 'August weekday 08: mean 142.6818 versus median 172.5; retain genuine low-power dates.']
        manifest['validation']='384 source-derived means retained; weighted recombination over months reconciled all 48 full-period day-type/hour means with section 2.1; common untruncated color range 0..210.'
        manifest['proposed_section_order']=['2.1 weekday/weekend daily pattern','2.2 monthly power changes','2.3 production/weather relationships']
    (TAB/('monthly_mean_heatmap_manifest.json' if statistic=='mean' else 'monthly_heatmap_manifest.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'figure':str(output),'cells':384,'warnings':bad,'manuscript_edited':False},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--statistic',choices=['mean','median'],default='mean')
    main(parser.parse_args().statistic)
