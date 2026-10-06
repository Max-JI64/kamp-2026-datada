"""Restart EDA section 2.1 only: weekday/weekend daily production/power lines.

Reuse hourly_overview; audit its 48 selected means against the supplied CSV.
Generate the two-panel section 2.1 PNG; image review is separate.
"""
# %% Source and scope
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import eda_figures as style

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'EDA'
SOURCE = ROOT/'data/origin/okm_augumented_2021.csv'
SHA = '8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'


def main():
    # %% Reuse existing summaries; independently check selected means and counts
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    slots = ['15분','30분','45분','60분']
    raw = pd.read_csv(SOURCE,encoding='utf-8-sig',usecols=['날짜','시간','생산량','평균',*slots])
    valid = raw.loc[raw['날짜'].lt(20210901) & raw['시간'].between(0,23)].copy()
    assert len(valid) == 5784 and valid['날짜'].nunique() == 241
    assert valid.groupby('날짜').size().eq(24).all()
    np.testing.assert_array_equal(valid['평균'],np.floor(valid[slots].mean(axis=1)+.5))
    valid['weekend'] = pd.to_datetime(valid['날짜'].astype(str),format='%Y%m%d').dt.dayofweek.ge(5)
    table = pd.read_csv(style.TABLES/'hourly_overview.csv',encoding='utf-8-sig',float_precision='round_trip')
    selected = table.loc[table.population.isin(['weekday','weekend']),
                         ['population','hour','n','production_mean','power_mean']].copy()
    assert len(selected) == 48
    facts = {}
    for population,weekend,days in [('weekday',False,171),('weekend',True,70)]:
        part = selected.loc[selected.population.eq(population)].sort_values('hour')
        expected = valid.loc[valid.weekend.eq(weekend)].groupby('시간').agg(
            n=('평균','size'),production_mean=('생산량','mean'),power_mean=('평균','mean'))
        np.testing.assert_array_equal(part.hour,np.arange(24))
        np.testing.assert_array_equal(part.n,np.full(24,days))
        np.testing.assert_allclose(part[['n','production_mean','power_mean']],expected,
                                   rtol=1e-12,atol=1e-12)
        facts[population] = dict(days=days,
                                 production_peak_hour=int(part.loc[part.production_mean.idxmax(),'hour']),
                                 power_peak_hour=int(part.loc[part.power_mean.idxmax(),'hour']))
    weekday=selected.loc[selected.population.eq('weekday')].set_index('hour')
    for metric in ['production_mean','power_mean']:
        assert weekday.loc[12,metric] < weekday.loc[11,metric]
        assert weekday.loc[13,metric] > weekday.loc[12,metric]
    weekend_19=valid.loc[valid.weekend & valid['시간'].eq(19)]
    assert weekend_19['생산량'].eq(0).all()
    assert round(float(weekend_19['평균'].mean()),2)==41.11
    selected.to_csv(style.TABLES/'daily_pattern_restart.csv',index=False,encoding='utf-8-sig')
    # %% One shared time axis; two independent measurement scales
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        font = style.setup()
        fig,axes = style.plt.subplots(2,1,figsize=(11.5,7.2),sharex=True)
        for ax,metric,title,label in zip(axes,['production_mean','power_mean'],
                                       ['시간대별 평균 생산량','시간대별 평균 전력'],
                                       ['평균 생산량','평균 전력']):
            for population,color,ls,name in [('weekday',style.BLUE,'-','평일 (171일)'),
                                            ('weekend',style.ORANGE,'--','주말 (70일)')]:
                part = selected.loc[selected.population.eq(population)].sort_values('hour')
                ax.plot(part.hour,part[metric],color=color,linestyle=ls,linewidth=2.4,
                        marker='o',markersize=3.5,label=name)
            ax.set(title=title,ylabel=label,ylim=(0,None))
            ax.grid(axis='y',alpha=.18,linewidth=.7)
            ax.set_axisbelow(True)
            ax.legend(loc='upper left',fontsize=10,ncol=2,frameon=False)
        axes[1].set(xticks=np.arange(24),xlim=(-.25,23.25),xlabel='시간대 (시)')
        style.finish(fig,'restart_01_daily_pattern_clean','평일과 주말의 생산량·전력 하루 패턴',
                     '2021년 1~8월 · 각 시각의 평일 171일 / 주말 70일 평균 · 공휴일 포함\n'
                     '전력은 CSV 평균 열(한 시간의 네 전력값을 평균·반올림한 값) · 두 그래프의 세로축 단위와 범위는 다름')
    important = [str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not important,important
    image_file=style.FIGURES/'restart_01_daily_pattern_clean.png'
    assert image_file.stat().st_size > 10000
    manifest=dict(status='passed',scope='Restarted EDA first section only',source_sha256=SHA,
                  checked_rows=48,checked_numeric_columns=3,groups=facts,
                  figure=image_file.name,bytes=image_file.stat().st_size,
                  figure_sha256=hashlib.sha256(image_file.read_bytes()).hexdigest(),font=font,
                  warnings=important,image_analysis_performed=False,visual_review=False,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (style.TABLES/'daily_pattern_restart_manifest.json').write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',checked_rows=48,figure=image_file.name,
                         font=font,warnings=important,image_analysis_performed=False)))


if __name__ == '__main__':
    main()
