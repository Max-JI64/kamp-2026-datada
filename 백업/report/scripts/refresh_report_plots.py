"""Apply the shared title/footer/axis-label style to all existing report PNGs.

Reuse numerical tables, regenerate plots only. No image decoding or review.
"""
from pathlib import Path
import hashlib
import json
import warnings
import pandas as pd
import matplotlib.dates as mdates
import eda_figures as style

REPORT=Path(__file__).resolve().parents[1]


def fingerprints(pattern):
    return {str(path.relative_to(REPORT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in REPORT.glob(pattern)}


def preserved_timeline():
    """Reproduce the previously generated timeline candidate from the same tables."""
    hourly=style.read('hourly_records')
    maxima=style.read('monthly_maxima')
    time=pd.to_datetime(hourly.timestamp)
    fig,axes=style.plt.subplots(2,1,figsize=(13,7.6),sharex=True,
                               gridspec_kw={'height_ratios':[2.5,1]})
    axes[0].scatter(time,hourly.P,s=5,color=style.GRAY,alpha=.18,rasterized=True)
    high=hourly.P.ge(187)
    axes[0].scatter(time[high],hourly.loc[high,'P'],s=11,color=style.ORANGE,alpha=.65)
    axes[0].axhline(187,color=style.ORANGE,ls='--',lw=.9)
    axes[0].set(ylabel='시간별 최대 전력',ylim=(0,245))
    for row in maxima.itertuples():
        axes[0].annotate(f'{row.month}월 {row.power:g}',(pd.Timestamp(row.timestamp),row.power),
                         xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    daily=hourly.assign(date=time.dt.normalize(),high=high).groupby('date').high.sum()
    daily=daily.reindex(pd.date_range('2021-01-01','2021-08-31',freq='D'))
    axes[1].plot(daily.index,daily,color=style.ORANGE,lw=1.4)
    axes[1].set(ylabel='고전력 시간 수\n(하루)',ylim=(0,None))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
    for ax in axes:
        ax.grid(axis='y',alpha=.15,linewidth=.7)
        ax.set_axisbelow(True)
        for date in ['2021-07-13','2021-07-15']:
            ax.axvspan(pd.Timestamp(date),pd.Timestamp(date)+pd.Timedelta(days=1),
                       color=style.GRAY,alpha=.18)
    style.finish(fig,'03B_peak_timeline','높은 전력값이 나타난 날짜와 월최대의 관계',
                 '위: 시간별 최대, 주황은 187 이상 / 아래: 하루의 고전력 시간 수 / 회색: 시간 오류 이틀')


def main():
    before_png=fingerprints('figures/**/*.png')
    before_csv=fingerprints('tables/**/*.csv')
    source=REPORT.parent/'data/origin/okm_augumented_2021.csv'
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    style.STYLE_CHECKS.clear()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        style.main()
        style.GENERATED.clear()
        import eda_story_figures
        eda_story_figures.main()
        style.GENERATED.clear()
        import eda_daily_pattern
        eda_daily_pattern.main()
        # Keep the earlier image path valid, while the manuscript uses the new
        # filename so desktop previews do not reuse the former image URL.
        current=style.FIGURES/'restart_01_daily_pattern_clean.png'
        earlier=style.FIGURES/'restart_01_daily_pattern.png'
        earlier.write_bytes(current.read_bytes())
        style.STYLE_CHECKS.append(dict(path=str(earlier),whole_figure_title=False,
                                       footer=False,original_value_label=False))
        preserved_timeline()
        style.GENERATED.clear()
        # This module switches the shared output path to figures/analysis.
        import analysis_figures
        analysis_figures.main()
    important=[str(w.message) for w in captured
               if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not important,important
    after_png=fingerprints('figures/**/*.png')
    assert before_png.keys()==after_png.keys(),'Unexpected change to figure inventory'
    assert before_csv==fingerprints('tables/**/*.csv'),'Numerical tables changed'
    assert source_sha==hashlib.sha256(source.read_bytes()).hexdigest(),'Source changed'
    checked={str(Path(item['path']).relative_to(REPORT)) for item in style.STYLE_CHECKS}
    assert checked==set(after_png),'Some report figures were not regenerated'
    manifest=dict(status='passed',unique_figures=len(checked),
                  changed_figures=sum(before_png[path]!=after_png[path] for path in before_png),
                  source_sha256=source_sha,numerical_tables_unchanged=True,
                  whole_figure_title=False,footer=False,original_value_axis_label=False,
                  panel_titles_and_legends_preserved=True,
                  warnings=important,image_reading=False,visual_review=False,
                  figures=[dict(file=path,sha256=sha) for path,sha in sorted(after_png.items())],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (REPORT/'figures/plot_style_manifest.json').write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in manifest.items() if key not in ['figures','script_sha256']}))


if __name__=='__main__':
    main()
