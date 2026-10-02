"""Restart EDA 2.3: explicit threshold provenance, shared weather bins and shape.
Generate figures and reuse/audit earlier tables; visual QA is performed separately.
"""
# %% Fixed definitions and source
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
import eda_figures as style

REPORT=Path(__file__).resolve().parents[1]
ROOT=REPORT.parent.parent
OUT=REPORT/'tables'
SLOTS=['15분','30분','45분','60분']
SOURCE=ROOT/'data/origin/okm_augumented_2021.csv'
SHA='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
THRESHOLD=187.
VARIABLES=['생산량','기온','풍속','습도','강수량']


def read(name):
    return pd.read_csv(OUT/f'{name}.csv',encoding='utf-8-sig',float_precision='round_trip')


def save(table,name):
    table.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')


def quantile_manual(values,q):
    ordered=sorted(values)
    position=(len(ordered)-1)*q
    lo=int(np.floor(position)); hi=int(np.ceil(position))
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


# %% Reuse earlier frequency/band/shape tables; check their exact source/support
def tables():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    raw=pd.read_csv(SOURCE,encoding='utf-8-sig')
    scope=raw.loc[raw['날짜'].lt(20210901)].copy()
    valid=scope.loc[scope['시간'].between(0,23)].copy()
    assert len(scope)==5832 and len(valid)==5784
    assert valid.groupby('날짜').size().eq(24).all()
    scope['P']=scope[SLOTS].max(axis=1)
    valid['P']=valid[SLOTS].max(axis=1)
    dates=pd.to_datetime(valid['날짜'].astype(str),format='%Y%m%d')
    valid['month']=dates.dt.month
    valid['weekend']=dates.dt.dayofweek.ge(5)
    valid['high']=valid.P.ge(THRESHOLD)
    threshold_rows=[]
    for population,part in [('Jan-Aug_5832',scope),('valid_hours_5784',valid)]:
        for q in [.9,.95,.99]:
            value=float(part.P.quantile(q,interpolation='linear'))
            np.testing.assert_allclose(value,quantile_manual(part.P.to_numpy(),q))
            threshold_rows.append(dict(population=population,source_n=len(part),quantile=q,
                                       threshold=value,valid_high_hours=int(valid.P.ge(value).sum()),
                                       valid_rate=valid.P.ge(value).mean()))
    thresholds=pd.DataFrame(threshold_rows)
    assert scope.P.quantile(.95)==187 and valid.P.quantile(.95)==186
    save(thresholds,'restart_peak_thresholds')
    monthly_rows=[]
    for month,part in valid.groupby('month'):
        for threshold in [186.,187.]:
            monthly_rows.append(dict(month=month,threshold=threshold,n=len(part),
                                     high_hours=int(part.P.ge(threshold).sum()),
                                     high_rate=part.P.ge(threshold).mean()))
    sensitivity=pd.DataFrame(monthly_rows)
    save(sensitivity,'restart_peak_month_sensitivity')
    monthly=read('monthly_hour_overview')
    for row in monthly.itertuples():
        part=valid.loc[valid.month.eq(row.month)&valid['시간'].eq(row.hour)]
        assert row.n==len(part) and row.high_hours==int(part.high.sum())
        np.testing.assert_allclose(row.high_rate,part.high.mean())
    assert int(valid.high.sum())==287
    bands=read('external_band_frequency')
    band_facts=json.loads((OUT/'external_manifest.json').read_text(encoding='utf-8'))
    base=valid.loc[valid.month.between(6,8)&~valid.weekend&valid['생산량'].gt(0)].copy()
    assert len(base)==1244
    for variable in VARIABLES:
        part=base.dropna(subset=[variable]).copy()
        edges=band_facts['bands'][variable]['edges']
        if variable=='강수량':
            part['band']=part[variable].gt(0).astype(int)
        else:
            np.testing.assert_allclose(edges,part[variable].quantile([.25,.5,.75]))
            part['band']=[sum(value>edge for edge in edges) for value in part[variable]]
        for row in bands.loc[bands.variable.eq(variable)].itertuples():
            cell=part.loc[part.month.eq(row.month)&part.band.eq(row.band)]
            assert row.n==len(cell) and row.high_hours==int(cell.high.sum())
            assert row.days==cell['날짜'].nunique()
            np.testing.assert_allclose(row.high_rate,cell.high.mean())
    shape=read('high_slot_records')
    expected=valid.loc[valid.high].set_index(['날짜','시간']).sort_index()
    actual=shape.set_index(['날짜','hour']).sort_index()
    np.testing.assert_array_equal(actual[SLOTS].to_numpy(),expected[SLOTS].to_numpy())
    assert len(shape)==287
    assert shape[SLOTS].ge(187).sum(axis=1).eq(shape.high_slots).all()
    assert int(shape.high_slots.ge(2).sum())==187
    assert int(shape.M.lt(187).sum())==194
    summary=read('high_slot_summary')
    for row in summary.itertuples():
        assert row.hours==int(shape.high_slots.eq(row.high_slots).sum())
    save(monthly,'restart_peak_month_hour')
    save(bands,'restart_peak_condition_rates')
    return scope,valid,thresholds,sensitivity,monthly,bands,band_facts,shape,summary


# %% Figure 1: position of threshold + alternate scope + frequency in time
def threshold_timing(scope,valid,sensitivity,monthly):
    plt=style.plt
    fig=plt.figure(figsize=(14,9))
    grid=fig.add_gridspec(2,2,height_ratios=[1,1.3])
    left=fig.add_subplot(grid[0,0]); right=fig.add_subplot(grid[0,1])
    bottom=fig.add_subplot(grid[1,:])
    for part,color,ls,label in [(scope,style.BLUE,'-','기준 산정 5,832행'),
                                (valid,style.ORANGE,'--','정상 시간 5,784행')]:
        ordered=np.sort(part.P.to_numpy())
        left.step(ordered,np.arange(1,len(ordered)+1)/len(ordered),where='post',
                  color=color,ls=ls,lw=2,label=label)
    left.axhline(.95,color=style.GRAY,ls=':',lw=1)
    left.axvline(187,color=style.ORANGE,ls='--',lw=1.2)
    left.axvspan(187,scope.P.max(),color=style.ORANGE,alpha=.10)
    left.annotate('95백분위 = 187',(187,.95),xytext=(122,.73),
                  arrowprops=dict(arrowstyle='->',color=style.ORANGE),fontsize=11,color=style.ORANGE)
    left.set(title='기준이 전체 분포에서 차지하는 위치',xlabel='시간별 최대 전력',
             ylabel='누적 비율',xlim=(0,230),ylim=(0,1.03))
    left.yaxis.set_major_formatter(PercentFormatter(1))
    left.legend(fontsize=9,loc='upper left',frameon=False)
    for threshold,color,ls in [(186,style.GRAY,'--'),(187,style.ORANGE,'-')]:
        part=sensitivity.loc[sensitivity.threshold.eq(threshold)].sort_values('month')
        right.plot(part.month,part.high_rate,color=color,ls=ls,marker='o',lw=2,
                   label=f'최대 {threshold} 이상')
        if threshold==187:
            for row in part.itertuples():
                right.annotate(f'{row.high_rate:.1%}',(row.month,row.high_rate),
                               xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    right.set(title='186·187 기준의 월별 발생 비율',xticks=range(1,9),
              xticklabels=[f'{m}월' for m in range(1,9)],ylabel='고전력 비율',ylim=(0,.20))
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.legend(fontsize=9,loc='upper left',frameon=False)
    right.grid(axis='y',alpha=.15)
    values=monthly.pivot(index='month',columns='hour',values='high_rate')
    image=bottom.imshow(values,aspect='auto',cmap='YlOrRd',vmin=0,vmax=.75)
    for i in range(8):
        for j in range(24):
            value=values.iloc[i,j]
            if value>0:
                bottom.text(j,i,f'{value:.0%}',ha='center',va='center',fontsize=8,
                            color='white' if value>.42 else '#283747')
    bottom.set(title='최대 187 이상인 시간의 비율',
               xticks=range(24),yticks=range(8),
               yticklabels=[f'{m}월' for m in range(1,9)],xlabel='시간대 (시)')
    fig.colorbar(image,ax=bottom,pad=.02,fraction=.03,format=PercentFormatter(1),label='고전력 비율')
    style.finish(fig,'restart_03_threshold_timing','고전력 기준과 발생 시간',
                 '기준 산정 5,832행 · 시간대 비교 정상 5,784행 · 187은 공식 설비 한계가 아닌 탐색 기준')


# %% Figure 2: fixed condition ranges, comparable colors, explicit denominators
def condition_ranges(bands,facts):
    fig,axes=style.plt.subplots(1,5,figsize=(17,4.8),sharey=True,
                               gridspec_kw={'width_ratios':[4,4,4,4,2]})
    for ax,variable in zip(axes,VARIABLES):
        part=bands.loc[bands.variable.eq(variable)]
        values=part.pivot(index='month',columns='band',values='high_rate').sort_index()
        n=part.pivot(index='month',columns='band',values='n').sort_index()
        high=part.pivot(index='month',columns='band',values='high_hours').sort_index()
        labels=facts['bands'][variable]['labels']
        ticks=[label.replace(', ','~').replace('(','').replace(']','')
               for label in labels]
        ticks=[tick.replace('~','~\n') for tick in ticks]
        image=ax.imshow(values,aspect='auto',cmap='YlOrRd',vmin=0,vmax=.70)
        for i in range(3):
            for j in range(values.shape[1]):
                value=values.iloc[i,j]
                ax.text(j,i,f'{value:.1%}\n{high.iloc[i,j]:.0f}/{n.iloc[i,j]:.0f}',
                        ha='center',va='center',fontsize=9.5,
                        color='white' if value>.40 else '#283747')
        ax.set(title=variable,xticks=range(values.shape[1]),xticklabels=ticks,
               yticks=range(3),yticklabels=['6월','7월','8월'])
        ax.tick_params(axis='x',labelsize=9)
        ax.set_xticks(np.arange(-.5,values.shape[1],1),minor=True)
        ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.5)
        ax.tick_params(which='minor',bottom=False,left=False)
    fig.colorbar(image,ax=axes[-1],pad=.08,format=PercentFormatter(1),label='고전력 비율')
    style.finish(fig,'restart_03_condition_ranges','조건별 고전력 비율',
                 '6~8월 생산량 양수인 평일 · 고정 통합 사분위 경계 · 강수 0/양수 · 칸은 비율과 고전력/전체 시간 수')


# %% Figure 3: actual within-hour values paired with hourly mean/maximum
def hour_shape(shape,summary):
    fig,axes=style.plt.subplots(1,2,figsize=(11,8),
                                gridspec_kw={'width_ratios':[1,1.25]},sharey=True)
    values=shape[SLOTS].to_numpy()
    image=axes[0].imshow(values,aspect='auto',cmap='YlOrRd',vmin=0,vmax=230)
    yy,xx=np.where(values>=187)
    axes[0].scatter(xx,yy,s=3,c='#16374B',marker='s')
    offset=0; centers=[]; labels=[]
    for row in summary.itertuples():
        centers.append(offset+(row.hours-1)/2)
        labels.append(f'{row.high_slots}개 구간 높음\n{row.hours}시간 ({row.share:.1%})')
        if offset:
            for ax in axes: ax.axhline(offset-.5,color='#283747',lw=1)
        offset+=row.hours
    axes[0].set(title='네 전력값 (점: 187 이상)',xticks=range(4),xticklabels=SLOTS,
                yticks=centers,yticklabels=labels)
    fig.colorbar(image,ax=axes[0],pad=.03,shrink=.6,label='전력')
    rows=np.arange(len(shape))
    axes[1].hlines(rows,shape.M,shape.P,color=style.GRAY,lw=.5,alpha=.4)
    axes[1].scatter(shape.M,rows,s=10,c=style.BLUE,label='평균 전력',alpha=.75)
    axes[1].scatter(shape.P,rows,s=10,c=style.ORANGE,label='최대 전력',alpha=.75)
    axes[1].axvline(187,color='#283747',ls='--',lw=1,label='기준 187')
    axes[1].set(title='같은 시간의 평균·최대',xlabel='전력',xlim=(0,235))
    axes[1].legend(loc='upper left',fontsize=9,frameon=False)
    style.finish(fig,'restart_03_hour_shape','고전력의 한 시간 안의 형태',
                 '고전력 287시간 모두 사용 · 순서는 높은 구간 수와 배열 기준, 시간순 아님 · 구간 수는 실제 초과시간 아님')


def main():
    result=tables()
    scope,valid,thresholds,sensitivity,monthly,bands,facts,shape,summary=result
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        font=style.setup()
        threshold_timing(scope,valid,sensitivity,monthly)
        condition_ranges(bands,facts)
        hour_shape(shape,summary)
    important=[str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not important,important
    assert len(style.GENERATED)==3 and all(item['bytes']>10000 for item in style.GENERATED)
    manifest=dict(status='passed',source_sha256=SHA,threshold=187,
                  threshold_source_rows=5832,normal_hour_rows=5784,normal_hour_p95=186,
                  quantile_rule='linear interpolation: zero-based position (n-1)*0.95',
                  threshold_reason='existing E004 upper-tail descriptive definition; not optimal/official/forecast threshold',
                  weather_method='univariate pooled quartile bins, not clusters or named weather states',
                  weather_population='June-August positive-production weekdays, 1244 rows; wind nonmissing 1242',
                  checked_rows=len(thresholds)+len(sensitivity)+len(monthly)+len(bands)+len(shape)+len(summary),
                  figures=[dict(file=item['file'],bytes=item['bytes'],sha256=hashlib.sha256((style.FIGURES/item['file']).read_bytes()).hexdigest()) for item in style.GENERATED],
                  warnings=important,font=font,image_analysis_performed=False,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'restart_peak_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in manifest.items() if key in ['status','threshold','threshold_source_rows','normal_hour_p95','checked_rows','warnings','image_analysis_performed']}))
    print(thresholds.to_json(orient='records'))
    print(sensitivity.loc[sensitivity.month.eq(7)].to_json(orient='records'))


if __name__=='__main__':
    main()