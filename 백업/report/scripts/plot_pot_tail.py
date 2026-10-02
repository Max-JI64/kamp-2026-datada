"""Scientific POT diagnostic figures and independent table verification."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from matplotlib.ticker import PercentFormatter
from PIL import Image
import eda_figures as style
from analyze_pot_tail import inputs,OUT,read,SHA,SOURCE

ROOT=Path(__file__).resolve().parents[2]
FIGURES=ROOT/'report/figures/eda/peak_definition_comparison/pot_tail'
style.FIGURES=FIGURES
BLUE,ORANGE,TEAL,GRAY=style.BLUE,style.ORANGE,style.TEAL,style.GRAY
REFS=[176,182,187,194]

def subset(scan,series):
    return scan.loc[scan.series.eq(series)&scan.gap_hours.eq(0 if series=='hourly' else 3)].sort_values('u')

def verify(scan,d,tr,event_table,boot):
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    count=0
    for u in range(145,206):
        raw=tr.loc[tr.P.gt(u)]
        # Independent vectorized grouping, checked against saved event summaries.
        mask=(raw.timestamp.diff()>pd.Timedelta(hours=3)).cumsum()
        maximum=raw.groupby(mask).P.max().to_numpy()
        e=event_table.loc[event_table.u.eq(u)]
        assert np.array_equal(maximum,e.P.to_numpy())
        assert e.exceed_hours.sum()==len(raw)
        for series,y in [('hourly',raw.P.to_numpy()-u),('event',maximum-u)]:
            row=subset(scan,series).set_index('u').loc[u]
            assert row.n==len(y) and row.unique==len(np.unique(y))
            assert np.isclose(row.mean_excess,np.mean(y))
            if row.finite_fit:
                z=np.sort(y);xi=row['shape'];sigma=row['scale']
                t=1+xi*z/sigma
                assert (t>0).all(),(series,u)
                cdf=-np.expm1(-np.log1p(xi*z/sigma)/xi) if abs(xi)>1e-9 else -np.expm1(-z/sigma)
                n=len(z);distance=max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n))
                assert np.isclose(distance,row.ks_distance,rtol=1e-7)
                assert np.isclose(sigma-xi*u,row.modified_scale)
            count+=1
    assert boot.groupby(['series','u']).size().eq(60).all()
    result=dict(status='passed',source_sha256=SHA,independent_threshold_series_checks=count,
        event_extraction='Independent gap-based vectorized grouping; every exceedance hour accounted for',
        gpd_checks='Analytic CDF/support, KS distance and modified-scale relation; no nominal p-value',
        boot_fits=len(boot),manuscript_edited=False)
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def ldc_plot(tr,ldc):
    fig,axes=style.plt.subplots(1,2,figsize=(14,5.7))
    for ax in axes:
        ax.plot(ldc.exceedance_fraction,ldc.P,c=BLUE,lw=2)
        ax.set_xlabel('大きい値からの累積比率' if False else '高い順に並べた時間の割合')
        ax.set_ylabel('時間別最大電力' if False else '시간별 최대 전력')
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(alpha=.12)
    axes[0].set(title='1~6월 전체 부하 지속 곡선',xlim=(0,1),ylim=(0,230))
    axes[1].set(title='높은 전력 구간 확대',xlim=(0,.35),ylim=(130,230))
    for u,color in zip(REFS,[BLUE,TEAL,ORANGE,GRAY]):
        rate=float(tr.P.gt(u).mean())
        axes[1].axhline(u,color=color,ls=':',lw=1,alpha=.7)
        axes[1].scatter(rate,u,c=color,s=45,zorder=4)
        axes[1].text(.343,u+1,f'{u} 초과 · {rate:.2%}',ha='right',va='bottom',color=color,fontsize=10)
    style.finish(fig,'01_load_duration_tail','','')

def excess_plot(scan,boot_summary,dependence):
    fig,axes=style.plt.subplots(2,2,figsize=(14,9))
    for ax,series,label,color in zip(axes[0],['hourly','event'],['초과한 모든 시간','초과 사건의 최대값 · 3시간 기준'],[BLUE,ORANGE]):
        p=subset(scan,series);b=boot_summary.loc[boot_summary.series.eq(series)].sort_values('u')
        ax.plot(p.u,p.mean_excess,'o-',ms=3,lw=1.4,c=color)
        ax.fill_between(b.u,b.mean_excess_low,b.mean_excess_high,color=color,alpha=.15)
        ax.set(title=f'평균 초과량: {label}',xlabel='후보 기준 u',ylabel='기준을 넘은 값의 평균 초과량',xlim=(145,205),ylim=(0,40))
        for u in REFS:ax.axvline(u,c=GRAY,ls=':',lw=.7,alpha=.5)
        ax.grid(alpha=.13)
    ax=axes[1,0]
    for gap,color in zip([1,3,6,24],[BLUE,ORANGE,TEAL,GRAY]):
        p=scan.loc[scan.series.eq('event')&scan.gap_hours.eq(gap)]
        ax.plot(p.u,p.n,c=color,lw=1.7,label=f'{gap}시간 이내 초과값 연결')
    ax.set(title='기준·사건 연결 간격에 따라 남는 사건 수',xlabel='후보 기준 u',ylabel='사건 수',xlim=(145,205))
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.13)
    ax=axes[1,1]
    for u,color in zip([176,182,187],[BLUE,TEAL,ORANGE]):
        p=dependence.loc[dependence.u.eq(u)]
        ax.plot(p.lag_hours,p.autocorrelation,c=color,label=f'{u} 초과 여부')
    ax.axhline(0,c=GRAY,lw=.7);ax.axvline(24,c=GRAY,ls=':',lw=1)
    ax.set(title='초과 여부에는 하루 주기의 반복도 남는다',xlabel='시간 간격',ylabel='초과 여부의 자기상관',xlim=(1,48),xticks=[1,6,12,24,36,48])
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.13)
    style.finish(fig,'02_mean_excess_and_events','','')

def parameters_plot(scan,bs):
    fig,axes=style.plt.subplots(2,2,figsize=(14,8.8))
    for col,(series,label,color) in enumerate([('hourly','시간별 초과값',BLUE),('event','사건 최대값 · 3시간',ORANGE)]):
        p=subset(scan,series);p=p.loc[p.u.between(160,194)]
        b=bs.loc[bs.series.eq(series)&bs.u.between(160,194)].sort_values('u')
        for row,metric,ylabel in [(0,'shape','꼬리 모양 ξ'),(1,'modified_scale','기준을 보정한 규모 σ(u) − ξu')]:
            ax=axes[row,col]
            ax.plot(p.u,p[metric],'o-',ms=3,c=color,lw=1.5,label='1~6월 적합')
            ax.fill_between(b.u,b[f'{metric}_low'],b[f'{metric}_high'],color=color,alpha=.16,
                            label='주별 재표본 2.5~97.5% 범위')
            for u in REFS:ax.axvline(u,c=GRAY,ls=':',lw=.7,alpha=.5)
            ax.set(title=f'{label}: {"꼬리 모양" if row==0 else "규모"} · 160~194 확대',
                   xlabel='후보 기준 u',ylabel=ylabel,xlim=(160,194))
            ax.grid(alpha=.13)
            if row==0:ax.axhline(0,c=GRAY,lw=.7)
        axes[0,col].legend(frameon=False,fontsize=9,loc='lower right')
    style.finish(fig,'03_gpd_parameter_stability','','')

def qq_plot(d,tr,scan):
    fig,axes=style.plt.subplots(2,2,figsize=(13,9))
    # All QQ quantiles use the SAME Jan-Jun fitted parameters at each fixed u.
    for ax,u in zip(axes.flat,REFS):
        row=subset(scan,'hourly').set_index('u').loc[u]
        highest=u+1;lowest=u
        for p,label,color in [(tr,'1~6월',BLUE),(d.loc[d.month.gt(6)],'7~8월 · 적합 고정',ORANGE)]:
            observed=np.sort(p.loc[p.P.gt(u),'P'].to_numpy());n=len(observed)
            model=u+genpareto.ppf((np.arange(n)+.5)/n,row['shape'],scale=row['scale'])
            ax.scatter(model,observed,s=19,c=color,alpha=.65,label=f'{label} · {n}시간')
            highest=max(highest,model.max(),observed.max())
        ax.plot([lowest,highest+2],[lowest,highest+2],c=GRAY,ls='--',lw=1)
        ax.set(title=f'{u} 초과: 모형 분위수와 실제 전력 비교',xlabel='1~6월 GPD가 설명하는 전력 분위수',
               ylabel='실제 전력 분위수',xlim=(lowest,highest+2),ylim=(lowest,highest+2))
        ax.legend(frameon=False,fontsize=9,loc='upper left');ax.grid(alpha=.13)
    style.finish(fig,'04_fixed_model_qq','','')

def run():
    style.setup();d,tr,_=inputs();scan=read('threshold_scan');bs=read('bootstrap_summary')
    result=verify(scan,d,tr,read('events_gap3'),read('weekly_resampling'))
    ldc_plot(tr,read('load_duration'));excess_plot(scan,bs,read('exceedance_dependence'))
    parameters_plot(scan,bs);qq_plot(d,tr,scan)
    thumbs=[]
    for row in style.GENERATED:
        with Image.open(FIGURES/row['file']) as im:
            thumbs.append(im.resize((im.width//2,im.height//2),Image.Resampling.LANCZOS).copy())
    w=max(im.width for im in thumbs);h=max(im.height for im in thumbs)
    sheet=Image.new('RGB',(w*2,h*2),'#EBEEF1')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*w,(i//2)*h))
    sheet.save(FIGURES/'contact_50.png')
    metadata=dict(status='passed',figures=style.GENERATED,style_checks=style.STYLE_CHECKS,
        manuscripts_edited=False,visual_review=False,
        parameter_view='160..194 zoom; complete fits145..205 saved, at u200 only12 events with nonregular boundary fit',
        bands='60 calendar-week block resamples; diagnostic empirical ranges, not independent-sample confidence intervals',
        QQ_application='All Jul-Aug QQ calculations use unchanged Jan-Jun fits',verification=result)
    (FIGURES/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(result,figures=[r['file'] for r in style.GENERATED]),ensure_ascii=False))

if __name__=='__main__':run()
