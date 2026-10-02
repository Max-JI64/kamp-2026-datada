"""Threshold distribution evidence for restarted EDA. Existing threshold stays fixed."""
from pathlib import Path
import hashlib,json,warnings
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter,MaxNLocator
from PIL import Image,ImageDraw
import eda_figures as style

REPORT=Path(__file__).resolve().parents[1]
ROOT=REPORT.parent
OUT=REPORT/'tables/eda'
SOURCE=ROOT/'data/origin/okm_augumented_2021.csv'
SHA='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
SLOTS=['15분','30분','45분','60분']
RED='#AB243B'
LEVELS=[(179,style.TEAL,'90백분위: 179'),(187,style.ORANGE,'기존 95백분위: 187'),(201,RED,'99백분위: 201')]

def save(table,name):
    table.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')

def load():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    data=pd.read_csv(SOURCE,encoding='utf-8-sig')
    scope=data.loc[data['날짜'].between(20210101,20210831)].copy()
    valid=scope.loc[scope['시간'].between(0,23)].copy()
    assert len(scope)==5832 and len(valid)==5784
    for part in [scope,valid]:
        part['P']=part[SLOTS].max(axis=1)
        assert part.P.notna().all() and np.equal(part.P,np.floor(part.P)).all()
    valid['timestamp']=pd.to_datetime(valid['날짜'].astype(str),format='%Y%m%d')+pd.to_timedelta(valid['시간'],unit='h')
    assert valid.timestamp.is_unique and valid.timestamp.dt.hour.eq(valid['시간']).all()
    valid=valid.sort_values('timestamp')
    previous=pd.read_csv(OUT/'restart_peak_thresholds.csv',encoding='utf-8-sig')
    for row in previous.itertuples():
        part=scope if row.population=='Jan-Aug_5832' else valid
        assert part.P.quantile(row.quantile,interpolation='linear')==row.threshold
        assert int(valid.P.ge(row.threshold).sum())==row.valid_high_hours
    maximum=int(scope.P.max())
    values=np.arange(0,maximum+1)
    frequency=pd.DataFrame(dict(power=values))
    frequency['scope_n']=scope.P.value_counts().reindex(values,fill_value=0).to_numpy()
    frequency['valid_n']=valid.P.value_counts().reindex(values,fill_value=0).to_numpy()
    frequency['scope_share']=frequency.scope_n/len(scope)
    frequency['valid_share']=frequency.valid_n/len(valid)
    assert frequency.scope_n.sum()==5832 and frequency.valid_n.sum()==5784
    save(frequency,'restart_peak_distribution_frequency')
    curve=pd.DataFrame([dict(threshold=t,n=int(valid.P.ge(t).sum()),
                            share=float(valid.P.ge(t).mean())) for t in values])
    for t in [179,186,187,201]:
        assert curve.loc[curve.threshold.eq(t),'n'].iloc[0]==int(valid.P.ge(t).sum())
    assert [int(valid.P.ge(t).sum()) for t in [179,186,187,201]]==[598,318,287,65]
    assert int(valid.P.eq(186).sum())==31
    save(curve,'restart_peak_threshold_curve')
    rows=[]
    for quantile in np.arange(90,100):
        for population,part in [('Jan-Aug_5832',scope),('valid_hours_5784',valid)]:
            t=float(part.P.quantile(quantile/100,interpolation='linear'))
            rows.append(dict(quantile=quantile,population=population,threshold=t,
                             valid_hours=int(valid.P.ge(t).sum())))
    save(pd.DataFrame(rows),'restart_peak_quantile_sensitivity')
    months=[]
    for month,part in valid.groupby(valid.timestamp.dt.month):
        for t in [179,186,187,201]:
            months.append(dict(month=month,threshold=t,n=len(part),
                               high_hours=int(part.P.ge(t).sum()),rate=float(part.P.ge(t).mean())))
    save(pd.DataFrame(months),'restart_peak_four_threshold_months')
    save(valid[['날짜','시간','timestamp','P']],'restart_peak_distribution_points')
    return scope,valid,frequency,curve,maximum

def reference(ax,normal=True):
    for value,color,label in LEVELS:
        ax.axvline(value,c=color,lw=1.5,ls='--',label=label)
    if normal:
        ax.axvline(186,c=style.GRAY,lw=1.3,ls=':',label='정상 기록 95백분위: 186')

def distribution(scope,valid,frequency,curve,maximum):
    fig=style.plt.figure(figsize=(14,8))
    grid=fig.add_gridspec(2,2,height_ratios=[1,1.1])
    whole=fig.add_subplot(grid[0,:])
    tail=fig.add_subplot(grid[1,0])
    sensitivity=fig.add_subplot(grid[1,1])
    whole.bar(frequency.power,frequency.valid_share,width=1,color=style.BLUE,
              alpha=.7,label='정상 시간 5,784행')
    whole.step(frequency.power,frequency.scope_share,where='mid',color=style.ORANGE,
               lw=1.1,label='기준 산정 5,832행')
    whole.axvspan(187,maximum+.5,color=style.ORANGE,alpha=.10)
    reference(whole,normal=False)
    whole.set(title='시간별 최대 전력의 전체 분포',xlabel='시간별 최대 전력',
              ylabel='전체 기록 중 비율',xlim=(-1,maximum+3))
    whole.yaxis.set_major_formatter(PercentFormatter(1))
    whole.legend(loc='upper right',fontsize=9,frameon=True,ncol=2,facecolor='white',framealpha=1)
    whole.grid(axis='y',alpha=.12)
    colors=np.select([frequency.power.ge(201),frequency.power.ge(187),frequency.power.ge(179)],
                     [RED,style.ORANGE,style.TEAL],default=style.BLUE)
    tail.bar(frequency.power,frequency.valid_n,width=.85,color=colors,alpha=.85)
    reference(tail)
    tail.set(title='높은 값 구간 확대 (정상 기록)',xlabel='시간별 최대 전력',
             ylabel='해당 값을 기록한 시간 수',xlim=(155,maximum+3))
    tail.set_ylim(0,max(frequency.loc[frequency.power.ge(155),'valid_n'])*1.36)
    tail.legend(loc='upper right',fontsize=8.8,frameon=False)
    tail.grid(axis='y',alpha=.12)
    part=curve.loc[curve.threshold.between(170,210)]
    sensitivity.step(part.threshold,part.n,where='post',c=style.BLUE,lw=2)
    for value,color,label in [(179,style.TEAL,'179: 598시간'),(186,style.GRAY,'186: 318시간'),
                               (187,style.ORANGE,'187: 287시간'),(201,RED,'201: 65시간')]:
        n=int(curve.loc[curve.threshold.eq(value),'n'].iloc[0])
        sensitivity.scatter(value,n,c=color,s=35,zorder=3)
        position={179:(172,950),186:(173,430),187:(195,350),201:(203,150)}[value]
        sensitivity.annotate(label,(value,n),xytext=position,color=color,fontsize=10,
                             arrowprops=dict(arrowstyle='-',color=color,lw=1))
    sensitivity.set(title='기준값을 움직이면 포함되는 시간 수는?',xlabel='고전력으로 분류할 최소 전력',
                    ylabel='기준 이상인 시간 수',xlim=(170,210),ylim=(0,int(part.n.max())*1.1))
    sensitivity.grid(axis='y',alpha=.12)
    style.finish(fig,'restart_03_peak_distribution','최대 전력 분포와 기준 민감도',
                 '정상 5784행과 기존 산정 5832행; 값 1 간격, 정상 전체 포함')

def timeline(valid,maximum):
    fig,axes=style.plt.subplots(2,1,figsize=(14,7.8),sharex=True,
                                gridspec_kw={'height_ratios':[1.25,1]})
    categories=[(valid.P.lt(179),style.GRAY,'179 미만',6,.26),
                (valid.P.ge(179)&valid.P.lt(187),style.TEAL,'179 이상~187 미만',12,.6),
                (valid.P.ge(187)&valid.P.lt(201),style.ORANGE,'187 이상~201 미만',18,.8),
                (valid.P.ge(201),RED,'201 이상',22,.9)]
    for ax in axes:
        for mask,color,label,size,alpha in categories:
            part=valid.loc[mask]
            ax.scatter(part.timestamp,part.P,c=color,s=size,alpha=alpha,edgecolors='none',
                       label=f'{label} ({len(part):,}시간)',rasterized=True)
        for value,color,label in LEVELS:
            ax.axhline(value,c=color,ls='--',lw=1)
        ax.set_ylabel('시간별 최대 전력')
        ax.grid(axis='y',alpha=.12)
    axes[0].set(title='시간순 최대 전력과 높은 값의 위치',ylim=(-4,maximum+22))
    axes[0].legend(loc='upper left',ncol=2,fontsize=9,frameon=True,facecolor='white')
    axes[1].set(title='같은 기록의 높은 값 구간 확대',ylim=(170,maximum+22),xlabel='날짜')

    axes[1].legend(loc='upper left',ncol=2,fontsize=9,frameon=True,facecolor='white')
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
    axes[1].set_xlim(pd.Timestamp('2021-01-01'),pd.Timestamp('2021-09-01'))
    style.finish(fig,'restart_03_peak_timeline','높은 최대 전력의 시간상 분포',
                 '정상 5784시간 전부 산점도; 선 연결 없음; 시간 오류 48행은 특정 시각에 배정하지 않음')

def monthly_distribution(valid,maximum):
    fig,axes=style.plt.subplots(2,4,figsize=(15,7.2),sharex=True,sharey=True)
    values=np.arange(155,maximum+1)
    parts=[]
    for month in range(1,9):
        part=valid.loc[valid.timestamp.dt.month.eq(month)]
        counts=part.P.value_counts().reindex(values,fill_value=0)
        parts.append((month,part,counts/len(part)))
    ymax=max(float(shares.max()) for _,_,shares in parts)*1.35
    for ax,(month,part,shares) in zip(axes.flat,parts):
        colors=np.select([values>=201,values>=187,values>=179],[RED,style.ORANGE,style.TEAL],default=style.BLUE)
        ax.bar(values,shares,width=.85,color=colors,alpha=.85)
        for value,color,label in LEVELS:
            ax.axvline(value,c=color,ls='--',lw=1)
        ax.set(title=f'{month}월 · 정상 {len(part)}시간',xlim=(154,maximum+3),ylim=(0,ymax))
        high=int(part.P.ge(187).sum())
        ax.text(.97,.95,f'187 이상: {high}/{len(part)}\n({high/len(part):.2%})',
                transform=ax.transAxes,ha='right',va='top',fontsize=10,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks([160,180,200,220])
        ax.grid(axis='y',alpha=.12)
    for ax in axes[:,0]:
        ax.set_ylabel('해당 월 전체 시간 중 비율')
    for ax in axes[1,:]:
        ax.set_xlabel('시간별 최대 전력')
    style.finish(fig,'restart_03_peak_month_distributions','월별 최대 전력 상위 구간 분포',
                 '정상 전체 월별 분모, 공통 축, 값 1 간격; 확대 범위 155 이상')

def contact():
    names=[item['file'] for item in style.GENERATED]
    panels=[]
    for name in names:
        with Image.open(style.FIGURES/name) as source:
            panels.append((name,source.convert('RGB').resize((source.width//2,source.height//2),Image.Resampling.LANCZOS)))
    sheet=Image.new('RGB',(max(im.width for _,im in panels)+32,sum(im.height+32 for _,im in panels)+16),'white')
    draw=ImageDraw.Draw(sheet)
    y=16
    for name,im in panels:
        draw.text((16,y),name,fill='#202020')
        sheet.paste(im,(16,y+18));y+=im.height+32
    qa=style.FIGURES/'qa'
    qa.mkdir(exist_ok=True)
    sheet.save(qa/'restart_03_distribution_contact_50.png')

def main():
    scope,valid,frequency,curve,maximum=load()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        font=style.setup()
        distribution(scope,valid,frequency,curve,maximum)
        timeline(valid,maximum)
        monthly_distribution(valid,maximum)
    important=[str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not important,important
    assert len(style.GENERATED)==3
    contact()
    manifest=dict(status='passed',source_sha256=SHA,normal_rows=len(valid),threshold_source_rows=len(scope),
        figures=[dict(file=item['file'],sha256=hashlib.sha256((style.FIGURES/item['file']).read_bytes()).hexdigest(),bytes=item['bytes']) for item in style.GENERATED],
        frequency_bin='each exact integer value; width 1; no smoothing/truncation in whole distribution',
        fixed_thresholds=[179,186,187,201],font=font,warnings=important,visual_review=False,
        threshold_changed=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'restart_peak_distribution_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',normal_rows=len(valid),maximum=maximum,figures=3,warnings=important)))
    print(frequency.loc[frequency.power.between(179,202),['power','valid_n']].to_json(orient='records'))
    print(curve.loc[curve.threshold.between(184,190)].to_json(orient='records'))
    print(pd.read_csv(OUT/'restart_peak_four_threshold_months.csv',encoding='utf-8-sig').groupby('threshold').apply(lambda p:int(p.loc[p.rate.idxmax(),'month']),include_groups=False).to_json())

if __name__=='__main__':
    main()