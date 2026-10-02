"""Session-only comparison figures. Do not edit EDA manuscripts."""
from pathlib import Path
import json,hashlib,warnings
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from PIL import Image,ImageDraw
import eda_figures as style

ROOT=Path(__file__).resolve().parents[3]
TABLES=ROOT/'report/EDA/tables/peak_definition_comparison'
FIGURES=ROOT/'report/EDA/figures/peak_definition_comparison'
FIGURES.mkdir(exist_ok=True)
style.FIGURES=FIGURES
BLUE,ORANGE,TEAL,GRAY=style.BLUE,style.ORANGE,style.TEAL,style.GRAY
RED='#AF304A'
ORDER=['GMM','KMeans','Q90','Q95','Q99','Legacy187','IQR','ReferenceMax','NewRecord','DailyMax','MonthlyMax']
LABELS={'GMM':'GMM 최고 집단','KMeans':'K-means 최고 집단','Q90':'90백분위 · 176 이상',
        'Q95':'95백분위 · 182 이상','Q99':'99백분위 · 194 이상','Legacy187':'기존 187 이상',
        'IQR':'IQR 상한 · 351 초과','ReferenceMax':'기준 최대 222 재도달','NewRecord':'기준 최대 222 초과',
        'DailyMax':'일최대 도달 (동점 포함)','MonthlyMax':'월최대 도달 (동점 포함)'}

def read(name):return pd.read_csv(TABLES/(name+'.csv'),encoding='utf-8-sig')

def components_figure(labels,components,selection,grid):
    fig,axes=style.plt.subplots(2,2,figsize=(14,8.2))
    train=labels.loc[labels.month.le(6)]
    values=np.arange(0,223)
    frequency=train.P.value_counts().reindex(values,fill_value=0)/len(train)
    colors=[BLUE,TEAL,GRAY,ORANGE]
    ax=axes[0,0]
    ax.bar(values,frequency,width=1,color='#D9E1E8',label='1~6월 분포')
    g=components.loc[components.scope.eq('Jan-Jun')&components.method.eq('GMM')].sort_values('rank')
    for color,row in zip(colors,g.itertuples()):
        ax.plot(grid.power,grid[f'component_{row.component}'],c=color,lw=1.8,
                label=f'{row.rank}집단 · 평균 {row.mean:.1f}')
    ax.axvline(138.45,c=ORANGE,ls='--',lw=1)
    ax.set(title='GMM: 4개 집단 · 최고 집단은 139 이상',xlim=(0,230),xlabel='시간별 최대 전력',ylabel='분포 밀도')
    ax.legend(loc='upper right',fontsize=9,frameon=False)
    ax=axes[0,1]
    k=components.loc[components.scope.eq('Jan-Jun')&components.method.eq('KMeans')].sort_values('rank')
    centers=k['mean'].to_numpy();bounds=(centers[:-1]+centers[1:])/2
    rank=np.argmin(abs(values[:,None]-centers),axis=1)
    palette=np.array([BLUE,TEAL,ORANGE])
    ax.bar(values,frequency,width=1,color=palette[rank],alpha=.75)
    for rank_id,(mu,color) in enumerate(zip(centers,palette),1):
        ax.scatter(mu,.052,c=color,marker='v',s=45)
        ax.text(mu,.057,f'{rank_id}집단\n평균 {mu:.1f}',ha='center',fontsize=9,color=color)
    for value in bounds:ax.axvline(value,c=GRAY,ls=':',lw=1)
    ax.set(title='K-means: 3개 집단 · 최고 집단은 142 이상',xlim=(0,230),ylim=(0,.075),
           xlabel='시간별 최대 전력',ylabel='전체 기록 중 비율')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax=axes[1,0]
    g=selection.loc[selection.method.eq('GMM')]
    difference=g.bic-g.bic.min()
    ax.plot(g.k,difference,marker='o',c=BLUE)
    ax.scatter([4],[0],c=ORANGE,s=90,zorder=4)
    ax.annotate('4개 선택',(4,0),xytext=(4.8,3),arrowprops=dict(arrowstyle='-',color=ORANGE),color=ORANGE)
    ax.set_yscale('symlog',linthresh=10)
    ax.set(title='집단 수 선택: GMM',xlabel='집단 수',ylabel='최저 BIC 대비 차이 (작을수록 좋음)',xticks=range(1,9))
    ax.grid(axis='y',alpha=.15)
    ax=axes[1,1]
    k=selection.loc[selection.method.eq('KMeans')]
    ax.plot(k.k,k.silhouette,marker='o',c=BLUE)
    ax.scatter([3],[float(k.loc[k.k.eq(3),'silhouette'].iloc[0])],c=ORANGE,s=90,zorder=4)
    ax.annotate('3개 선택',(3,.77181),xytext=(4.1,.783),arrowprops=dict(arrowstyle='-',color=ORANGE),color=ORANGE)
    ax.set(title='집단 수 선택: K-means',xlabel='집단 수',ylabel='평균 실루엣 (클수록 좋음)',xticks=range(2,9),ylim=(.65,.80))
    ax.grid(axis='y',alpha=.15)
    style.finish(fig,'01_load_groups','고부하 집단과 집단 수의 근거','1~6월 정상 기록으로 적합; 생산·날씨는 군집 입력에서 제외')

def membership_figure(labels,summary,monthly):
    fig,axes=style.plt.subplots(1,2,figsize=(16,8),gridspec_kw={'width_ratios':[1.4,1]})
    levels=np.arange(0,223)
    matrix=np.stack([labels.groupby('P')[name].mean().reindex(levels).to_numpy() for name in ORDER])
    cmap=style.plt.get_cmap('Blues').copy();cmap.set_bad('#E6E6E6')
    image=axes[0].imshow(matrix,aspect='auto',origin='upper',extent=[-.5,222.5,10.5,-.5],cmap=cmap,vmin=0,vmax=1)
    s=summary.set_index('method')
    ticks=[f'{LABELS[name]}\n{int(s.loc[name,"all_n"]):,}시간 ({s.loc[name,"all_rate"]:.2%})' for name in ORDER]
    axes[0].set(title='각 방법이 고르는 전력 범위',yticks=range(11),yticklabels=ticks,
                xlabel='시간별 최대 전력',xticks=[0,25,50,75,100,125,150,175,200,222])
    axes[0].tick_params(axis='y',labelsize=9.2)
    fig.colorbar(image,ax=axes[0],pad=.02,fraction=.035,format=PercentFormatter(1),label='같은 전력값 중 선택된 비율')
    for ax in axes:ax.set_yticks(np.arange(-.5,11,1),minor=True);ax.grid(which='minor',axis='y',c='white',lw=1.5)
    rates=monthly.pivot(index='method',columns='month',values='rate').reindex(ORDER)
    image=axes[1].imshow(rates,aspect='auto',cmap='YlOrRd',vmin=0,vmax=.5)
    for i in range(11):
        for j in range(8):
            rate=rates.iloc[i,j]
            axes[1].text(j,i,f'{rate:.1%}',ha='center',va='center',fontsize=9,
                         color='white' if rate>.29 else '#283747')
    axes[1].set(title='동일 기준을 적용한 월별 선택 비율',xticks=range(8),xticklabels=[f'{x}월' for x in range(1,9)],
                yticks=range(11),yticklabels=['']*11)
    fig.colorbar(image,ax=axes[1],pad=.03,fraction=.05,format=PercentFormatter(1),label='해당 월 전체 시간 중 비율')
    style.finish(fig,'02_definition_comparison','피크 정의별 선택 범위와 월별 패턴','전체 5784시간에 고정 적용; 일·월최대는 사후 참조 정의; 회색은 해당 전력 관측 없음')

def stability_figure(k_effect,boot,summary):
    fig,axes=style.plt.subplots(1,2,figsize=(14,5.6),gridspec_kw={'width_ratios':[1,1.3]})
    for name,color in [('GMM',ORANGE),('KMeans',BLUE)]:
        part=k_effect.loc[k_effect.method.eq(name)&k_effect.monotone]
        axes[0].plot(part.k,part.threshold,marker='o',c=color,label=name)
    axes[0].scatter([4,3],[138.45,141.4],marker='*',s=150,c=[ORANGE,BLUE],zorder=4)
    axes[0].set(title='집단 수를 바꾸면 고부하 경계도 바뀐다',xticks=range(2,9),xlabel='집단 수',
                ylabel='가장 높은 집단의 시작 전력',ylim=(75,200))
    axes[0].legend(loc='lower right',frameon=False)
    axes[0].grid(axis='y',alpha=.15)
    methods=['GMM','KMeans','Q90','Q95','Q99'];rng=np.random.default_rng(18)
    for i,name in enumerate(methods):
        part=boot.loc[boot.method.eq(name)]
        color=ORANGE if name=='GMM' else BLUE if name=='KMeans' else GRAY
        axes[1].hlines(i,part.threshold.min(),part.threshold.max(),color=color,lw=2)
        axes[1].scatter(part.threshold,i+rng.uniform(-.13,.13,len(part)),c=color,s=18,alpha=.6)
        original=float(summary.loc[summary.method.eq(name),'threshold'].iloc[0])
        axes[1].scatter([original],[i],marker='D',s=48,c='#142B3D',zorder=4)
        axes[1].text(201,i,f'{part.threshold.min():.1f}~{part.threshold.max():.1f}',va='center',fontsize=10,color=color)
    axes[1].set(title='주 단위로 다시 뽑아 적합한 경계 40개',yticks=range(5),
                yticklabels=['GMM','K-means','90백분위','95백분위','99백분위'],
                xlabel='고정 기준의 후보값',xlim=(130,221),ylim=(4.6,-.6))
    axes[1].plot([],[],marker='D',ls='None',c='#142B3D',label='원래 1~6월 자료의 기준')
    axes[1].legend(loc='lower left',frameon=False,fontsize=9)
    axes[1].grid(axis='x',alpha=.15)
    style.finish(fig,'03_boundary_stability','집단 수와 주별 재표집에 따른 경계 안정성','범위는 40회 관측 최소~최대이며 신뢰구간 아님; 매회 집단 수 재선택')

def cost_figure(labels,costs,conditions):
    fig,axes=style.plt.subplots(2,2,figsize=(14,8),gridspec_kw={'width_ratios':[1,1.2]})
    ax=axes[0,0]
    ax.plot(costs.month,costs.maximum,c=ORANGE,lw=2,marker='o',label='월별 최대 전력')
    ax.axhline(222,c=GRAY,ls='--',label='1~6월 관측 최대: 222')
    for row in costs.itertuples():ax.annotate(f'{row.maximum:.0f}',(row.month,row.maximum),xytext=(0,7),textcoords='offset points',ha='center')
    ax.set(title='비용 목적의 참조: 월별 최대 전력',xticks=range(1,9),xticklabels=[f'{x}월' for x in range(1,9)],
           ylabel='최대 전력',ylim=(190,233))
    ax.legend(loc='lower left',frameon=False,fontsize=9)
    ax=axes[1,0]
    ax.step(costs.month,costs.tariff,where='mid',c=BLUE,lw=2)
    ax.scatter(costs.month,costs.tariff,c=BLUE,s=25)
    for x,value in [(1.5,109.8),(4,167.2),(7,191.6)]:ax.text(x,value+5,f'{value:.1f}',ha='center',color=BLUE)
    ax.set(title='CSV의 계절 요금값 (실제 청구액 아님)',xticks=range(1,9),xticklabels=[f'{x}월' for x in range(1,9)],
           ylabel='전기요금(계절)',ylim=(90,215))
    names=['KMeans','GMM','Q90','Q95','Q99','Legacy187']
    for ax,variable,band,title in [(axes[0,1],'기온',3,'기온 25.9 초과 구간'),(axes[1,1],'습도',0,'습도 75 이하 구간')]:
        part=conditions.loc[conditions.variable.eq(variable)&conditions.band.eq(band)&conditions.method.isin(names)]
        rates=part.pivot(index='method',columns='month',values='rate').reindex(names)
        hits=part.pivot(index='method',columns='month',values='high_hours').reindex(names)
        ns=part.pivot(index='method',columns='month',values='n').reindex(names)
        image=ax.imshow(rates,aspect='auto',cmap='YlOrRd',vmin=0,vmax=1)
        for i in range(6):
            for j in range(3):
                value=rates.iloc[i,j]
                ax.text(j,i,f'{value:.1%}  ({hits.iloc[i,j]:.0f}/{ns.iloc[i,j]:.0f})',ha='center',va='center',fontsize=9,
                        color='white' if value>.55 else '#283747')
        ax.set(title=title,xticks=range(3),xticklabels=['6월','7월','8월'],yticks=range(6),
               yticklabels=['K-means','GMM','90백분위','95백분위','99백분위','기존 187'])
        fig.colorbar(image,ax=ax,fraction=.04,pad=.02,format=PercentFormatter(1),label='선택 비율')
    style.finish(fig,'04_cost_and_conditions','최대 전력과 계절 요금 및 조건별 정의 차이','조건 표본은 6~8월 양수 생산 평일; 비용 인과·청구액 추정 없음')

def contact():
    panels=[]
    for item in style.GENERATED:
        with Image.open(FIGURES/item['file']) as source:
            panels.append((item['file'],source.convert('RGB').resize((source.width//2,source.height//2),Image.Resampling.LANCZOS)))
    width=max(p.width for _,p in panels);height=max(p.height for _,p in panels)
    sheet=Image.new('RGB',(width*2+48,(height+30)*2+16),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,panel) in enumerate(panels):
        x=16+(i%2)*(width+16);y=16+(i//2)*(height+30)
        draw.text((x,y),name,fill='#222222');sheet.paste(panel,(x,y+18))
    sheet.save(FIGURES/'contact_50.png')

def main():
    labels=read('assignments');summary=read('comparison')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');style.setup()
        components_figure(labels,read('components'),read('model_selection'),read('fitted_distribution'))
        membership_figure(labels,summary,read('monthly_rates'))
        stability_figure(read('k_sensitivity'),read('weekly_resampling'),summary)
        cost_figure(labels,read('cost_evidence'),read('condition_rates'))
    important=[str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not important,important
    assert len(style.GENERATED)==4
    contact()
    manifest=dict(status='generated',figures=[dict(file=x['file'],sha256=hashlib.sha256((FIGURES/x['file']).read_bytes()).hexdigest()) for x in style.GENERATED],
                  warnings=important,visual_review=False,manuscript_edited=False)
    (TABLES/'figures.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(figures=4,warnings=important,manuscript_edited=False)))

if __name__=='__main__':main()
