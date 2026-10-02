"""Show k=3..7 group subdivisions as full and upper-range histograms."""
from pathlib import Path
import json
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from PIL import Image
import eda_figures as style
from plot_fixed_cluster_counts import read,verify

ROOT=Path(__file__).resolve().parents[3]
FIGURES=ROOT/'report/EDA/figures/peak_definition_comparison/fixed_k/histograms_3_to_7'
style.FIGURES=FIGURES
ORANGE=style.ORANGE

def plot(method,d,s,c):
    train=d.loc[d.month.le(6),'P'].to_numpy()
    edges=np.arange(-.5,226.5,2);mid=(edges[:-1]+edges[1:])/2
    total=np.histogram(train,bins=edges)[0]/len(train)
    ymax=np.ceil(total.max()/.05)*.05+.025
    fig,axes=style.plt.subplots(5,2,figsize=(15,17.5))
    palette=['#71A39B','#B4C6C1','#9DAFBD','#A08CAC','#426D95','#6B9FBA']
    for row_id,k in enumerate(range(3,8)):
        params=c.loc[c.method.eq(method)&c.k.eq(k)].sort_values('rank')
        means=params['mean'].to_numpy();x=train[:,None]
        if method=='KMeans':labels=abs(x-means).argmin(axis=1)
        else:
            sd=params['std'].to_numpy();w=params.weight.to_numpy()
            labels=(np.log(w/sd)-.5*((x-means)/sd)**2).argmax(axis=1)
        colors=palette[:k-1]+[ORANGE]
        frequencies=[np.histogram(train[labels==rank],bins=edges)[0]/len(train) for rank in range(k)]
        assert np.allclose(np.sum(frequencies,axis=0),total)
        info=s.loc[s.method.eq(method)&s.k.eq(k)].iloc[0]
        for j,ax in enumerate(axes[row_id]):
            bottom=np.zeros_like(mid)
            for rank,freq in enumerate(frequencies):
                ax.bar(mid,freq,width=2,bottom=bottom,color=colors[rank],alpha=.88,linewidth=0)
                bottom+=freq
            ax.axvline(info.boundary,color=ORANGE,lw=1.6,ls='--')
            ax.set_xlim((0,226) if j==0 else (100,226))
            ax.set_ylim((0,ymax) if j==0 else (0,.05))
            ax.set_xticks([0,50,100,150,200] if j==0 else [100,125,150,175,200,225])
            ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            if j==1:ax.set_yticks(np.arange(0,.051,.01))
            ax.set_ylabel('전체 시간 중 비율',fontsize=11)
            ax.grid(axis='y',alpha=.13)
            ax.tick_params(labelsize=10)
        left,right=axes[row_id]
        left.set_title(f'{k}개 군집 · 전체 분포',loc='left')
        right.set_title(f'{k}개 군집 · 높은 전력 구간 확대',loc='left')
        handles=[Patch(facecolor=color,alpha=.88,label=f'집단 {rank+1} · 중심 {mu:.1f}')
                 for rank,(color,mu) in enumerate(zip(colors,means))]
        left.legend(handles=handles,loc='upper right',frameon=False,ncol=2,fontsize=9.5,
                    columnspacing=1,handlelength=1.1)
        right.text(.97,.95,f'최고 집단: {int(info.min_selected)} 이상\n1~8월 {info.all_n:,}시간 · {info.all_rate:.1%}',
                   transform=right.transAxes,ha='right',va='top',color=ORANGE,fontsize=11.5,
                   bbox=dict(facecolor='white',edgecolor='none',alpha=.93,pad=4))
    for ax in axes[-1]:ax.set_xlabel('시간별 최대 전력')
    name='kmeans_histograms_3_to_7' if method=='KMeans' else 'gmm_histograms_3_to_7'
    style.finish(fig,name,'','')

def run():
    style.setup();d=read('assignments');s=read('comparison');c=read('components');boot=read('weekly_resampling')
    result=verify(d,s,c,boot)
    plot('KMeans',d,s,c);plot('GMM',d,s,c)
    thumbs=[]
    for row in style.GENERATED:
        with Image.open(FIGURES/row['file']) as im:
            thumbs.append(im.resize((im.width//2,im.height//2),Image.Resampling.LANCZOS).copy())
    contact=Image.new('RGB',(sum(im.width for im in thumbs),max(im.height for im in thumbs)),'white')
    offset=0
    for im in thumbs:contact.paste(im,(offset,0));offset+=im.width
    contact.save(FIGURES/'contact_50.png')
    manifest=dict(status='passed',k_candidates=list(range(3,8)),histogram_fit_period='Jan-Jun',fit_n=4344,
                  classification_application_period='Jan-Aug',application_n=5784,
                  bin_width=2,denominator='4344 in both full and upper zoom; zoom does not renormalize',
                  colours='mean-ordered groups, highest orange; GMM histograms use maximum-posterior assignments',
                  visual_review=False,manuscripts_edited=False,figures=style.GENERATED,style_checks=style.STYLE_CHECKS)
    (FIGURES/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',images=[row['file'] for row in style.GENERATED]),ensure_ascii=False))

if __name__=='__main__':run()
