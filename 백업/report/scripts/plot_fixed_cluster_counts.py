"""Session-only fixed-k comparison and independent membership verification."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from PIL import Image,ImageDraw
import eda_figures as style
from compare_peak_definitions import SOURCE,SHA

ROOT=Path(__file__).resolve().parents[2]
TABLES=ROOT/'report/tables/eda/peak_definition_comparison/fixed_k'
FIGURES=ROOT/'report/figures/eda/peak_definition_comparison/fixed_k'
style.FIGURES=FIGURES
BLUE,ORANGE,TEAL,GRAY=style.BLUE,style.ORANGE,style.TEAL,style.GRAY

def read(name):return pd.read_csv(TABLES/(name+'.csv'),encoding='utf-8-sig')

def verify(d,summary,components,boot):
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    assert len(d)==5784 and d.month.le(6).sum()==4344
    verified=0
    for row in summary.itertuples():
        params=components.loc[components.method.eq(row.method)&components.k.eq(row.k)].sort_values('rank')
        means=params['mean'].to_numpy();x=d.P.to_numpy()[:,None]
        if row.method=='KMeans':labels=np.argmin(abs(x-means),axis=1)
        else:
            sd=params['std'].to_numpy();weight=params.weight.to_numpy()
            score=np.log(weight/sd)-.5*((x-means)/sd)**2
            labels=score.argmax(axis=1)
        mask=labels==len(means)-1
        original=d[f'{row.method}_{row.k}'].to_numpy(dtype=bool)
        assert np.array_equal(mask,original),(row.method,row.k)
        assert mask.sum()==row.all_n
        assert np.array_equal(mask,d.P.to_numpy()>=row.min_selected)
        part=boot.loc[boot.method.eq(row.method)&boot.k.eq(row.k)]
        assert len(part)==40 and part.observed_monotone.all() and part.includes_observed_max.all()
        verified+=len(d)
    result=dict(status='passed',source_sha256=SHA,independently_verified_membership_cells=verified,
                models=12,weekly_resampling_fits=480,all_observed_classifications_equivalent_to_integer_cut=True,
                verification_method='Independent nearest-centroid / unnormalized Gaussian log posterior from saved parameters',
                manuscripts_edited=False)
    (TABLES/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def kmeans_plot(d,s,c):
    fig,axes=style.plt.subplots(2,3,figsize=(16,8.8),sharex=True,sharey=True)
    train=d.loc[d.month.le(6),'P'].to_numpy()
    edges=np.arange(99.5,226.5,2);mid=(edges[:-1]+edges[1:])/2
    palette=['#A9BDD0','#83B8B1','#AEC0BB','#7899AF','#6B9F9A','#91A8BB','#436C93',ORANGE]
    for ax,k in zip(axes.flat,range(3,9)):
        row=s.loc[s.method.eq('KMeans')&s.k.eq(k)].iloc[0]
        means=c.loc[c.method.eq('KMeans')&c.k.eq(k)].sort_values('rank')['mean'].to_numpy()
        labels=np.argmin(abs(train[:,None]-means),axis=1)
        colors=palette[:k-1]+[ORANGE]
        # Bins straddling a decision boundary must stack, rather than cover one another.
        bottom=np.zeros(len(mid))
        for rank,color in enumerate(colors):
            freq=np.histogram(train[labels==rank],bins=edges)[0]/len(train)
            ax.bar(mid,freq,width=2,bottom=bottom,color=color,alpha=.82)
            bottom+=freq
        for mu in means[means>=100]:ax.scatter(mu,.031,marker='v',s=35,c=ORANGE if mu==means[-1] else BLUE)
        ax.axvline(row.boundary,c=ORANGE,ls='--',lw=1.5)
        ax.text(.97,.93,f'{int(row.min_selected)} 이상\n{row.all_n:,}시간 · {row.all_rate:.1%}',
                transform=ax.transAxes,ha='right',va='top',fontsize=12,color=ORANGE,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=3))
        ax.set(title=f'{k}개 · 전체 실루엣 {row.silhouette:.3f}',xlim=(100,226),ylim=(0,.05),
               xticks=[100,125,150,175,200,225],yticks=np.arange(0,.051,.01))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(axis='y',alpha=.12)
    for ax in axes[-1]:ax.set_xlabel('시간별 최대 전력')
    for ax in axes[:,0]:ax.set_ylabel('전체 1~6월 시간 중 비율 (2단위 간격)')
    style.finish(fig,'01_kmeans_fixed_k','','')

def gmm_plot(d,s,c):
    fig,axes=style.plt.subplots(2,3,figsize=(16,8.8),sharex=True,sharey=True)
    train=d.loc[d.month.le(6),'P'].to_numpy();edges=np.arange(99.5,226.5,2)
    freq=np.histogram(train,bins=edges)[0]/len(train)/2
    x=np.linspace(100,226,1500)
    for ax,k in zip(axes.flat,range(3,9)):
        row=s.loc[s.method.eq('GMM')&s.k.eq(k)].iloc[0]
        params=c.loc[c.method.eq('GMM')&c.k.eq(k)].sort_values('rank')
        ax.bar((edges[:-1]+edges[1:])/2,freq,width=2,color='#DCE3E9',alpha=.8)
        total=np.zeros_like(x)
        for p in params.itertuples():
            y=p.weight*np.exp(-.5*((x-p.mean)/p.std)**2)/(p.std*np.sqrt(2*np.pi));total+=y
            color=ORANGE if p.rank==k else BLUE if p.rank==k-1 else '#B9C6CE'
            ax.plot(x,y,c=color,lw=2 if p.rank==k else 1.4)
        ax.plot(x,total,c='#344553',ls=':',lw=1.3)
        ax.axvline(row.boundary,c=ORANGE,ls='--',lw=1.5)
        ax.text(.97,.93,f'{int(row.min_selected)} 이상 · {row.all_rate:.1%}\n인접 분포 겹침 {row.overlap_high_neighbor:.1%}',
                transform=ax.transAxes,ha='right',va='top',fontsize=11.5,color=ORANGE,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=3))
        ax.set(title=f'{k}개 · 최저 BIC 대비 +{row.bic_difference:.1f}',xlim=(100,226),ylim=(0,.026),
               xticks=[100,125,150,175,200,225])
        ax.grid(axis='y',alpha=.12)
    for ax in axes[-1]:ax.set_xlabel('시간별 최대 전력')
    for ax in axes[:,0]:ax.set_ylabel('분포 밀도')
    style.finish(fig,'02_gmm_fixed_k','','')

def scores_plot(s):
    fig,axes=style.plt.subplots(2,2,figsize=(13,8.2))
    km=s.loc[s.method.eq('KMeans')];gm=s.loc[s.method.eq('GMM')]
    ax=axes[0,0]
    ax.plot(km.k,km.silhouette,'o-',c=BLUE,label='전체 집단')
    ax.plot(km.k,km.highest_group_silhouette,'o-',c=ORANGE,label='최고 집단만')
    ax.legend(frameon=False)
    ax.set(title='K-means: 전체 분리와 최고 집단 분리',ylabel='실루엣 (클수록 잘 분리됨)',ylim=(.45,.82))
    ax=axes[0,1]
    ax.plot(gm.k,gm.bic_difference,'o-',c=BLUE)
    ax.scatter([4,5],gm.loc[gm.k.isin([4,5]),'bic_difference'],c=ORANGE,s=55,zorder=5)
    ax.annotate('4개: 0  /  5개: +7.0',(5,6.97),xytext=(4.65,95),
                arrowprops=dict(arrowstyle='-',color=ORANGE),color=ORANGE)
    ax.set(title='GMM: 분포 적합도와 복잡성',ylabel='최저 BIC 대비 차이 (작을수록 좋음)',ylim=(-20,480))
    ax=axes[1,0]
    ax.plot(gm.k,gm.overlap_high_neighbor,'o-',c=ORANGE)
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set(title='GMM: 최고·인접 집단의 분포 겹침',ylabel='정규화한 두 분포의 겹치는 면적',ylim=(0,.5))
    ax=axes[1,1]
    for name,part,color in [('K-means',km,BLUE),('GMM',gm,ORANGE)]:
        ax.plot(part.k,part.all_rate,'o-',c=color,label=name)
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set(title='최고 집단에 들어가는 시간의 비율',ylabel='1~8월 전체 5,784시간 중 비율',ylim=(0,.36))
    for ax in axes.flat:
        ax.set_xlabel('집단 수');ax.set_xticks(range(3,9));ax.grid(alpha=.13)
    style.finish(fig,'03_scores_and_upper_groups','','')

def stability_plot(s,boot):
    fig,axes=style.plt.subplots(1,2,figsize=(15,6.8),sharex=True,sharey=True)
    for ax,method,color in zip(axes,['KMeans','GMM'],[BLUE,ORANGE]):
        for k in range(3,9):
            vals=boot.loc[boot.method.eq(method)&boot.k.eq(k),'boundary'].to_numpy()
            ax.plot([vals.min(),vals.max()],[k,k],c=color,lw=1.4,alpha=.7)
            ax.plot(np.quantile(vals,[.1,.9]),[k,k],c=color,lw=6,alpha=.35)
            jitter=np.linspace(-.13,.13,len(vals))
            ax.scatter(vals,k+jitter,c=color,s=14,alpha=.65)
            ref=s.loc[s.method.eq(method)&s.k.eq(k),'boundary'].iloc[0]
            ax.scatter(ref,k,c='black',marker='D',s=45,zorder=6,label='기준 적합의 경계' if k==3 else None)
        ax.set(title=f'{"K-means" if method=="KMeans" else "GMM"}: 같은 집단 수로 40회 재적합',
               xlabel='최고 집단의 경계 전력값',xlim=(130,198),yticks=range(3,9),
               yticklabels=[f'{k}개' for k in range(3,9)],ylim=(8.6,2.4))
        ax.legend(frameon=False,loc='lower left')
        ax.grid(axis='x',alpha=.15)
    axes[0].set_ylabel('집단 수')
    style.finish(fig,'04_fixed_k_stability','','')

def run():
    style.setup();d=read('assignments');s=read('comparison');c=read('components');boot=read('weekly_resampling')
    result=verify(d,s,c,boot)
    kmeans_plot(d,s,c);gmm_plot(d,s,c);scores_plot(s);stability_plot(s,boot)
    names=[row['file'] for row in style.GENERATED]
    thumbs=[]
    for name in names:
        with Image.open(FIGURES/name) as im:
            thumbs.append(im.resize((im.width//2,im.height//2),Image.Resampling.LANCZOS).copy())
    cellw=max(im.width for im in thumbs);cellh=max(im.height for im in thumbs)
    contact=Image.new('RGB',(cellw*2,cellh*2),'#EBEEF1')
    for i,im in enumerate(thumbs):contact.paste(im,((i%2)*cellw,(i//2)*cellh))
    contact.save(FIGURES/'contact_50.png')
    manifest=dict(result,font=style.plt.rcParams['font.family'],figures=style.GENERATED,style_checks=style.STYLE_CHECKS,
                  distribution_view='100..226 upper-load zoom; fits use ALL Jan-Jun valid hours; histogram denominator unchanged',
                  resampling_view='40 calendar-week resamples; thin line min-max, thick segment p10-p90; NOT confidence intervals',
                  visual_review=False)
    (FIGURES/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(result,generated_images=names),ensure_ascii=False))

if __name__=='__main__':run()
