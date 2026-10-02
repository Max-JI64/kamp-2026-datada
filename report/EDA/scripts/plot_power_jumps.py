"""Make a small set of readable level/rise/spike figures from saved results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / 'report/EDA/tables/power_jumps_review'
OUT = ROOT / 'report/EDA/figures/power_jumps_review'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'Malgun Gothic', 'axes.unicode_minus':False,
    'font.size':12, 'axes.titlesize':14, 'axes.labelsize':12,
    'axes.spines.top':False, 'axes.spines.right':False, 'figure.facecolor':'white',
    'savefig.facecolor':'white'})
BLUE='#245a85'; RED='#cf533e'; TEAL='#258779'; GRAY='#818b94'

def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

def main():
    d=pd.read_csv(TAB/'slots.csv', encoding='utf-8-sig', parse_dates=['time'])
    s=json.loads((TAB/'summary.json').read_text(encoding='utf-8'))
    c=s['candidate_rise_cut']
    # Actual examples selected by explicit descriptive rules, not estimates of
    # category frequency. Save the windows so the displayed values are auditable.
    candidate=d.loc[d.rise & ~d.legacy_high & ~d.single_slot_spike & d.prev.gt(40)]
    a=((candidate.prev-60).abs()+(candidate.power-90).abs()).idxmin()
    sustained=d.power.rolling(7,center=True).min()
    rises=d.rise.astype(int).rolling(7,center=True).sum()
    candidate=d.loc[sustained.ge(187)&rises.eq(0)]
    assert len(candidate), 'No seven-slot sustained-high example'
    b=(candidate.power-candidate.power.median()).abs().idxmin()
    candidate=d.loc[d.single_slot_spike & d.prev.gt(40)]
    z=(candidate.delta-candidate.delta.median()).abs().idxmin()
    selected=[a,b,z]
    labels=['급상승: 낮은 전력에서도 잡힌다','고전력 유지: 높지만 급상승은 아니다','한 구간 돌출: 상승 직후 다시 내려온다']
    fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
    exports=[]; examples=[]
    for ax,i,label in zip(axes,selected,labels):
        part=d.iloc[i-4:i+5].copy()
        assert part.time.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
        x=np.arange(len(part)); k=4
        ax.plot(x,part.power,'o-',c=BLUE,lw=2.3,ms=5,label='전력')
        ax.plot(x,part.prev+c,'--',c=RED,lw=1.4,alpha=.8,label='직전 전력 + 20.76 (시험선)')
        ax.axhline(187,c=GRAY,ls=':',lw=1.6,label='기존 고전력 비교선 187 (미확정)')
        flags=part.rise.to_numpy()
        ax.scatter(x[flags],part.power.to_numpy()[flags],c=RED,s=68,zorder=5,label='21 이상 상승')
        current=d.loc[i]
        arrow=f'{int(current.prev)} → {int(current.power)}'
        if current.single_slot_spike: arrow+=f' → {int(current["next"])}'
        ax.annotate(arrow,xy=(k,current.power),xytext=(k+.5,min(current.power+27,236)),
            fontsize=13,fontweight='bold',arrowprops={'arrowstyle':'->','color':BLUE},color=BLUE)
        ax.set(title=f'{label}  |  {current.time:%Y-%m-%d}',ylabel='전력',ylim=(0,245),
            xticks=x,xticklabels=part.time.dt.strftime('%H:%M'))
        ax.grid(axis='y',alpha=.12)
        part['example']=label;exports.append(part)
        examples.append({'label':label,'time':str(current.time),'previous':int(current.prev),
            'power':int(current.power),'next':int(current['next']),'delta':int(current.delta)})
    axes[0].legend(loc='upper left',ncol=2,fontsize=9,frameon=False)
    axes[-1].set_xlabel('15분 구간의 시작 시각')
    save(fig,'01_actual_examples')
    pd.concat(exports).to_csv(TAB/'example_windows.csv',index=False,encoding='utf-8-sig')
    (TAB/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')

    # Joint distribution: same points, both questions visible at once. Repeated
    # identical coordinates are counted in point size, rather than hidden.
    p=d.loc[d.delta.notna()].groupby(['power','delta','category']).size().reset_index(name='n')
    fig,ax=plt.subplots(figsize=(10,6.4),layout='constrained')
    colors={'neither':'#c4cbd0','rise_only':RED,'high_only':BLUE,'both':'#8c4ba3'}
    names={'neither':'둘 다 아님','rise_only':'급상승만','high_only':'고전력만','both':'둘 다'}
    for cat in ['neither','high_only','rise_only','both']:
        q=p.loc[p.category.eq(cat)]
        ax.scatter(q.power,q.delta,s=8+11*np.sqrt(q.n),alpha=.48 if cat=='neither' else .75,
            c=colors[cat],edgecolors='none',label=f'{names[cat]} · {s["category_counts_valid_pairs"][cat]:,}구간')
    ax.axhline(c,c=RED,ls='--',lw=1.5)
    ax.axvline(187,c=BLUE,ls=':',lw=1.7)
    ax.axhline(0,c=GRAY,lw=.7)
    ax.set(xlabel='현재 전력',ylabel='직전 15분 구간보다 증가한 값',xlim=(-3,232),ylim=(-170,175))
    ax.text(4,161,'위쪽: 21 이상 상승',color=RED,fontsize=12)
    ax.text(190,-156,'오른쪽:\n187 이상',color=BLUE,fontsize=11)
    ax.legend(frameon=False,loc='upper right',fontsize=10,markerscale=.7)
    ax.grid(alpha=.1)
    save(fig,'02_level_vs_change')

    # Supporting figure: don't imply +21 is a naturally separated component.
    x=d.delta.dropna().to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    ax=axes[0]
    edges=np.arange(np.floor(x.min())-1,np.ceil(x.max())+3,2)
    count,bins,patch=ax.hist(x,bins=edges,color=BLUE,edgecolor='white',lw=.25)
    for rect,left in zip(patch,bins[:-1]):
        if left>c: rect.set_facecolor(RED)
    ax.axvline(c,c=RED,ls='--',lw=1.6,label='시험 경계 +20.76')
    ax.set(title='전력 자체가 아니라 변화량의 분포',xlabel='현재 전력 - 직전 구간 전력',ylabel='구간 수 (로그 눈금)',yscale='log')
    ax.legend(frameon=False,fontsize=10);ax.grid(axis='y',alpha=.12)
    ax=axes[1]
    slots=pd.read_csv(TAB/'clock_slots.csv',encoding='utf-8-sig')
    ax.plot(slots.slot/4,slots.rate,'o-',ms=3,lw=1.4,c=RED)
    top=slots.sort_values('rises',ascending=False).iloc[0]
    ax.annotate(f'{int(top.slot)//4:02}:{int(top.slot)%4*15:02}\n{top.rate:.1%}',
        xy=(top.slot/4,top.rate),xytext=(15.5,.69),fontsize=12,
        arrowprops={'arrowstyle':'->','color':RED},color=RED)
    ax.set(title='같은 시각에 반복되는 상승도 포함',xlabel='15분 구간의 시작 시각',ylabel='해당 시각에서 21 이상 상승한 비율',
        xlim=(0,24),ylim=(0,.8),xticks=np.arange(0,25,3))
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(alpha=.12)
    save(fig,'03_rule_diagnostics')

    paths=[OUT/f'{name}.png' for name in ['01_actual_examples','02_level_vs_change','03_rule_diagnostics']]
    ims=[]
    for path in paths:
        with Image.open(path) as im:
            ims.append(im.convert('RGB').resize((im.width//2,im.height//2),Image.Resampling.LANCZOS))
    canvas=Image.new('RGB',(max(im.width for im in ims),sum(im.height for im in ims)+20*(len(ims)-1)),'white')
    y=0
    for im in ims:canvas.paste(im,(0,y));y+=im.height+20
    canvas.save(OUT/'contact_50.png')
    print(json.dumps({'examples':examples,'figures':[str(p) for p in paths]},ensure_ascii=False))

if __name__=='__main__':main()
