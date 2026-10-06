"""Publication figures from frozen results; no fitting or rescaling to hide errors."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from regime_forecast import sha, save_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Modeling/tables/modeling_close'
FIG=ROOT/'Modeling/figures/modeling_close'
AGE=ROOT/'Modeling/tables/regime_age_ablation'
BLUE='#275D8C'; ORANGE='#C66B31'; GREY='#8995A0'; DARK='#253746'


def keep(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=200,facecolor='white',bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),facecolor='white',bbox_inches='tight')
    plt.close(fig)


def pipeline():
    fig,ax=plt.subplots(figsize=(10.6,3.6));ax.set_xlim(0,10);ax.set_ylim(-.2,3.7);ax.axis('off')
    def box(x,y,w,h,label,color='#F0F4F7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',facecolor=color,edgecolor='#9CACB8',lw=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11,color=DARK)
    def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':GREY,'lw':1.4})
    box(.1,2.35,2.15,1,'자료 진단·시차 구성\n정확한 과거 시각 연결')
    box(2.65,2.35,2.5,1,'시간순 모델·입력 비교\n전체·일별 최대 오차 평가')
    box(5.55,2.35,2.05,1,'기본 HGB 선택\n상승 오류 진단')
    box(8,2.35,1.8,1,'지속 상승 정의\n과거 정보로 분류')
    for a,b in [((2.25,2.85),(2.65,2.85)),((5.15,2.85),(5.55,2.85)),((7.6,2.85),(8,2.85))]:arrow(a,b)
    box(.1,.35,2.15,1,'매시간 t-1 관측 완료\n다음 t 시간 예측')
    box(2.65,.35,2.5,1,'지속 상승 경고 판정\n과거 OOF로 정한 경계')
    box(5.7,1,1.8,.65,'경고 있음: 보완 B1','#E5EFF8')
    box(5.7,0,1.8,.65,'경고 없음: 기본 B0')
    box(8,.35,1.8,1,'최대 전력 예측\n담당자 확인')
    arrow((2.25,.85),(2.65,.85));arrow((5.15,.9),(5.7,1.3));arrow((5.15,.8),(5.7,.3))
    arrow((7.5,1.3),(8,1));arrow((7.5,.3),(8,.65))
    ax.plot([8.9,8.9,3.9],[2.35,1.88,1.88],color=GREY,lw=1.4)
    arrow((3.9,1.88),(3.9,1.35))
    ax.text(.1,3.55,'개발 흐름',fontsize=11,weight='bold');ax.text(.1,1.62,'적용 흐름',fontsize=11,weight='bold')
    keep(fig,'pipeline')


def performance(axes=None):
    m=pd.read_csv(AGE/'followup_metrics.csv',encoding='utf-8-sig')
    standalone=axes is None
    if standalone:fig,axes=plt.subplots(1,2,figsize=(10.6,3.3),gridspec_kw={'width_ratios':[1.6,1]})
    ax,bx=axes
    cols=['all','up_start','daily_peak_day_equal']; y=np.arange(3); height=.33
    for name,dy,color,label in [('B0',-height/2,GREY,'기본 HGB'),('G1_noage',height/2,BLUE,'선택적 보완')]:
        vals=[float(m.loc[(m.variant==name)&(m.period=='Jul-Aug')&(m.condition==x),'mae'].iloc[0]) for x in cols]
        ax.barh(y+dy,vals,height,color=color,label=label)
        for yy,v in zip(y+dy,vals):ax.text(v+.25,yy,f'{v:.2f}',va='center',fontsize=10)
    ax.set_yticks(y,['전체','상승 시작','일별 최대 시간']);ax.invert_yaxis();ax.set_xlim(0,24)
    ax.set_xlabel('MAE (원자료 전력 척도)');ax.set_title('7~8월 동일 1,344시간 비교',fontsize=12)
    ax.legend(frameon=False,fontsize=10,loc='lower right');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    bx.barh([0,1],[7,7],color=BLUE,label='적중');bx.barh([0,1],[38,0],left=[7,7],color=ORANGE,label='오경보')
    bx.set_yticks([0,1],['수정 전','경과시간 제거']);bx.invert_yaxis();bx.set_xlim(0,50)
    bx.text(3.5,0,'7',ha='center',va='center',color='white');bx.text(26,0,'38',ha='center',va='center',color='white')
    bx.text(3.5,1,'7',ha='center',va='center',color='white');bx.text(8,1,'오경보 0',va='center',fontsize=10)
    bx.set_title('적중 유지·오경보 억제',fontsize=12);bx.set_xlabel('경고 수 (공통 정답 1,340시간)')
    bx.legend(frameon=False,fontsize=10,loc='lower right')
    if standalone:
        fig.tight_layout(w_pad=2);keep(fig,'performance_and_alarms')


def cases(axes=None):
    p=pd.read_csv(OUT/'case_windows.csv',encoding='utf-8-sig',parse_dates=['timestamp','onset_time'])
    standalone=axes is None
    if standalone:fig,axes=plt.subplots(1,2,figsize=(10.6,3.65),sharey=True)
    for ax,name,label in zip(axes,['detected_improved','missed'],['탐지·오차 개선 사례','미탐지 사례']):
        g=p.loc[(p.case==name)&(p.variant=='B0')].sort_values('timestamp')
        other=p.loc[(p.case==name)&(p.variant=='G1_noage')].set_index('timestamp').loc[g.timestamp]
        t=g.onset_time.iloc[0];x=(g.timestamp-t).dt.total_seconds()/3600
        ax.axvspan(0,2,color='#EAF1F6');ax.axvline(-1,color=GREY,lw=.9,ls=':')
        ax.plot(x,g.actual,'o-',color=DARK,label='실제 최대',ms=3)
        ax.plot(x,g.observed_mean,':',color=GREY,label='실제 평균')
        ax.plot(x,g.prediction,'--',color=ORANGE,label='기본 HGB',lw=1.6)
        ax.plot(x,other.prediction,'-',color=BLUE,label='선택적 보완',lw=1.4,marker='.',ms=4)
        ax.set_title(f'{label} | {t:%m/%d %H시}',fontsize=12)
        ax.set_xticks([-4,-2,0,2,4]);ax.set_xlabel('상승 시작 기준 시간 (0 = 시작)')
        ax.set_ylim(0,240);ax.grid(alpha=.15);ax.text(.03,.91,'음영: 지속 정답의 3시간',transform=ax.transAxes,fontsize=9)
    axes[0].set_ylabel('전력 (원자료 척도)')
    if standalone:
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',frameon=False,fontsize=10)
        fig.tight_layout(rect=[0,.1,1,1]);keep(fig,'transition_cases')


def compact_evidence():
    """Draw metrics, alarms and contrasting cases together from the source tables."""
    fig,axes=plt.subplots(2,2,figsize=(10.6,6.7))
    performance(axes[0]);cases(axes[1])
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',frameon=False,fontsize=10)
    fig.tight_layout(rect=[0,.055,1,1],h_pad=2,w_pad=2)
    keep(fig,'compact_evidence')


def contribution():
    f=pd.read_csv(OUT/'contribution_summary.csv',encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(8,3.6));x=np.arange(3)
    for i,(col,label,color) in enumerate([('calendar_contribution','달력',GREY),('power_contribution','과거 전력',BLUE),('production_contribution','생산량',ORANGE),('state_contribution','추적 상태','#6A8B72')]):
        ax.bar(x+(i-1.5)*.18,f[col],width=.18,color=color,label=label)
    ax.set_xticks(x,['상승 적중 (7건)','상승 미탐지 (6건)','월요일 08시 비사건 (1건)'])
    ax.axhline(0,color=DARK,lw=.7);ax.set_ylabel('평균 로짓 기여 (확률·인과효과 아님)')
    ax.legend(frameon=False,ncol=4,loc='upper right',fontsize=10)
    ax.set_title('고정 Logistic의 실제 계산 분해 | 공통 절편 -10.104',fontsize=12)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();keep(fig,'logit_contributions')


def main():
    r=json.loads((OUT/'run.json').read_text(encoding='utf-8'))
    assert r['status']=='passed'
    for n,h in r['outputs_sha256'].items():assert sha(OUT/n)==h
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,'svg.fonttype':'path'})
    pipeline();performance();cases();contribution();compact_evidence()
    preview=ROOT/'tmp/modeling_visual_review/closing';preview.mkdir(parents=True,exist_ok=True)
    for p in FIG.glob('*.png'):
        with Image.open(p) as im:im.resize((im.width//2,im.height//2),Image.Resampling.LANCZOS).save(preview/p.name)
    save_json(OUT/'figures.json',{'status':'generated','script_sha256':sha(__file__),'run_sha256':sha(OUT/'run.json'),
                               'figures_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in FIG.iterdir() if p.suffix in ['.png','.svg']},
                               'visual_review':'pending','new_fits':0})
    print(json.dumps({'status':'generated','figures':5,'preview_directory':str(preview)}))


if __name__=='__main__':main()
