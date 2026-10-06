"""Plot fixed common-row weather comparisons; no new analysis/model fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from a03_conditioned_weather import ROOT,OUT,WEATHER


def main():
    manifest=json.loads((OUT/'diagnostic_verification.json').read_text(encoding='utf-8'))
    source=OUT/'common_row_associations.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['outputs_sha256']['common_row_associations']
    data=pd.read_csv(source,encoding='utf-8-sig')
    scope=data.loc[data.scope.eq('weekday')]
    raw=scope.loc[scope.adjustment.eq('raw')].set_index('variable').loc[WEATHER]
    adjusted=scope.loc[scope.adjustment.eq('joint')].set_index('variable').loc[WEATHER]
    assert (raw.n==4102).all() and (adjusted.n==4102).all()
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':12})
    fig,ax=plt.subplots(figsize=(8.7,4.8),layout='constrained')
    x=np.arange(4); width=.34
    a=ax.bar(x-width/2,raw.correlation,width,label='단순 관계',color='#A3B6C2')
    b=ax.bar(x+width/2,adjusted.correlation,width,label='월·시간대·생산량·다른 날씨를 고려',color='#267D8D')
    for bars in (a,b):
        ax.bar_label(bars,labels=[f'{bar.get_height():+.3f}' for bar in bars],padding=4,fontsize=11)
    ax.set_xticks(x,WEATHER); ax.set_ylim(-.18,.33)
    ax.set_ylabel('전력과의 순위 관계'); ax.axhline(0,color='#777777',linewidth=.8)
    ax.legend(loc='upper center',ncol=1,frameon=False)
    ax.spines[['top','right']].set_visible(False)
    figures=ROOT/'Analysis/figures'; path=figures/'a03_weather_conditions.png'
    fig.savefig(path,dpi=160); plt.close(fig)
    with Image.open(path) as img:
        img.resize((img.width//2,img.height//2),Image.Resampling.LANCZOS).save(figures/'a03_weather_conditions_review.png')
    verify=dict(status='passed',same4102weekday_records=True,
        note='Raw Spearman vs adjusted rank-residual partial association; neither is accuracy or causal effect.',
        path=str(path.relative_to(ROOT)),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        visually_reviewed=False)
    (OUT/'figure_verification.json').write_text(json.dumps(verify,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
