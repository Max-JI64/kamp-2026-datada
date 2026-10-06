"""Simple observed-change figure from verified matched comparison table."""
from pathlib import Path
import json
import hashlib
import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
TABLE=ROOT/'Analysis/tables/a01_production_changes/transition_matched_contrasts.csv'
OUT=ROOT/'Analysis/figures'


def main():
    table=pd.read_csv(TABLE,encoding='utf-8-sig')
    row=table.loc[(table.weighting=='date_hour')&(table.a=='zero_to_positive')
                  &(table.b=='positive_to_positive')&(table.metric=='abs_power_delta')].iloc[0]
    vals=[row.adjusted_b,row.adjusted_a]
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
        'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.4,4.6))
    bars=ax.bar([0,1],vals,color=['#7F9BB7','#D07834'],width=.5)
    ax.set_xticks([0,1],['직전·현재 모두 생산량 양수','직전 생산량 0 → 현재 양수'])
    ax.set_ylabel('직전 시간 대비 전력 변화 크기')
    ax.set_ylim(0,max(vals)*1.25)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,v+1.1,f'{v:.2f}',ha='center',fontsize=14)
    fig.tight_layout(pad=1.5)
    OUT.mkdir(exist_ok=True,parents=True)
    path=OUT/'a01_production_transition.png'
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)
    assert not any('Glyph' in str(w.message) for w in ws)
    with Image.open(path) as img:
        img.resize((img.width//2,img.height//2)).save(OUT/'a01_production_transition_review.png')
    verification=dict(status='passed',input_sha256=hashlib.sha256(TABLE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),values=vals,
        actual_observations=True,forecast_or_oracle=False,visual_review=False)
    (ROOT/'Analysis/tables/a01_production_changes/figure_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification))


if __name__=='__main__':
    main()
