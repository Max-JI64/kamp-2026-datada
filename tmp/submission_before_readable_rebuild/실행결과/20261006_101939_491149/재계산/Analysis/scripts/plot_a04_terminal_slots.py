"""Plot verified descriptive conditional associations, not forecast accuracy."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from a04_maximum_conditions import ROOT, OUT


def main():
    table = pd.read_csv(OUT/'terminal_slot_robustness.csv', encoding='utf-8-sig')
    rows = table.loc[(table.check == 'main') & (table.weighting == 'date_hour') &
                     (table.target == 'target_maximum')].set_index('feature')
    features = ['prior_first', 'prior_second', 'prior_third', 'prior_last']
    values = rows.loc[features, 'correlation'].to_numpy()
    assert rows.loc[features, 'n'].eq(5778).all()
    plt.rcParams.update({'font.family':'Malgun Gothic', 'axes.unicode_minus':False,
                         'font.size':13, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(8.8,4.5))
    x = np.arange(4)
    ax.bar(x, values, width=.55, color=['#98A3AF']*3+['#227F78'], zorder=3)
    ax.axhline(0,color='#6B7280',linewidth=1)
    ax.set_xticks(x, ['15분 값', '30분 값', '45분 값', '60분 값\n(마지막)'])
    ax.set_xlabel('관측을 마친 직전 시간의 전력값', labelpad=12)
    ax.set_ylabel('다음 시간 최댓값과의 순위 관계', labelpad=12)
    ax.set_ylim(-.22,.52)
    ax.set_yticks([-.2,0,.2,.4])
    ax.grid(axis='y',alpha=.16,zorder=0)
    for xi, value in zip(x,values):
        ax.text(xi,value+(.025 if value>=0 else -.03), f'{value:+.3f}',
                ha='center', va='bottom' if value>=0 else 'top', fontsize=14)
    fig.tight_layout(pad=1.6)
    output = ROOT/'Analysis/figures/a04_terminal_slots.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor='white')
    fig.set_size_inches(4.4,2.25)
    # Preserve layout and text exactly in a 50% pixel review, via Pillow.
    from PIL import Image
    with Image.open(output) as source:
        source.resize((source.width//2,source.height//2),Image.Resampling.LANCZOS).save(output.with_name('a04_terminal_slots_review.png'))
    plt.close(fig)
    audit=dict(status='passed',same5778records=True,
               controls='Target calendar and known production plus observed prior mean cubic ranks; no target mean/production controls.',
               interpretation='Partial rank association, not accuracy, percentage improvement or causal effect.',
               path=str(output.relative_to(ROOT)),image_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),visually_reviewed=False)
    (OUT/'figure_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',path=str(output)),ensure_ascii=False))


if __name__ == '__main__':
    main()
