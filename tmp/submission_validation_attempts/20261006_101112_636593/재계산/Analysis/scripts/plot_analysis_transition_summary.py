"""Combine two distinct, verified transition analyses without recomputing them."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    change_path = ROOT / 'Analysis/tables/a01_production_changes/transition_matched_contrasts.csv'
    class_path = ROOT / 'Analysis/tables/a02_transition_forecast/classification_test.csv'
    changes = pd.read_csv(change_path, encoding='utf-8-sig')
    row = changes.loc[(changes.weighting == 'date_hour') &
                      (changes.a == 'zero_to_positive') &
                      (changes.b == 'positive_to_positive') &
                      (changes.metric == 'abs_power_delta')]
    assert len(row) == 1
    row = row.iloc[0]
    assert int(row.n_a) == 150 and int(row.n_b) == 567
    scores = pd.read_csv(class_path, encoding='utf-8-sig')
    scores = scores.loc[scores.scope == 'all'].set_index('features').loc[['calendar', 'level', 'shape']]
    assert scores.n.eq(617).all() and scores.events.eq(52).all()
    assert scores.tp.tolist() == [35, 50, 50] and scores.fp.tolist() == [51, 25, 25]
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={'width_ratios': [1, 1.6]})
    left, right = axes
    bars = left.bar([0, 1], [row.adjusted_b, row.adjusted_a], width=.5,
                    color=['#7F9BB7', '#D07834'])
    left.bar_label(bars, fmt='%.2f', padding=5)
    left.set_xticks([0, 1], ['양수 지속\n567시간', '0→양수 전환\n150시간'])
    left.set_ylabel('시간 사이 평균 전력 변화 크기')
    left.set_ylim(0, 47)
    x = np.arange(3)
    width = .32
    hits = right.bar(x-width/2, scores.tp, width, color='#24788A', label='실제 전환 적중')
    false = right.bar(x+width/2, scores.fp, width, color='#D78947', label='전환 오탐')
    right.bar_label(hits, padding=4)
    right.bar_label(false, padding=4)
    right.axhline(52, color='#777777', linestyle=':', linewidth=1)
    right.text(2.48, 53, '실제 전환 52시간', ha='right', fontsize=10, color='#666666')
    right.set_xticks(x, ['달력만', '달력 +\n직전 평균', '달력 + 직전 평균\n+ 마지막-첫 값'])
    right.set_ylabel('시간 수')
    right.set_ylim(0, 69)
    right.legend(loc='upper center', ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(pad=1.0, w_pad=2)
    output = ROOT / 'Analysis/figures/a01_a02_transition_summary.png'
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)
    manifest = {'status': 'passed', 'panels_use_distinct_samples_and_metrics': True,
                'left_n': 717, 'right_n': 617, 'right_events': 52,
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (change_path, class_path)},
                'image_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    audit_path = ROOT / 'Analysis/tables/transition_summary_figure_verification.json'
    audit_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
