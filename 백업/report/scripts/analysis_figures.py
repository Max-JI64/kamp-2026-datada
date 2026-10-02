"""Ten alternative report figures, built from local analysis tables. No image reading."""
# %% Shared report style (no EDA analysis runs on import)
from pathlib import Path
import json
import numpy as np
import pandas as pd
import eda_figures as style

REPORT = Path(__file__).resolve().parents[1]
TABLES = REPORT/'tables/analysis'
style.FIGURES = REPORT/'figures/analysis'
BLUE, ORANGE, TEAL, GRAY = style.BLUE, style.ORANGE, style.TEAL, style.GRAY
plt, finish = style.plt, style.finish


def read(name):
    return pd.read_csv(TABLES/f'{name}.csv', encoding='utf-8-sig')


# %% 3.1 Distribution composition
def distribution():
    s = read('distribution').query('profile_weighted == False').set_index('month')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, col, label, color in zip(axes, ['max_average', 'high_182'],
            ['시간별 최대값의 평균', '최대 ≥ 182인 시간 비중 (%)'], [BLUE, ORANGE]):
        v = s[col] * (100 if col == 'high_182' else 1)
        ax.bar(range(3), v, color=color, width=.55)
        for i, val in enumerate(v):
            ax.text(i, val, f'{val:.2f}', ha='center', va='bottom')
        ax.set(xticks=range(3), xticklabels=['6월\n469시간', '7월\n415시간', '8월\n360시간'],
               ylabel=label, ylim=(0, 180 if col == 'max_average' else 38))
    finish(fig, '01A_mean_and_high_rate', '평균 수준이 비슷해도 높은 최대값의 비중은 다르다',
           '생산량 양수인 평일 · 6~8월 통합 시간대 구성으로 표준화 · 날짜 가중 · 182는 분석 기준')
    b = read('distribution_bins').query("profile_weighted == False and metric == 'max'")
    wide = b.pivot(index='low', columns='month', values='mean_contribution')
    delta = wide[8] - wide[6]
    labels = ['0~26 미만', '26~112.5 미만', '112.5~156 미만', '156~176 미만',
              '176~182 미만', '182~188 미만', '188 이상']
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(labels, delta, color=[ORANGE if x >= 0 else BLUE for x in delta])
    ax.axvline(0, color=GRAY, lw=1)
    for i, v in enumerate(delta):
        ax.text(v + (.3 if v >= 0 else -.3), i, f'{v:+.2f}', va='center', ha='left' if v >= 0 else 'right')
    ax.set(xlabel='8월 - 6월: 표준화 평균에 대한 구간별 기여 변화', xlim=(-20, 20))
    ax.invert_yaxis()
    finish(fig, '01B_distribution_contribution', f'높은 구간의 증가와 중간 구간의 감소가 상쇄된다 (합계 {delta.sum():+.2f})',
           '기여 = 구간의 표준화 비중 × 구간 내 평균 · 인과효과가 아닌 관측 분포의 산술 분해')


# %% 3.2 Transitions and shared support
def transitions():
    s = read('transition_summary').query('profile_weighted == False').set_index('period')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    x = np.arange(3)
    for ax, (period, row) in zip(axes, s.iterrows()):
        for i, (name, label, color) in enumerate([('start', '생산 0→양수', ORANGE), ('continuing', '양수→양수', BLUE)]):
            v = np.array([row[f'{name}_{m}_rate'] for m in ['crude', 'overlap', 'standardized']])*100
            ax.bar(x + (i-.5)*.34, v, width=.34, label=label, color=color)
            for j, value in enumerate(v):
                ax.text(j+(i-.5)*.34, value+.5, f'{value:.2f}', ha='center', fontsize=9)
        ax.set(title={'Jan-Jun': '1~6월', 'Jul-Aug': '7~8월'}[period],
               xticks=x, xticklabels=['단순 비교', '공통 조건만', '월×시각 표준화'], ylim=(0, 31))
        ax.legend(fontsize=9)
    axes[0].set_ylabel('고전력 시작 비율 (%)')
    finish(fig, '02A_transition_comparison', '생산 시작과 고전력의 관계는 비교 조건에 따라 달라진다',
           '직전 1시간 최대 < 182, 현재 생산 양수인 시간 · 공통 월×시각 구성으로 표준화 · 인과효과 아님')
    fig, ax = plt.subplots(figsize=(10, 5.6))
    labels, kept, excluded = [], [], []
    for period, row in s.iterrows():
        for name, label in [('start', '0→양수'), ('continuing', '양수→양수')]:
            labels.append(f"{'1~6월' if period == 'Jan-Jun' else '7~8월'} {label}")
            kept.append(row[f'{name}_overlap_n'])
            excluded.append(row[f'{name}_all_n']-row[f'{name}_overlap_n'])
    ax.barh(labels, kept, color=BLUE, label='공통 월×시각에 포함')
    ax.barh(labels, excluded, left=kept, color='#D9DFE5', label='비교 상대가 없어 제외')
    for i, (k, e) in enumerate(zip(kept, excluded)):
        ax.text(k+e+20, i, f'{int(k):,}/{int(k+e):,} ({k/(k+e):.1%})', va='center', fontsize=10)
    ax.invert_yaxis()
    ax.set(xlabel='비교 가능한 시간 수 / 전체 분석 대상 시간 수', xlim=(0, 2750))
    ax.legend(loc='lower right', fontsize=9)
    finish(fig, '02B_transition_support', '조건을 맞춘 결과의 적용 범위도 함께 확인해야 한다',
           '공통 월×시각 65개 / 19개 · 둘 중 한 집단이 2시간 이하인 조건 42개 / 12개 · 날짜 가중')


# %% 3.3 More than the single highest slot
def caps():
    slots, caps = read('daily_slots'), read('direct_caps')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, date in zip(axes, ['2021-01-29', '2021-07-19']):
        v = np.sort(slots.loc[slots.date.eq(date), 'value'].to_numpy())[::-1]
        row = caps.loc[caps.date.eq(date) & caps.cap.eq(182)].iloc[0]
        ax.plot(np.arange(1, 97), v, color=BLUE, lw=2)
        ax.fill_between(np.arange(1, 97), 182, v, where=v>182, color=ORANGE, alpha=.35)
        ax.axhline(182, ls='--', color=ORANGE)
        ax.set(title=f'{date}\n182 초과 {int(row.cells_gt)}칸 · 초과분 합 {row.excess_value_sum:.0f}',
               xlabel='하루 96칸을 값이 큰 순서로 정렬한 순위', xlim=(1, 96), ylim=(0, 235))
    axes[0].set_ylabel('15분 구간 전력')
    finish(fig, '03A_sorted_daily_profiles', '최고 한 칸을 낮춰도 다음으로 높은 값이 최대가 된다',
           '시간 순서가 아닌 크기 순서 · 182 이하 달성에는 초과 칸 전체의 조정 필요 · 합계는 에너지 단위가 아님')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, date in zip(axes, ['2021-01-29', '2021-07-19']):
        g = caps.loc[caps.date.eq(date)].sort_values('cap')
        ax.bar(g.cap.astype(str), g.cells_gt, color=[ORANGE, BLUE, TEAL], width=.55)
        for i, row in enumerate(g.itertuples()):
            ax.text(i, row.cells_gt+.3, f'{row.cells_gt}칸\n초과합 {row.excess_value_sum:.0f}', ha='center', fontsize=10)
        ax.set(title=date, xlabel='목표 상한', ylabel='목표보다 높은 15분 구간 수', ylim=(0, 35))
    finish(fig, '03B_cap_requirements', '목표 상한에 따라 필요한 조정 범위가 달라진다',
           '176·182·188은 기존 분석 기준 · 상한과 같은 값은 조정 대상에서 제외 · 실제 이동 가능량은 미식별')


# %% 3.4 Daily reductions under the idealized assumptions
def daily_scenarios():
    s = read('group_summary').query("period == 'Jan-Aug' and weighting == 'observed_days' and receive_zero == True")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    groups = ['all', 'high_ge_182', 'below_182']
    for ax, budget in zip(axes, [.05, .1]):
        g = s.loc[s.budget_fraction.eq(budget)].set_index('group').loc[groups]
        med = g.median_reduction.to_numpy()
        ax.errorbar(range(3), med, yerr=[med-g.q25_reduction, g.q75_reduction-med],
                    fmt='o', color=BLUE, capsize=7, markersize=8)
        for i, val in enumerate(med):
            ax.text(i+.12, val, f'{val:.2f}', va='center', fontsize=10)
        ax.set(title=f'기록값 합의 {budget:.0%} 이동 예산',
               xticks=range(3), xticklabels=['전체\n241일', '원래 최대 ≥182\n105일', '원래 최대 <182\n136일'],
               xlim=(-.4, 2.65), ylim=(0, 75))
    axes[0].set_ylabel('일최대 감소량')
    finish(fig, '04A_daily_reduction', '고최대 날짜는 별도 집단으로 감소량을 보고한다',
           '점: 중앙값, 막대: 25~75백분위 · 일별 합 보존·자유 재배치 가정 · 0칸 수신 허용 · 실측 절감 아님')
    fig, ax = plt.subplots(figsize=(10, 5.4))
    all_s = read('group_summary').query("period == 'Jan-Aug' and group == 'high_ge_182' and receive_zero == True")
    for wt, label, color in [('observed_days', '관측 날짜 105일', BLUE), ('unique_profiles', '고유 프로필 61개', ORANGE)]:
        g = all_s.loc[all_s.weighting.eq(wt)].sort_values('budget_fraction')
        ax.plot(g.budget_fraction*100, g.median_reduction, 'o-', label=label, color=color)
        for row in g.itertuples():
            if row.budget_fraction:
                ax.annotate(f'{row.median_reduction:.2f}', (row.budget_fraction*100, row.median_reduction),
                            xytext=(0, -16 if wt == 'observed_days' else 9), textcoords='offset points', ha='center')
    ax.set(xlabel='일별 기록값 합 대비 이동 예산 (%)', ylabel='고최대 집단의 일최대 감소량 중앙값',
           xticks=[0, 5, 10], ylim=(0, 66), xlim=(-.5, 10.6))
    ax.legend()
    finish(fig, '04B_profile_sensitivity', '반복 프로필을 한 번씩 세어도 감소 방향은 유지된다',
           '세 예산 조건만 계산 · 선은 조건 간 연결이며 중간 예산의 계산 결과가 아님 · 실제 운영 성과 아님')


# %% 3.5 Monthly maximum and excluded-day sensitivity
def monthly():
    m = read('monthly_max').query('receive_zero == True')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={'width_ratios': [1.6, 1]})
    for budget, label, color in [(0, '원래 최대', GRAY), (.05, '5% 재배치', BLUE), (.1, '10% 재배치', ORANGE)]:
        g = m.loc[m.budget_fraction.eq(budget)].sort_values('month')
        axes[0].plot(range(1, 9), g.adjusted_month_max, 'o-', label=label, color=color)
    axes[0].set(xticks=range(1, 9), xlabel='월', ylabel='월최대', ylim=(0, 240))
    axes[0].legend(fontsize=9)
    j = read('july_sensitivity').query('receive_zero == True').sort_values('budget_fraction')
    axes[1].plot([0, 5, 10], j.adjusted_month_max, 'o-', color=BLUE, label='정상 29일만')
    axes[1].plot([0, 5, 10], j.mixed_month_max, 's--', color=ORANGE, label='오류 이틀 원값 유지')
    axes[1].set(title='7월 범위 민감도', xticks=[0, 5, 10], xlabel='이동 예산 (%)', ylim=(0, 240))
    axes[1].legend(fontsize=9, loc='lower left')
    for x, val in zip([5, 10], j.mixed_month_max.to_numpy()[1:]):
        axes[1].annotate(f'{val:.0f}', (x, val), xytext=(0, 8), textcoords='offset points', ha='center')
    finish(fig, '05A_monthly_maximum', '월최대는 모든 날짜를 다시 비교해 계산한다',
           '7월 본 결과는 정상 29일 · 오류 날짜 원값 유지 시 보조 최대 202 · 사후 재배치 가정, 실제 성과 아님')
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    rows = []
    for month, g in m.groupby('month'):
        g = g.set_index('budget_fraction')
        row = [f'{int(month[-2:])}월']
        for budget in [0, .05, .1]:
            r = g.loc[budget]
            dates = '·'.join(str(int(d[-2:])) for d in r.adjusted_max_dates.split('|'))+'일'
            row.append(f'{r.adjusted_month_max:.2f} / {dates}')
        rows.append(row)
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=['월', '원래 최대 / 날짜', '5% 후 최대 / 날짜', '10% 후 최대 / 날짜'],
                     cellLoc='center', colLoc='center', loc='center', colWidths=[.10, .28, .31, .31])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#E1E6EB')
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color='white', weight='bold')
        elif r % 2:
            cell.set_facecolor('#F1F5F8')
    finish(fig, '05B_maximum_dates', '최대 발생 날짜가 5%에서 6개월, 10%에서 7개월 바뀐다',
           '정상 날짜만 비교 · 7월 28·30일 동률 보존 · 0칸 정책 두 조건의 월최대 동일 · 조정 후 시각은 산출 안 함')


def main():
    font = style.setup()
    for fn in [distribution, transitions, caps, daily_scenarios, monthly]:
        fn()
    assert len(style.GENERATED) == 10
    (style.FIGURES/'manifest.json').write_text(json.dumps(dict(font=font, figures=style.GENERATED,
        image_reading=False, visual_review=False), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figures=len(style.GENERATED), font=font, image_reading=False)))


if __name__ == '__main__':
    main()
