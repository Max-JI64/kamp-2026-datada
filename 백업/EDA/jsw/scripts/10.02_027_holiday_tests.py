"""E027: daily calendar-group tests, multiplicity and temporal sensitivity."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from itertools import combinations
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent.parent
SOURCE = ROOT / 'data/origin/okm_augumented_2021.csv'
PREFIX = '10.02_027'
OUT = BASE / 'tables'
PROCESSED = ROOT / 'data/processed/jsw'
PROCESSED.mkdir(parents=True, exist_ok=True)
SHA = '8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
LABELS = ['평일 비공휴일', '평일 공휴일', '주말 비공휴일', '주말 공휴일']
METRICS = ['daily_mean', 'daily_peak']
KASI = 'https://astro.kasi.re.kr/life/post/calendardata'
MPM = 'https://www.mpm.go.kr/mpm/comm/newsPress/newsPressRelease/?boardId=bbs_0000000000000029&category=&cntId=3237&mode=view&pageIdx=9'
HOLIDAYS = [(20210101, '신정'), (20210211, '설 연휴'), (20210212, '설날'),
            (20210213, '설 연휴'), (20210301, '삼일절'), (20210505, '어린이날'),
            (20210519, '부처님오신날'), (20210606, '현충일'),
            (20210815, '광복절'), (20210816, '광복절 대체공휴일')]
CONFIG = {'source_sha256': SHA, 'period': '2021-01-01..2021-08-31',
          'invalid_hours': 'exclude outside 0..23; no reconstruction',
          'unit': 'complete 24-hour date', 'metrics': METRICS,
          'Q1': 'weekday nonholiday versus all named public holidays; nonholiday weekends excluded',
          'Q2': 'four groups, omnibus and all six pairwise comparisons',
          'holiday_definition': 'national named holidays and substitute holidays; Sundays alone are weekend, not named holiday; May 1 is not a national public holiday',
          'alpha': .05, 'primary_tests': 'Welch two-sample and Welch one-way ANOVA',
          'multiplicity': 'Holm: four primary tests; separately twelve pairwise tests',
          'temporal_sensitivity': '9999 independent month-wise circular rotations of the complete four-group label sequence, preserving outcomes in order and within-month group counts; absolute mean difference for two-group contrasts, between-group sum of squares for omnibus; exchangeability/stationarity assumption required',
          'rotation_B': 9999, 'seed': 20261002,
          'repeat_sensitivity': 'inverse daily-profile frequency within full Jan-Aug scope; descriptive only',
          'production': 'composition diagnosis, not a causal mediator adjustment',
          'external_scope': 'user-authorized calendar labels only; no external manufacturing observations',
          'holidays': HOLIDAYS, 'calendar_sources': [KASI, MPM]}

def save(frame, suffix):
    frame.to_csv(OUT / f'{PREFIX}_{suffix}.csv', index=False, encoding='utf-8-sig')

def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))))
    out = np.empty_like(values)
    out[order] = adjusted
    return out

def welch_anova(ns, means, variances):
    """Welch (1951) F approximation; supports leading simulation dimension."""
    k = len(ns)
    w = ns / variances
    total = w.sum(axis=-1)
    center = (w * means).sum(axis=-1) / total
    numerator = (w * (means - center[..., None]) ** 2).sum(axis=-1) / (k - 1)
    term = (((1 - w / total[..., None]) ** 2) / (ns - 1)).sum(axis=-1)
    f = numerator / (1 + 2 * (k - 2) * term / (k * k - 1))
    df2 = (k * k - 1) / (3 * term)
    return f, df2

def moments(y, codes):
    ns, means, variances = [], [], []
    for g in range(4):
        mask = codes == g
        n = mask.sum(axis=-1)
        sums = (mask * y).sum(axis=-1)
        squares = (mask * y ** 2).sum(axis=-1)
        ns.append(n)
        means.append(sums / n)
        variances.append((squares - sums ** 2 / n) / (n - 1))
    return np.stack(ns, -1), np.stack(means, -1), np.stack(variances, -1)

def two_moments(ns, means, variances, a, b):
    na, nb = ns[..., a], ns[..., b]
    difference = means[..., b] - means[..., a]
    va, vb = variances[..., a] / na, variances[..., b] / nb
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return difference, se, df, np.abs(difference / se)

def pooled_holiday(ns, means, variances):
    ids = [1, 3]
    n = ns[..., ids].sum(-1)
    m = (ns[..., ids] * means[..., ids]).sum(-1) / n
    v = (((ns[..., ids] - 1) * variances[..., ids] + ns[..., ids] * (means[..., ids] - m[..., None]) ** 2).sum(-1)) / (n - 1)
    return np.stack([ns[..., 0], n], -1), np.stack([means[..., 0], m], -1), np.stack([variances[..., 0], v], -1)

def main():
    # Freeze before reading outcomes and before performing tests.
    (OUT / f'{PREFIX}_frozen.json').write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding='utf-8')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    raw = pd.read_csv(SOURCE, encoding='utf-8-sig')
    scope = raw.loc[raw['날짜'].lt(20210901)].copy()
    valid = scope.loc[scope['시간'].between(0, 23)].copy()
    assert len(scope) == 5832 and len(valid) == 5784
    assert not valid.duplicated(['날짜', '시간']).any()
    assert valid.groupby('날짜').size().eq(24).all()
    assert not valid[['평균', '15분', '30분', '45분', '60분', '생산량']].isna().any().any()
    valid['hour_peak'] = valid[['15분', '30분', '45분', '60분']].max(axis=1)
    valid['producing'] = valid['생산량'].gt(0)
    daily = valid.groupby('날짜').agg(daily_mean=('평균', 'mean'), daily_peak=('hour_peak', 'max'),
                                      production_total=('생산량', 'sum'), producing_hours=('producing', 'sum')).reset_index().rename(columns={'날짜': 'date'})
    dates = pd.to_datetime(daily.date.astype(str), format='%Y%m%d')
    daily['month'] = dates.dt.month
    daily['weekday_mon0'] = dates.dt.dayofweek
    daily['is_weekend'] = daily.weekday_mon0.ge(5)
    daily['holiday_name'] = daily.date.map(dict(HOLIDAYS)).fillna('')
    daily['is_public_holiday'] = daily.holiday_name.ne('')
    daily['group'] = 2 * daily.is_weekend.astype(int) + daily.is_public_holiday.astype(int)
    daily['day_type'] = daily.group.map(dict(enumerate(LABELS)))
    profiles = pd.read_csv(OUT / '09.23_011_daily_profile_summary.csv', encoding='utf-8-sig')
    daily = daily.merge(profiles[['date', 'profile_id', 'profile_days']], on='date', validate='one_to_one')
    daily['profile_weight'] = 1 / daily.profile_days
    assert len(daily) == 241 and daily.is_public_holiday.sum() == 10
    assert daily.loc[daily.is_public_holiday, 'date'].tolist() == [x[0] for x in HOLIDAYS]
    daily.to_csv(PROCESSED / f'{PREFIX}_daily_calendar.csv', index=False, encoding='utf-8-sig')
    calendar = pd.DataFrame(HOLIDAYS, columns=['date', 'holiday_name'])
    calendar['source_url'] = [MPM if d == 20210816 else KASI for d, _ in HOLIDAYS]
    calendar['verified_on'] = '2026-10-02'
    save(calendar, 'holiday_calendar')
    summaries = []
    for g, part in daily.groupby('group'):
        for metric in METRICS:
            summaries.append({'group': int(g), 'day_type': LABELS[g], 'metric': metric,
                              'days': len(part), 'hours': 24 * len(part),
                              'mean': part[metric].mean(), 'median': part[metric].median(), 'sd': part[metric].std(),
                              'profile_weighted_mean': np.average(part[metric], weights=part.profile_weight),
                              'producing_hour_share': part.producing_hours.sum() / (len(part) * 24),
                              'unique_profiles': part.profile_id.nunique()})
    save(pd.DataFrame(summaries), 'group_summary')
    save(daily.loc[daily.is_public_holiday], 'holiday_days')
    save(daily.groupby(['month', 'day_type']).agg(days=('date', 'size'), mean_power=('daily_mean', 'mean'),
                                                 peak_mean=('daily_peak', 'mean')).reset_index(), 'monthly_groups')
    rng = np.random.default_rng(CONFIG['seed'])
    codes = daily.group.to_numpy()
    B = CONFIG['rotation_B']
    rotated = np.empty((B, len(daily)), dtype=np.int8)
    for month in range(1, 9):
        inds = np.flatnonzero(daily.month.to_numpy() == month)
        shifts = rng.integers(0, len(inds), size=B)
        rotated[:, inds] = codes[inds][(np.arange(len(inds))[None, :] - shifts[:, None]) % len(inds)]
    assert np.all(np.stack([(rotated == g).sum(1) for g in range(4)], -1) == np.bincount(codes))
    primary, pairs = [], []
    for metric in METRICS:
        y = daily[metric].to_numpy()
        n, m, v = moments(y, codes)
        rn, rm, rv = moments(y, rotated)
        qn, qm, qv = pooled_holiday(n, m, v)
        rqn, rqm, rqv = pooled_holiday(rn, rm, rv)
        diff, se, df, t = two_moments(qn, qm, qv, 0, 1)
        rdifference = np.abs(rqm[:, 1] - rqm[:, 0])
        crit = stats.t.ppf(.975, df)
        primary.append({'question': 'Q1', 'metric': metric, 'test': 'Welch t',
                        'statistic': float(t), 'df2': float(df), 'p_raw': float(2 * stats.t.sf(t, df)),
                        'rotation_p': (1 + np.count_nonzero(rdifference >= abs(diff))) / (B + 1),
                        'difference_holiday_minus_weekday': float(diff), 'ci_low_independent_days': float(diff - crit * se),
                        'ci_high_independent_days': float(diff + crit * se)})
        f, df2 = welch_anova(n, m, v)
        center = np.average(m, weights=n)
        ss = np.sum(n * (m - center) ** 2)
        rcenter = (rn * rm).sum(-1) / rn.sum(-1)
        rss = (rn * (rm - rcenter[:, None]) ** 2).sum(-1)
        assert np.isfinite(rss).all()
        scipy_f = stats.f_oneway(*(y[codes == g] for g in range(4)), equal_var=False)
        assert np.isclose(scipy_f.statistic, f) and np.isclose(scipy_f.pvalue, stats.f.sf(f, 3, df2))
        primary.append({'question': 'Q2', 'metric': metric, 'test': 'Welch ANOVA',
                        'statistic': float(f), 'df2': float(df2), 'p_raw': float(stats.f.sf(f, 3, df2)),
                        'rotation_p': (1 + np.count_nonzero(rss >= ss)) / (B + 1)})
        # Independent cross-check against SciPy's two-sample implementation.
        a = y[codes == 0]; b = y[np.isin(codes, [1, 3])]
        st = stats.ttest_ind(a, b, equal_var=False)
        assert np.isclose(abs(st.statistic), t) and np.isclose(st.pvalue, primary[-2]['p_raw'])
        for a, b in combinations(range(4), 2):
            diff, se, df, t = two_moments(n, m, v, a, b)
            rdifference = np.abs(rm[:, b] - rm[:, a])
            crit = stats.t.ppf(.975, df)
            pairs.append({'metric': metric, 'group_a': LABELS[a], 'group_b': LABELS[b],
                          'days_a': int(n[a]), 'days_b': int(n[b]), 'difference_b_minus_a': float(diff),
                          'ci_low_independent_days': float(diff - crit * se), 'ci_high_independent_days': float(diff + crit * se),
                          'p_raw': float(2 * stats.t.sf(t, df)),
                          'rotation_p': (1 + np.count_nonzero(rdifference >= abs(diff))) / (B + 1)})
    for results in (primary, pairs):
        frame = pd.DataFrame(results)
        frame['p_holm'] = holm(frame.p_raw)
        frame['rotation_p_holm'] = holm(frame.rotation_p)
        frame['both_below_05'] = (frame.p_holm < .05) & (frame.rotation_p_holm < .05)
        save(frame, 'primary_tests' if results is primary else 'pairwise_tests')
    # Leave one holiday out to quantify sparse-holiday influence; no p-value fishing.
    influence = []
    for metric in METRICS:
        base = daily.loc[daily.group.eq(0), metric].mean()
        holidays = daily.loc[daily.is_public_holiday]
        for _, day in holidays.iterrows():
            influence.append({'metric': metric, 'omitted_date': int(day.date),
                              'holiday_minus_weekday': holidays.loc[holidays.date.ne(day.date), metric].mean() - base})
    save(pd.DataFrame(influence), 'holiday_leave_one_out')
    facts = {'run_at': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
             'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
             'source_sha256': SHA, 'raw_rows': len(raw), 'scope_rows': len(scope), 'valid_rows': len(valid),
             'days': len(daily), 'group_days': daily.groupby('day_type').size().to_dict(),
             'zero_power_rows_preserved': int(valid[['15분', '30분', '45분', '60분']].eq(0).all(axis=1).sum()),
             'validation': 'hash, prior E011 profiles/date set, 24-hour completeness, 10 holiday dates, rotation group counts, finite unstudentized rotation statistics, SciPy Welch t and ANOVA cross-check'}
    assert facts['zero_power_rows_preserved'] == 17
    (OUT / f'{PREFIX}_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(facts, ensure_ascii=False))
    print(pd.read_csv(OUT / f'{PREFIX}_primary_tests.csv', encoding='utf-8-sig').to_string(index=False))

if __name__ == '__main__':
    main()
