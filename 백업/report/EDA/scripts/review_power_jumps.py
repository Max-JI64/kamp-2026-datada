"""Descriptive review of level vs adjacent-slot jumps, not a deployed detector.

Run with regular CPython 3.13. No manuscript edits or source-row deletion.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'data/origin/okm_augumented_2021.csv'
OUT = ROOT / 'report/EDA/tables/power_jumps_review'
SHA = '8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
OUT.mkdir(parents=True, exist_ok=True)

def dump(name, obj):
    (OUT / (name + '.json')).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    dump('plan', {
        'previous_evidence': 'POT found a possible absolute tail region, not a unique physical boundary. User now asks about 60→90 and 100→140 regardless of absolute level.',
        'question': 'Can adjacent 15-minute slot changes describe sudden rises simply, and do these rises immediately fall back?',
        'scope': 'Jan-Aug 2021 valid hours, whole-period descriptive EDA; no chronological train/test split.',
        'definition': 'delta = current slot - immediately preceding slot; gaps never bridged. The four columns are ordered interval values, not instantaneous sensor readings.',
        'candidate_rule': 'One-sided modified Z > 3.5 on signed adjacent changes. Threshold C = median(delta) + 3.5 * MAD(delta) / 0.6745; also require positive delta.',
        'selection_reason': 'Published potential-outlier labeling convention, not a desired percentage, an electricity standard, or a calibrated false-alarm guarantee. Mixture/dependence can make it inappropriate.',
        'validation': 'Inspect distribution and concentration by clock slot; compare k=3/3.5/4 and profile-weight sensitivity. Never tune for a desired flagged rate.',
        'immediate_spike': 'current-prev>C AND current-next>C with both neighbors adjacent; this is a single-slot rise-and-return, not every possible peak shape. Uses subsequent observation, retrospective only.',
        'sources': ['https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html'],
        'manuscript_edited': False,
    })
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    raw = pd.read_csv(SOURCE, encoding='utf-8-sig')
    period = raw.loc[raw['날짜'].between(20210101, 20210831)]
    valid = period.loc[period['시간'].between(0, 23)].sort_values(['날짜', '시간']).copy()
    cols = ['15분', '30분', '45분', '60분']
    assert len(valid) == 5784 and len(period) - len(valid) == 48
    assert not valid.duplicated(['날짜', '시간']).any()
    assert not valid[cols].isna().any().any()
    hours = pd.to_datetime(valid['날짜'].astype(str)) + pd.to_timedelta(valid['시간'], unit='h')
    times = np.repeat(hours.to_numpy(), 4) + np.tile(np.arange(4) * np.timedelta64(15, 'm'), len(valid))
    d = pd.DataFrame({'time': times, 'power': valid[cols].to_numpy().ravel()})
    d['date'] = d.time.dt.strftime('%Y-%m-%d')
    d['slot'] = d.time.dt.hour * 4 + d.time.dt.minute // 15
    d['weekday'] = d.time.dt.dayofweek < 5
    adjacent = d.time.diff().eq(pd.Timedelta(minutes=15))
    d['prev'] = d.power.shift().where(adjacent)
    d['delta'] = d.power - d.prev
    next_adjacent = d.time.shift(-1).sub(d.time).eq(pd.Timedelta(minutes=15))
    d['next'] = d.power.shift(-1).where(next_adjacent)
    x = d.delta.dropna().to_numpy()
    med = float(np.median(x)); mad = float(np.median(np.abs(x - med)))
    assert mad > 0
    c = med + 3.5 * mad / .6745
    d['rise'] = d.delta.gt(c) & d.delta.gt(0)
    d['single_slot_spike'] = d.rise & (d.power - d['next']).gt(c)
    d['legacy_high'] = d.power.ge(187)
    d['category'] = np.select([d.rise & d.legacy_high, d.rise, d.legacy_high], ['both', 'rise_only', 'high_only'], default='neither')
    d['boundary'] = d.time.dt.minute.eq(0)
    d['z'] = .6745 * (d.delta - med) / mad
    d['percent_rise'] = d.delta.div(d.prev.where(d.prev.gt(0))) * 100
    sensitivity = []
    for k in [3, 3.5, 4]:
        cut = med + k * mad / .6745
        flag = d.delta.gt(cut) & d.delta.gt(0)
        sensitivity.append({'k': k, 'cut': cut, 'n': int(flag.sum()), 'rate': float(flag.sum() / len(x))})
    # Equal-weight daily-profile diagnostic; only within-day pairs so omitted
    # dates never become neighbors. Original dataset and all rows retained.
    profiles = []; seen = set()
    for _, group in valid.groupby('날짜'):
        p = tuple(group[cols].to_numpy().ravel())
        if p not in seen:
            seen.add(p); profiles.extend(np.diff(p))
    pm = float(np.median(profiles)); p_mad = float(np.median(np.abs(np.array(profiles) - pm)))
    within = d.loc[d.date.eq(d.date.shift()), 'delta'].dropna()
    wm = float(within.median()); wmad = float((within - wm).abs().median())
    # Slot composition describes repetition; not a claim of abnormal operation.
    slot = d.groupby('slot').agg(valid_pairs=('delta', 'count'), rises=('rise', 'sum'), spikes=('single_slot_spike', 'sum'), median_delta=('delta', 'median'))
    slot['rate'] = slot.rises / slot.valid_pairs
    slot.to_csv(OUT / 'clock_slots.csv', encoding='utf-8-sig')
    pd.DataFrame(sensitivity).to_csv(OUT / 'sensitivity.csv', index=False, encoding='utf-8-sig')
    d.to_csv(OUT / 'slots.csv', index=False, encoding='utf-8-sig')
    top_slots = slot.sort_values(['rises', 'slot'], ascending=[False, True]).head(5).reset_index()
    # Independently reconstruct each eligible difference directly from the
    # original 2D array, including valid hour/day boundaries.
    arr = valid[cols].to_numpy()
    independent = list(np.diff(arr, axis=1).ravel())
    for i in range(1, len(arr)):
        if hours.iloc[i] - hours.iloc[i-1] == pd.Timedelta(hours=1):
            independent.append(arr[i, 0] - arr[i-1, -1])
    assert np.array_equal(np.sort(independent), np.sort(x))
    assert int(d.rise.sum()) == int((x > c).sum())
    assert not d.loc[~adjacent, 'rise'].any()
    categories = d.loc[d.delta.notna(), 'category'].value_counts().to_dict()
    summary = {
        'source_sha256': SHA, 'hours': len(valid), 'slot_values': len(d), 'valid_adjacent_pairs': len(x),
        'segments': int((~adjacent).sum()), 'time_invalid_hours_excluded': 48,
        'timestamp_convention': 'Display each labeled interval at its start (0,15,30,45 minutes). This is a plotting convention, not verification of physical sensor timestamps.',
        'median_change': med, 'mad_change': mad, 'candidate_rise_cut': c,
        'integer_minimum_rise': int(np.floor(c)+1),
        'rise_count': int(d.rise.sum()), 'rise_rate': float(d.rise.sum()/len(x)),
        'single_slot_spikes': int(d.single_slot_spike.sum()),
        'rise_with_next_observed': int((d.rise & d['next'].notna()).sum()),
        'category_counts_valid_pairs': categories,
        'nonpositive_previous_pairs': int(d.prev.le(0).sum()),
        'quantiles_signed_change': {str(q): float(np.quantile(x,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]},
        'top_rise_slots': top_slots.to_dict('records'),
        'boundary_counts': d.groupby('boundary').agg(pairs=('delta','count'), rises=('rise','sum')).reset_index().to_dict('records'),
        'sensitivity': sensitivity,
        'profile_sensitivity': {'unique_days':len(seen),'within_day_all_cut':wm+3.5*wmad/.6745,'within_day_unique_cut':pm+3.5*p_mad/.6745},
        'verification': 'passed: independently reconstructed all changes from original four-column values; no gap crossing, no duplicate hours; source hash matched',
        'decision': 'Descriptive candidate for large upward changes, not a confirmed anomaly/physical peak threshold. Compare high-level and rising events separately; no p95 endorsement.',
        'manuscript_edited': False,
    }
    dump('summary', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
