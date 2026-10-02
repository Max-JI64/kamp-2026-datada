"""Numerical reference checks and independent feasibility checks; no image decoding."""
# %% Imports and keyed comparisons
from pathlib import Path
from hashlib import sha256
import json
import re
from urllib.parse import unquote
import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[1]
TABLES = REPORT/'tables/analysis'
OLD = REPORT.parent/'analysis/jsw/tables'
RESULTS = []


def read(path):
    return pd.read_csv(path, encoding='utf-8-sig')


def compare(name, reference, keys, rename=None):
    actual, expected = read(TABLES/f'{name}.csv'), read(OLD/f'{reference}.csv')
    if rename:
        expected = expected.rename(columns=rename)
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    a, e = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, e.index, exact=False)
    columns = [c for c in a.columns if c in e and c != 'profile']
    assert columns
    for col in columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            np.testing.assert_allclose(a[col], e[col], rtol=1e-10, atol=1e-8, equal_nan=True,
                                       err_msg=f'{name}.{col}')
        else:
            pd.testing.assert_series_equal(a[col].fillna(''), e[col].fillna(''), check_dtype=False)
    RESULTS.append(dict(table=name, reference=reference, rows=len(a), compared_columns=columns, passed=True))


# %% Reproduced results vs original source tables
def main():
    compare('distribution', '09.27_037_summary', ['profile_weighted', 'month'])
    compare('distribution_bins', '09.27_037_bins', ['profile_weighted', 'month', 'metric', 'low', 'high'])
    compare('distribution_sensitivity', '09.27_037_sensitivity', ['profile_weighted', 'unit', 'comparison_month'])
    compare('distribution_exclusions', '09.27_037_exclusions', ['profile_weighted', 'unit', 'omitted', 'comparison_month'])
    compare('transition_strata', '09.27_003_strata', ['period', 'month', 'hour'])
    compare('transition_summary', '09.27_003_summary', ['period', 'profile_weighted'])
    compare('direct_caps', '09.28_044_direct_caps', ['date', 'cap'], {'cutoff': 'cap'})
    compare('redistribution', '09.28_044_redistribution', ['date', 'budget_fraction', 'receive_zero'])
    compare('group_summary', '09.28_045_group_summary', ['period', 'group', 'weighting', 'budget_fraction', 'receive_zero'])
    compare('monthly_max', '09.28_045_monthly_max', ['month', 'budget_fraction', 'receive_zero'])
    compare('july_sensitivity', '09.28_045_excluded_date_sensitivity', ['month', 'budget_fraction', 'receive_zero'],
            {'adjusted_normal_days': 'complete_days', 'error_candidate_max': 'excluded_original_max',
             'adjusted_normal_month_max': 'adjusted_month_max', 'auxiliary_max_errors_unchanged': 'mixed_month_max',
             'auxiliary_reduction_from_222': 'mixed_reduction'})

    # Independently reconstruct a feasible 96-slot allocation for each computed cap.
    # It is a numerical witness only, never an operational schedule or a report chart.
    raw = read(REPORT.parent/'data/origin/okm_augumented_2021.csv')
    valid = raw.loc[raw['날짜'].lt(20210901) & raw['시간'].between(0, 23)].sort_values(['날짜', '시간'])
    slots = ['15분', '30분', '45분', '60분']
    arrays = {pd.to_datetime(str(d), format='%Y%m%d').strftime('%Y-%m-%d'): g[slots].to_numpy(dtype=float).ravel()
              for d, g in valid.groupby('날짜')}
    sims = read(TABLES/'redistribution.csv')
    for row in sims.itertuples():
        v, cap = arrays[row.date], row.optimal_max
        q = np.minimum(v, cap)
        required = (v-q).sum()
        eligible = np.ones(96, dtype=bool) if row.receive_zero else v > 0
        capacity = np.where(eligible, np.maximum(cap-q, 0), 0)
        if required > 1e-9:
            q += capacity * (required/capacity.sum())
        assert np.isclose(q.sum(), v.sum(), atol=1e-7, rtol=0)
        assert q.min() >= -1e-9 and q.max() <= cap+1e-7
        assert (np.maximum(v-q, 0)).sum() <= row.budget_value_sum+1e-7
        if not row.receive_zero:
            assert np.all(q[v == 0] == 0)
        # Just below the answer must violate the movement budget or receive capacity.
        lower = cap-1e-6
        need = np.maximum(v-lower, 0).sum()
        room = np.maximum(lower-v[eligible], 0).sum()
        assert need > row.budget_value_sum+1e-9 or need > room+1e-9
    np.testing.assert_allclose(sims.loc[sims.budget_fraction.eq(0), 'max_reduction'], 0)
    for _, group in sims.groupby(['date', 'receive_zero']):
        assert (np.diff(group.sort_values('budget_fraction').optimal_max) <= 1e-8).all()

    manifest = json.loads((TABLES/'run_manifest.json').read_text(encoding='utf-8'))
    for filename, info in manifest['files'].items():
        assert sha256((TABLES/filename).read_bytes()).hexdigest() == info['sha256']
    images = json.loads((REPORT/'figures/analysis/manifest.json').read_text(encoding='utf-8'))
    assert len(images['figures']) == 10
    for fig in images['figures']:
        p = REPORT/'figures/analysis'/fig['file']
        assert p.is_file() and p.stat().st_size == fig['bytes'] and fig['bytes'] > 1000
        assert not fig['visual_review']
    doc = REPORT/'10.02_003_analysis.md'
    links = 0
    if doc.exists():
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if target.startswith(('http:', 'https:')):
                continue
            path = Path(unquote(target.strip('<>')).split('#')[0])
            if not path.is_absolute():
                path = doc.parent/path
            assert path.exists(), f'Missing link {path}'
            links += 1
        assert doc.read_text(encoding='utf-8').count('![') == 10
    result = dict(passed=True, reference_tables=len(RESULTS), compared_rows=sum(r['rows'] for r in RESULTS),
                  comparisons=RESULTS, feasibility_and_local_optimality_checks=len(sims),
                  zero_budget_and_monotonicity=True, png_files=10, image_reading=False,
                  visual_review=False, markdown_links_checked=links)
    (TABLES/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'comparisons'}))


if __name__ == '__main__':
    main()
