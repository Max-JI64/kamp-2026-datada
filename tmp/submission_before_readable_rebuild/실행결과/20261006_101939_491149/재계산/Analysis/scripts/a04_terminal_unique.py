"""A04 third comparison: terminal-slot information beyond observed mean/max."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from a02_prior_power_signals import residuals, weighted_corr
from a04_maximum_conditions import OUT, KEYS, control_array, save

FEATURES = ['prior_last', 'prior_slot_trend', 'prior_last_above_mean', 'prior_maximum_excess']


def compare(sample, mode='date_hour', adjustment='mean_maximum'):
    keys = KEYS if adjustment != 'exact_mean' else KEYS + ['past_level', 'known_production_positive']
    sample = sample.loc[sample.groupby(keys).date.transform('nunique').ge(3)].copy()
    if sample.empty:
        return []
    weights = np.ones(len(sample)) if mode == 'date_hour' else sample.profile_weight.to_numpy()
    values = np.column_stack([rankdata(sample[key]) / len(sample) for key in FEATURES + ['target_maximum']])
    cells = pd.factorize(pd.MultiIndex.from_frame(sample[keys]))[0]
    controls = control_array(sample, include_level=adjustment != 'exact_mean')
    if adjustment in ['mean_maximum', 'mean_maximum_trend']:
        maximum = rankdata(sample.prior_maximum) / len(sample)
        controls = np.column_stack([controls, maximum, maximum ** 2, maximum ** 3])
    if adjustment == 'mean_maximum_trend':
        controls = np.column_stack([controls, rankdata(sample.prior_slot_trend) / len(sample)])
    res = residuals(values, cells, weights, controls)
    return [dict(feature=feature, adjustment=adjustment, weighting=mode, n=len(sample),
                 days=sample.date.nunique(), cells=int(cells.max() + 1),
                 correlation=weighted_corr(res[:, i], res[:, -1], weights))
            for i, feature in enumerate(FEATURES)
            if not (adjustment == 'mean_maximum_trend' and feature == 'prior_slot_trend')]


def main():
    source = OUT / 'observations_triples.csv'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT / 'unique_contract.json').write_text(json.dumps(dict(
        reason='Terminal slot association robust, but also associated with next mean. Need check redundancy with prior maximum and mean-adjustment assumptions before forecasting candidate.',
        comparison='Retain continuous next maximum; add observed prior maximum cubic ranks to fixed controls, then prior trend; exact prior mean+production0state matched cells as non-binned sensitivity.',
        periods='Jan-Apr,May-Jun,Jul-Aug separate descriptive associations only, no fitting or independent test claim.',
        stop='Candidate retained only as observed antecedent if direction survives; no forecast training/threshold change or favorable subgroup selection.'), ensure_ascii=False, indent=2), encoding='utf-8')
    sample = pd.read_csv(source, encoding='utf-8-sig', parse_dates=['timestamp','date','input_timestamp'])
    rows = []
    for mode in ['date_hour','profile_balanced']:
        for adjustment in ['mean_maximum', 'mean_maximum_trend', 'exact_mean']:
            for row in compare(sample, mode, adjustment):
                rows.append(dict(scope='all', **row))
    for scope, cell in dict(jan_apr=sample.loc[sample.month.le(4)],
                            may_jun=sample.loc[sample.month.between(5,6)],
                            jul_aug=sample.loc[sample.month.ge(7)]).items():
        for adjustment in ['mean_maximum', 'exact_mean']:
            for row in compare(cell, adjustment=adjustment):
                rows.append(dict(scope=scope, **row))
    result = pd.DataFrame(rows)
    save(result, 'terminal_unique_information')
    # Verify the new mean+maximum residual calculation against explicit controls.
    tiny = sample.loc[(sample.month == 1) & (sample.weekend == 0)].copy()
    cells = pd.factorize(pd.MultiIndex.from_frame(tiny[KEYS]))[0]
    maximum = rankdata(tiny.prior_maximum) / len(tiny)
    controls = np.column_stack([control_array(tiny), maximum, maximum ** 2, maximum ** 3])
    values = np.column_stack([rankdata(tiny.prior_last), rankdata(tiny.target_maximum)])
    weights = tiny.profile_weight.to_numpy()
    computed = residuals(values, cells, weights, controls)
    design = np.column_stack([np.eye(cells.max()+1)[cells], controls])
    fitted = np.linalg.lstsq(design*np.sqrt(weights[:,None]), values*np.sqrt(weights[:,None]), rcond=None)[0]
    np.testing.assert_allclose(computed, values-design@fitted, atol=1e-9)
    assert source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    audit = dict(status='passed', source_observations_unchanged=True, explicit_dummy_maximum_control_verified=True,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 output_sha256=hashlib.sha256((OUT/'terminal_unique_information.csv').read_bytes()).hexdigest())
    (OUT/'unique_verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result.to_json(orient='records'))


if __name__ == '__main__':
    main()
