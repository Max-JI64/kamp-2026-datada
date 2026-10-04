"""A04 final diagnosis: exact-mean support composition, not a threshold search."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from a01_production_changes import load
from a04_maximum_conditions import OUT, KEYS, association, save
from a04_terminal_unique import compare


def main():
    source = OUT / 'observations_triples.csv'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT / 'support_contract.json').write_text(json.dumps(dict(
        reason='Exact prior-mean matching has1039of5778hours, Jan-Apr r-.009 vs whole-control r+.394. Check support and observed low-power composition rather than alter matching bins.',
        comparison='Exact same calendar, prior unrounded mean and known0/positive state; require3dates. Describe retained/omitted records and cells with varying terminal values. Compare same selected subset under original controls.',
        low_power='Use existing EDA original rounded average20..26 at observed t-1, including zeros/extremes in other group. No new peak threshold or selection on future outcomes.',
        stop='Report support restriction and remaining heterogeneity; no more cutoff/transformation searches.'), ensure_ascii=False, indent=2), encoding='utf-8')
    sample = pd.read_csv(source, encoding='utf-8-sig', parse_dates=['timestamp','date','input_timestamp'])
    data, _ = load()
    previous = data.set_index('timestamp').reindex(sample.input_timestamp)
    sample['known_low_power'] = previous['평균'].between(20,26).to_numpy()
    exact_keys = KEYS + ['past_level','known_production_positive']
    keep = sample.groupby(exact_keys).date.transform('nunique').ge(3)
    vary = sample.groupby(exact_keys).prior_last.transform('nunique').gt(1)
    sample['exact_supported'] = keep
    sample['terminal_varies'] = vary
    support = []
    for period, scope in dict(all=sample, jan_apr=sample.loc[sample.month.le(4)],
                             may_jun=sample.loc[sample.month.between(5,6)],
                             jul_aug=sample.loc[sample.month.ge(7)]).items():
        for group, cell in [('available',scope),('exact_mean_supported',scope.loc[scope.exact_supported]),
                            ('supported_terminal_varies',scope.loc[scope.exact_supported & scope.terminal_varies])]:
            support.append(dict(period=period,group=group,n=len(cell),days=cell.date.nunique(),
                                known_low_hours=int(cell.known_low_power.sum()),
                                known_low_fraction=cell.known_low_power.mean(),
                                known_zero_hours=int(cell.past_production.eq(0).sum()),
                                prior_mean_median=cell.past_level.median(),prior_mean_q90=cell.past_level.quantile(.9),
                                target_maximum_median=cell.target_maximum.median(),
                                target_maximum_q90=cell.target_maximum.quantile(.9)))
    save(pd.DataFrame(support),'exact_mean_support')
    results=[]
    groups = dict(exact_supported=sample.loc[keep], known_low=sample.loc[sample.known_low_power],
                  known_other=sample.loc[~sample.known_low_power])
    for scope, cell in groups.items():
        for mode in ['date_hour','profile_balanced']:
            for row in compare(cell, mode, adjustment='mean_maximum'):
                results.append(dict(scope=scope,**row))
            for row in compare(cell, mode, adjustment='exact_mean'):
                results.append(dict(scope=scope,**row))
    for scope, cell in dict(jan_apr_exact=sample.loc[keep & sample.month.le(4)],
                            may_jun_exact=sample.loc[keep & sample.month.between(5,6)],
                            jul_aug_exact=sample.loc[keep & sample.month.ge(7)]).items():
        for row in compare(cell,adjustment='mean_maximum'):
            results.append(dict(scope=scope,**row))
    frame = pd.DataFrame(results)
    save(frame,'support_associations')
    assert len(sample) == 5778 and int(keep.sum()) == 1039
    assert source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    audit = dict(status='passed',source_observations_unchanged=True,
                 support_hours=1039,total_hours=5778,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 outputs_sha256={name:hashlib.sha256((OUT/f'{name}.csv').read_bytes()).hexdigest()
                                 for name in ['exact_mean_support','support_associations']})
    (OUT/'support_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(support).to_json(orient='records'))
    print(frame.loc[frame.feature.eq('prior_last')].to_json(orient='records'))


if __name__ == '__main__':
    main()
