"""Independent source/count/weight checks for the July weather follow-up."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from eda_analysis import OUT


def verify(raw, checks):
    data = raw.sort_values(['날짜','시간']).copy()
    dates = pd.to_datetime(data['날짜'].astype(str),format='%Y%m%d')
    data['month'] = dates.dt.month
    data['hour'] = data['시간']
    data['weekend'] = dates.dt.dayofweek.ge(5)
    data['high'] = data[['15분','30분','45분','60분']].to_numpy().max(axis=1)>=187
    profiles = {day:tuple(part[['15분','30분','45분','60분']].to_numpy().ravel())
                for day,part in data.groupby('날짜')}
    counts = Counter(profiles.values())
    data['profile_weight'] = data['날짜'].map({day:1/counts[profile] for day,profile in profiles.items()})
    read = lambda name: pd.read_csv(OUT/f'{name}.csv',encoding='utf-8-sig')
    monthly = read('external_month_frequency')
    for row in monthly.itertuples():
        part = data.loc[data.month.eq(row.month)]
        assert row.n==len(part) and row.high_hours==int(part.high.sum())
        np.testing.assert_allclose(row.high_rate,part.high.mean())
        hour_rates = part.groupby('hour').high.mean()
        assert row.hours_with_any_high==int(hour_rates.gt(0).sum())
        assert row.hours_with_rate_ge_20pct==int(hour_rates.ge(.2).sum())
    calendar = read('external_calendar_frequency')
    for row in calendar.itertuples():
        part = data.loc[data.month.eq(row.month) & data.weekend.eq(row.weekend)
                        & data['생산량'].gt(0).eq(row.producing)]
        assert row.n==len(part) and row.high_hours==int(part.high.sum())
        np.testing.assert_allclose(row.high_rate,part.high.mean())
    base = data.loc[data.month.between(6,8) & ~data.weekend & data['생산량'].gt(0)].copy()
    months = read('external_positive_weekday_months')
    for row in months.itertuples():
        part=base.loc[base.month.eq(row.month)]
        assert row.n==len(part) and row.days==part['날짜'].nunique() and row.high_hours==int(part.high.sum())
        rate=sum(len(base.loc[base.hour.eq(h)])/len(base)*part.loc[part.hour.eq(h)].high.mean() for h in range(24))
        np.testing.assert_allclose(row.hour_standardized,rate)
    weeks=read('external_week_sensitivity')
    base['week'] = pd.to_datetime(base['날짜'].astype(str),format='%Y%m%d').dt.isocalendar().week.to_numpy()
    for row in weeks.itertuples():
        retained=base.loc[base.month.isin([6,7]) & base.week.ne(row.excluded_week)]
        assert row.supported
        for month in [6,7]:
            current=retained.loc[retained.month.eq(month)]
            assert getattr(row,f'n_{month}')==len(current)
            rate=sum(sum(base.hour==h)/len(base)*current.loc[current.hour.eq(h)].high.mean() for h in range(24))
            np.testing.assert_allclose(getattr(row,f'rate_{month}'),rate)
        np.testing.assert_allclose(row.delta,row.rate_7-row.rate_6)
    facts=json.loads((OUT/'external_manifest.json').read_text(encoding='utf-8'))
    bands, contrasts, cell_table = read('external_band_frequency'),read('external_common_support'),read('external_common_cells')
    for row in contrasts.itertuples():
        definition=facts['bands'][row.variable]
        if row.variable=='기온+습도':
            part=base.dropna(subset=['기온','습도']).copy()
            cuts=[float(np.median(part['기온'])),float(np.median(part['습도']))]
            np.testing.assert_allclose(cuts,[definition['temperature_median'],definition['humidity_median']])
            part['band']=2*(part['기온']>cuts[0]).astype(int)+(part['습도']>cuts[1]).astype(int)
        else:
            part=base.dropna(subset=[row.variable]).copy()
            edges=definition['edges']
            if row.variable=='강수량':
                part['band']=(part[row.variable]>0).astype(int)
            else:
                np.testing.assert_allclose(edges,sorted(set(np.quantile(part[row.variable],[.25,.5,.75]))))
                part['band']=[sum(value>edge for edge in edges) for value in part[row.variable]]
            for rate in bands.loc[bands.variable.eq(row.variable)].itertuples():
                block=part.loc[part.month.eq(rate.month) & part.band.eq(rate.band)]
                assert rate.n==len(block) and rate.high_hours==int(block.high.sum())
                np.testing.assert_allclose(rate.high_rate,block.high.mean())
        pair=part.loc[part.month.isin([6,7])]
        blocks={key:cell for key,cell in pair.groupby(['hour','band'])
                if sum(cell.month==6)>=5 and sum(cell.month==7)>=5}
        total=sum(len(block) for block in blocks.values())
        assert row.common_cells==len(blocks)
        retained=pd.concat(blocks.values())
        assert row.common_hours==retained.hour.nunique()
        for month in [6,7]:
            current=retained.loc[retained.month.eq(month)]
            assert getattr(row,f'n_{month}')==len(current) and getattr(row,f'days_{month}')==current['날짜'].nunique()
            byhour=sum(len(group)/total*current.loc[current.hour.eq(h)].high.mean()
                       for h,group in retained.groupby('hour'))
            rate,weighted=0.,0.
            for (hour,band),block in blocks.items():
                cell=block.loc[block.month.eq(month)]
                weight=len(block)/total
                rate+=weight*cell.high.mean()
                weighted+=weight*np.dot(cell.high,cell.profile_weight)/cell.profile_weight.sum()
                saved=cell_table.loc[cell_table.variable.eq(row.variable) & cell_table.month.eq(month)
                                     & cell_table.hour.eq(hour) & cell_table.band.eq(band)].iloc[0]
                assert saved.n==len(cell) and saved.high_hours==int(cell.high.sum())
                np.testing.assert_allclose(saved.pooled_weight,weight)
            for metric,value in [('raw',current.high.mean()),('hour_only',byhour),('hour_band',rate),('profile_hour_band',weighted)]:
                np.testing.assert_allclose(getattr(row,f'{metric}_{month}'),value)
            np.testing.assert_allclose(getattr(row,f'coverage_{month}'),len(current)/sum(pair.month==month))
        for metric in ['raw','hour_only','hour_band','profile_hour_band']:
            np.testing.assert_allclose(getattr(row,f'delta_{metric}'),getattr(row,f'{metric}_7')-getattr(row,f'{metric}_6'))
    checks.append(dict(name='weather_high_frequency_raw_and_support',rows=len(monthly)+len(calendar)+len(months)+len(bands)+len(contrasts)+len(cell_table)+len(weeks),
                       numeric_columns=6,status='passed'))
