"""Connected descriptive audit; January-August only, no model fitting."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import argparse
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
SOURCE = BASE / '../../data/origin/okm_augumented_2021.csv'
OUT = BASE / 'tables'
P = '09.23_007'
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['relations', 'temporal'], required=True)
args = parser.parse_args()
raw = pd.read_csv(SOURCE, encoding='utf-8-sig')
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
old = json.loads((OUT/'09.23_004_facts.json').read_text(encoding='utf-8'))
assert source_hash == old['sha256'], 'Previous summaries are stale'
df = raw.loc[raw['날짜'].lt(20210901) & raw['시간'].between(0, 23)].copy()
df.index = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d') + pd.to_timedelta(df['시간'], unit='h')
df = df.sort_index()
assert df.index.is_unique and len(df) == 5784
df['month'] = df.index.month
df['hour'] = df.index.hour
df['weekday'] = df.index.dayofweek
df['producing'] = df['생산량'].gt(0)
WEATHER = ['기온', '풍속', '습도', '강수량']
created = []

def save(frame, name):
    frame = pd.DataFrame(frame)
    file = OUT/f'{P}_{name}.csv'
    frame.to_csv(file, index=False, encoding='utf-8-sig')
    check = pd.read_csv(file, encoding='utf-8-sig')
    assert check.shape == frame.shape
    created.append(file.name)

def correlation(x, y):
    b = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()
    valid = len(b) > 2 and b.x.nunique() > 1 and b.y.nunique() > 1
    return {'n': len(b), 'pearson': b.x.corr(b.y) if valid else np.nan,
            'spearman': b.x.corr(b.y, method='spearman') if valid else np.nan}

if args.stage == 'relations':
    # Measurement/derived-value block; calendar codes are represented by strata.
    cols = ['15분', '30분', '45분', '60분', '평균', '생산량', *WEATHER,
            '전기요금(계절)', '공장인원', '인건비']
    for method in ['pearson', 'spearman']:
        matrix = df[cols].corr(method=method)
        assert np.allclose(matrix, matrix.T, equal_nan=True)
        save(matrix.reset_index(names='variable'), method+'_matrix')
    present = df[cols].notna().astype('int64')
    counts = present.T @ present
    assert np.array_equal(np.diag(counts), df[cols].count())
    save(counts.reset_index(names='variable'), 'pair_counts')
    # Explicit arithmetic relationships are more informative than a VIF cutoff.
    power_sum = df[['15분', '30분', '45분', '60분']].sum(axis=1)
    valid = df['공장인원'].notna() & power_sum.gt(0)
    identities = [
        {'rule':'mean_half_up', 'n':len(df), 'max_absolute_error':float((df['평균']-np.floor(power_sum/4+.5)).abs().max())},
        {'rule':'staff_production_over_power_sum', 'n':int(valid.sum()),
         'max_absolute_error':float((df.loc[valid,'공장인원']-df.loc[valid,'생산량']/power_sum.loc[valid]).abs().max())}]
    save(identities, 'identities')
    groups = [[], ['hour','weekday'], ['month','hour','weekday'], ['month','hour','weekday','producing']]
    conditional, bins, monthly = [], [], []
    for pop, block in [('all',df), ('positive',df.loc[df.producing])]:
        for col in ['생산량', *WEATHER]:
            b = block.dropna(subset=[col,'평균']).copy()
            for keys in groups:
                if keys:
                    sizes = b.groupby(keys)['평균'].transform('size')
                    keep = sizes.ge(5)
                    usable = b.loc[keep]
                    centered = usable[[col,'평균']] - usable.groupby(keys)[[col,'평균']].transform('mean')
                    values = correlation(centered[col], centered['평균'])
                    raw_same = correlation(usable[col], usable['평균'])
                    ng = usable.groupby(keys).ngroups
                else:
                    usable = b
                    values = correlation(b[col],b['평균'])
                    raw_same = values
                    ng = 1
                conditional.append({'population':pop,'variable':col,'controls':'+'.join(keys) or 'none',
                    'eligible_n':len(b),'dropped_sparse_n':len(b)-len(usable),'groups':ng,
                    **values, 'same_sample_raw_pearson':raw_same['pearson']})
            for month, mb in b.groupby('month'):
                center = mb[[col,'평균']] - mb.groupby(['hour','weekday'])[[col,'평균']].transform('mean')
                monthly.append({'population':pop,'variable':col,'month':month,
                    **correlation(mb[col],mb['평균']),
                    'within_hour_weekday_pearson':correlation(center[col],center['평균'])['pearson'],
                    'days':mb['날짜'].nunique()})
            if col not in WEATHER:
                continue
            # Fixed pooled quintile edges, not group-specific thresholds or model preprocessing.
            edges = np.unique(df[col].dropna().quantile(np.linspace(0,1,6)).to_numpy())
            b['bin'] = pd.cut(b[col], edges, include_lowest=True, duplicates='drop')
            keys = ['month','hour','weekday','producing']
            sizes = b.groupby(keys)['평균'].transform('size')
            b['power_centered'] = (b['평균']-b.groupby(keys)['평균'].transform('mean')).where(sizes.ge(5))
            for label, part in b.groupby('bin', observed=True):
                bins.append({'population':pop,'variable':col,'bin':str(label),'n':len(part),
                    'x_min':part[col].min(),'x_max':part[col].max(),'power_mean':part['평균'].mean(),
                    'power_median':part['평균'].median(),'power_p10':part['평균'].quantile(.1),
                    'power_p90':part['평균'].quantile(.9),'producing_share':part.producing.mean(),
                    'centered_n':int(part.power_centered.notna().sum()),
                    'centered_power_mean':part.power_centered.mean()})
            assert b['bin'].notna().all()
    save(conditional, 'conditional_correlations')
    save(monthly, 'monthly_correlations')
    save(bins, 'weather_bins')
    # Metadata clarifies that rain ties collapse bins; no zeros are discarded.
    print(pd.DataFrame(conditional).loc[lambda x:x.population.eq('all'),
        ['variable','controls','n','pearson','same_sample_raw_pearson']].round(4).to_string(index=False))
    print(pd.DataFrame(bins).loc[lambda x:x.variable.eq('기온')].round(3).to_string(index=False))

else:
    # Positive lag h means x(t-h) paired with power(t), with actual timestamp joins.
    # Calendar centering is descriptive and uses the exploratory period, not a deployable feature.
    cols = ['평균','생산량',*WEATHER]
    residual = df[cols] - df.groupby(['month','hour','weekday'])[cols].transform('mean')
    rows = []
    lags = [0, *range(1,49), 72, 168, 336]
    for lag in lags:
        for scale, frame in [('raw',df[cols]), ('calendar_centered',residual)]:
            shifted = frame.copy()
            shifted.index += pd.Timedelta(hours=lag)
            prior = shifted.reindex(df.index)
            for col in cols:
                rows.append({'scale':scale,'variable':col,'lag_hours':lag,
                    **correlation(prior[col],frame['평균'])})
    save(rows,'lag_correlations')
    # Full-day power signatures are retrospective audit labels, never model inputs.
    powercols = ['15분','30분','45분','60분']
    wide = df.pivot(index='날짜',columns='hour',values=powercols).dropna()
    assert wide.shape == (241,96)
    signatures = wide.apply(lambda row:tuple(row),axis=1)
    train_signatures = set(signatures.loc[signatures.index < 20210701])
    days = pd.DataFrame({'date':signatures.index,'signature':signatures.values})
    days['period'] = np.where(days.date.lt(20210701),'train','validation')
    days['seen_in_train'] = days.signature.isin(train_signatures)
    codes,_ = pd.factorize(days.signature)
    days['profile'] = codes
    profile = days.groupby('profile').agg(days=('date','size'),train_days=('period',lambda x:x.eq('train').sum()),
        validation_days=('period',lambda x:x.eq('validation').sum()),
        first_date=('date','min'),last_date=('date','max')).reset_index()
    save(profile,'profile_overlap')
    valdays = days.loc[days.period.eq('validation')]
    save(valdays.groupby('seen_in_train').agg(days=('date','size'),profiles=('profile','nunique')).reset_index(), 'validation_profile_coverage')
    old3 = json.loads((OUT/'09.23_003_facts.json').read_text(encoding='utf-8'))
    assert old3['source_sha256'] == source_hash
    pred = pd.read_csv(BASE/'../../data/processed/jsw/09.23_003_validation_predictions.csv',encoding='utf-8-sig',parse_dates=['timestamp'])
    assert pred.timestamp.is_unique and pred.timestamp.max() < pd.Timestamp('2021-09-01')
    dates = pred.timestamp.dt.strftime('%Y%m%d').astype(int)
    lookup = valdays.set_index('date')
    pred['seen'] = dates.map(lookup.seen_in_train)
    pred['profile'] = dates.map(lookup.profile)
    pred['month'] = pred.timestamp.dt.month
    assert pred.seen.notna().all()
    actual = df.reindex(pd.DatetimeIndex(pred.timestamp))
    assert np.array_equal(pred['mean'].to_numpy(),actual['평균'].to_numpy())
    scores = []
    for keys in [['seen'], ['month','seen']]:
        for key,b in pred.groupby(keys):
            key = key if isinstance(key,tuple) else (key,)
            fields = dict(zip(keys,key))
            for target in ['mean','peak']:
                for lag in [1,24,168]:
                    err = b[f'{target}_lag{lag}']-b[target]
                    pe = err.abs().groupby(b.profile).mean()
                    scores.append({'scope':'+'.join(keys),**fields,'target':target,'lag':lag,
                        'n':len(b),'days':b.timestamp.dt.normalize().nunique(),'profiles':b.profile.nunique(),
                        'mae':err.abs().mean(),'rmse':np.sqrt((err**2).mean()),'bias':err.mean(),
                        'equal_profile_mae':pe.mean()})
    save(scores,'profile_baseline_scores')
    previous = pd.read_csv(OUT/'09.23_003_scores.csv',encoding='utf-8-sig')
    for target in ['mean','peak']:
        for lag in [1,24,168]:
            expected = previous.loc[previous.target.eq(target)&previous.method.eq(f'lag{lag}'),'mae'].iloc[0]
            assert np.isclose((pred[f'{target}_lag{lag}']-pred[target]).abs().mean(),expected)
    print(pd.DataFrame(rows).loc[lambda x:x.lag_hours.isin([0,1,24,168])].round(4).to_string(index=False))
    print(valdays.groupby('seen_in_train').size().to_string())
    print(pd.DataFrame(scores).loc[lambda x:x.scope.eq('seen')&x.target.eq('mean')].round(3).to_string(index=False))

facts = {'time':datetime.now().astimezone().isoformat(timespec='seconds'),'stage':args.stage,
    'source_sha256':source_hash,'raw_rows':len(raw),'analysis_rows':len(df),'days':df['날짜'].nunique(),
    'reserved_september_rows':int(raw['날짜'].ge(20210901).sum()),'files':created,
    'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
    'checks':'source hash, unique valid timestamps, CSV roundtrip shapes; stage-specific numerical checks passed'}
(OUT/f'{P}_{args.stage}_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(facts,ensure_ascii=False,indent=2))
