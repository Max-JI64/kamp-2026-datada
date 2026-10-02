"""July high-power frequency follow-up, supplied data only, no model or image reading.

Question from monthly_hour_overview (E015/E024): does July's broadly high
frequency remain at the same hour and at shared production/weather levels?
Rules fixed before reading new outputs: P>=187; positive-production weekdays;
June-August pooled quartile boundaries (rain: zero/positive); common June/July
hour x band cells with >=5 records in EACH month; pooled-cell exposure weights.
Two standardized rates use exactly the same support. Variables are assessed
separately; temperature/humidity is also assessed jointly with median splits.
These are descriptive comparisons, not causal adjustments or model evaluation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from eda_analysis import OUT, save

VARIABLES = ["생산량", "기온", "풍속", "습도", "강수량"]


def standardized(part, keys, weights, weighted=False):
    rates = {}
    for key, cell in part.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        rates[key] = np.average(cell.high, weights=cell.profile_weight) if weighted else cell.high.mean()
    return sum(weight*rates[key] for key, weight in weights.items())


def analyze(frame, daily):
    data = frame.copy()
    data["high"] = data.P.ge(187)
    data["profile_weight"] = data['날짜'].map(daily.profile_weight)
    monthly = data.groupby("month").agg(n=("high", "size"), high_hours=("high", "sum"),
                                        high_rate=("high", "mean"))
    hourly = data.groupby(["month", "hour"]).high.mean().unstack("hour")
    monthly["hours_with_any_high"] = hourly.gt(0).sum(axis=1)
    monthly["hours_with_rate_ge_20pct"] = hourly.ge(.20).sum(axis=1)
    save(monthly.reset_index(), "external_month_frequency")
    contexts = []
    for (month, weekend, producing), part in data.groupby(["month", "weekend", "producing"]):
        contexts.append(dict(month=month, weekend=weekend, producing=producing,
                             n=len(part), high_hours=int(part.high.sum()), high_rate=part.high.mean()))
    save(pd.DataFrame(contexts), "external_calendar_frequency")
    base = data.loc[data.month.between(6, 8) & ~data.weekend & data.producing].copy()
    baseline = base.groupby(["month", "hour"]).high.agg(["size", "mean"])
    hour_weights = base.groupby("hour").size()/len(base)
    month_rows = []
    for month, part in base.groupby("month"):
        month_rows.append(dict(month=month, n=len(part), days=part['날짜'].nunique(),
                               high_hours=int(part.high.sum()), raw_rate=part.high.mean(),
                               hour_standardized=sum(hour_weights[h]*baseline.loc[(month,h), "mean"]
                                                     for h in hour_weights.index)))
    save(pd.DataFrame(month_rows), "external_positive_weekday_months")
    bands, records, rates, contrasts, cells = {}, [], [], [], []
    for variable in VARIABLES:
        part = base.dropna(subset=[variable]).copy()
        if variable == "강수량":
            edges = [0.0]
            part["band"] = part[variable].gt(0).astype(int)
            labels = ["0", ">0"]
        else:
            edges = sorted(set(float(x) for x in part[variable].quantile([.25,.5,.75])))
            part["band"] = np.searchsorted(edges, part[variable], side="left")
            labels = [f"≤{edges[0]:g}"] + [f"({a:g}, {b:g}]" for a,b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:g}"]
        bands[variable] = dict(edges=edges, labels=labels, pooled_n=len(part))
        part["variable"] = variable
        part["value"] = part[variable]
        records.append(part.reset_index()[["timestamp", "날짜", "month", "hour", "variable", "value", "band", "high", "profile_weight"]])
        for (month, band), cell in part.groupby(["month", "band"]):
            rates.append(dict(variable=variable, month=month, band=band, band_label=labels[band],
                              n=len(cell), days=cell['날짜'].nunique(), high_hours=int(cell.high.sum()),
                              high_rate=cell.high.mean(), median_value=cell[variable].median()))
        assess(part, variable, contrasts, cells)
    # Joint weather follow-up chosen beforehand, not after trying many cutoffs.
    part = base.dropna(subset=["기온", "습도"]).copy()
    temp_cut, humid_cut = float(part['기온'].median()), float(part['습도'].median())
    part["band"] = 2*part['기온'].gt(temp_cut).astype(int)+part['습도'].gt(humid_cut).astype(int)
    bands["기온+습도"] = dict(temperature_median=temp_cut, humidity_median=humid_cut,
                              rule="2*(temperature>median)+(humidity>median)")
    assess(part, "기온+습도", contrasts, cells)
    save(pd.concat(records, ignore_index=True), "external_band_records")
    save(pd.DataFrame(rates), "external_band_frequency")
    save(pd.DataFrame(contrasts), "external_common_support")
    save(pd.DataFrame(cells), "external_common_cells")
    facts = dict(threshold=187, population="June-August positive-production weekdays",
                 band_source="pooled June-August, exploratory, not training/validation cutoffs",
                 min_records_per_month_cell=5, comparison_months=[6,7], bands=bands,
                 sample_weights="pooled exposure on common hour x band cells",
                 profile_sensitivity="same cells and cell weights; inverse daily profile frequency within each cell",
                 inference="descriptive only; one variable at a time except predeclared temperature/humidity joint bands",
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 image_analysis_performed=False)
    facts['week_sensitivity'] = week_sensitivity()
    (OUT/"external_manifest.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status='external_conditions_completed',
                         july_high_hours=int(monthly.loc[7,'high_hours']),
                         positive_weekday_rows=len(base),common_support_comparisons=len(contrasts),
                         image_analysis_performed=False)))


def assess(part, variable, contrasts, cells):
    pair = part.loc[part.month.isin([6,7])].copy()
    count = pair.groupby(["hour", "band", "month"]).size().unstack("month", fill_value=0)
    support = count.index[count.reindex(columns=[6,7], fill_value=0).ge(5).all(axis=1)]
    row_keys = pd.MultiIndex.from_frame(pair[["hour", "band"]])
    kept = pair.loc[row_keys.isin(support)].copy()
    totals = kept.groupby(["hour", "band"]).size()
    cell_weights = {key: n/len(kept) for key,n in totals.items()} if len(kept) else {}
    hour_weights = {(h,): n/len(kept) for h,n in kept.groupby("hour").size().items()} if len(kept) else {}
    row = dict(variable=variable, common_cells=len(support), common_hours=kept.hour.nunique())
    for month in [6,7]:
        current = kept.loc[kept.month.eq(month)]
        row.update({f"n_{month}":len(current), f"days_{month}":current['날짜'].nunique(),
                    f"coverage_{month}":len(current)/int(pair.month.eq(month).sum()),
                    f"raw_{month}":current.high.mean(),
                    f"hour_only_{month}":standardized(current,["hour"],hour_weights) if len(current) else np.nan,
                    f"hour_band_{month}":standardized(current,["hour","band"],cell_weights) if len(current) else np.nan,
                    f"profile_hour_band_{month}":standardized(current,["hour","band"],cell_weights,True) if len(current) else np.nan})
    for metric in ["raw","hour_only","hour_band","profile_hour_band"]:
        row[f"delta_{metric}"] = row[f"{metric}_7"]-row[f"{metric}_6"]
    contrasts.append(row)
    for (hour, band, month), cell in kept.groupby(["hour","band","month"]):
        cells.append(dict(variable=variable,hour=hour,band=band,month=month,n=len(cell),
                          high_hours=int(cell.high.sum()),high_rate=cell.high.mean(),
                          profile_rate=np.average(cell.high, weights=cell.profile_weight),
                          pooled_weight=cell_weights[(hour,band)]))


def week_sensitivity():
    """Follow-up after the July difference survives hour composition/profile checks.

    Hold original pooled June-August hour weights fixed. Remove each observed
    ISO week from the June/July comparison once, without changing any threshold.
    A zero-support hour invalidates that run; do not silently reweight it.
    """
    data = pd.read_csv(OUT/'hourly_records.csv',encoding='utf-8-sig',float_precision='round_trip')
    base = data.loc[data.month.between(6,8) & ~data.weekend & data.producing].copy()
    base['high'] = base.P.ge(187)
    time = pd.to_datetime(base.timestamp)
    base['week'] = time.dt.isocalendar().week.to_numpy()
    weights = base.groupby('hour').size()/len(base)
    pair = base.loc[base.month.isin([6,7])]
    rows = []
    for week in sorted(pair.week.unique()):
        kept = pair.loc[pair.week.ne(week)]
        rates = kept.groupby(['month','hour']).high.mean()
        supported = all((m,h) in rates.index for m in [6,7] for h in weights.index)
        row = dict(excluded_week=int(week),n_6=int(sum(kept.month==6)),n_7=int(sum(kept.month==7)),supported=supported)
        for month in [6,7]:
            row[f'rate_{month}'] = sum(w*rates.loc[(month,h)] for h,w in weights.items()) if supported else np.nan
        row['delta'] = row['rate_7']-row['rate_6']
        rows.append(row)
    save(pd.DataFrame(rows),'external_week_sensitivity')
    values = [row['delta'] for row in rows if row['supported']]
    print(json.dumps(dict(week_runs=len(rows),supported_runs=len(values),
                         min_delta=min(values),max_delta=max(values))))
    return dict(question='Does the hour-standardized July-June gap rely on one ISO week?',
                weights='Original pooled June-August hour exposure weights, fixed in all runs',
                runs=len(rows),supported_runs=len(values),min_delta=min(values),max_delta=max(values))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--week-sensitivity',action='store_true')
    args = parser.parse_args()
    if not args.week_sensitivity:
        parser.error('For full generation use eda_analysis.py; this entry is the targeted week follow-up')
    week_sensitivity()
