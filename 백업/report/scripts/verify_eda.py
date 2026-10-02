"""Compare report reproduction against stored EDA evidence without reading images.

Run after eda_analysis.py. Reference files are needed only for this audit,
not for analysis or plotting. No reference file is modified.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
OUT = REPORT/"tables/eda"
REF = ROOT/"EDA/jsw/tables"
CHECKS = []


def read(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def compare(label, actual, expected, keys, values):
    a = actual.set_index(keys).sort_index()[values]
    b = expected.set_index(keys).sort_index()[values]
    pd.testing.assert_index_equal(a.index, b.index, exact=False)
    np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                               rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=label)
    CHECKS.append(dict(name=label, rows=len(a), numeric_columns=len(values), status="passed"))


def main():
    for name in ["calendar_summary", "calendar_tests"]:
        expected = read(REF/f"10.02_028_{'summary' if name.endswith('summary') else 'tests'}.csv")
        actual = read(OUT/f"{name}.csv")
        keys = ["contrast", "group", "metric"] if name.endswith("summary") else ["contrast", "metric"]
        values = ["days", "mean", "median", "profile_weighted_mean", "producing_hour_share"] if name.endswith("summary") else [
            "days_a", "days_b", "difference_b_minus_a", "t", "p_raw", "rotation_p", "p_holm", "rotation_p_holm"]
        compare(name, actual, expected, keys, values)
    old = read(REF/"09.29_025_visibility.csv")
    old = old.loc[old.scope.isin(["Jan-Jun_EDA_reference", "Jul-Aug_EDA_reference"])
                  & old.weighting.eq("date_hour")].copy()
    old["period"] = old.scope.str.replace("_EDA_reference", "", regex=False)
    old = old.rename(columns={"peak_high_units": "high_hours", "hidden_units": "hidden_hours"})
    compare("mean_peak", read(OUT/"mean_peak_visibility.csv"), old,
            ["period", "mean_definition"], ["normal_hours", "threshold", "high_hours", "hidden_hours", "hidden_rate"])
    compare("hour_frequency", read(OUT/"hour_patterns.csv"), read(REF/"09.28_024_hour_comparison.csv"),
            ["period", "hour"], ["valid_hours", "high_hours_ge_182", "monthly_max_tied_slots"])
    old = read(REF/"09.28_024_monthly_max_ties.csv").rename(columns={"date": "날짜"})
    old["month"] %= 100
    compare("monthly_maximum", read(OUT/"monthly_maxima.csv"), old,
            ["month", "날짜", "hour", "slot"], ["power"])
    old = read(REF/"09.23_007_strata_sensitivity.csv")
    old = old.loc[old.controls.eq("month+hour+weekend+producing") & old.minimum_group_n.eq(5)]
    compare("conditional_relations", read(OUT/"conditional_correlations.csv"), old,
            ["population", "variable"], ["eligible_n", "n", "groups", "raw_same_sample", "centered_pearson"])
    old = read(REF/"09.23_007_monthly_lags.csv")
    old = old.loc[old.variable.eq("평균") & old.lag_hours.isin([1, 24, 168])]
    now = read(OUT/"lag_correlations.csv")
    compare("monthly_lags", now.loc[now.month.gt(0)], old,
            ["month", "lag_hours"], ["n", "pearson"])
    old = read(REF/"09.23_007_lag_correlations.csv")
    old = old.loc[old.scale.eq("raw") & old.variable.eq("평균") & old.lag_hours.isin([1, 24, 168])]
    compare("overall_lags", now.loc[now.month.eq(0)], old, ["lag_hours"], ["n", "pearson"])
    old = read(REF/"09.28_024_low_load_monthly.csv")
    old["month"] %= 100
    compare("low_load", read(OUT/"low_load_monthly.csv"), old, ["month"],
            ["days", "unique_power_profiles", "q10_min", "q10_q25", "q10_median",
             "q10_q75", "q10_max", "q10_unique_profile_median"])
    # Calendar independence check: raw-derived report dates and outcomes vs previous daily data.
    old = read(ROOT/"data/processed/jsw/10.02_027_daily_calendar.csv")
    compare("daily_calendar", read(OUT/"daily_records.csv"), old, ["date"],
            ["daily_mean", "daily_peak", "producing_hours", "month", "group", "profile_weight"])
    old = read(REF/"10.02_026_cjh_audit/hourly_policy_comparison.csv")
    old = old.loc[old.policy.eq("jsw_valid_jan_aug")].rename(columns={
        "시간": "hour", "rows": "n", "mean_power": "power_mean", "mean_production": "production_mean"})
    compare("whole_period_hourly_rhythm", read(OUT/"hourly_overview.csv").query("population == 'all'"),
            old, ["hour"], ["n", "power_mean", "production_mean"])
    old = read(REF/"09.23_015_high_quarter_counts.csv").rename(columns={
        "high_quarters": "high_slots", "share_of_high_hours": "share"})
    compare("high_slot_counts", read(OUT/"high_slot_summary.csv"), old, ["high_slots"], ["hours", "share"])
    old = read(REF/"09.23_015_high_segment_patterns.csv")
    old["pattern"] = old.high_pattern.astype(str).str.zfill(4)
    now = read(OUT/"high_slot_patterns.csv")
    now["pattern"] = now.pattern.astype(str).str.zfill(4)
    compare("high_slot_patterns", now, old, ["pattern"], ["hours"])
    old = read(REF/"09.23_015_hour_context.csv").rename(columns={
        "시간": "hour", "all_hours": "n", "high_share_of_all_hours": "high_rate"})
    compare("whole_period_high_frequency", read(OUT/"hour_frequency_187.csv"), old,
            ["hour"], ["n", "high_hours", "high_rate"])
    old = read(REF/"09.23_007_pearson_matrix.csv").set_index("variable")
    variables = ["평균", "생산량", "기온", "풍속", "습도", "강수량"]
    expected = old.loc[variables, variables].rename_axis("x").reset_index().melt(
        id_vars="x", var_name="y", value_name="pearson")
    expected["x"] = expected.x.replace({"평균": "M"})
    expected["y"] = expected.y.replace({"평균": "M"})
    actual = read(OUT/"overall_correlations.csv")
    compare("overall_pearson", actual.loc[actual.x.ne("P") & actual.y.ne("P")], expected,
            ["x", "y"], ["pearson"])
    old = read(REF/"09.23_007_spearman_matrix.csv").set_index("variable")
    expected = old.loc[variables,variables].rename_axis("x").reset_index().melt(
        id_vars="x",var_name="y",value_name="spearman")
    expected['x'] = expected.x.replace({'평균':'M'})
    expected['y'] = expected.y.replace({'평균':'M'})
    compare("overall_spearman",actual.loc[actual.x.ne('P') & actual.y.ne('P')],expected,
            ['x','y'],['spearman'])
    points = read(OUT/'conditional_points.csv')
    summary = read(OUT/'conditional_correlations.csv').query("population == 'positive'")
    for row in summary.itertuples():
        part = points.loc[points.variable.eq(row.variable)]
        assert len(part) == row.n
        for x,y,metric in [('x','y','raw_spearman'),('x_centered','y_centered','centered_spearman')]:
            expected_r = np.corrcoef(rankdata(part[x]),rankdata(part[y]))[0,1]
            np.testing.assert_allclose(getattr(row,metric),expected_r,atol=1e-12)
    CHECKS.append(dict(name='conditional_rank_correlations',rows=len(summary),numeric_columns=2,status='passed'))
    # Independent raw-value checks for new summaries and threshold provenance.
    raw = read(ROOT/"data/origin/okm_augumented_2021.csv")
    scope = raw.loc[raw['날짜'].lt(20210901)]
    valid = scope.loc[scope['시간'].between(0, 23)].copy()
    slot_columns = ['15분', '30분', '45분', '60분']
    peak = valid[slot_columns].to_numpy().max(axis=1)
    def manual_quantile(values, q):
        values = sorted(values)
        rank = (len(values)-1)*q
        lo = int(np.floor(rank)); hi = int(np.ceil(rank))
        return values[lo]+(values[hi]-values[lo])*(rank-lo)
    assert manual_quantile(scope[slot_columns].to_numpy().max(axis=1), .95) == 187
    assert manual_quantile(peak, .95) == 186
    records = read(OUT/'high_slot_records.csv')
    source_high = valid.loc[peak >= 187].set_index(['날짜', '시간']).sort_index()
    actual_high = records.set_index(['날짜', 'hour']).sort_index()
    np.testing.assert_array_equal(actual_high[slot_columns].to_numpy(), source_high[slot_columns].to_numpy())
    assert len(records) == 287 and (records[slot_columns].ge(187).sum(axis=1) == records.high_slots).all()
    assert int(records.M.lt(187).sum()) == 194
    monthly = read(OUT/'monthly_hour_overview.csv')
    assert monthly.n.sum() == 5784 and monthly.high_hours.sum() == 287
    hourly = read(OUT/'hourly_overview.csv')
    for group in ('all', 'weekday', 'weekend'):
        assert hourly.loc[hourly.population.eq(group)].shape[0] == 24
    grouped = monthly.groupby('hour').apply(lambda x: np.average(x.power_mean, weights=x.n))
    np.testing.assert_allclose(grouped, hourly.query("population == 'all'").sort_values('hour').power_mean)
    for variable in ['M', '생산량', '기온', '풍속', '습도', '강수량']:
        source_x = valid['평균'].to_numpy() if variable == 'M' else valid[variable].to_numpy()
        mask = np.isfinite(source_x)
        row = actual.loc[actual.x.eq('P') & actual.y.eq(variable)].iloc[0]
        assert row.n == int(mask.sum())
        np.testing.assert_allclose(row.pearson, np.corrcoef(peak[mask], source_x[mask])[0,1])
        np.testing.assert_allclose(row.spearman,np.corrcoef(rankdata(peak[mask]),rankdata(source_x[mask]))[0,1])
    CHECKS.append(dict(name='new_raw_scope_and_shape_checks', rows=len(records), numeric_columns=4, status='passed'))
    from verify_external_conditions import verify
    verify(valid, CHECKS)
    # File existence/size only. No decoding, contact sheet, screenshot or visual review.
    manifest = json.loads((REPORT/"figures/eda/story_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["images"]) == 16
    for item in manifest["images"]:
        assert (REPORT/"figures/eda"/item["file"]).stat().st_size == item["bytes"] > 10000
    source = ROOT/"data/origin/okm_augumented_2021.csv"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
    result = dict(status="passed", checked_tables=len(CHECKS),
                  checked_rows=sum(x["rows"] for x in CHECKS), checks=CHECKS,
                  images_generated=16, image_analysis_performed=False,
                  validated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"))
    (OUT/"verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
