"""Targeted independent reconciliations and representation-only output repairs."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_common import BASE, TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, json_value


def read(name):
    return pd.read_csv(TABLES / name, encoding="utf-8-sig")


def main():
    assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
    assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    source = raw.loc[raw["시간"].between(0, 23) & raw["날짜"].lt(20210901)].copy()
    keys = pd.to_datetime(source["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(source["시간"], unit="h")
    peak = source[["15분", "30분", "45분", "60분"]].max(axis=1)
    lookup = dict(zip(keys, peak))
    prev = np.array([lookup.get(key - pd.Timedelta(hours=1), np.nan) for key in keys])
    eligible = np.isfinite(prev) & (prev < 182)
    onset = eligible & (peak.to_numpy() >= 182)
    early = keys < pd.Timestamp("2021-07-01")
    assert int(onset[early].sum()) == 139 and int(onset[~early].sum()) == 85
    assert int(eligible[early].sum()) == 4108 and int(eligible[~early].sum()) == 1218
    record = read("09.27_004_records.csv")
    assert np.array_equal(record["peak"], peak.to_numpy())
    assert np.array_equal(record["onset"], onset)
    event = read("09.27_004_events.csv")
    assert event["hours"].sum() == int(peak.ge(182).sum()) == 455
    assert event["excess_sum"].sum() == (peak-182).clip(lower=0).sum() == 4215
    std = read("09.27_005_standardization.csv")
    cells = read("09.27_005_strata.csv")
    for _, row in std.iterrows():
        subset = cells.loc[cells["metric"].eq(row["metric"]) & cells["profile_weighted"].eq(row["profile_weighted"]) & cells["common"]]
        exposure = subset["earlier_exposure_weight"] + subset["later_exposure_weight"]
        for tag in ("earlier", "later"):
            expected = np.sum(exposure * subset[f"{tag}_event_weight"] / subset[f"{tag}_exposure_weight"]) / exposure.sum()
            assert np.isclose(expected, row[f"{tag}_standardized_rate"])
    clusters = read("09.27_008_cluster_days.csv")
    assert clusters.groupby(["representation", "k", "profile_weighted"]).size().eq(241).all()
    loadings = read("09.27_008_pca_loadings.csv")
    for _, part in loadings.groupby("representation", sort=False):
        components = part[[f"pc{k}" for k in range(1, 7)]].to_numpy().T
        assert np.allclose(components @ components.T, np.eye(6), atol=1e-10)
    quality = read("09.27_009_rounding_redundancy.csv").iloc[0]
    assert quality["classification_disagreement"] == 13
    assert quality["staff_formula_matches"] == 6151
    extended = read("09.27_009_extended_periods.csv")
    assert extended.loc[extended["period"].eq("Sep"), "hours"].eq(336).all()
    # A small output refinement: tied top-10 cutoffs are not unique rankings.
    daily = read("09.27_004_daily.csv")
    ranks = read("09.27_004_metric_rank_sensitivity.csv")
    for i, row in ranks.iterrows():
        part = daily.loc[daily["period"].eq(row["period"])]
        selected = []
        for label, column in (("peak", "daily_peak"), ("mean", "exact_daily_mean")):
            cutoff = part[column].nlargest(10).min()
            ranks.loc[i, f"{label}_top10_cutoff"] = cutoff
            ranks.loc[i, f"{label}_dates_at_cutoff"] = int(part[column].eq(cutoff).sum())
            ranks.loc[i, f"{label}_inclusive_dates"] = int(part[column].ge(cutoff).sum())
            selected.append(set(part.loc[part[column].ge(cutoff), "date"]))
        ranks.loc[i, "inclusive_intersection"] = len(selected[0] & selected[1])
        ranks.loc[i, "inclusive_jaccard"] = len(selected[0] & selected[1]) / len(selected[0] | selected[1])
    ranks.to_csv(TABLES / "09.27_004_metric_rank_sensitivity.csv", index=False, encoding="utf-8-sig")
    conditions = event.groupby(["period", "hour", "start_production_positive"]).agg(events=("hours", "size"), median_hours=("hours", "median"), mean_hours=("hours", "mean"), multi_hour=("hours", lambda x: x.gt(1).sum())).reset_index()
    conditions.to_csv(TABLES / "09.27_004_duration_conditions.csv", index=False, encoding="utf-8-sig")
    for sequence in range(4, 10):
        prefix = f"09.27_{sequence:03}"
        path = TABLES / f"{prefix}_facts.json"
        facts = json.loads(path.read_text(encoding="utf-8"))
        assert facts["source_sha256"] == SOURCE_HASH
        assert facts["frozen_sha256"] == sha256((TABLES / f"{prefix}_frozen.json").read_bytes()).hexdigest()
        facts["profile_sha256"] = PROFILE_HASH
        path.write_text(json.dumps(json_value(facts), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    results = {"source_sha256": SOURCE_HASH, "profile_sha256": PROFILE_HASH,
               "independent_onsets": [139, 85], "independent_eligible": [4108, 1218],
               "high_hour_reconciliation": 455, "excess_reconciliation": 4215,
               "standardized_rates_reconciled": len(std), "cluster_assignments_per_fit": 241,
               "pca_orthogonality": True, "rounding_disagreements": 13,
               "refinements": ["top10 cutoff ties", "duration by production context", "strict JSON null for nonfinite metadata"],
               "rank_sensitivity": ranks.to_dict("records")}
    (TABLES / "09.27_010_verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
