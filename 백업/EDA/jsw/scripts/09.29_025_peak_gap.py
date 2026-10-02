"""E025: descriptive exact-mean gap summaries; no model fitting or scoring."""

from __future__ import annotations

import ast
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


EDA = Path(__file__).resolve().parent.parent
ROOT = EDA.parent.parent
TABLES = EDA / "tables"
PREFIX = "09.29_025"
SLOTS = ["15분", "30분", "45분", "60분"]
EXPECTED = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
INPUTS = {
    "raw": ROOT / "data/origin/okm_augumented_2021.csv",
    "profiles": EDA / "tables/09.23_011_daily_profile_summary.csv",
    "legacy": EDA / "tables/09.23_015_mean_visibility.csv",
    "daily": EDA / "tables/09.28_024_daily_concentration.csv",
    "manifest": ROOT / "data/processed/jsw/09.26_021_coverage_manifest.csv",
    "catalog": ROOT / "modeling/jsw/tables/09.26_001_feature_catalog.csv",
    "model_code": ROOT / "modeling/jsw/scripts/09.26_001_power_models.py",
    "frozen": ROOT / "modeling/jsw/tables/09.26_001_frozen.json",
    "model_facts": ROOT / "modeling/jsw/tables/09.26_001_facts.json",
    "development": ROOT / "modeling/jsw/predictions/09.26_001_development_predictions.csv",
    "review": ROOT / "modeling/jsw/predictions/09.26_001_review_predictions.csv",
}


def read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(INPUTS[name], encoding="utf-8-sig", **kwargs)


def save(rows, suffix: str) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(TABLES / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def once(frame: pd.DataFrame, weighting: str) -> pd.DataFrame:
    if weighting == "profile_hour_once":
        return frame.drop_duplicates(["profile_id", "hour"])
    return frame


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    hashes = {name: sha256(path.read_bytes()).hexdigest() for name, path in INPUTS.items()}
    assert hashes["raw"] == EXPECTED
    raw = read("raw", usecols=["날짜", "시간", *SLOTS, "평균"])
    raw["source_data_row"] = np.arange(1, len(raw) + 1)
    scope = raw.loc[raw["날짜"].lt(20210901)]
    frame = scope.loc[scope["시간"].between(0, 23)].copy()
    assert (len(raw), len(scope), len(frame)) == (6168, 5832, 5784)
    frame = frame.rename(columns={"날짜": "date", "시간": "hour", "평균": "M"})
    frame["record_key"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d") + pd.to_timedelta(frame["hour"], unit="h")
    frame = frame.sort_values("record_key")
    assert frame["record_key"].is_unique and frame.groupby("date").size().eq(24).all()
    assert frame[[*SLOTS, "M"]].notna().all().all()
    frame["month"] = frame["record_key"].dt.month
    frame["period"] = np.where(frame["month"].le(6), "Jan-Jun_design", "Jul-Aug_exploratory")
    frame["A"] = frame[SLOTS].mean(axis=1)
    frame["P"] = frame[SLOTS].max(axis=1)
    frame["D"] = frame["P"] - frame["A"]
    frame["R"] = frame["P"] - frame[SLOTS].min(axis=1)
    frame["S"] = frame[SLOTS[-1]] - frame[SLOTS[0]]
    frame["rounding_delta"] = frame["M"] - frame["A"]
    assert frame["M"].eq(np.floor(frame["A"] + 0.5)).all()
    assert frame[SLOTS].min(axis=1).le(frame["A"]).all()
    assert frame["A"].le(frame["P"]).all() and frame[["D", "R"]].ge(0).all().all()
    assert frame["D"].le(0.75 * frame["R"]).all()
    profiles = read("profiles", usecols=["date", "profile_id"])
    assert len(profiles) == 241 and profiles["date"].is_unique
    frame = frame.merge(profiles, on="date", how="left", validate="many_to_one")
    assert frame["profile_id"].notna().all()
    assert frame.groupby(["profile_id", "hour"])[SLOTS].nunique().eq(1).all().all()

    manifest = read("manifest")
    assert len(manifest) == len(frame) and manifest["record_key"].is_unique
    manifest["eligible"] = manifest["complete_past_24h"] & manifest["exact_lag_168h"]
    frame = frame.merge(manifest[["source_data_row", "eligible"]], on="source_data_row", validate="one_to_one")
    assert int(frame["eligible"].sum()) == 5520
    catalog = read("catalog")
    frozen = json.loads(INPUTS["frozen"].read_text(encoding="utf-8"))
    assert catalog["feature"].tolist() == frozen["all_features"]
    assert catalog.loc[catalog["group"].eq("past_power"), "latest_offset_hours"].lt(0).all()
    code = INPUTS["model_code"].read_text(encoding="utf-8")
    ast.parse(code)  # Parse only; importing the model could write files or run training.
    assert '"mean": valid["평균"]' in code and '"last_slot": valid["60분"]' in code
    assert 'valid[slots].max(axis=1) - valid[slots].min(axis=1)' in code
    assert 'grid[variable].shift(lag)' in code and 'shift(1).rolling(24, min_periods=24)' in code

    # E015 reproduction is an input-definition check, not a repeat of its analysis.
    high = frame.loc[frame["P"].ge(187)].copy()
    high["high_quarters"] = high[SLOTS].ge(187).sum(axis=1)
    high["mean_below_same_number"] = high["M"].lt(187)
    legacy = high.assign(gap=high["P"] - high["M"]).groupby(
        ["high_quarters", "mean_below_same_number"]).agg(
            hours=("P", "size"), days=("date", "nunique"), median_max_mean_gap=("gap", "median")
        ).reset_index()
    pd.testing.assert_frame_equal(legacy, read("legacy"), check_dtype=False)
    assert len(high) == 287 and int(high["M"].lt(187).sum()) == 194
    legacy["period"] = "Jan-Aug_valid"
    legacy["threshold"] = 187
    legacy["definition"] = "P>=187 and M<187; denominator=count(P>=187)"
    save(legacy, "legacy_visibility")

    # Reuse saved training thresholds and target keys; never consume model scores.
    keys = pd.concat([read(name, usecols=["record_key", "stage", "high_threshold_from_training"])
                      for name in ("development", "review")]).drop_duplicates()
    keys["record_key"] = pd.to_datetime(keys["record_key"])
    assert keys.groupby("stage")["high_threshold_from_training"].nunique().eq(1).all()
    thresholds = keys.groupby("stage")["high_threshold_from_training"].first().to_dict()
    assert thresholds == {"May": 183.0, "Jun": 181.0, "Jul-Aug": 182.0}
    stage_specs = [("May", "2021-05-01", 2712), ("Jun", "2021-06-01", 3456), ("Jul-Aug", "2021-07-01", 4176)]

    design = frame.loc[frame["month"].le(6)]
    quantiles = np.unique(design["A"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).to_numpy())
    internal = quantiles[1:-1]
    frame["A_bin"] = np.searchsorted(internal, frame["A"], side="left") + 1
    frame["below_design_min"] = frame["A"].lt(quantiles[0])
    frame["above_design_max"] = frame["A"].gt(quantiles[-1])
    bin_rows = []
    for b in range(1, len(internal) + 2):
        bin_rows.append({"A_bin": b, "lower_exclusive": -np.inf if b == 1 else internal[b-2],
                         "upper_inclusive": np.inf if b == len(internal)+1 else internal[b-1],
                         "fit_period": "Jan-Jun_design", "fit_hours": len(design),
                         "fit_min_A": quantiles[0], "fit_max_A": quantiles[-1]})
    save(bin_rows, "bins")

    summary = []
    coverage = []
    examples = []
    exact = []
    rounding = []
    groups = [(p, "all", part) for p, part in frame.groupby("period")]
    groups += [(p, str(m), part) for (p, m), part in frame.groupby(["period", "month"])]
    for period, month, part in groups:
        coverage.append({"period": period, "month": month, "normal_hours": len(part), "days": part["date"].nunique(),
                         "profiles": part["profile_id"].nunique(), "M001_eligible_hours": int(part["eligible"].sum()),
                         "M001_ineligible_hours": int((~part["eligible"]).sum()),
                         "zero_hours_preserved": int(part["P"].eq(0).sum()),
                         "below_design_min_hours": int(part["below_design_min"].sum()),
                         "above_design_max_hours": int(part["above_design_max"].sum())})
        bins = [(0, part), *list(part.groupby("A_bin"))]
        for b, cell in bins:
            for weighting in ("date_hour", "profile_hour_once"):
                weighted = once(cell, weighting)
                result = {"period": period, "month": month, "A_bin": b, "weighting": weighting,
                          "normal_hours": len(cell), "weighted_units": len(weighted),
                          "days": cell["date"].nunique(), "profiles": cell["profile_id"].nunique(),
                          "small_cell": len(weighted) < 30 or cell["date"].nunique() < 5}
                for name in ("A", "P", "D"):
                    for q in (0.25, 0.5, 0.75, 0.9, 0.95):
                        result[f"{name}_q{int(q*100):02d}"] = weighted[name].quantile(q)
                summary.append(result)
        if month != "all":
            continue
        for weighting in ("date_hour", "profile_hour_once"):
            weighted = once(part, weighting)
            for delta, cell in weighted.groupby("rounding_delta"):
                rounding.append({"period": period, "weighting": weighting, "M_minus_A": delta,
                                 "weighted_units": len(cell), "denominator_units": len(weighted)})
            exact_groups = weighted.groupby("A")["D"].agg(["size", "nunique", "min", "max"])
            repeated = exact_groups.loc[exact_groups["size"].ge(2)]
            varying = repeated.loc[repeated["nunique"].gt(1)]
            exact.append({"period": period, "weighting": weighting, "A_values": len(exact_groups),
                          "repeated_A_values": len(repeated), "varying_D_A_values": len(varying),
                          "repeated_A_units": int(repeated["size"].sum()), "varying_D_A_units": int(varying["size"].sum()),
                          "D_span_q50_over_varying_A": (varying["max"]-varying["min"]).median(),
                          "D_span_q95_over_varying_A": (varying["max"]-varying["min"]).quantile(0.95)})
        # Predetermined choices: nearest observed D to median and q95, ties by timestamp.
        for b, cell in part.groupby("A_bin"):
            for selection, pool in [("bin", cell)]:
                for q in (0.5, 0.95):
                    target = pool["D"].quantile(q)
                    chosen = pool.assign(distance=(pool["D"]-target).abs()).sort_values(["distance", "record_key"]).iloc[0]
                    examples.append({"period": period, "A_bin": b, "selection": selection, "D_quantile": q,
                                     "quantile_value": target, **chosen[["source_data_row", "record_key", "date", "hour", "profile_id", *SLOTS, "M", "A", "P", "D", "R", "S"]].to_dict()})
            varying_a = cell.groupby("A")["D"].nunique()
            eligible_a = varying_a.loc[varying_a.gt(1)].index
            if len(eligible_a):
                chosen_a = min(eligible_a, key=lambda a: (abs(a-cell["A"].median()), a))
                pool = cell.loc[cell["A"].eq(chosen_a)]
                for q in (0.5, 0.95):
                    target = pool["D"].quantile(q)
                    chosen = pool.assign(distance=(pool["D"]-target).abs()).sort_values(["distance", "record_key"]).iloc[0]
                    examples.append({"period": period, "A_bin": b, "selection": "exact_A_nearest_bin_median", "D_quantile": q,
                                     "quantile_value": target, **chosen[["source_data_row", "record_key", "date", "hour", "profile_id", *SLOTS, "M", "A", "P", "D", "R", "S"]].to_dict()})
    save(summary, "gap_summary")
    save(coverage, "coverage")
    save(examples, "examples")
    save(exact, "exact_A_spread")
    save(rounding, "rounding")

    visibility = []
    comparisons = [("legacy_period_new_A", frame, 187.0, "Jan-Aug exploration", 5784),
                   ("Jan-Jun_EDA_reference", design, thresholds["Jul-Aug"], "Jan-Jun M001 eligible", 4176),
                   ("Jul-Aug_EDA_reference", frame.loc[frame["month"].ge(7)], thresholds["Jul-Aug"], "Jan-Jun M001 eligible", 4176)]
    for stage, cutoff, training_n in stage_specs:
        target_keys = keys.loc[keys["stage"].eq(stage), "record_key"]
        scored = frame.loc[frame["record_key"].isin(target_keys)]
        train = frame.loc[frame["eligible"] & frame["record_key"].lt(pd.Timestamp(cutoff))]
        assert len(train) == training_n
        assert len(scored) == {"May": 744, "Jun": 720, "Jul-Aug": 1344}[stage] and scored["eligible"].all()
        comparisons += [(f"{stage}_M001_training", train, thresholds[stage], f"before {cutoff} M001 eligible", training_n),
                        (f"{stage}_M001_target", scored, thresholds[stage], f"before {cutoff} M001 eligible", training_n)]
    for name, part, u, basis, training_n in comparisons:
        for weighting in ("date_hour", "profile_hour_once"):
            weighted = once(part, weighting)
            peak_high = weighted["P"].ge(u)
            for mean in ("M", "A"):
                numerator = int((peak_high & weighted[mean].lt(u)).sum())
                denominator = int(peak_high.sum())
                visibility.append({"scope": name, "period_start": part["record_key"].min(), "period_end": part["record_key"].max(),
                                   "normal_hours": len(part), "days": part["date"].nunique(), "profiles": part["profile_id"].nunique(),
                                   "weighting": weighting, "weighted_units": len(weighted), "threshold": u,
                                   "threshold_basis": basis, "threshold_training_units": training_n, "mean_definition": mean,
                                   "numerator_condition": f"P>=u and {mean}<u", "denominator_condition": "P>=u",
                                   "hidden_units": numerator, "peak_high_units": denominator,
                                   "hidden_rate": numerator / denominator if denominator else np.nan})
    save(visibility, "visibility")

    overlaps = [
        ("past_M", "CSV rounded mean", "mean_lag1/2/3/24/168; mean_past24_*", "existing", "reuse"),
        ("past_A", "(q1+q2+q3+q4)/4", "M differs by rounding only", "rounding precision", "do not silently replace M001 mean target"),
        ("past_P", "max(q1,q2,q3,q4)", "peak_lag1/2/3/24/168; peak_past24_*", "existing", "reuse; not a new candidate"),
        ("past_R", "P-min(q1,q2,q3,q4)", "range_lag1", "existing at lag1", "reuse; no additional lags without a question"),
        ("past_D_M", "P-M", "peak_lagk - mean_lagk, k=1/2/3/24/168", "fully derived", "optional representation only"),
        ("past_D", "P-A = (P-M)+(M-A)", "past_D_M plus rounding remainder", "mostly derived", "conditional alternative to S; no independent sensor claim"),
        ("past_S", "q4-q1", "last_slot_lag1 exists; first_slot absent", "not algebraically determined", "one lag1 candidate only if M005 identifies relevant onset failure"),
        ("target_A_P_D_R_S", "target-hour descriptors", "target power", "future leakage", "EDA explanation only; prohibit prediction input"),
        ("current_day_96_FA_profile", "requires complete target date", "future records within day", "future leakage", "prohibit intra-day prediction input"),
    ]
    save([dict(zip(["candidate", "definition", "M001_overlap", "classification", "handoff"], row)) for row in overlaps], "feature_overlap")
    assert all(sha256(path.read_bytes()).hexdigest() == hashes[name] for name, path in INPUTS.items())
    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"), "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__, "randomness": "none",
        "input_sha256": hashes, "input_paths": {n: p.relative_to(ROOT).as_posix() for n, p in INPUTS.items()},
        "raw_rows": len(raw), "Jan_Aug_rows": len(scope), "invalid_time_rows_excluded": 48, "September_rows_excluded": 336,
        "normal_hours": len(frame), "normal_days": frame["date"].nunique(), "M001_eligible": int(frame["eligible"].sum()),
        "zero_hours_preserved": int(frame["P"].eq(0).sum()), "A_fit_quantiles": quantiles.tolist(),
        "bin_rule": "internal boundaries from Jan-Jun A quintiles; lower exclusive, upper inclusive; outside fit min/max retained",
        "weight_rule": "date_hour: each observed hour; profile_hour_once: unique profile_id/hour within reported group",
        "small_cell_rule": "weighted_units<30 or observed_days<5; descriptive only",
        "quantile_method": "linear", "thresholds_reused_from_M001": thresholds,
        "Q1": "legacy conclusion reused; missing exact hourly A/D summaries calculated",
        "Q2": "catalog/frozen/code static audit; no new relationship/correlation/lag dataset calculated",
        "unexecuted": ["model training", "performance comparison", "alert evaluation", "relocation simulation", "clustering", "FA", "correlation", "hour interaction tables", "September EDA"],
    }
    (TABLES / f"{PREFIX}_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"normal_hours": len(frame), "zero_hours": facts["zero_hours_preserved"],
                      "A_boundaries": quantiles.tolist(), "Q1": "calculated", "Q2": "static reuse", "csv_tables": 9}, ensure_ascii=False))


if __name__ == "__main__":
    main()
