"""M005: audit saved predictions only. No estimator imports, fitting or inference."""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
PREFIX = "09.29_005"
HOUR = pd.Timedelta(hours=1)
SLOTS = ["15분", "30분", "45분", "60분"]
KEY = ["stage", "target", "method", "record_key"]
PERIODS = {
    "May": ("2021-05-01", "2021-06-01"),
    "Jun": ("2021-06-01", "2021-07-01"),
    "Jul-Aug": ("2021-07-01", "2021-09-01"),
}
DIAG_METHODS = ["lag1", "hgb_calendar", "hgb_q75_calendar"]


def read(relative: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, encoding="utf-8-sig", **kwargs)


def save(frame: pd.DataFrame, suffix: str, rows: bool = False) -> None:
    frame.to_csv(HERE / ("predictions" if rows else "tables") / f"{PREFIX}_{suffix}.csv",
                 index=False, encoding="utf-8-sig")


def write_json(value: dict, suffix: str) -> None:
    (HERE / "tables" / f"{PREFIX}_{suffix}.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def metric(g: pd.DataFrame) -> dict:
    e = g.prediction.to_numpy() - g.actual.to_numpy()
    short = np.maximum(-e, 0)
    over = np.maximum(e, 0)
    return {"n": len(g), "dates": g.date.nunique() if "date" in g else 0,
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "under_n": int((e < 0).sum()),
            "under_rate": float((e < 0).mean()), "mean_shortfall": float(short.mean()),
            "mean_over": float(over.mean()), "shortfall_p90": float(np.quantile(short, .90)),
            "shortfall_p95": float(np.quantile(short, .95)),
            "actual_mean": float(g.actual.mean()), "prediction_mean": float(g.prediction.mean())}


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **metric(g)}
                         for k, g in frame.groupby(keys, dropna=False, sort=True)])


def compare_score_table(calculated: pd.DataFrame, relative: str, keys: list[str],
                        aliases: dict | None = None, columns: list[str] | None = None) -> dict:
    old = read(relative)
    new = calculated.rename(columns=aliases or {})
    new = new.set_index(keys).sort_index()
    old = old.set_index(keys).sort_index()
    assert new.index.equals(old.index), relative
    fields = columns or [x for x in old.columns if x in new.columns]
    assert fields and "n" in fields, relative
    max_diff = 0.0
    for field in fields:
        np.testing.assert_allclose(new[field], old[field], rtol=1e-10, atol=1e-9, equal_nan=True,
                                   err_msg=f"{relative}: {field}")
        if len(old):
            max_diff = max(max_diff, float((new[field] - old[field]).abs().max()))
    return {"file": relative, "groups": len(old), "columns": ",".join(fields),
            "max_abs_difference": max_diff}


def reproduce(dev: pd.DataFrame, review: pd.DataFrame, quantile: pd.DataFrame) -> list[dict]:
    checks = []
    specs = [
        (dev, ["target", "method"], "development_scores"),
        (dev, ["stage", "target", "method"], "development_by_month"),
        (review, ["target", "method"], "review_scores"),
        (review, ["month", "target", "method"], "review_by_month"),
        (review, ["peak_state", "target", "method"], "review_by_peak_state"),
        (review, ["hour", "target", "method"], "review_by_hour"),
        (review, ["zero_window_posthoc", "target", "method"], "review_zero_windows"),
        (dev[dev.method.isin(["lag1", "hgb_calendar"])],
         ["stage", "target", "method", "peak_state"], "development_peak_states"),
        (review[review.actual.eq(0) & review.method.isin(["lag1", "hgb_calendar"])],
         ["target", "method"], "zero_target_scores"),
    ]
    for frame, keys, name in specs:
        checks.append(compare_score_table(summarize(frame, keys),
                      f"modeling/jsw/tables/09.26_001_{name}.csv", keys))
    qrows = []
    for period, f in [("pooled", quantile), *list(quantile.groupby("stage"))]:
        for method, g in f.groupby("method"):
            for state, sub in [("all", g), *list(g.groupby("peak_state"))]:
                e = sub.prediction - sub.actual
                qrows.append({"period": period, "method": method, "peak_state": state,
                              **metric(sub), "shortfall_when_under": float((-e[e.lt(0)]).mean()) if e.lt(0).any() else 0,
                              "over_n": int(e.gt(0).sum()), "over_rate": float(e.gt(0).mean()),
                              "over_when_over": float(e[e.gt(0)].mean()) if e.gt(0).any() else 0})
    checks.append(compare_score_table(pd.DataFrame(qrows), "modeling/jsw/tables/09.26_002_scores.csv",
                  ["period", "method", "peak_state"],
                  {"mean_shortfall": "shortfall_per_hour", "mean_over": "over_per_hour"}))
    p3 = read("modeling/jsw/predictions/09.26_003_development_predictions.csv")
    combo = pd.concat([quantile, p3[p3.target.eq("peak") & p3.method.ne("hgb_calendar")]])
    selection = []
    high_states = []
    for period, f in [("pooled", combo), *list(combo.groupby("stage"))]:
        for method, g in f.groupby("method"):
            actual = g.actual.ge(g.high_threshold_from_training)
            pred = g.prediction.ge(g.high_threshold_from_training)
            hit, miss, false = int((actual & pred).sum()), int((actual & ~pred).sum()), int((~actual & pred).sum())
            selection.append({"period": period, "method": method, "n": len(g),
                              "high_actual_n": int(actual.sum()), "not_high_actual_n": int((~actual).sum()),
                              "selected_n": int(pred.sum()), "hit_n": hit, "miss_n": miss,
                              "false_selection_n": false, "recall": hit / (hit + miss),
                              "precision": hit / (hit + false)})
            for state in ["high_onset", "high_continued"]:
                sub = g[g.peak_state.eq(state)]
                hits = int(sub.prediction.ge(sub.high_threshold_from_training).sum())
                high_states.append({"period": period, "method": method, "peak_state": state,
                                    "actual_n": len(sub), "hit_n": hits, "miss_n": len(sub) - hits})
    checks.append(compare_score_table(pd.DataFrame(selection),
                  "modeling/jsw/tables/09.26_004_selection_scores.csv", ["period", "method"]))
    # M004's state table uses actual_n instead of n.
    state_new = pd.DataFrame(high_states).set_index(["period", "method", "peak_state"]).sort_index()
    state_old = read("modeling/jsw/tables/09.26_004_by_high_state.csv").set_index(state_new.index.names).sort_index()
    pd.testing.assert_frame_equal(state_new, state_old, check_dtype=False)
    checks.append({"file": "modeling/jsw/tables/09.26_004_by_high_state.csv", "groups": len(state_old),
                   "columns": "actual_n,hit_n,miss_n", "max_abs_difference": 0.0})
    save(pd.DataFrame(checks), "reproduction")
    return checks


def main() -> None:
    print(json.dumps({"runtime_started": platform.python_version(), "operation": "saved_prediction_audit"}), flush=True)
    inputs = ["data/origin/okm_augumented_2021.csv", "data/processed/jsw/09.26_021_coverage_manifest.csv",
              "EDA/jsw/tables/09.23_011_daily_profile_summary.csv", "EDA/jsw/tables/09.28_024_daily_max_ties.csv",
              "EDA/jsw/tables/09.28_024_monthly_max_ties.csv", "analysis/jsw/tables/09.28_042_hour_event_comparison.csv"]
    inputs += [str(p.relative_to(ROOT)).replace("\\", "/") for directory in [HERE / "tables", HERE / "predictions"]
               for p in directory.glob("09.26_*.*")]
    hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(set(inputs))}
    assert hashes[inputs[0]] == "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
    write_json({"time": datetime.now().astimezone().isoformat(timespec="seconds"), "input_sha256": hashes,
                "training": "none; saved predictions only", "periods": PERIODS,
                "state_threshold": "preserve saved stage threshold; high means >=",
                "event_contract": "maximal exact-hour high runs within saved evaluation stage; unknown boundaries retained",
                "daily_peak": "E024 full normal day; deduplicate tied slots to hours; retain every tied hour",
                "profile": "exact full-day 96 values, diagnostic only; observed and eligible-contributed libraries before cutoff",
                "paired_difference": "HGB minus lag1; date/week and leave-one-week-out descriptive sensitivity",
                "partial_days": "complete/partial/no saved prediction dates separated; full-day all capture requires every original peak hour available"}, "frozen")
    raw = read(inputs[0], usecols=["날짜", "시간", *SLOTS, "평균", "생산량"])
    raw["source_data_row"] = np.arange(1, len(raw) + 1)
    raw = raw[raw["날짜"].lt(20210901)]
    valid = raw[raw["시간"].between(0, 23)].copy()
    valid["record_key"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(valid["시간"], unit="h")
    valid = valid.set_index("record_key").sort_index()
    assert valid.index.is_unique and len(valid) == 5784
    valid["date"] = valid.index.strftime("%Y-%m-%d")
    valid["month"] = valid.index.month
    valid["peak"] = valid[SLOTS].max(axis=1)
    valid["mean"] = valid["평균"]
    valid["previous_peak"] = valid.peak.reindex(valid.index - HOUR).to_numpy()
    manifest = read(inputs[1], parse_dates=["record_key"]).set_index("record_key").reindex(valid.index)
    assert manifest.index.equals(valid.index) and manifest.source_data_row.eq(valid.source_data_row).all()
    # Verify the cached input-availability contract independently from source keys.
    grid = valid.peak.reindex(pd.date_range("2021-01-01", "2021-08-31 23:00", freq="h"))
    past24 = grid.shift(1).rolling(24, min_periods=24).count().eq(24).reindex(valid.index)
    lag168 = pd.Series((valid.index - 168 * HOUR).isin(valid.index), index=valid.index)
    lag1 = valid.previous_peak.notna()
    assert manifest.complete_past_24h.eq(past24).all()
    assert manifest.exact_lag_168h.eq(lag168).all()
    assert manifest.exact_lag_1h.eq(lag1).all()
    eligible = past24 & lag168
    assert int(eligible.sum()) == 5520
    valid["exact_lag1"] = lag1
    valid["past24_available"] = past24
    valid["lag168_available"] = lag168
    valid["eligible"] = eligible
    valid["input_missing_reason"] = np.select(
        [~past24 & ~lag168, ~past24, ~lag168], ["past24_and_lag168", "past24", "lag168"], default="none")
    profiles = read(inputs[2], usecols=["date", "profile_id"])
    profiles.date = pd.to_datetime(profiles.date.astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    assert profiles.date.is_unique
    profile_map = profiles.set_index("date").profile_id
    vectors = {}
    for date, g in valid.groupby("date"):
        if len(g) == 24 and list(g.index.hour) == list(range(24)):
            vectors[date] = tuple(g[SLOTS].to_numpy().ravel().tolist())
    assert set(vectors) == set(profile_map.index)
    id_vectors = {}
    vector_ids = {}
    for date, vector in vectors.items():
        pid = profile_map[date]
        assert pid not in id_vectors or id_vectors[pid] == vector
        assert vector not in vector_ids or vector_ids[vector] == pid
        id_vectors[pid], vector_ids[vector] = vector, pid
    valid["profile_id"] = valid.date.map(profile_map).fillna("unknown")
    dev = read("modeling/jsw/predictions/09.26_001_development_predictions.csv", parse_dates=["record_key"])
    review = read("modeling/jsw/predictions/09.26_001_review_predictions.csv", parse_dates=["record_key"])
    q = read("modeling/jsw/predictions/09.26_002_development_predictions.csv", parse_dates=["record_key"])
    reproduction = reproduce(dev, review, q)  # Existing scores precede additional aggregation.
    base = pd.concat([dev, review], ignore_index=True)
    assert not base.duplicated(KEY).any() and not q.duplicated(KEY).any()
    duplicate = base.merge(q, on=KEY, suffixes=("_base", "_q"), validate="one_to_one")
    for name in ["actual", "prediction", "high_threshold_from_training", "source_data_row"]:
        np.testing.assert_allclose(duplicate[f"{name}_base"], duplicate[f"{name}_q"], atol=1e-10, rtol=1e-12)
    p = pd.concat([base, q[q.method.eq("hgb_q75_calendar")]], ignore_index=True)
    assert not p.duplicated(KEY).any()
    for (stage, target, method), g in p.groupby(KEY[:-1]):
        start, end = map(pd.Timestamp, PERIODS[stage])
        expected = valid.index[(valid.index >= start) & (valid.index < end) & eligible]
        assert g.sort_values("record_key").record_key.tolist() == expected.tolist()
        np.testing.assert_array_equal(g.actual, valid[target].reindex(g.record_key))
        np.testing.assert_array_equal(g.source_data_row, valid.source_data_row.reindex(g.record_key))
    thresholds = p.groupby("stage").high_threshold_from_training.agg(["first", "nunique"])
    assert thresholds["nunique"].eq(1).all()
    for stage, (start, _) in PERIODS.items():
        cutoff = pd.Timestamp(start)
        train = valid[(valid.index < cutoff) & eligible]
        assert train.peak.quantile(.95) == thresholds.loc[stage, "first"]
    libraries = []
    statuses = []
    for stage, (start, end) in PERIODS.items():
        cutoff = pd.Timestamp(start)
        complete_past = [d for d in vectors if pd.Timestamp(d) + pd.Timedelta(days=1) <= cutoff]
        contributed_dates = set(valid.loc[(valid.index < cutoff) & eligible, "date"])
        observed = set(profile_map.reindex(complete_past))
        contributed = set(profile_map.reindex(sorted(set(complete_past) & contributed_dates)))
        libraries.append({"stage": stage, "training_cutoff_exclusive": start,
                          "eligible_training_rows": int(((valid.index < cutoff) & eligible).sum()),
                          "observed_complete_dates": len(complete_past), "observed_profiles": len(observed),
                          "contributing_complete_dates": len(set(complete_past) & contributed_dates),
                          "contributing_profiles": len(contributed), "threshold": thresholds.loc[stage, "first"]})
        v = valid.loc[(valid.index >= cutoff) & (valid.index < pd.Timestamp(end))].copy()
        v["stage"] = stage
        v["threshold"] = thresholds.loc[stage, "first"]
        v["profile_seen_observed"] = np.where(v.profile_id.eq("unknown"), "unknown", np.where(v.profile_id.isin(observed), "seen", "unseen"))
        v["profile_seen_contributed"] = np.where(v.profile_id.eq("unknown"), "unknown", np.where(v.profile_id.isin(contributed), "seen", "unseen"))
        v["state"] = np.select([v.previous_peak.isna(), v.peak.ge(v.threshold) & v.previous_peak.lt(v.threshold),
                                v.peak.ge(v.threshold) & v.previous_peak.ge(v.threshold),
                                v.peak.lt(v.threshold) & v.previous_peak.ge(v.threshold)],
                               ["boundary_unknown", "high_onset", "high_continued", "descending"], default="ordinary")
        statuses.append(v)
    v = pd.concat(statuses).rename_axis("record_key").reset_index()
    assert v.record_key.is_unique and len(v) == 2904
    labels = v[["record_key", "date", "previous_peak", "peak", "state", "profile_id",
                "profile_seen_observed", "profile_seen_contributed", "input_missing_reason"]]
    n_before = len(p)
    p = p.drop(columns=[c for c in ["month", "hour"] if c in p]).merge(labels, on="record_key", validate="many_to_one")
    assert len(p) == n_before
    p["month"] = p.record_key.dt.month
    p["hour"] = p.record_key.dt.hour
    p["error"] = p.prediction - p.actual
    p["shortfall"] = (-p.error).clip(lower=0)
    p["over"] = p.error.clip(lower=0)
    p["week"] = p.record_key.dt.to_period("W-SUN").astype(str)
    save(pd.DataFrame(libraries), "training_profile_libraries")
    groups = ["stage", "month", "target", "method", "high_threshold_from_training"]
    all_scores = summarize(p, groups)
    save(all_scores, "scores")
    state_scores = summarize(p, groups + ["state"])
    assert state_scores.groupby(groups)["n"].sum().tolist() == all_scores.set_index(groups)["n"].tolist()
    save(state_scores, "state_scores")
    save(summarize(p[p.actual.eq(0)], groups), "zero_scores")
    prod = p.assign(production_condition=np.where(p.production_posthoc.eq(0), "zero", "positive"))
    save(summarize(prod, groups + ["production_condition"]), "production_scores")
    profile_scores = summarize(p, groups + ["profile_seen_contributed"])
    n_profiles = p.groupby(groups + ["profile_seen_contributed"]).profile_id.nunique().reset_index(name="profiles")
    high_short = p[p.peak.ge(p.high_threshold_from_training)].groupby(groups + ["profile_seen_contributed"]).agg(
        high_n=("record_key", "size"), high_mean_shortfall=("shortfall", "mean"))
    profile_scores = profile_scores.merge(n_profiles, on=groups + ["profile_seen_contributed"], validate="one_to_one").merge(
        high_short.reset_index(), on=groups + ["profile_seen_contributed"], how="left", validate="one_to_one")
    profile_scores.high_n = profile_scores.high_n.fillna(0).astype(int)
    save(profile_scores, "profile_scores")
    save(summarize(p, groups + ["profile_seen_observed"]), "profile_observed_scores")
    ties = read(inputs[3])
    ties["date"] = pd.to_datetime(ties.date.astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    ties["record_key"] = pd.to_datetime(ties.date) + pd.to_timedelta(ties.hour, unit="h")
    peak_hours = ties.drop_duplicates("record_key")[["record_key", "date", "power"]]
    np.testing.assert_array_equal(peak_hours.power, valid.peak.reindex(peak_hours.record_key))
    assert peak_hours.groupby("date").power.first().equals(valid.groupby("date").peak.max().rename("power"))
    v["is_daily_peak"] = v.record_key.isin(peak_hours.record_key)
    p["is_daily_peak"] = p.record_key.isin(peak_hours.record_key)
    # Calendar month maxima are original E024 locations, independent of eligibility.
    mt = read(inputs[4])
    mt["date"] = pd.to_datetime(mt.date.astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    mt["record_key"] = pd.to_datetime(mt.date) + pd.to_timedelta(mt.hour, unit="h")
    mt = mt.drop_duplicates("record_key")
    v["is_monthly_peak"] = v.record_key.isin(mt.record_key)
    monthly_cases = []
    daily_cases = []
    coverage = []
    date_coverage = []
    for (stage, month), g in v.groupby(["stage", "month"]):
        peak = g[g.is_daily_peak]
        coverage.append({"stage": stage, "month": month, "threshold": g.threshold.iloc[0],
                         "normal_hours": len(g), "dates": g.date.nunique(),
                         "exact_lag1_hours": int(g.exact_lag1.sum()),
                         "past24_hours": int(g.past24_available.sum()),
                         "all_features_hours": int(g.eligible.sum()), "common_saved_hours": int(g.eligible.sum()),
                         "excluded_hours": int((~g.eligible).sum()),
                         "daily_peak_hours": len(peak), "eligible_peak_hours": int(peak.eligible.sum()),
                         "excluded_peak_hours": int((~peak.eligible).sum()),
                         "peak_dates": peak.date.nunique(), "dates_any_peak_available": peak[peak.eligible].date.nunique(),
                         "excluded_high_hours": int((~g.eligible & g.peak.ge(g.threshold)).sum()),
                         "monthly_peak_hours": int(g.is_monthly_peak.sum()),
                         "excluded_monthly_peak_hours": int((g.is_monthly_peak & ~g.eligible).sum())})
    for (stage, date), g in v.groupby(["stage", "date"]):
        peak = g[g.is_daily_peak]
        saved_n = int(g.eligible.sum())
        status = "complete" if saved_n == len(g) else ("partial" if saved_n else "none")
        date_coverage.append({"stage": stage, "month": int(g.month.iloc[0]), "date": date,
                              "normal_hours": len(g), "saved_hours": saved_n, "prediction_day_status": status,
                              "predictable_rate": saved_n / len(g), "original_peak_hours": len(peak),
                              "available_peak_hours": int(peak.eligible.sum()),
                              "all_original_peaks_available": bool(peak.eligible.all()),
                              "profile_id": g.profile_id.iloc[0], "missing_reasons": ",".join(sorted(set(g.input_missing_reason) - {"none"}))})
        for method in sorted(p[p.stage.eq(stage) & p.target.eq("peak")].method.unique()):
            pred = p[p.stage.eq(stage) & p.target.eq("peak") & p.method.eq(method)].set_index("record_key").prediction
            predicted = pred.reindex(peak.record_key)
            available = predicted.notna()
            hit = predicted.ge(g.threshold.iloc[0])
            errors = predicted.to_numpy() - peak.peak.to_numpy()
            daily_cases.append({"stage": stage, "month": int(g.month.iloc[0]), "date": date, "method": method,
                                "threshold": g.threshold.iloc[0], "prediction_day_status": status,
                                "original_peak_hours": len(peak), "available_peak_hours": int(available.sum()),
                                "peak_tied_across_hours": len(peak) > 1,
                                "day_mean_shortfall_available": float(np.maximum(-errors[available], 0).mean()) if available.any() else np.nan,
                                "day_mae_available": float(np.abs(errors[available]).mean()) if available.any() else np.nan,
                                "any_peak_hour_alarm": bool(hit.any()) if available.any() else np.nan,
                                "all_original_peak_hours_alarm": bool(hit.all() and available.all()) if available.all() else np.nan,
                                "day_peak_ge_threshold": bool(peak.peak.ge(g.threshold.iloc[0]).all()),
                                "all_original_peaks_available": bool(available.all())})
    for _, row in v[v.is_monthly_peak].iterrows():
        for target in ["peak", "mean"]:
            for method in sorted(p[p.stage.eq(row.stage) & p.target.eq(target)].method.unique()):
                scored = p[p.record_key.eq(row.record_key) & p.target.eq(target) & p.method.eq(method)]
                prediction = float(scored.prediction.iloc[0]) if len(scored) else np.nan
                actual = float(row[target])
                monthly_cases.append({"stage": row.stage, "month": row.month, "record_key": row.record_key,
                                      "target": target, "method": method, "actual": actual, "prediction": prediction,
                                      "shortfall": max(actual - prediction, 0) if len(scored) else np.nan,
                                      "available": bool(len(scored)), "input_missing_reason": row.input_missing_reason})
    save(pd.DataFrame(coverage), "coverage")
    dc = pd.DataFrame(date_coverage)
    save(dc, "date_coverage")
    dcase = pd.DataFrame(daily_cases)
    save(dcase, "daily_peak_dates", rows=True)
    save(pd.DataFrame(monthly_cases), "monthly_peak_cases")
    save(summarize(p[p.is_daily_peak], groups), "daily_peak_scores")
    daily_summary = []
    for k, g in dcase.groupby(["stage", "month", "method", "prediction_day_status"]):
        high = g[g.day_peak_ge_threshold]
        daily_summary.append({**dict(zip(["stage", "month", "method", "prediction_day_status"], k)),
                              "dates": len(g), "dates_with_prediction_at_peak": int(g.available_peak_hours.gt(0).sum()),
                              "tied_peak_dates": int(g.peak_tied_across_hours.sum()),
                              "date_weighted_mean_shortfall_available": g.day_mean_shortfall_available.mean(),
                              "date_weighted_mae_available": g.day_mae_available.mean(),
                              "high_peak_dates": len(high),
                              "high_peak_dates_any_alarm": int(high.any_peak_hour_alarm.fillna(False).astype(bool).sum()),
                              "high_peak_dates_all_peaks_available": int(high.all_original_peaks_available.sum()),
                              "high_peak_dates_all_alarms": int(high.all_original_peak_hours_alarm.fillna(False).astype(bool).sum())})
    save(pd.DataFrame(daily_summary), "daily_peak_date_scores")
    # Form events on ALL normal targets, not just model-eligible targets.
    a42 = read(inputs[5], parse_dates=["start", "end"])
    event_rows = []
    event_members = []
    for stage, (start, end) in PERIODS.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        sub = v[v.stage.eq(stage)].set_index("record_key")
        threshold = float(thresholds.loc[stage, "first"])
        high_keys = sub.index[sub.peak.ge(threshold)]
        batches = []
        for t in high_keys:
            if not batches or t - batches[-1][-1] != HOUR:
                batches.append([t])
            else:
                batches[-1].append(t)
        for seq, keys in enumerate(batches, 1):
            s, e = keys[0], keys[-1]
            prev, nxt = valid.peak.get(s - HOUR, np.nan), valid.peak.get(e + HOUR, np.nan)
            left_reason = "time_gap" if pd.isna(prev) else ("evaluation_left_continued" if s == start and prev >= threshold else "known")
            right_reason = "time_gap" if pd.isna(nxt) else ("evaluation_right_continued" if e + HOUR >= end and nxt >= threshold else "known")
            left_unknown = left_reason != "known"
            right_unknown = right_reason != "known"
            eid = f"{stage}|ge{threshold:g}|{seq:03d}"
            full_match = a42[a42.start.eq(s) & a42.end.eq(e)] if threshold == 182 else a42.iloc[:0]
            event_members.extend({"stage": stage, "threshold": threshold, "event_id": eid, "record_key": t} for t in keys)
            for method in sorted(p[p.stage.eq(stage) & p.target.eq("peak")].method.unique()):
                saved = p[p.stage.eq(stage) & p.target.eq("peak") & p.method.eq(method)].set_index("record_key")
                predictions = saved.prediction.reindex(keys)
                captured = predictions[predictions.ge(threshold)]
                start_available = bool(pd.notna(predictions.iloc[0]))
                first = captured.index[0] if len(captured) else pd.NaT
                if left_unknown:
                    outcome = "start_unknown"
                elif not start_available:
                    outcome = "start_unpredictable"
                elif predictions.iloc[0] >= threshold:
                    outcome = "pre_start_warning"
                elif len(captured):
                    outcome = "late_capture"
                else:
                    outcome = "missed"
                event_rows.append({"stage": stage, "month": s.month, "threshold": threshold, "event_id": eid,
                                   "method": method, "start": s, "end": e, "hours_observed": len(keys),
                                   "dates_observed": len(set(t.date() for t in keys)), "maximum": float(sub.peak.reindex(keys).max()),
                                   "left_unknown": left_unknown, "left_reason": left_reason,
                                   "right_unknown": right_unknown, "right_reason": right_reason,
                                   "complete_event": not (left_unknown or right_unknown),
                                   "start_input_available": start_available,
                                   "start_eligible_for_warning_score": not left_unknown and start_available,
                                   "start_prediction": predictions.iloc[0], "predicted_hours": int(predictions.notna().sum()),
                                   "missing_prediction_hours": int(predictions.isna().sum()), "outcome": outcome,
                                   "any_capture_observed": bool(len(captured)), "first_warning_target": first,
                                   "warning_after_completed_record": first - HOUR if len(captured) else pd.NaT,
                                   "delay_hours_from_observed_start": (first - s) / HOUR if len(captured) else np.nan,
                                   "delay_interpretation": "from_unknown_boundary" if left_unknown else "from_known_start",
                                   "start_input_missing_reason": sub.loc[s, "input_missing_reason"],
                                   "start_date": s.strftime("%Y-%m-%d"), "start_profile": sub.loc[s, "profile_id"],
                                   "profiles_observed": ",".join(sorted(set(sub.profile_id.reindex(keys)))),
                                   "A042_exact_event_id": int(full_match.event_id.iloc[0]) if len(full_match) else np.nan,
                                   "A042_link_status": "exact_same_definition" if len(full_match) else (
                                       "different_threshold_not_linked" if threshold != 182 else "boundary_requires_separate_event")})
    events = pd.DataFrame(event_rows)
    members = pd.DataFrame(event_members)
    assert not members.duplicated(["stage", "record_key"]).any()
    assert len(members) == int(v.peak.ge(v.threshold).sum())
    for (stage, method), total in events.groupby(["stage", "method"]).hours_observed.sum().items():
        sub = v[v.stage.eq(stage)]
        assert total == int(sub.peak.ge(sub.threshold).sum()), (stage, method)
    save(events, "event_audit", rows=True)
    save(members, "event_members", rows=True)
    event_scores = []
    for k, g in events.groupby(["stage", "month", "threshold", "method"]):
        counts = g.outcome.value_counts()
        row = {**dict(zip(["stage", "month", "threshold", "method"], k)), "events": len(g),
               "dates": g.start_date.nunique(), "complete_events": int(g.complete_event.sum()),
               "right_unknown_events": int(g.right_unknown.sum()),
               "start_evaluable_events": int(g.start_eligible_for_warning_score.sum()),
               "events_any_observed_capture": int(g.any_capture_observed.sum())}
        row.update({name: int(counts.get(name, 0)) for name in ["pre_start_warning", "late_capture", "missed", "start_unknown", "start_unpredictable"]})
        row["pre_start_warning_rate_evaluable"] = row["pre_start_warning"] / row["start_evaluable_events"] if row["start_evaluable_events"] else np.nan
        event_scores.append(row)
    save(pd.DataFrame(event_scores), "event_scores")
    # False alarms retained as the counter-effect; no threshold tuning.
    selection = []
    for k, g in p[p.target.eq("peak")].groupby(["stage", "month", "method", "high_threshold_from_training"]):
        actual, predicted = g.actual.ge(g.high_threshold_from_training), g.prediction.ge(g.high_threshold_from_training)
        selection.append({**dict(zip(["stage", "month", "method", "threshold"], k)), "n": len(g),
                          "high_actual_n": int(actual.sum()), "alarm_n": int(predicted.sum()),
                          "hit_n": int((actual & predicted).sum()), "miss_n": int((actual & ~predicted).sum()),
                          "false_alarm_n": int((~actual & predicted).sum())})
    save(pd.DataFrame(selection), "selection_scores")
    p = p.merge(members[["stage", "record_key", "event_id"]], on=["stage", "record_key"], how="left", validate="many_to_one")
    save(v, "target_coverage", rows=True)
    save(p[p.method.isin(DIAG_METHODS)], "row_audit", rows=True)
    save(p[p.method.eq("hgb_calendar")].sort_values("shortfall", ascending=False).head(40), "large_shortfalls", rows=True)
    # Descriptive paired evidence and leave-one-week-out sensitivity for core comparisons.
    paired = p[p.target.eq("peak") & p.method.isin(["lag1", "hgb_calendar"])].pivot(
        index="record_key", columns="method", values=["error", "shortfall"])
    x = p[p.target.eq("peak") & p.method.eq("hgb_calendar")][
        ["record_key", "stage", "month", "date", "week", "state", "profile_seen_contributed", "is_daily_peak", "event_id"]].set_index("record_key")
    x["mae_delta"] = paired["error"]["hgb_calendar"].abs() - paired["error"]["lag1"].abs()
    x["shortfall_delta"] = paired["shortfall"]["hgb_calendar"] - paired["shortfall"]["lag1"]
    paired_rows, sensitivity = [], []
    for stage, g in x.groupby("stage"):
        subsets = {"all": g, "high_onset": g[g.state.eq("high_onset")],
                   "high_continued": g[g.state.eq("high_continued")],
                   "daily_peak": g[g.is_daily_peak], "unseen": g[g.profile_seen_contributed.eq("unseen")]}
        for name, section in subsets.items():
            if section.empty:
                continue
            for unit in ["date", "week"]:
                for value, sg in section.groupby(unit):
                    paired_rows.append({"stage": stage, "subset": name, "unit": unit, "value": value,
                                        "n": len(sg), "mae_delta": sg.mae_delta.mean(), "shortfall_delta": sg.shortfall_delta.mean()})
            for week in sorted(section.week.unique()):
                remain = section[section.week.ne(week)]
                sensitivity.append({"stage": stage, "subset": name, "excluded_week": week,
                                    "remaining_n": len(remain), "remaining_dates": remain.date.nunique(),
                                    "mae_delta": remain.mae_delta.mean(), "shortfall_delta": remain.shortfall_delta.mean()})
    save(pd.DataFrame(paired_rows), "paired_differences")
    save(pd.DataFrame(sensitivity), "week_sensitivity")
    priorities = []
    for (stage, month, state), g in p[p.target.eq("peak") & p.method.eq("hgb_calendar")].groupby(["stage", "month", "state"]):
        priorities.append({"stage": stage, "month": month, "state": state, **metric(g),
                           "events": g.event_id.nunique(), "sum_shortfall": g.shortfall.sum(),
                           "daily_peak_hours": int(g.is_daily_peak.sum()),
                           "daily_peak_dates": g[g.is_daily_peak].date.nunique(),
                           "daily_peak_sum_shortfall": g.loc[g.is_daily_peak, "shortfall"].sum()})
    save(pd.DataFrame(priorities), "failure_priority")
    assert all(sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    facts = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "python": platform.python_version(),
             "pandas": pd.__version__, "numpy": np.__version__, "training_executed": False,
             "new_predictions_generated": False, "original_inputs_preserved": len(hashes),
             "invalid_time_rows_Jan_Aug": int(len(raw) - len(valid)), "normal_Jan_Aug": len(valid),
             "eligible_Jan_Aug": int(eligible.sum()), "normal_audited_May_Aug": len(v),
             "duplicate_M002_baseline_rows_verified_and_removed": len(duplicate), "unique_saved_prediction_rows": len(p),
             "saved_thresholds": thresholds["first"].to_dict(), "reproduced_score_tables": len(reproduction),
             "distinct_full_day_profiles": len(id_vectors), "coverage": coverage,
             "event_counts": pd.DataFrame(event_scores).query("method == 'hgb_calendar'").to_dict(orient="records"),
             "completed": ["Q1", "Q2", "Q3", "Q4"], "M006": "selection and handoff recorded in M005 report; no M006 execution"}
    write_json(facts, "facts")
    print(json.dumps(facts, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
