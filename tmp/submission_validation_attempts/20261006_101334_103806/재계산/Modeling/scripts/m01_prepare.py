"""M01: exact-time feature construction and audit, without model fitting.

Run from project root: regular Python 3.13 Modeling/scripts/m01_prepare.py
Use load_frame() and feature_columns() for subsequent development comparisons.
"""
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "Modeling/config/m01_contract.json"
OUT = ROOT / "Modeling/tables/m01"
SLOTS = ["15분", "30분", "45분", "60분"]
WEATHER = {"temperature": "기온", "wind": "풍속", "humidity": "습도", "rain": "강수량"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(values):
    # Stable across numpy byte order and independently reproducible with csv.
    return hashlib.sha256(",".join(format(float(v), ".17g") for v in values).encode()).hexdigest()


def read_contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def feature_columns(group, contract=None):
    contract = contract or read_contract()
    columns = contract["calendar_features"] + contract["feature_groups"]["A"]
    if group in ("B", "C_B", "D"):
        columns += contract["feature_groups"]["B_add"]
    if group in ("C_A", "C_B", "D"):
        columns += contract["feature_groups"]["C_add"]
    if group == "D":
        columns += contract["feature_groups"]["D_add"]
    if group not in ("A", "B", "C_A", "C_B", "D"):
        raise ValueError(group)
    return columns


def load_frame(contract=None):
    c = contract or read_contract()
    source = ROOT / c["source"]
    assert sha(source) == c["source_sha256"], "Source changed: review contract before execution"
    raw = pd.read_csv(source, encoding="utf-8-sig")
    scoped = raw.loc[raw["날짜"].between(*c["period"])].copy()
    invalid = scoped.loc[~scoped["시간"].between(*c["valid_hour"])].copy()
    valid = scoped.loc[scoped["시간"].between(*c["valid_hour"])].copy()
    valid["timestamp"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(valid["시간"], unit="h")
    valid = valid.sort_values("timestamp").reset_index(drop=True)
    assert not valid.timestamp.duplicated().any()
    assert valid.groupby("날짜")["시간"].apply(lambda s: set(s) == set(range(24))).all()
    assert valid[SLOTS + ["생산량"]].notna().all().all()
    f = pd.DataFrame({"timestamp": valid.timestamp})
    f["date"] = f.timestamp.dt.normalize()
    f["target_maximum"] = valid[SLOTS].max(axis=1)
    f["target_mean"] = valid[SLOTS].mean(axis=1)
    f["diag_target_production"] = valid["생산량"]
    f["diag_target_rounded_mean"] = valid["평균"]
    f["month"] = f.timestamp.dt.month
    f["weekend"] = (f.timestamp.dt.dayofweek >= 5).astype(int)
    f["hour_sin"] = np.sin(2 * np.pi * f.timestamp.dt.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f.timestamp.dt.hour / 24)
    for dow in range(7):
        f[f"dow_{dow}"] = (f.timestamp.dt.dayofweek == dow).astype(int)
    valid["mean"] = f.target_mean
    valid["maximum"] = f.target_maximum
    valid["last"] = valid[SLOTS[-1]]
    index = valid.set_index("timestamp")
    for lag in c["lags_hours"]:
        times = f.timestamp - pd.Timedelta(hours=lag)
        prior = index.reindex(pd.DatetimeIndex(times))
        f[f"lag{lag}_timestamp"] = times.where(times.isin(index.index))
        f[f"available_lag{lag}"] = times.isin(index.index)
        for metric in ("mean", "maximum", "last"):
            f[f"lag{lag}_{metric}"] = prior[metric].to_numpy()
        f[f"lag{lag}_production"] = prior["생산량"].to_numpy()
    f["lag1_production_zero"] = (f.lag1_production == 0).astype(float).where(f.available_lag1)
    for name, source_col in WEATHER.items():
        f[f"lag1_{name}"] = index.reindex(pd.DatetimeIndex(f.timestamp - pd.Timedelta(hours=1)))[source_col].to_numpy()
    for pool, lags in c["pools"].items():
        f[f"eligible_{pool}"] = f[[f"available_lag{lag}" for lag in lags]].all(axis=1)
    daily = {date: profile(cell[SLOTS].to_numpy().reshape(-1)) for date, cell in valid.groupby("날짜")}
    f["diag_target_profile"] = valid["날짜"].map(daily)
    previous_positive = f.lag1_production > 0
    current_positive = f.diag_target_production > 0
    f["diag_production_transition"] = np.select(
        [~f.available_lag1, ~previous_positive & current_positive,
         previous_positive & ~current_positive, previous_positive & current_positive],
        ["unavailable", "zero_to_positive", "positive_to_zero", "positive_to_positive"],
        default="zero_to_zero")
    return raw, invalid, valid, f


def independent_check(c, raw, invalid, valid, f):
    # Independent standard-library source parse; no reuse of pandas aggregates.
    source = ROOT / c["source"]
    observations, excluded = {}, []
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        date, hour = int(row["날짜"]), int(row["시간"])
        if not c["period"][0] <= date <= c["period"][1]:
            continue
        if not 0 <= hour <= 23:
            excluded.append((date, hour))
            continue
        ts = datetime.strptime(str(date), "%Y%m%d") + timedelta(hours=hour)
        assert ts not in observations
        slots = [float(row[s]) for s in SLOTS]
        assert int(row["평균"]) == int(sum(slots) / 4 + 0.5)
        observations[ts] = {"slots": slots, "production": float(row["생산량"]),
                            "weather": {name: float(row[col]) if row[col] else np.nan for name, col in WEATHER.items()}}
    checked_values = 0
    for _, row in f.iterrows():
        ts = row.timestamp.to_pydatetime()
        own = observations[ts]
        assert row.target_maximum == max(own["slots"])
        assert row.target_mean == sum(own["slots"]) / 4
        for lag in c["lags_hours"]:
            prior_ts = ts - timedelta(hours=lag)
            exists = prior_ts in observations
            assert bool(row[f"available_lag{lag}"]) == exists
            if exists:
                prior = observations[prior_ts]
                assert row[f"lag{lag}_timestamp"] == prior_ts < ts
                for metric, expected in (("mean", sum(prior["slots"]) / 4), ("maximum", max(prior["slots"])),
                                         ("last", prior["slots"][-1]), ("production", prior["production"])):
                    assert row[f"lag{lag}_{metric}"] == expected
                    checked_values += 1
                if lag == 1:
                    for name, expected in prior["weather"].items():
                        got = row[f"lag1_{name}"]
                        assert (pd.isna(got) and np.isnan(expected)) or got == expected
            else:
                assert pd.isna(row[f"lag{lag}_timestamp"])
                assert pd.isna(row[f"lag{lag}_mean"])
        for pool, lags in c["pools"].items():
            assert bool(row[f"eligible_{pool}"]) == all(ts - timedelta(hours=x) in observations for x in lags)
    independent_profiles = {}
    for day in sorted({t.date() for t in observations}):
        slots = [v for h in range(24) for v in observations[datetime.combine(day, datetime.min.time()) + timedelta(hours=h)]["slots"]]
        independent_profiles[day] = profile(slots)
    assert all(row.diag_target_profile == independent_profiles[row.timestamp.date()] for row in f.itertuples())
    for key, got in {"raw_rows": len(raw), "scope_rows": len(valid) + len(invalid), "invalid_rows": len(excluded),
                     "valid_rows": len(f), "valid_days": f.date.nunique(), "lag1_rows": int(f.available_lag1.sum()),
                     "lag1_and_lag2_rows": int(f.eligible_core.sum()), "unique_profiles": len(set(independent_profiles.values())),
                     "repeated_profile_days": sum(n for n in Counter(independent_profiles.values()).values() if n > 1)}.items():
        assert got == c["expected_scope"][key], (key, got)
    for group in ("A", "B", "C_A", "C_B", "D"):
        features = feature_columns(group, c)
        assert not set(features) & {col for col in f if col.startswith(("diag_", "target_", "eligible_", "available_"))}
        assert f.loc[f.eligible_common, feature_columns("C_B", c)].notna().all().all()
    return {"independent_source_rows": len(rows), "targets_checked": len(f), "lag_feature_values_checked": checked_values,
            "source_time_lookup_checked": True, "feature_allowlist_checked": True, "profile_values_checked": len(independent_profiles)}


def save(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled(), "Use regular CPython 3.13"
    c = read_contract()
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = {"executable": sys.executable, "version": sys.version, "gil_enabled": sys._is_gil_enabled(), "platform": platform.platform(), "packages": {}}
    for package in ("numpy", "pandas", "scipy", "scikit-learn"):
        runtime["packages"][package] = importlib.metadata.version(package)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    runtime["required_model_imports"] = [Ridge.__name__, HistGradientBoostingRegressor.__name__, StandardScaler.__name__]
    raw, invalid, valid, f = load_frame(c)
    verification = independent_check(c, raw, invalid, valid, f)
    save(invalid[["날짜", "시간"]].assign(reason="invalid_hour_unidentified_actual_time"), "excluded_rows")
    save(f, "hourly_frame")
    coverage = []
    for label, subset in [("all", f)] + [(f"month_{m}", g) for m, g in f.groupby("month")]:
        for lag in c["lags_hours"]:
            coverage.append({"scope": label, "rule": f"lag{lag}", "total_hours": len(subset), "eligible_hours": int(subset[f"available_lag{lag}"].sum())})
        for pool in c["pools"]:
            coverage.append({"scope": label, "rule": pool, "total_hours": len(subset), "eligible_hours": int(subset[f"eligible_{pool}"].sum())})
    save(pd.DataFrame(coverage), "lag_coverage")
    unavailable = []
    earliest = f.timestamp.min()
    for lag in c["lags_hours"]:
        for row in f.loc[~f[f"available_lag{lag}"]].itertuples():
            requested = row.timestamp - pd.Timedelta(hours=lag)
            unavailable.append({"target_timestamp": row.timestamp, "lag_hours": lag, "requested_timestamp": requested,
                                "reason": "before_scope_start" if requested < earliest else "excluded_invalid_date"})
    save(pd.DataFrame(unavailable), "unavailable_lags")
    splits, repetition, daily_rows, memberships, fallback = [], [], [], [], []
    for split in c["splits"]:
        train_mask = f.timestamp <= pd.Timestamp(split["train_end"])
        eval_mask = f.timestamp.between(pd.Timestamp(split["eval_start"]), pd.Timestamp(split["eval_end"]))
        assert f.loc[train_mask].timestamp.max() < f.loc[eval_mask].timestamp.min()
        for pool in c["pools"]:
            train = f.loc[train_mask & f[f"eligible_{pool}"]]
            evaluate = f.loc[eval_mask & f[f"eligible_{pool}"]]
            row = {"split": split["name"], "role": split["role"], "pool": pool, "train_hours": len(train), "eval_hours": len(evaluate),
                   "train_dates": train.date.nunique(), "eval_dates": evaluate.date.nunique(),
                   "eval_available_target_hours": int(eval_mask.sum()), "eval_excluded_hours": int(eval_mask.sum()) - len(evaluate)}
            profiles = set(train.diag_target_profile)
            diag = evaluate[["timestamp", "date", "diag_target_profile", "target_maximum"]].copy()
            diag["train_profile_overlap"] = diag.diag_target_profile.isin(profiles)
            counts = diag[["date", "diag_target_profile"]].drop_duplicates().diag_target_profile.value_counts()
            diag["profile_weight"] = 1 / diag.diag_target_profile.map(counts)
            diag["daily_maximum_weight"] = 0.0
            full_days = set()
            for day, group in diag.groupby("date"):
                complete = len(group) == 24
                daily_rows.append({"split": split["name"], "pool": pool, "date": day.date(), "eligible_hours": len(group),
                                   "complete_day": complete, "observed_daily_maximum": float(f.loc[f.date == day, "target_maximum"].max())})
                if complete:
                    full_days.add(day)
                    is_max = group.target_maximum == group.target_maximum.max()
                    diag.loc[group.index[is_max], "daily_maximum_weight"] = 1 / int(is_max.sum())
            assert abs(diag.daily_maximum_weight.sum() - len(full_days)) < 1e-10
            row["complete_eval_days"] = len(full_days)
            row["incomplete_eval_days"] = int(f.loc[eval_mask].date.nunique()) - len(full_days)
            row["daily_maximum_eval_hours"] = int((diag.daily_maximum_weight > 0).sum())
            splits.append(row)
            repetition.append({"split": split["name"], "pool": pool, "eval_days": diag.date.nunique(), "eval_profiles": len(counts),
                               "overlap_days": diag.loc[diag.train_profile_overlap].date.nunique(), "overlap_hours": int(diag.train_profile_overlap.sum()),
                               "new_profile_hours": int((~diag.train_profile_overlap).sum())})
            diag["split"], diag["pool"] = split["name"], pool
            memberships.append(diag)
        ev = f.loc[eval_mask]
        for baseline, order in c["missing_policy"]["secondary_baseline_fallback"].items():
            selected = pd.Series("abstain", index=ev.index)
            for lag in order:
                selected.loc[(selected == "abstain") & ev[f"available_lag{lag}"]] = f"lag{lag}"
            for used, n in selected.value_counts().items():
                fallback.append({"split": split["name"], "baseline": baseline, "used": used, "hours": int(n)})
    save(pd.DataFrame(splits), "split_counts")
    save(pd.DataFrame(repetition), "profile_overlap")
    save(pd.DataFrame(daily_rows), "daily_coverage")
    save(pd.concat(memberships, ignore_index=True), "evaluation_diagnostics")
    save(pd.DataFrame(fallback), "baseline_fallback_coverage")
    weather_missing = [{"feature": col, "pool": pool, "missing_hours": int(f.loc[f[f"eligible_{pool}"], col].isna().sum())}
                       for pool in c["pools"] for col in c["feature_groups"]["D_add"]]
    save(pd.DataFrame(weather_missing), "weather_missing")
    features = [{"group": group, "feature": col, "availability": "target calendar known" if col in c["calendar_features"] else "observed lag only"}
                for group in ("A", "B", "C_A", "C_B", "D") for col in feature_columns(group, c)]
    save(pd.DataFrame(features), "feature_dictionary")
    assert sha(ROOT / c["source"]) == c["source_sha256"]
    verification.update({"status": "passed", "contract_sha256": sha(CONTRACT), "script_sha256": sha(Path(__file__)),
                         "source_sha256": c["source_sha256"], "model_fitting_performed": False,
                         "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
                         "outputs_sha256": {p.name: sha(p) for p in sorted(OUT.glob("*.csv"))}})
    (OUT / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"valid_hours": len(f), "lag_counts": {str(lag): int(f[f"available_lag{lag}"].sum()) for lag in c["lags_hours"]},
               "core_hours": int(f.eligible_core.sum()), "common_hours": int(f.eligible_common.sum()),
               "split_counts": splits, "profile_overlap": repetition, "weather_missing": weather_missing,
               "model_fitting_performed": False}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
