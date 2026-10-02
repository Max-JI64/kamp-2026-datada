"""Shared read-only source and exact-time helpers for connected analyses."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
TABLES = BASE / "tables"
SOURCE = BASE / "../../data/origin/okm_augumented_2021.csv"
PROFILES = BASE / "../../EDA/jsw/tables/09.23_011_daily_profile_summary.csv"
SOURCE_HASH = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
PROFILE_HASH = "917b05d100c5ccbcfdacac58d9586b85fe2b842f573a4487742ff5afb566dd3c"
SLOTS = ["15분", "30분", "45분", "60분"]


def load(include_september=False):
    assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
    assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    assert raw.shape == (6168, 18)
    data = raw.loc[raw["시간"].between(0, 23)].copy()
    if not include_september:
        data = data.loc[data["날짜"].lt(20210901)].copy()
    data["date"] = pd.to_datetime(data["날짜"].astype(str), format="%Y%m%d")
    data["timestamp"] = data["date"] + pd.to_timedelta(data["시간"], unit="h")
    data = data.set_index("timestamp").sort_index()
    assert data.index.is_unique and data.groupby("date").size().eq(24).all()
    assert len(data) == (6120 if include_september else 5784)
    data["peak"] = data[SLOTS].max(axis=1)
    data["exact_mean"] = data[SLOTS].mean(axis=1)
    assert np.floor(data["exact_mean"] + .5).eq(data["평균"]).all()
    data["month"] = data.index.month
    data["hour"] = data.index.hour
    data["weekday"] = data.index.dayofweek
    data["weekend"] = (data["weekday"] >= 5).astype(int)
    data["period"] = np.select([data.index < pd.Timestamp("2021-07-01"), data.index < pd.Timestamp("2021-09-01")], ["Jan-Jun", "Jul-Aug"], default="Sep")
    data["production_positive"] = data["생산량"].gt(0).astype(int)
    data["prev_peak"] = data["peak"].reindex(data.index - pd.Timedelta(hours=1)).to_numpy()
    data["prev_production"] = data["생산량"].reindex(data.index - pd.Timedelta(hours=1)).to_numpy()
    data["segment"] = data.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum().to_numpy()
    profiles = pd.read_csv(PROFILES, encoding="utf-8-sig")
    profiles["date"] = pd.to_datetime(profiles["date"].astype(str), format="%Y%m%d")
    profiles["period"] = np.where(profiles["date"] < pd.Timestamp("2021-07-01"), "Jan-Jun", "Jul-Aug")
    profiles["freq"] = profiles.groupby(["period", "profile_id"])["date"].transform("size")
    pmap = profiles.set_index("date")
    data["profile"] = data["date"].map(pmap["profile_id"])
    data["profile_weight"] = 1 / data["date"].map(pmap["freq"])
    if not include_september:
        assert data["profile_weight"].notna().all()
    return raw, data


def events(data, threshold):
    high = data["peak"].ge(threshold)
    starts = high & ~data["prev_peak"].ge(threshold)
    eligible = data["prev_peak"].notna() & data["prev_peak"].lt(threshold)
    onset = high & eligible
    work = data.loc[high].copy()
    work["event_id"] = starts.cumsum().loc[work.index].to_numpy()
    work["excess"] = (work["peak"] - threshold).clip(lower=0)
    records = []
    for event_id, group in work.groupby("event_id"):
        first, last = group.index[0], group.index[-1]
        following = last + pd.Timedelta(hours=1)
        records.append({"event_id": int(event_id), "start": first, "end": last,
                        "period": group["period"].iloc[0], "date": first.normalize(),
                        "hour": first.hour, "hours": len(group), "peak_max": group["peak"].max(),
                        "excess_sum": group["excess"].sum(), "segment": int(group["segment"].iloc[0]),
                        "start_production_positive": int(group["production_positive"].iloc[0]),
                        "left_censored": pd.isna(group["prev_peak"].iloc[0]),
                        "right_censored": following not in data.index,
                        "crosses_period": group["period"].nunique() > 1})
    event_table = pd.DataFrame(records)
    assert int(event_table["hours"].sum()) == int(high.sum())
    assert len(event_table) == int(starts.sum())
    assert int(onset.sum()) + int(event_table["left_censored"].sum()) == len(event_table)
    return high, eligible, onset, event_table


def save(frame, prefix, name):
    frame.to_csv(TABLES / f"{prefix}_{name}.csv", index=False, encoding="utf-8-sig")


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def finish(prefix, facts):
    assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
    config_path = TABLES / f"{prefix}_frozen.json"
    facts["source_sha256"] = SOURCE_HASH
    facts["profile_sha256"] = sha256(PROFILES.read_bytes()).hexdigest()
    facts["frozen_sha256"] = sha256(config_path.read_bytes()).hexdigest()
    facts = json_value(facts)
    (TABLES / f"{prefix}_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=True, allow_nan=False), flush=True)
