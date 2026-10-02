"""Compare production transitions descriptively within month and exact hour."""

from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
TABLES = BASE / "tables"
PREFIX = "09.27_003"
SOURCE = BASE / "../../data/origin/okm_augumented_2021.csv"
PROFILES = BASE / "../../EDA/jsw/tables/09.23_011_daily_profile_summary.csv"
FROZEN = TABLES / f"{PREFIX}_frozen.json"
GROUPS = ("zero->positive", "positive->positive")


def main():
    config_bytes = FROZEN.read_bytes()
    config = json.loads(config_bytes)
    source_hash = sha256(SOURCE.read_bytes()).hexdigest()
    profile_hash = sha256(PROFILES.read_bytes()).hexdigest()
    assert source_hash == config["source_sha256"]
    slots = ["15분", "30분", "45분", "60분"]
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", "생산량", *slots])
    valid = raw.loc[raw["날짜"].lt(20210901) & raw["시간"].between(0, 23)].copy()
    assert len(valid) == 5784 and valid["생산량"].ge(0).all()
    valid["date"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d")
    valid["record_key"] = valid["date"] + pd.to_timedelta(valid["시간"], unit="h")
    valid = valid.set_index("record_key").sort_index()
    assert valid.index.is_unique
    peak = valid[slots].max(axis=1)
    previous_keys = valid.index - pd.Timedelta(hours=1)
    previous_peak = peak.reindex(previous_keys).to_numpy()
    previous_production = valid["생산량"].reindex(previous_keys).to_numpy()
    valid["eligible"] = np.isfinite(previous_peak) & (previous_peak < 182)
    valid["onset"] = valid["eligible"] & peak.ge(182)
    valid["previous_production"] = previous_production
    valid["period"] = np.where(valid.index < pd.Timestamp("2021-07-01"), "Jan-Jun", "Jul-Aug")
    valid["month"] = valid.index.month
    valid["hour"] = valid.index.hour
    pair = valid.loc[valid["eligible"] & valid["생산량"].gt(0)].copy()
    assert pair["previous_production"].notna().all()
    pair["transition"] = np.where(pair["previous_production"].eq(0), GROUPS[0], GROUPS[1])
    profiles = pd.read_csv(PROFILES, encoding="utf-8-sig")
    profiles["date"] = pd.to_datetime(profiles["date"].astype(str), format="%Y%m%d")
    profiles["period"] = np.where(profiles["date"] < pd.Timestamp("2021-07-01"), "Jan-Jun", "Jul-Aug")
    profiles["frequency"] = profiles.groupby(["period", "profile_id"])["date"].transform("size")
    pmap = profiles.set_index("date")
    pair["profile_id"] = pair["date"].map(pmap["profile_id"])
    pair["profile_weight"] = 1 / pair["date"].map(pmap["frequency"])
    assert pair["profile_weight"].notna().all()

    previous = pd.read_csv(TABLES / "09.27_002_onset_by_production_transition.csv", encoding="utf-8-sig")
    for period, part in pair.groupby("period"):
        for transition, group in part.groupby("transition"):
            old = previous.loc[previous["period"].eq(period) & previous["production_transition"].eq(transition)].iloc[0]
            assert len(group) == old["eligible_hours"] and int(group["onset"].sum()) == old["onset_hours"]

    rows = []
    for (period, month, hour), part in pair.groupby(["period", "month", "hour"]):
        row = {"period": period, "month": month, "hour": hour}
        for name, transition in (("start", GROUPS[0]), ("continuing", GROUPS[1])):
            group = part.loc[part["transition"].eq(transition)]
            row[f"{name}_n"] = len(group)
            row[f"{name}_events"] = int(group["onset"].sum())
            row[f"{name}_weight_n"] = float(group["profile_weight"].sum())
            row[f"{name}_weight_events"] = float(group.loc[group["onset"], "profile_weight"].sum())
        row["common_support"] = row["start_n"] > 0 and row["continuing_n"] > 0
        rows.append(row)
    strata = pd.DataFrame(rows)
    summaries = []
    for period, part in strata.groupby("period"):
        support = part.loc[part["common_support"]]
        assert len(support) > 0
        for profile_weighted in (False, True):
            exposure_suffix = "weight_n" if profile_weighted else "n"
            event_suffix = "weight_events" if profile_weighted else "events"
            pooled = support[f"start_{exposure_suffix}"] + support[f"continuing_{exposure_suffix}"]
            common_weights = pooled / pooled.sum()
            out = {"period": period, "profile_weighted": profile_weighted,
                   "common_strata": len(support), "total_strata": len(part),
                   "strata_with_either_cell_n_le_2": int((support[["start_n", "continuing_n"]].min(axis=1) <= 2).sum()),
                   "min_start_n": int(support["start_n"].min()),
                   "min_continuing_n": int(support["continuing_n"].min())}
            for name in ("start", "continuing"):
                total_n = int(part[f"{name}_n"].sum())
                overlap_n = int(support[f"{name}_n"].sum())
                out[f"{name}_all_n"] = total_n
                out[f"{name}_all_events"] = int(part[f"{name}_events"].sum())
                out[f"{name}_overlap_n"] = overlap_n
                out[f"{name}_overlap_events"] = int(support[f"{name}_events"].sum())
                out[f"{name}_retention"] = overlap_n / total_n
                out[f"{name}_excluded_n"] = total_n - overlap_n
                out[f"{name}_excluded_events"] = out[f"{name}_all_events"] - out[f"{name}_overlap_events"]
                out[f"{name}_crude_rate"] = part[f"{name}_{event_suffix}"].sum() / part[f"{name}_{exposure_suffix}"].sum()
                out[f"{name}_overlap_rate"] = support[f"{name}_{event_suffix}"].sum() / support[f"{name}_{exposure_suffix}"].sum()
                within_rates = support[f"{name}_{event_suffix}"] / support[f"{name}_{exposure_suffix}"]
                out[f"{name}_standardized_rate"] = float((common_weights * within_rates).sum())
            out["standardized_difference"] = out["start_standardized_rate"] - out["continuing_standardized_rate"]
            out["crude_difference"] = out["start_crude_rate"] - out["continuing_crude_rate"]
            assert abs(common_weights.sum() - 1) < 1e-10
            summaries.append(out)
    detail = pair[["date", "period", "month", "hour", "transition", "onset", "profile_id", "profile_weight"]].reset_index()
    detail.to_csv(TABLES / f"{PREFIX}_comparison_records.csv", index=False, encoding="utf-8-sig")
    strata.to_csv(TABLES / f"{PREFIX}_strata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(TABLES / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    assert sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    assert sha256(PROFILES.read_bytes()).hexdigest() == profile_hash
    assert FROZEN.read_bytes() == config_bytes
    facts = {"python": platform.python_version(), "source_sha256": source_hash,
             "profile_sha256": profile_hash, "frozen_sha256": sha256(config_bytes).hexdigest(),
             "pair_rows": len(pair), "summaries": summaries}
    (TABLES / f"{PREFIX}_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
