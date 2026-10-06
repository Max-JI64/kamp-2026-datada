"""S01/S02: causal onset labels, future persistence, past-only precursor diagnosis.
No classifier, regression, July/August evaluation, or manuscript generation.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_diagnosis"
SLOTS = ["15분", "30분", "45분", "60분"]
SOURCE = ROOT / "data/origin/okm_augumented_2021.csv"
SOURCE_SHA = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
FEATURES = ["prior_mean", "past_median6", "past_iqr6", "past_range6",
            "past_slope6", "prior_delta", "prior_last_minus_mean",
            "prior_slot_range", "prior_production", "prior_production_delta",
            "prior_state_age", "prior_state_left_censored"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save(frame, name):
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def load():
    assert sha(SOURCE) == SOURCE_SHA
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    raw = raw.loc[raw["날짜"].between(20210101, 20210630) & raw["시간"].between(0, 23)].copy()
    raw["timestamp"] = pd.to_datetime(raw["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(raw["시간"], unit="h")
    raw = raw.sort_values("timestamp").set_index("timestamp")
    assert raw.index.is_unique
    d = raw.reindex(pd.date_range("2021-01-01", "2021-06-30 23:00", freq="h"))
    d.index.name = "timestamp"
    d["valid"] = d[SLOTS].notna().all(axis=1)
    d["level"] = d[SLOTS].mean(axis=1).where(d.valid)
    d["maximum"] = d[SLOTS].max(axis=1).where(d.valid)
    d["date"] = d.index.normalize()
    d["month"] = d.index.month
    d["hour"] = d.index.hour
    d["weekend"] = (d.index.dayofweek >= 5).astype(int)
    profiles = {date: hashlib.sha256(g[SLOTS].to_numpy(dtype=np.int64).tobytes()).hexdigest()
                for date, g in raw.groupby(raw.index.normalize())}
    d["profile"] = d.date.map(profiles)
    history = pd.concat([d.level.shift(lag).rename(str(lag)) for lag in range(1, 7)], axis=1)
    d["eligible_onset"] = d.valid & history.notna().all(axis=1)
    d["prior_mean"] = history["1"]
    d["past_median6"] = history.median(axis=1).where(d.eligible_onset)
    d["past_iqr6"] = (history.quantile(.75, axis=1)-history.quantile(.25, axis=1)).where(d.eligible_onset)
    d["past_range6"] = (history.max(axis=1)-history.min(axis=1)).where(d.eligible_onset)
    d["past_slope6"] = ((history["1"]-history["6"])/5).where(d.eligible_onset)
    d["prior_delta"] = d.level.shift(1)-d.level.shift(2)
    d["prior_last_minus_mean"] = d["60분"].shift(1)-d.prior_mean
    d["prior_slot_range"] = (d[SLOTS].max(axis=1)-d[SLOTS].min(axis=1)).shift(1).where(d.valid.shift(1, fill_value=False))
    d["prior_production"] = d["생산량"].shift(1)
    d["prior_production_delta"] = d["생산량"].shift(1)-d["생산량"].shift(2)
    d["delta"] = d.level-d.prior_mean
    d["deviation"] = d.level-d.past_median6
    return d


def thresholds(d, quantile=.90):
    jan = d.loc[d.month.eq(1) & d.eligible_onset]
    return {"up": float(jan.loc[jan.delta.gt(0), "delta"].quantile(quantile)),
            "down": float((-jan.loc[jan.delta.lt(0), "delta"]).quantile(quantile))}


def label(d, th, multiplier=1.5):
    """Onset identity uses only current/past values; future is used only afterwards."""
    z = d.copy()
    up = z.eligible_onset & z.delta.ge(th["up"]) & z.deviation.ge(th["up"]) & z.deviation.ge(multiplier*z.past_iqr6)
    down = z.eligible_onset & (-z.delta).ge(th["down"]) & (-z.deviation).ge(th["down"]) & (-z.deviation).ge(multiplier*z.past_iqr6)
    candidate = np.select([up, down], [1, -1], default=0)
    starts = np.zeros(len(z), dtype=int)
    ages, censored, directions, available = [], [], [], []
    active = None
    last_gap = None
    event_rows = []
    for i, (ts, row) in enumerate(z.iterrows()):
        # State at t-1, before current t measurement.
        ages.append(i-active["i"] if active else 0)
        directions.append(active["sign"] if active else 0)
        censored.append(int(active is None and (last_gap is None or i-last_gap <= 6)))
        available.append(ts-pd.Timedelta(hours=1))
        if not row.valid:
            active = None
            last_gap = i
            continue
        sign = int(candidate[i])
        if active is not None:
            if active["sign"]*(row.level-active["baseline"]) < .5*th[active["direction"]]:
                active = None
            elif sign and sign != active["sign"]:
                active = None
        if sign and active is None:
            direction = "up" if sign == 1 else "down"
            active = {"i": i, "baseline": row.past_median6, "sign": sign, "direction": direction}
            starts[i] = sign
            event_rows.append({"event_id": len(event_rows)+1, "position": i,
                              "timestamp": ts, "date": row.date, "month": row.month,
                              "hour": row.hour, "weekend": row.weekend,
                              "direction": direction, "baseline": row.past_median6,
                              "first_mean": row.level, "first_maximum": row.maximum,
                              "first_delta": row.delta, "first_deviation": row.deviation,
                              "past_iqr6": row.past_iqr6,
                              "profile": row.profile,
                              "previous_mean": row.prior_mean,
                              "production_before": row.prior_production,
                              "production_at_start": row["생산량"]})
    z["candidate"] = candidate
    z["onset"] = starts
    z["prior_state_age"] = ages
    z["prior_state_direction"] = directions
    z["prior_state_left_censored"] = censored
    z["latest_input_timestamp"] = available
    z["label_confirmed_at"] = z.index + pd.Timedelta(hours=2)
    z["persistence_known"] = z.valid & z.valid.shift(-1, fill_value=False) & z.valid.shift(-2, fill_value=False)
    z["sustained_onset"] = 0.
    z.loc[~z.persistence_known, "sustained_onset"] = np.nan
    for event in event_rows:
        i, direction = event["position"], event["direction"]
        sign = 1 if direction == "up" else -1
        boundary = .5*th[direction]
        n = 0
        while i+n < len(z) and bool(z.valid.iloc[i+n]) and sign*(z.level.iloc[i+n]-event["baseline"]) >= boundary:
            n += 1
        end = i+n
        reason = "scope_end" if end == len(z) else ("gap" if not bool(z.valid.iloc[end]) else "return")
        event["reference_departure_hours"] = n
        event["right_censored"] = reason != "return"
        event["end_reason"] = reason
        event["last_departure_timestamp"] = z.index[i+n-1]
        event["duration_confirmed_at"] = z.index[end] if end < len(z) else pd.NaT
        for horizon in [1, 3, 6]:
            window = z.iloc[i:i+horizon]
            known = len(window) == horizon and bool(window.valid.all())
            sustained = bool((sign*(window.level-event["baseline"]) >= boundary).all()) if known else None
            event[f"sustained{horizon}"] = sustained
        event["label_confirmed_at"] = z.index[i]+pd.Timedelta(hours=2) if event["sustained3"] is not None else pd.NaT
        first3 = z.iloc[i:i+3]
        event["next3_mean"] = float(first3.level.mean()) if event["sustained3"] is not None else np.nan
        event["next3_min_mean"] = float(first3.level.min()) if event["sustained3"] is not None else np.nan
        event["next3_max_mean"] = float(first3.level.max()) if event["sustained3"] is not None else np.nan
        z.iloc[i, z.columns.get_loc("sustained_onset")] = sign if event["sustained3"] else (0. if event["sustained3"] is not None else np.nan)
    events = pd.DataFrame(event_rows)
    return z, events


def probe():
    assert not (OUT/"probe.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    jan = load().loc[lambda x: x.month.eq(1)].copy()
    result = {"source_sha256": sha(SOURCE), "script_sha256": sha(__file__), "scope": "January only, no model fit",
              "threshold_quantiles": {}, "primary_preference": "q90 predefined large hourly changes plus median departure and 1.5IQR filter; q95 sensitivity only"}
    for q in [.90,.95]:
        th = thresholds(jan,q)
        _, ev = label(jan,th)
        result["threshold_quantiles"][str(q)] = {"thresholds": th,
            "onsets": ev.groupby("direction").size().to_dict(),
            "sustained3": ev[ev.sustained3.eq(True)].groupby("direction").size().to_dict()}
    result["ranges"] = {"mean_min":float(jan.level.min()),"mean_max":float(jan.level.max()),
                        "maximum":float(jan.maximum.max())}
    dump(result,OUT/"probe.json")
    save(jan.loc[jan.eligible_onset, ["level","prior_mean","past_median6","past_iqr6","delta","deviation"]].reset_index(),"january_probe")
    print(json.dumps(result,ensure_ascii=False),flush=True)


def freeze():
    assert not (OUT/"contract.json").exists()
    p = json.loads((OUT/"probe.json").read_text(encoding="utf-8"))
    assert p["script_sha256"] == sha(__file__)
    th = p["threshold_quantiles"]["0.9"]["thresholds"]
    c = {"recorded_at":datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
         "entrypoint_sha256":sha(__file__),"source_sha256":SOURCE_SHA,
         "probe_sha256":sha(OUT/"probe.json"),
         "scope":"January-June only. S02 condition diagnosis January-March; April-June onset counts only, no performance selection.",
         "thresholds":th,"quantile":.90,"calibration":"January valid rows with six exact past observations; separate positive/negative one-hour mean deltas",
         "onset":"delta and departure from preceding6 median each >=direction threshold; departure>=1.5 preceding6 IQR",
         "episode":"freeze baseline at onset; suppress same-direction starts while departure>=half threshold; allow opposite qualifying start; reset on return or gap. State features prior-only. Large same-direction steps saved as suppressed candidates, not new starts.",
         "persistence":"at t,t+1,t+2 all observed levels in original departure direction at least half threshold. Unknown if gap/end; confirmed t+2.",
         "duration":"consecutive hours on one side of original frozen baseline until return/gap/end; opposite-event reference durations can overlap; not a disjoint state segmentation or constant box guarantee",
         "sensitivity":"q95 and horizon1/6 descriptive; no model-dependent definition choice",
         "features":FEATURES+["prior_state_direction"],
         "conditions":"January-March only month/hour/previous-level-bin; January prior mean quintiles; min2event and min5control records; weights=min(event,control); one-event cells sensitivity",
         "controls":"eligible onset=0 rows with the same complete persistence observation window; controls include ongoing states, not all stable",
         "no_future_inputs":"target level/delta/deviation, event outcomes, full-day profile and future duration excluded from features",
         "splits":"for validation months2..6 use timestamp before month and confirmed_at before month. No class prediction fitted.",
         "no_manuscript_edit":True,"fits":0,"later_evaluated":False}
    dump(c,OUT/"contract.json")
    print("S01/S02 contract frozen; thresholds="+json.dumps(th),flush=True)


def precursor(z, events):
    eligible = z.eligible_onset & z.persistence_known & z.month.le(3)
    early = z.loc[eligible].copy()
    edges = sorted(set(early.loc[early.month.eq(1),"prior_mean"].quantile([.2,.4,.6,.8]).tolist()))
    early["level_bin"] = np.searchsorted(edges,early.prior_mean,side="right")
    rows, cells, coverage = [], [], []
    targets = {"up":early.onset.eq(1),"down":early.onset.eq(-1),
               "sustained_up":early.sustained_onset.eq(1),"sustained_down":early.sustained_onset.eq(-1)}
    for name, mask in targets.items():
        a = early.loc[mask]
        b = early.loc[early.onset.eq(0)]
        for feature in FEATURES:
            rows.append({"target":name,"feature":feature,"events":len(a),"controls":len(b),
                         "event_mean":float(a[feature].mean()),"control_mean":float(b[feature].mean()),
                         "event_median":float(a[feature].median()),"control_median":float(b[feature].median())})
        keys = ["month","hour","level_bin"]
        event_groups = {k:g for k,g in a.groupby(keys)}
        control_groups = {k:g for k,g in b.groupby(keys)}
        for minimum in [1,2]:
            shared = [k for k in event_groups if k in control_groups and len(event_groups[k])>=minimum and len(control_groups[k])>=5]
            count_a = sum(len(event_groups[k]) for k in shared)
            count_b = sum(len(control_groups[k]) for k in shared)
            coverage.append({"target":name,"minimum_events_per_cell":minimum,"matched_cells":len(shared),
                             "events_total":len(a),"events_covered":count_a,
                             "event_coverage":count_a/len(a) if len(a) else 0.,
                             "controls_covered":count_b})
            for feature in FEATURES:
                diffs, weights = [], []
                for key in shared:
                    ag,bg = event_groups[key],control_groups[key]
                    w = min(len(ag),len(bg))
                    av,bv = float(ag[feature].mean()),float(bg[feature].mean())
                    cells.append({"target":name,"minimum_events_per_cell":minimum,
                        "month":key[0],"hour":key[1],"level_bin":key[2],"feature":feature,
                        "event_n":len(ag),"control_n":len(bg),"event_mean":av,"control_mean":bv,"weight":w})
                    diffs.append(av-bv);weights.append(w)
                rows.append({"target":name,"feature":feature,"weighting":"matched",
                    "minimum_events_per_cell":minimum,"events":count_a,"controls":count_b,
                    "matched_cells":len(shared),"matched_difference":float(np.average(diffs,weights=weights)) if diffs else np.nan})
    calendar = early.groupby(["month","weekend","hour"]).agg(hours=("onset","size"),
        up=("onset",lambda x:int(x.eq(1).sum())),down=("onset",lambda x:int(x.eq(-1).sum())),
        sustained_up=("sustained_onset",lambda x:int(x.eq(1).sum())),
        sustained_down=("sustained_onset",lambda x:int(x.eq(-1).sum()))).reset_index()
    for field in ["up","down","sustained_up","sustained_down"]:
        calendar[field+"_rate"] = calendar[field]/calendar.hours
    splits = []
    for month in range(2,7):
        boundary = pd.Timestamp(2021,month,1)
        train = z.loc[z.eligible_onset & z.persistence_known & (z.index < boundary) & z.label_confirmed_at.lt(boundary)]
        valid = z.loc[z.eligible_onset & z.persistence_known & z.month.eq(month)]
        for group, data in [("train",train),("validation_count_only",valid)]:
            for sign,direction in [(1,"up"),(-1,"down")]:
                part = data.loc[data.onset.eq(sign)]
                splits.append({"month":month,"partition":group,"direction":direction,"hours":len(data),
                    "onsets":len(part),"sustained3":int(part.sustained_onset.eq(sign).sum()),
                    "event_dates":part.date.nunique(),"event_profiles":part.profile.nunique(),
                    "last_label_confirmation":str(data.label_confirmed_at.max())})
    save(pd.DataFrame(rows),"precursor_summary")
    save(pd.DataFrame(cells),"matched_cells")
    save(pd.DataFrame(coverage),"matched_coverage")
    save(calendar,"calendar_conditions")
    save(pd.DataFrame(splits),"split_counts")
    dump({"prior_level_edges":edges,"scope":"Jan-Mar","comparison":"association, not forecasting or causal evidence"},OUT/"precursor_contract.json")


def run():
    assert not (OUT/"run.json").exists()
    c = json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    assert c["entrypoint_sha256"] == sha(__file__) and c["source_sha256"] == sha(SOURCE)
    assert c["probe_sha256"] == sha(OUT/"probe.json")
    raw = load()
    z, events = label(raw,c["thresholds"])
    rows = []
    for (month,direction),g in events.groupby(["month","direction"]):
        complete = g.loc[~g.right_censored]
        rows.append({"month":month,"direction":direction,"events":len(g),"dates":g.date.nunique(),
            "profiles":g.profile.nunique(),"sustained3":int(g.sustained3.eq(True).sum()),
            "short3":int(g.sustained3.eq(False).sum()),"unknown3":int(g.sustained3.isna().sum()),
            "sustained6":int(g.sustained6.eq(True).sum()),"censored_duration":int(g.right_censored.sum()),
            "complete_duration_median":float(complete.reference_departure_hours.median()) if len(complete) else None})
    save(pd.DataFrame(rows),"event_summary")
    save(events,"events")
    keep = ["valid","eligible_onset","level","maximum","date","month","hour","weekend","profile",
            "candidate","onset","sustained_onset","persistence_known","label_confirmed_at",
            "latest_input_timestamp","delta","deviation"]+FEATURES+["prior_state_direction"]
    save(z[keep].reset_index(),"hourly_labels")
    sens = []
    for q in [.90,.95]:
        th = thresholds(raw,q)
        _,ev = label(raw,th)
        for (month,direction),g in ev.groupby(["month","direction"]):
            sens.append({"quantile":q,"month":month,"direction":direction,
                         "threshold":th[direction],"events":len(g),
                         "sustained3":int(g.sustained3.eq(True).sum()),
                         "sustained6":int(g.sustained6.eq(True).sum())})
    save(pd.DataFrame(sens),"definition_sensitivity")
    precursor(z,events)
    # Deterministic representative windows, using Jan-Mar only.
    selected = events.loc[events.month.le(3)].sort_values("first_deviation")
    selected = pd.concat([selected.head(2),selected.tail(2),
                          selected.loc[selected.sustained3.eq(False)].head(2)]).drop_duplicates("event_id")
    windows = []
    for _,ev in selected.iterrows():
        part = z.loc[ev.timestamp-pd.Timedelta(hours=6):ev.timestamp+pd.Timedelta(hours=8),
                     ["level","maximum","valid"]].reset_index()
        part["event_id"] = ev.event_id
        part["direction"] = ev.direction
        part["baseline"] = ev.baseline
        windows.append(part)
    save(pd.concat(windows,ignore_index=True),"representative_windows")
    result = {"status":"completed","contract_sha256":sha(OUT/"contract.json"),
              "hours":int(z.valid.sum()),"eligible_onset_hours":int(z.eligible_onset.sum()),
              "events":len(events),"sustained3":int(events.sustained3.eq(True).sum()),
              "thresholds":c["thresholds"],"new_fits":0,"later_evaluated":False,
              "recorded_ranges":{"mean_max":float(z.level.max()),"maximum":float(z.maximum.max())},
              "outputs_sha256":{p.name:sha(p) for p in OUT.glob("*.csv")} }
    result["outputs_sha256"]["precursor_contract.json"] = sha(OUT/"precursor_contract.json")
    dump(result,OUT/"run.json")
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    assert sys.version_info[:2] == (3,13) and sys._is_gil_enabled()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["probe","freeze","run"])
    globals()[parser.parse_args().stage]()
