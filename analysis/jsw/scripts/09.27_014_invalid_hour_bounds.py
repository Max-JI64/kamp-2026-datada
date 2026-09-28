"""Finite-file sensitivity of prevalence to the 48 invalid-hour records."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, load, save, finish

PREFIX = "09.27_014"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
raw, _ = load()
peak = raw[["15분", "30분", "45분", "60분"]].max(axis=1)
threshold = cfg["threshold"]
earlier = raw["날짜"].lt(20210701)
later = raw["날짜"].between(20210701, 20210831)
valid = raw["시간"].between(0, 23)
n0, h0 = int(earlier.sum()), int(peak.loc[earlier].ge(threshold).sum())
nv = int((later & valid).sum())
hv = int(peak.loc[later & valid].ge(threshold).sum())
ni = int((later & ~valid).sum())
hi = int(peak.loc[later & ~valid].ge(threshold).sum())
assert (n0, h0, nv, hv, ni, hi) == (4344, 235, 1440, 220, 48, 10)
baseline = h0 / n0
rows = []
for label, h, n in (("valid_hours_only", hv, nv), ("all_dated_records", hv + hi, nv + ni),
                     ("all_invalid_records_assumed_low", hv, nv + ni),
                     ("all_invalid_records_assumed_high", hv + ni, nv + ni)):
    rows.append({"scenario": label, "earlier_n": n0, "earlier_high": h0, "earlier_fraction": baseline,
                 "later_n": n, "later_high": h, "later_fraction": h / n,
                 "later_minus_earlier": h / n - baseline,
                 "interpretation": "record fraction; actual hour interpretation requires provider definitions"})
assert all(r["later_minus_earlier"] > 0 for r in rows)
daily = []
for date, group in raw.loc[later & ~valid].groupby("날짜"):
    daily.append({"date": date, "n": len(group), "high": int(peak.loc[group.index].ge(threshold).sum())})
save(pd.DataFrame(rows), PREFIX, "bounds")
save(pd.DataFrame(daily), PREFIX, "invalid_date_counts")
finish(PREFIX, {"scenarios": rows, "invalid_record_count": ni, "known_invalid_high_count": hi,
                "timestamp_repair": False, "event_duration_or_onset_identification": False})
