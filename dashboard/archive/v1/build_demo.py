"""Build an offline replay from existing observations/predictions; no model fitting."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard"
RAW = ROOT / "data/origin/okm_augumented_2021.csv"
DEV = ROOT / "Modeling/tables/regime_age_ablation/development_predictions.csv"
FOLLOW = ROOT / "Modeling/tables/regime_age_ablation/followup_predictions.csv"
CLASS = ROOT / "Modeling/tables/regime_age_ablation/followup_classification.csv"
SLOTS = ["15분", "30분", "45분", "60분"]
VARIANT = "G1_noage"
HOUR = 3_600_000


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def epoch(value: str) -> int:
    # Synthetic UTC keeps the source's wall-clock labels identical in all browsers.
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def number(value: str):
    if value == "":
        return None
    try:
        n = float(value)
        return int(n) if n.is_integer() else n
    except ValueError:
        return value


def coverage(rows, q):
    hits = sum(abs(float(r["actual"]) - float(r["prediction"])) <= q for r in rows)
    return {"n": len(rows), "covered": hits, "coverage": hits / len(rows) if rows else None}


def main():
    raw = read(RAW)
    columns = list(raw[0])
    full_raw = [[number(r[c]) for c in columns] for r in raw]
    observed = {}
    invalid = []
    for r in raw:
        h = int(r["시간"])
        if not 0 <= h <= 23:
            invalid.append({"date": r["날짜"], "hour": h})
            continue
        t = epoch(datetime.strptime(r["날짜"], "%Y%m%d").isoformat()) + h * HOUR
        assert t not in observed
        observed[t] = max(float(r[c]) for c in SLOTS)

    dev = [r for r in read(DEV) if r["variant"] == VARIANT]
    follow = [r for r in read(FOLLOW) if r["variant"] == VARIANT]
    classifiers = [r for r in read(CLASS) if r["method"] == "noage"]
    class_by_time = {epoch(r["timestamp"]): r for r in classifiers}
    assert len(dev) == 1462 and len(follow) == 1344 and len(classifiers) == 1428
    assert max(epoch(r["timestamp"]) for r in dev) < min(epoch(r["timestamp"]) for r in follow)

    # Fixed before reading follow-up errors: nearest-rank finite-sample absolute residual.
    residuals = sorted(abs(float(r["actual"]) - float(r["prediction"])) for r in dev)
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * 0.90))
    q = residuals[rank - 1]
    dev_coverage = coverage(dev, q)
    follow_coverage = coverage(follow, q)
    forecasts = []
    checked_maxima = 0
    for r in follow:
        t = epoch(r["timestamp"])
        assert float(r["actual"]) == observed[t]
        checked_maxima += 1
        c = class_by_time[t]
        p = float(r["prediction"])
        alarm = int(c["alarm"])
        routed = int(r["routed"])
        assert alarm == routed
        forecasts.append([t, p, p - q, p + q, float(c["score"]), alarm, routed])

    scores = [[epoch(r["timestamp"]), float(r["score"]), int(r["alarm"])] for r in classifiers]
    dates = sorted({r["날짜"] for r in raw if 0 <= int(r["시간"]) <= 23})
    slot_count = len(observed) * 4
    report = {
        "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "scope": "Static offline historical replay; no model inference or refitting in browser.",
        "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [RAW, DEV, FOLLOW, CLASS]},
        "raw_rows_embedded": len(raw), "raw_columns_embedded": len(columns),
        "valid_hour_rows": len(observed), "invalid_hour_rows": len(invalid),
        "invalid_rows_preserved_in_payload": True,
        "quarter_hour_observations": slot_count,
        "forecast_rows": len(forecasts), "classification_rows": len(scores),
        "prediction_variant": VARIANT,
        "interval": {
            "target": "Maximum of the FOUR quarter-hour observations in the target hour, not individual slots.",
            "name": "Nominal 90% empirical prediction interval",
            "method": "prediction +/- fixed 90% absolute-residual order statistic; no clipping",
            "calibration_period": "2021-05-01 to 2021-06-30, prior-month OOF development predictions",
            "rank": rank, "half_width": q, "development": dev_coverage,
            "followup_2021_07_08": follow_coverage,
            "limits": "Development selected after exploratory modeling; dependent time series and reuse of development data. No formal 90% coverage guarantee and no confidence interval for a parameter.",
        },
        "verification": {"raw_hour_maxima_checked": checked_maxima,
                         "alarm_route_matches": True, "future_error_used_to_set_width": False,
                         "forecasts_only_published_at_target_hour_start": True},
        "time_contract": "hour row h unfolds at h:15, h:30, h:45, (h+1):00; display convention consistent with hour blocks, not a verified meter acquisition timestamp.",
        "forecast_missing_policy": "Show unavailable; do not impute or forward-fill predictions.",
    }
    payload = {"columns": columns, "raw": full_raw, "forecasts": forecasts, "scores": scores,
               "dates": dates, "meta": report}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    template = (OUT / "template.html").read_text(encoding="utf-8")
    assert template.count("__EMBEDDED_DATA__") == 1
    html = template.replace("__EMBEDDED_DATA__", encoded)
    assert "__EMBEDDED_DATA__" not in html
    assert "fetch(" not in html and '<script src=' not in html
    (OUT / "전력_흐름_시연.html").write_text(html, encoding="utf-8")
    (OUT / "검증_결과.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"raw_rows": len(raw), "slots": slot_count, "forecasts": len(forecasts),
                      "interval_half_width": q, "followup_coverage": follow_coverage,
                      "html_bytes": len(html.encode("utf-8")), "checks": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
