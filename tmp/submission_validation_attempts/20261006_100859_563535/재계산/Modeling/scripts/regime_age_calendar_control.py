"""Posthoc Monday08 control after S06: diagnosis only, never adopted or tuned."""
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from regime_age_ablation_verify import ROOT, OUT, sha, errstats


def main():
    assert not (OUT / "calendar_control_contract.json").exists()
    names = ["followup_classification.csv", "followup_predictions.csv", "run.json", "independent_verification.json"]
    c = {"created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
         "entrypoint_sha256": sha(__file__), "inputs_sha256": {n: sha(OUT / n) for n in names},
         "reason": "All7 candidate alarms at Monday08; check whether a weekday/hour rule explains the selection and value gain.",
         "rule": "Monday hour08, no other weekday/hour trials or threshold changes",
         "scope": "Posthoc control from already-seen July-August; not independent validation or a new adopted candidate",
         "fits": 0}
    (OUT / "calendar_control_contract.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    cp = pd.read_csv(OUT / "followup_classification.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    cp = cp.loc[cp.method == "noage"]
    p = pd.read_csv(OUT / "followup_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    base = p.loc[p.variant == "B0"].copy()
    b1 = p.loc[p.variant == "B1_all"].set_index("timestamp").loc[base.timestamp]
    mask = (base.timestamp.dt.dayofweek == 0) & (base.timestamp.dt.hour == 8)
    g = base.copy(); g["prediction"] = np.where(mask, b1.prediction, base.prediction)
    g["variant"] = "Monday08_posthoc"; g["routed"] = mask.astype(int)
    det = []; rows = []
    common = base.timestamp
    for period in ["Jul-Aug", "7", "8"]:
        subset = cp.loc[cp.event.notna() & cp.timestamp.isin(common)]
        part = g
        if period != "Jul-Aug":
            subset = subset.loc[subset.month == int(period)]; part = part.loc[part.month == int(period)]
        a = (subset.timestamp.dt.dayofweek == 0) & (subset.timestamp.dt.hour == 8)
        tp = int((a & subset.event.eq(1)).sum()); fp = int((a & subset.event.eq(0)).sum())
        fn = int(subset.event.sum()) - tp; tn = len(subset) - tp - fp - fn
        det.append({"period": period, "hours": len(subset), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "recall": tp / (tp + fn), "precision": tp / (tp + fp), "fpr": fp / (fp + tn)})
        for condition, pp in [("all", part), ("up_start", part.loc[part.sustained_onset == 1])]:
            rows.append({"period": period, "condition": condition, "hours": len(pp), **errstats(pp)})
        days = []
        for date, pp in part.groupby("date"):
            if len(pp) == 24:
                peak = pp.loc[pp.actual == pp.actual.max()]
                days.append(errstats(peak))
        rows.append({"period": period, "condition": "daily_peak_day_equal", "units": len(days),
                     **{n: float(np.mean([d[n] for d in days])) for n in ["mae", "under", "over", "bias"]}})
    # Validate mask against Python datetime and routed values against scalar selection.
    for r in g.itertuples():
        selected = r.timestamp.weekday() == 0 and r.timestamp.hour == 8
        assert r.routed == int(selected)
        expected = b1.loc[r.timestamp, "prediction"] if selected else base.set_index("timestamp").loc[r.timestamp, "prediction"]
        assert abs(r.prediction - expected) < 1e-9
    artifacts = {"calendar_control_detection.csv": pd.DataFrame(det), "calendar_control_metrics.csv": pd.DataFrame(rows),
                 "calendar_control_predictions.csv": g}
    for name, frame in artifacts.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    result = {"status": "completed", "contract_sha256": sha(OUT / "calendar_control_contract.json"),
              "script_sha256": sha(__file__), "new_fits": 0, "scalar_routing_check": "passed",
              "outputs_sha256": {n: sha(OUT / n) for n in artifacts}, "scope": c["scope"]}
    (OUT / "calendar_control_run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"detection": det, "metrics": [r for r in rows if r["period"] == "Jul-Aug"]}), flush=True)


if __name__ == "__main__":
    main()
