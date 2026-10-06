"""Minimum M04 follow-up explaining maximum-hour versus rise errors; no fitting."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from m01_prepare import ROOT, sha
from m02_compare import metric
from m03_verify import rows, verify_metric


def main():
    tables = ROOT / "Modeling/tables/m03/c"
    out = ROOT / "Modeling/tables/m03/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads((tables / "run.json").read_text(encoding="utf-8"))
    v = json.loads((tables / "independent_verification.json").read_text(encoding="utf-8"))
    assert v["status"] == "passed" and v["run_sha256"] == sha(tables / "run.json")
    for name in ["predictions.csv", "paired_predictions.csv"]:
        assert sha(tables / name) == run["outputs_sha256"][name]
    p = pd.read_csv(tables / "predictions.csv", encoding="utf-8-sig", float_precision="round_trip")
    p = p.loc[p.target.eq("target_maximum") & p.group.isin(["A", "B"])].copy()
    p["prior_state"] = p.power_transition.str.split("->").str[0]
    prior, peak = [], []
    for group, g in p.groupby("group"):
        for state, part in g.groupby("prior_state"):
            prior.append({"group": group, "prior_state": state, **metric(part)})
        maximum = g.loc[g.daily_maximum_weight.gt(0)]
        for transition, part in maximum.groupby("power_transition"):
            peak.append({"group": group, "power_transition": transition, **metric(part, part.daily_maximum_weight)})
    pd.DataFrame(prior).to_csv(out / "prior_state_errors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(peak).to_csv(out / "maximum_transition_errors.csv", index=False, encoding="utf-8-sig")
    paired = pd.read_csv(tables / "paired_predictions.csv", encoding="utf-8-sig", float_precision="round_trip")
    rise = paired.loc[paired.target.eq("target_maximum") & paired.before.eq("A") & paired.after.eq("B") & paired.power_transition.eq("low->above26")].copy()
    assert len(rise) == 20 and rise.timestamp.nunique() == 20
    cols = ["timestamp", "date", "split", "actual_before", "prediction_before", "prediction_after", "under_before", "under_after", "delta_abs", "daily_maximum_weight"]
    rise[cols].to_csv(out / "rise_cases.csv", index=False, encoding="utf-8-sig")
    summary = {"status": "completed", "finished": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
               "no_fitting": True, "question": "Why daily maximum underprediction improved while low->above26 did not; examine overlap and observable prior-state bias.",
               "rise_hours": len(rise), "rise_dates": rise.date.nunique(),
               "rise_maximum_hours": int(rise.daily_maximum_weight.gt(0).sum()),
               "rise_maximum_dates": int(rise.loc[rise.daily_maximum_weight.gt(0), "date"].nunique()),
               "rise_maximum_weight": float(rise.daily_maximum_weight.sum()),
               "maximum_total_hours": int(p.loc[p.group.eq("B") & p.daily_maximum_weight.gt(0)].shape[0]),
               "maximum_total_date_weight": float(p.loc[p.group.eq("B"), "daily_maximum_weight"].sum()),
               "rise_by_month": rise.groupby("split").size().to_dict(),
               "rise_B_underpredicted_hours": int(rise.under_after.gt(0).sum()),
               "input_sha256": {name: sha(tables / name) for name in ["predictions.csv", "paired_predictions.csv"]},
               "outputs_sha256": {x.name: sha(x) for x in out.glob("*.csv")}, "script_sha256": sha(Path(__file__))}
    assert summary["maximum_total_date_weight"] == 91
    # Independent direct differences check against saved columns, no model call.
    for r in rise.itertuples():
        assert abs(r.under_after - max(r.actual_before-r.prediction_after, 0)) < 1e-9
        assert abs(r.delta_abs - (abs(r.prediction_after-r.actual_before)-abs(r.prediction_before-r.actual_before))) < 1e-9
    summary["rise_cases_arithmetic_verified"] = True
    csv_predictions = rows(tables / "predictions.csv")
    independent_checks = 0
    for row in rows(out / "prior_state_errors.csv"):
        subset = [r for r in csv_predictions if r["target"] == "target_maximum" and r["group"] == row["group"]
                  and r["power_transition"].split("->")[0] == row["prior_state"]]
        verify_metric(row, subset)
        independent_checks += 1
    for row in rows(out / "maximum_transition_errors.csv"):
        subset = [r for r in csv_predictions if r["target"] == "target_maximum" and r["group"] == row["group"]
                  and r["power_transition"] == row["power_transition"] and float(r["daily_maximum_weight"]) > 0]
        verify_metric(row, subset, "daily_maximum")
        independent_checks += 1
    summary["independently_recomputed_metric_rows"] = independent_checks
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(pd.DataFrame(prior)[["group", "prior_state", "hours", "MAE", "mean_under", "mean_over"]].to_string(index=False), flush=True)
    print(pd.DataFrame(peak)[["group", "power_transition", "hours", "weight_sum", "MAE", "mean_under", "mean_over"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
