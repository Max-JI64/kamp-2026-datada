"""Independent scalar verification of the single causal S04 routing follow-up."""
import hashlib
import json
import math
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_routing"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8), (a, b)


def stats(g):
    errors = [float(r.prediction) - float(r.actual) for r in g.itertuples()]
    return {"hours": len(g), "mae": math.fsum(abs(e) for e in errors) / len(g),
            "rmse": math.sqrt(math.fsum(e * e for e in errors) / len(g)),
            "bias": math.fsum(errors) / len(g), "under": math.fsum(max(-e, 0) for e in errors) / len(g),
            "over": math.fsum(max(e, 0) for e in errors) / len(g)}


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["contract_sha256"] == sha(OUT / "contract.json")
    assert c["entrypoint_sha256"] == run["script_sha256"] == sha(ROOT / "Modeling/scripts/regime_route.py")
    for name, digest in c["inputs_sha256"].items():
        assert sha(ROOT / name) == digest
    for name, digest in run["outputs_sha256"].items():
        assert sha(OUT / name) == digest
    parent_verify = json.loads((ROOT / "Modeling/tables/regime_integration/independent_verification.json").read_text(encoding="utf-8"))
    assert parent_verify["status"] == "passed" and parent_verify["run_sha256"] == sha(ROOT / "Modeling/tables/regime_integration/run.json")
    parent = pd.read_csv(ROOT / "Modeling/tables/regime_integration/predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    base = {(r.variant, r.timestamp): r for r in parent.itertuples()}
    scores = pd.read_csv(ROOT / "Modeling/tables/regime_forecast/predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    choices = pd.read_csv(ROOT / "Modeling/tables/regime_forecast/selection.csv", encoding="utf-8-sig")
    alarms = {}
    for month in [5, 6]:
        choice = choices.loc[(choices.direction == "up") & (choices.month == month)].iloc[0]
        assert pd.Timestamp(choice.calibration_latest_confirmed) < pd.Timestamp(2021, month, 1)
        for family in ["ml", "simple"]:
            for r in scores.loc[(scores.direction == "up") & (scores.month == month) & (scores.method == choice["selected_" + family])].itertuples():
                assert not pd.isna(r.alarm) and int(r.alarm) == int(r.score >= r.threshold)
                alarms[(family, r.timestamp)] = int(r.alarm)
    p = pd.read_csv(OUT / "predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    assert len(p) == 8772 and not p.duplicated(["variant", "timestamp"]).any()
    for r in p.itertuples():
        b0, b1, b2 = [base[(v, r.timestamp)] for v in ["B0", "B1", "B2_up"]]
        a, simple = alarms[("ml", r.timestamp)], alarms[("simple", r.timestamp)]
        expected = {"B0": b0.prediction, "B1_all": b1.prediction, "G1": b1.prediction if a else b0.prediction,
                    "G2": b2.prediction if a else b0.prediction,
                    "hour08": b1.prediction if r.timestamp.hour == 8 else b0.prediction,
                    "simple_gate": b1.prediction if simple else b0.prediction}[r.variant]
        routed = {"B0": 0, "B1_all": 1, "G1": a, "G2": a, "hour08": int(r.timestamp.hour == 8), "simple_gate": simple}[r.variant]
        close(r.prediction, expected); close(r.actual, b0.actual)
        assert r.routed == routed and r.alarm_ml == a and r.alarm_simple == simple
    daily = pd.read_csv(OUT / "daily_peak_errors.csv", encoding="utf-8-sig")
    assert len(daily) == 6 * 60
    for r in daily.itertuples():
        own = p.loc[(p.variant == r.variant) & (p.date == r.date)]
        assert len(own) == 24
        peak = own.loc[own.actual == own.actual.max()]
        assert len(peak) == r.ties
        for name, value in stats(peak).items():
            close(getattr(r, name), value)
    metrics = pd.read_csv(OUT / "metrics.csv", encoding="utf-8-sig", dtype={"period": str})
    for r in metrics.itertuples():
        own = p.loc[p.variant == r.variant]
        if r.period in ["5", "6"]:
            own = own.loc[own.month == int(r.period)]
        if r.condition == "up_start":
            own = own.loc[own.sustained_onset == 1]
        elif r.condition == "other_hours":
            own = own.loc[own.sustained_onset != 1]
        if r.condition == "daily_peak_day_equal":
            own = daily.loc[daily.variant == r.variant]
            expected = {n: math.fsum(own[n]) / len(own) for n in ["mae", "bias", "under", "over"]}
            expected["rmse"] = math.sqrt(math.fsum(v * v for v in own.rmse) / len(own))
            expected["hours"] = int(own.ties.sum())
            assert len(own) == r.days
        else:
            expected = stats(own)
        for name, value in expected.items():
            close(getattr(r, name), value)
    for r in pd.read_csv(OUT / "routing_burden.csv", encoding="utf-8-sig").itertuples():
        own = p.loc[p.variant == r.variant]
        assert r.hours == len(own) and r.routed_hours == own.routed.sum()
        assert r.routed_up_starts == len(own.loc[(own.routed == 1) & (own.sustained_onset == 1)])
        assert r.routed_other_hours == r.routed_hours - r.routed_up_starts
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    def get(v, cond, period="May-Jun"):
        return metrics.loc[(metrics.variant == v) & (metrics.condition == cond) & (metrics.period == period)].iloc[0]
    for r in decision["comparisons"]:
        v = r["variant"]; a, b = get(v, "up_start"), get("B0", "up_start")
        gain = 1 - a.mae / b.mae; under = 1 - a.under / b.under
        all_change = get(v, "all").mae / get("B0", "all").mae - 1
        peak_change = get(v, "daily_peak_day_equal").mae / get("B0", "daily_peak_day_equal").mae - 1
        wins = sum(get(v, "up_start", m).mae < get("B0", "up_start", m).mae for m in ["5", "6"])
        for field, expected in [("mae_reduction", gain), ("under_reduction", under), ("overall_mae_change", all_change), ("daily_peak_mae_change", peak_change)]:
            close(r[field], expected)
        assert wins == r["monthly_wins"]
        assert r["development_gate_passed"] == (gain >= .05 and under >= .05 and all_change <= .01 and peak_change <= .01 and wins >= 2)
    # Future prediction changes cannot alter any earlier routing or resulting values.
    checks = 0
    for cut in [pd.Timestamp("2021-05-15"), pd.Timestamp("2021-06-15")]:
        earlier = p.loc[p.timestamp < cut]
        trimmed_parent = {key: value for key, value in base.items() if key[1] < cut}
        trimmed_alarms = {key: value for key, value in alarms.items() if key[1] < cut}
        for r in earlier.loc[earlier.variant == "G1"].itertuples():
            used = "B1" if trimmed_alarms[("ml", r.timestamp)] else "B0"
            close(r.prediction, trimmed_parent[(used, r.timestamp)].prediction)
        checks += 1
    g1 = p.loc[(p.variant == "G1") & (p.sustained_onset == 1)]
    hour = p.loc[(p.variant == "hour08") & (p.sustained_onset == 1)]
    assert g1.timestamp.tolist() == hour.timestamp.tolist() and g1.prediction.tolist() == hour.prediction.tolist()
    cases = p.loc[(p.variant == "G1") & (p.routed == 1)].copy()
    cases["B0_absolute_error"] = abs(cases.b0 - cases.actual)
    cases["G1_absolute_error"] = abs(cases.prediction - cases.actual)
    cases.to_csv(OUT / "verified_routed_cases.csv", index=False, encoding="utf-8-sig")
    result = {"status": "passed", "run_sha256": sha(OUT / "run.json"), "verifier_sha256": sha(__file__),
              "predictions_checked": len(p), "daily_rows_checked": len(daily), "metrics_checked": len(metrics),
              "future_removal_checks": checks, "new_fits": 0, "alarm_origin_past_only": True,
              "calendar_routing_event_predictions_identical": True,
              "extra_outputs_sha256": {"verified_routed_cases.csv": sha(OUT / "verified_routed_cases.csv")}}
    (OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
