"""One S04 follow-up: preserve B0 outside causal S03 upward alarms, no new fits."""
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "Modeling/tables/regime_integration"
S03 = ROOT / "Modeling/tables/regime_forecast"
OUT = ROOT / "Modeling/tables/regime_routing"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def inputs():
    names = ["Modeling/tables/regime_integration/predictions.csv", "Modeling/tables/regime_integration/run.json",
             "Modeling/tables/regime_integration/independent_verification.json", "Modeling/tables/regime_integration/metrics.csv",
             "Modeling/tables/regime_forecast/predictions.csv", "Modeling/tables/regime_forecast/selection.csv",
             "Modeling/tables/regime_forecast/independent_verification.json"]
    return {n: sha(ROOT / n) for n in names}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "contract.json").exists()
    verified = json.loads((PARENT / "independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"] == "passed" and verified["run_sha256"] == sha(PARENT / "run.json")
    write(OUT / "contract.json", {
        "stage": "S04 one diagnostic follow-up", "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "entrypoint_sha256": sha(__file__), "inputs_sha256": inputs(),
        "reason": "Uniform B2_up reduced up underprediction but worsened global MAE by5.07% and daily peak MAE by4.74%. B1 lowered up-start MAE in both May and June. Restrict supplementary regression use to predicted upward onset, without changing fitted values or alarms.",
        "scope": "May-Jun development only, same1462hours12sustainedup; analysis-selected follow-up, not independent confirmation",
        "variants": {"B0": "unchanged base", "B1_all": "uniform raw-state input model", "G1": "B1 when S03 past-selected up ML alarm, otherwise B0",
                     "G2": "B2_up when identical alarm, otherwise B0", "hour08": "B1 at target hour08, otherwise B0",
                     "simple_gate": "B1 when S03 past-selected simple upward alarm, otherwise B0"},
        "primary": "G1", "no_training": True,
        "alarms": "Frozen S03 selection and 1% calibration FPR thresholds from earlier OOF only. No new boundary, top-k, oracle event mask, threshold tuning or production-origin assertion.",
        "gate": {"up_start_mae_reduction_vs_B0": .05, "up_under_reduction_vs_B0": .05,
                 "overall_mae_worsening_vs_B0_at_most": .01, "daily_peak_mae_worsening_vs_B0_at_most": .01,
                 "monthly_start_improvements_vs_B0_required": 2},
        "control_interpretation": "G1 vs B0 is combined routing/raw-input effect; numeric probability input remains unproven. G1/hour08 have identical values at08sustained event starts, so do not claim event-error improvement beyond calendar routing. Compare all1462hour errors and changed-hour burden too.",
        "stop": "No further same-period routing/model/threshold search after this comparison; retain S03 detection separately from value integration."})
    print(json.dumps({"status": "frozen", "contract_sha256": sha(OUT / "contract.json")}), flush=True)


def stats(g):
    e = g.prediction.to_numpy() - g.actual.to_numpy()
    return {"hours": len(g), "mae": float(np.mean(abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "under": float(np.mean(np.maximum(-e, 0))), "over": float(np.mean(np.maximum(e, 0))), "bias": float(np.mean(e))}


def run():
    assert not (OUT / "run.json").exists()
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    assert c["inputs_sha256"] == inputs() and c["entrypoint_sha256"] == sha(__file__)
    p = pd.read_csv(PARENT / "predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    p = p.loc[p.month >= 5]
    f = p.loc[p.variant == "B0"].copy().set_index("timestamp")
    assert len(f) == 1462
    f["b0"] = f.prediction
    for variant, name in [("B1", "b1"), ("B2_up", "b2")]:
        other = p.loc[p.variant == variant].set_index("timestamp").loc[f.index]
        assert np.allclose(f.actual, other.actual)
        f[name] = other.prediction
    scores = pd.read_csv(S03 / "predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    selection = pd.read_csv(S03 / "selection.csv", encoding="utf-8-sig")
    f["alarm_ml"] = 0; f["alarm_simple"] = 0
    for month in [5, 6]:
        r = selection.loc[(selection.direction == "up") & (selection.month == month)].iloc[0]
        assert r.calibration_available
        for family in ["ml", "simple"]:
            ss = scores.loc[(scores.direction == "up") & (scores.month == month) & (scores.method == r[f"selected_{family}"])].set_index("timestamp")
            own = f.index[f.month == month]
            assert ss.loc[own, "alarm"].notna().all()
            f.loc[own, "alarm_" + family] = ss.loc[own, "alarm"].to_numpy(int)
    frames = []; metrics = []; burden = []; daily_rows = []
    options = {"B0": f.b0, "B1_all": f.b1,
               "G1": np.where(f.alarm_ml == 1, f.b1, f.b0), "G2": np.where(f.alarm_ml == 1, f.b2, f.b0),
               "hour08": np.where(f.index.hour == 8, f.b1, f.b0),
               "simple_gate": np.where(f.alarm_simple == 1, f.b1, f.b0)}
    masks = {"B0": np.zeros(len(f), bool), "B1_all": np.ones(len(f), bool), "G1": f.alarm_ml == 1,
             "G2": f.alarm_ml == 1, "hour08": f.index.hour == 8, "simple_gate": f.alarm_simple == 1}
    for variant, yhat in options.items():
        g = f.copy(); g["variant"] = variant; g["prediction"] = yhat; g["routed"] = np.asarray(masks[variant], int)
        g = g.reset_index(); frames.append(g)
        for period, gg in [("May-Jun", g), ("5", g.loc[g.month == 5]), ("6", g.loc[g.month == 6])]:
            for name, part in [("all", gg), ("up_start", gg.loc[gg.sustained_onset == 1]),
                               ("other_hours", gg.loc[gg.sustained_onset != 1])]:
                metrics.append({"variant": variant, "period": period, "condition": name, **stats(part)})
        burden.append({"variant": variant, "hours": len(g), "routed_hours": int(g.routed.sum()),
                       "routed_up_starts": int(((g.routed == 1) & (g.sustained_onset == 1)).sum()),
                       "routed_other_hours": int(((g.routed == 1) & (g.sustained_onset != 1)).sum())})
        for date, gg in g.groupby("date"):
            if len(gg) == 24:
                peak = gg.loc[gg.actual == gg.actual.max()]
                daily_rows.append({"variant": variant, "date": date, "ties": len(peak), **stats(peak)})
    predictions = pd.concat(frames, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    for variant, g in daily.groupby("variant"):
        metrics.append({"variant": variant, "period": "May-Jun", "condition": "daily_peak_day_equal", "hours": int(g.ties.sum()),
                        "days": len(g), **{n: float(g[n].mean()) for n in ["mae", "under", "over", "bias"]},
                        "rmse": float(np.sqrt(np.mean(g.rmse ** 2)))})
    table = pd.DataFrame(metrics)
    def get(v, cond, period="May-Jun"):
        return table.loc[(table.variant == v) & (table.condition == cond) & (table.period == period)].iloc[0]
    decisions = []
    for variant in ["G1", "G2", "hour08"]:
        a, b = get(variant, "up_start"), get("B0", "up_start")
        gain = 1 - a.mae / b.mae; under_gain = 1 - a.under / b.under
        all_change = get(variant, "all").mae / get("B0", "all").mae - 1
        peak_change = get(variant, "daily_peak_day_equal").mae / get("B0", "daily_peak_day_equal").mae - 1
        wins = sum(get(variant, "up_start", m).mae < get("B0", "up_start", m).mae for m in ["5", "6"])
        decisions.append({"variant": variant, "mae_reduction": gain, "under_reduction": under_gain,
                          "overall_mae_change": all_change, "daily_peak_mae_change": peak_change, "monthly_wins": int(wins),
                          "development_gate_passed": bool(gain >= .05 and under_gain >= .05 and all_change <= .01 and peak_change <= .01 and wins >= 2)})
    outputs = {"predictions.csv": predictions, "metrics.csv": table, "routing_burden.csv": pd.DataFrame(burden),
               "daily_peak_errors.csv": daily}
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    write(OUT / "decision.json", {"primary": "G1", "comparisons": decisions, "stop": c["stop"], "numeric_probability_ablation": "Original B2 variants failed. Routing does not overturn that result."})
    write(OUT / "run.json", {"status": "completed", "script_sha256": sha(__file__), "contract_sha256": sha(OUT / "contract.json"),
                             "inputs_sha256": inputs(), "new_fits": 0,
                             "outputs_sha256": {n: sha(OUT / n) for n in list(outputs) + ["decision.json"]}})
    print(json.dumps({"status": "completed", "decisions": decisions}), flush=True)


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    {"freeze": freeze, "run": run}[sys.argv[1]]()
