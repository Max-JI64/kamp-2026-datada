"""M04: one predeclared observable-state rule and paired error diagnosis; no fitting."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from m01_prepare import ROOT, sha
from m02_compare import metric, state

CONTRACT = ROOT / "Modeling/config/m04_contract.json"
OUT = ROOT / "Modeling/tables/m04"


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes")


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def subsets(g):
    yield "all", "all", g, None
    peak = g.loc[g.daily_maximum_weight.gt(0)]
    yield "daily_maximum", "all", peak, peak.daily_maximum_weight
    yield "profile_reweighted", "all", g, g.profile_weight
    for column in ["split", "prior_state", "power_transition", "diag_production_transition",
                   "train_profile_overlap", "weekend", "hour"]:
        for value, part in g.groupby(column):
            yield column, str(value), part, None
    for column in ["split", "power_transition"]:
        for value, part in peak.groupby(column):
            yield "peak_" + column, str(value), part, part.daily_maximum_weight


def paired(before, after, name):
    fields = ["timestamp", "date", "split", "power_transition", "prior_state",
              "daily_maximum_weight", "train_profile_overlap"]
    q = before[fields + ["actual", "prediction", "absolute_error", "under_amount", "over_amount"]].merge(
        after[["timestamp", "actual", "prediction", "absolute_error", "under_amount", "over_amount"]],
        on="timestamp", suffixes=("_before", "_after"), validate="one_to_one")
    assert np.array_equal(q.actual_before, q.actual_after)
    q["comparison"] = name
    for original, label in [("absolute_error", "abs"), ("under_amount", "under"), ("over_amount", "over")]:
        q["delta_" + label] = q[original + "_after"] - q[original + "_before"]
    return q


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    assert not (OUT / "run.json").exists(), "Preserve completed M04; version a new comparison instead."
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for key in ["parent_run", "parent_predictions", "frame"]:
        assert sha(ROOT / c[key]) == c[key + "_sha256"], key
    parent = json.loads((ROOT / c["parent_run"]).read_text(encoding="utf-8"))
    verified = json.loads((ROOT / "Modeling/tables/m03/c/independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"] == "passed" and verified["run_sha256"] == sha(ROOT / c["parent_run"])
    assert parent["outputs_sha256"]["predictions.csv"] == c["parent_predictions_sha256"]
    m1 = json.loads((ROOT / "Modeling/tables/m01/verification.json").read_text(encoding="utf-8"))
    assert m1["status"] == "passed" and m1["outputs_sha256"]["hourly_frame.csv"] == c["frame_sha256"]
    p = read_csv(ROOT / c["parent_predictions"])
    p = p.loc[p.target.eq(c["target"]) & p.group.isin(["A", "B"])].copy()
    f = read_csv(ROOT / c["frame"])
    features = [x for x in c["origin_input_diagnostics"] if x not in ["hour", "lag1_last_minus_mean", "lag1_last_minus_lag2_last"]]
    f = f[["timestamp", "lag1_timestamp"] + features]
    p = p.merge(f, on="timestamp", validate="many_to_one")
    p["prior_state"] = state(p.lag1_maximum, np.floor(p.lag1_mean + 0.5))
    assert (p.prior_state == p.power_transition.str.split("->").str[0]).all()
    p["lag1_last_minus_mean"] = p.lag1_last - p.lag1_mean
    p["lag1_last_minus_lag2_last"] = p.lag1_last - p.lag2_last
    p["rule_applied"] = False
    a, b = [p.loc[p.group.eq(group)].sort_values("timestamp").copy() for group in ["A", "B"]]
    assert len(b) == 2184 and b.timestamp.nunique() == 2184 and b.date.nunique() == 91
    assert set(b.split) == {"dev_apr", "dev_may", "dev_jun"}
    rule = b.copy()
    rule["group"] = c["rule"]["name"]
    rule["rule_applied"] = rule.prior_state.eq("low")
    rule["prediction"] = np.where(rule.rule_applied, rule.lag1_maximum, rule.prediction)
    baseline = b.copy()
    baseline["group"] = "lag1"
    baseline["prediction"] = baseline.lag1_maximum
    p = pd.concat([a, b, rule, baseline], ignore_index=True)
    p["signed_error"] = p.prediction - p.actual
    p["absolute_error"] = p.signed_error.abs()
    p["under_amount"] = (-p.signed_error).clip(lower=0)
    p["over_amount"] = p.signed_error.clip(lower=0)
    OUT.mkdir(parents=True, exist_ok=True)
    save(p, "predictions.csv")
    metrics = []
    for group, g in p.groupby("group"):
        total_error = g.absolute_error.sum()
        for kind, value, part, weights in subsets(g):
            metrics.append({"group": group, "kind": kind, "value": value,
                            **metric(part, weights),
                            "absolute_error_share": float(part.absolute_error.sum() / total_error)})
    m = pd.DataFrame(metrics)
    save(m, "metrics.csv")
    pairs = pd.concat([paired(a, b, "A_to_B"), paired(b, rule, "B_to_rule")], ignore_index=True)
    save(pairs, "paired_predictions.csv")
    q = pairs.loc[pairs.comparison.eq("B_to_rule")]
    daily = q.groupby(["split", "date"], as_index=False).agg(
        hours=("timestamp", "size"), MAE_B=("absolute_error_before", "mean"),
        MAE_rule=("absolute_error_after", "mean"), delta_MAE=("delta_abs", "mean"),
        delta_under=("delta_under", "mean"), delta_over=("delta_over", "mean"))
    save(daily, "daily_errors.csv")
    low = b.loc[b.prior_state.eq("low")].merge(
        rule[["timestamp", "prediction", "absolute_error", "under_amount", "over_amount"]],
        on="timestamp", suffixes=("_B", "_rule"), validate="one_to_one")
    save(low, "prior_low_cases.csv")
    summary = []
    for transition, part in low.groupby("power_transition"):
        for feature in c["origin_input_diagnostics"]:
            values = part[feature]
            summary.append({"power_transition": transition, "feature": feature, "hours": len(part),
                            "mean": values.mean(), "min": values.min(), "q25": values.quantile(.25),
                            "median": values.median(), "q75": values.quantile(.75), "max": values.max()})
    save(pd.DataFrame(summary), "origin_feature_summary.csv")
    counts = low.groupby(["split", "hour", "power_transition"], as_index=False).size()
    save(counts, "origin_hour_counts.csv")
    # Complete observed B input signatures; posthoc collisions cannot be deployed as a rule.
    low["day_of_week"] = pd.to_datetime(low.timestamp).dt.dayofweek
    signatures = ["month", "day_of_week", "weekend", "hour", "lag1_production", "lag1_mean", "lag1_maximum",
                  "lag2_mean", "lag2_maximum", "lag1_last"]
    low["signature"] = low[signatures].astype(str).agg("|".join, axis=1)
    collisions = low.groupby("signature").agg(hours=("timestamp", "size"), dates=("date", "nunique"),
        actual_min=("actual", "min"), actual_max=("actual", "max"), transitions=("power_transition", "nunique"))
    collisions = collisions.loc[collisions.transitions.gt(1)].reset_index()
    save(collisions, "origin_signature_collisions.csv")
    def cell(group, kind, value="all"):
        r = m.loc[m.group.eq(group) & m.kind.eq(kind) & m.value.eq(value)]
        assert len(r) == 1
        return r.iloc[0]
    tol = c["selection"]["numeric_tolerance"]
    rule_name = c["rule"]["name"]
    gates = []
    for kind, value, field, strict in [("all", "all", "MAE", True),
            ("daily_maximum", "all", "MAE", False), ("daily_maximum", "all", "mean_under", False),
            ("power_transition", "low->above26", "mean_under", False)]:
        ref, candidate = float(cell("B", kind, value)[field]), float(cell(rule_name, kind, value)[field])
        delta = candidate - ref
        gates.append({"kind": kind, "value": value, "metric": field, "B": ref, "rule": candidate,
                      "delta": delta, "strict_decrease": strict, "passed": delta < -tol if strict else delta <= tol})
    accepted = all(g["passed"] for g in gates)
    selection = {"status": "accepted" if accepted else "rejected", "selected": rule_name if accepted else "B",
        "gates": gates, "rule_applied_hours": int(rule.rule_applied.sum()),
        "unchanged_hours": int((~rule.rule_applied).sum()),
        "days_improved": int(daily.delta_MAE.lt(-tol).sum()), "days_worsened": int(daily.delta_MAE.gt(tol).sum()),
        "days_unchanged": int(daily.delta_MAE.abs().le(tol).sum()),
        "origin_signature_collisions": len(collisions),
        "note": "A smaller aggregate error does not override a failed predeclared safety-of-accuracy gate; no operational safety claim."}
    (OUT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    run = {"status": "completed", "finished": now(), "no_fitting": True, "no_july_august_evaluation": True,
        "contract_sha256": sha(CONTRACT), "script_sha256": sha(Path(__file__)),
        "inputs_sha256": {c[key]: sha(ROOT / c[key]) for key in ["parent_run", "parent_predictions", "frame"]},
        "outputs_sha256": {x.name: sha(x) for x in sorted(OUT.iterdir()) if x.suffix in [".csv", ".json"] and x.name != "run.json"},
        "prediction_rows": len(p), "paired_rows": len(pairs), "metric_rows": len(m),
        "unique_evaluation_hours": len(b), "daily_maximum_hours": int(b.daily_maximum_weight.gt(0).sum()),
        "daily_maximum_date_weight": float(b.daily_maximum_weight.sum()),
        "runtime": {"executable": sys.executable, "version": sys.version}}
    (OUT / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False), flush=True)
    print(m.loc[m.kind.isin(["all", "daily_maximum"]) | (m.kind.eq("power_transition") & m.value.isin(["low->low", "low->above26"]))][
        ["group", "kind", "value", "hours", "MAE", "RMSE", "mean_under", "mean_over"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
