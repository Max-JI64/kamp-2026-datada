"""Independent CSV/raw-time/metric/selection checks for M04; no model fitting."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from m03_verify import close, rows, verify_metric

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/m04"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(maximum, mean):
    if maximum == 0:
        return "zero"
    rounded = math.floor(mean + .5)
    return "below20" if rounded < 20 else "low" if rounded <= 26 else "above26"


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    cpath = ROOT / "Modeling/config/m04_contract.json"
    c = json.loads(cpath.read_text(encoding="utf-8"))
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["no_fitting"] and run["no_july_august_evaluation"]
    assert sha(cpath) == run["contract_sha256"]
    assert sha(ROOT / "Modeling/scripts/m04_compare.py") == run["script_sha256"]
    for path, expected in run["inputs_sha256"].items():
        assert sha(ROOT / path) == expected
    for name, expected in run["outputs_sha256"].items():
        assert sha(OUT / name) == expected
    m1 = json.loads((ROOT / "Modeling/config/m01_contract.json").read_text(encoding="utf-8"))
    source = ROOT / m1["source"]
    assert sha(source) == m1["source_sha256"]
    raw = {}
    for r in rows(source):
        if not m1["period"][0] <= int(r["날짜"]) <= m1["period"][1] or not 0 <= int(r["시간"]) <= 23:
            continue
        t = datetime.strptime(r["날짜"], "%Y%m%d") + timedelta(hours=int(r["시간"]))
        assert t not in raw
        values = [float(r[name]) for name in ["15분", "30분", "45분", "60분"]]
        raw[t] = {"mean": sum(values) / 4, "maximum": max(values), "last": values[-1], "production": float(r["생산량"])}
    parent = {(r["group"], r["timestamp"]): r for r in rows(ROOT / c["parent_predictions"])
              if r["target"] == c["target"] and r["group"] in ["A", "B"]}
    pred = rows(OUT / "predictions.csv")
    groups = defaultdict(list)
    index = {}
    rule_name = c["rule"]["name"]
    changed = 0
    origin_values_checked = 0
    for r in pred:
        group, timestamp = r["group"], r["timestamp"]
        assert (group, timestamp) not in index
        index[group, timestamp] = r
        groups[group].append(r)
        t = datetime.fromisoformat(timestamp)
        assert t.month in [4, 5, 6] and r["split"] == {4: "dev_apr", 5: "dev_may", 6: "dev_jun"}[t.month]
        current, previous, previous2 = raw[t], raw[t - timedelta(hours=1)], raw[t - timedelta(hours=2)]
        close(r["actual"], current["maximum"])
        assert r["lag1_timestamp"] == str(t - timedelta(hours=1))
        for lag, record in [(1, previous), (2, previous2)]:
            for name in ["mean", "maximum", "last"]:
                close(r[f"lag{lag}_{name}"], record[name])
                origin_values_checked += 1
        close(r["lag1_production"], previous["production"])
        close(r["lag1_last_minus_mean"], previous["last"] - previous["mean"])
        close(r["lag1_last_minus_lag2_last"], previous["last"] - previous2["last"])
        prior = state(previous["maximum"], previous["mean"])
        assert r["prior_state"] == prior
        # Target state's historical diagnostic definition uses the supplied rounded mean.
        target_raw = math.floor(current["mean"] + .5)
        assert r["power_transition"] == prior + "->" + state(current["maximum"], target_raw)
        parent_group = group if group in ["A", "B"] else "B"
        original = parent[parent_group, timestamp]
        for field in ["date", "split", "power_transition", "diag_production_transition", "train_profile_overlap", "diag_target_profile"]:
            assert r[field] == original[field]
        for field in ["daily_maximum_weight", "profile_weight"]:
            close(r[field], original[field])
        applied = group == rule_name and prior == "low"
        assert (r["rule_applied"] == "True") == applied
        prediction = previous["maximum"] if group == "lag1" or applied else float(original["prediction"])
        close(r["prediction"], prediction)
        if applied:
            changed += 1
        error = prediction - current["maximum"]
        for field, value in [("signed_error", error), ("absolute_error", abs(error)),
                             ("under_amount", max(-error, 0)), ("over_amount", max(error, 0))]:
            close(r[field], value)
    assert set(groups) == {"A", "B", "lag1", rule_name}
    times = {r["timestamp"] for r in groups["B"]}
    assert len(times) == 2184 and all({r["timestamp"] for r in g} == times for g in groups.values())
    dates = defaultdict(list)
    for r in groups["B"]:
        dates[r["date"]].append(r)
    assert len(dates) == 91
    peak_hours = 0
    for date, g in dates.items():
        assert len(g) == 24
        maximum = max(float(r["actual"]) for r in g)
        ties = sum(float(r["actual"]) == maximum for r in g)
        for r in g:
            expected = 1 / ties if float(r["actual"]) == maximum else 0
            close(r["daily_maximum_weight"], expected)
            peak_hours += expected > 0
    assert peak_hours == 137 and changed == 650
    metric_rows = rows(OUT / "metrics.csv")
    for r in metric_rows:
        g = groups[r["group"]]
        kind, value = r["kind"], r["value"]
        weight_kind = "all"
        if kind in ["daily_maximum", "profile_reweighted"]:
            weight_kind = kind
            subset = [x for x in g if float(x["daily_maximum_weight"]) > 0] if kind == "daily_maximum" else g
        elif kind.startswith("peak_"):
            weight_kind = "daily_maximum"
            subset = [x for x in g if float(x["daily_maximum_weight"]) > 0 and x[kind[5:]] == value]
        else:
            subset = g if kind == "all" else [x for x in g if x[kind] == value]
        verify_metric(r, subset, weight_kind)
        close(r["absolute_error_share"], math.fsum(float(x["absolute_error"]) for x in subset) /
              math.fsum(float(x["absolute_error"]) for x in g))
    pairs = rows(OUT / "paired_predictions.csv")
    assert len(pairs) == 4368
    for r in pairs:
        before, after = ("A", "B") if r["comparison"] == "A_to_B" else ("B", rule_name)
        for side, group in [("before", before), ("after", after)]:
            for field in ["actual", "prediction", "absolute_error", "under_amount", "over_amount"]:
                close(r[field + "_" + side], index[group, r["timestamp"]][field])
        for field, short in [("absolute_error", "abs"), ("under_amount", "under"), ("over_amount", "over")]:
            close(r["delta_" + short], float(r[field + "_after"]) - float(r[field + "_before"]))
    daily = rows(OUT / "daily_errors.csv")
    assert len(daily) == 91
    for r in daily:
        subset = [x for x in pairs if x["comparison"] == "B_to_rule" and x["date"] == r["date"]]
        assert len(subset) == int(r["hours"]) == 24
        for field, source_field in [("MAE_B", "absolute_error_before"), ("MAE_rule", "absolute_error_after"),
                ("delta_MAE", "delta_abs"), ("delta_under", "delta_under"), ("delta_over", "delta_over")]:
            close(r[field], math.fsum(float(x[source_field]) for x in subset) / 24)
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    tol = c["selection"]["numeric_tolerance"]
    def cell(group, kind, value):
        result = [r for r in metric_rows if r["group"] == group and r["kind"] == kind and r["value"] == value]
        assert len(result) == 1
        return result[0]
    expected_gates = [("all", "all", "MAE", True), ("daily_maximum", "all", "MAE", False),
                      ("daily_maximum", "all", "mean_under", False), ("power_transition", "low->above26", "mean_under", False)]
    assert len(selection["gates"]) == len(expected_gates)
    for gate, expected in zip(selection["gates"], expected_gates):
        kind, value, name, strict = expected
        assert (gate["kind"], gate["value"], gate["metric"], gate["strict_decrease"]) == expected
        reference, candidate = float(cell("B", kind, value)[name]), float(cell(rule_name, kind, value)[name])
        close(gate["B"], reference)
        close(gate["rule"], candidate)
        close(gate["delta"], candidate - reference)
        assert gate["passed"] == (candidate - reference < -tol if strict else candidate - reference <= tol)
    accepted = all(g["passed"] for g in selection["gates"])
    assert selection["status"] == ("accepted" if accepted else "rejected")
    assert selection["selected"] == (rule_name if accepted else "B")
    assert selection["rule_applied_hours"] == changed
    low = rows(OUT / "prior_low_cases.csv")
    assert len(low) == 650 and len({r["timestamp"] for r in low}) == 650
    for r in low:
        assert index["B", r["timestamp"]]["prior_state"] == "low"
        for group, suffix in [("B", "B"), (rule_name, "rule")]:
            for field in ["prediction", "absolute_error", "under_amount", "over_amount"]:
                close(r[field + "_" + suffix], index[group, r["timestamp"]][field])
    for r in rows(OUT / "origin_feature_summary.csv"):
        values = sorted(float(x[r["feature"]]) for x in low if x["power_transition"] == r["power_transition"])
        assert len(values) == int(r["hours"])
        close(r["mean"], math.fsum(values) / len(values))
        close(r["min"], values[0])
        close(r["max"], values[-1])
        for field, fraction in [("q25", .25), ("median", .5), ("q75", .75)]:
            pos = (len(values) - 1) * fraction
            lo, hi = math.floor(pos), math.ceil(pos)
            close(r[field], values[lo] + (values[hi] - values[lo]) * (pos - lo))
    for r in rows(OUT / "origin_hour_counts.csv"):
        assert int(r["size"]) == sum(x["split"] == r["split"] and x["hour"] == r["hour"] and
                                    x["power_transition"] == r["power_transition"] for x in low)
    signatures = defaultdict(list)
    for r in low:
        t = datetime.fromisoformat(r["timestamp"])
        key = (t.month, t.weekday(), t.hour) + tuple(float(r[x]) for x in [
            "lag1_production", "lag1_mean", "lag1_maximum", "lag2_mean", "lag2_maximum", "lag1_last"])
        signatures[key].append(r)
    conflicting = [g for g in signatures.values() if len({x["power_transition"] for x in g}) > 1]
    collision_rows = rows(OUT / "origin_signature_collisions.csv")
    assert len(conflicting) == len(collision_rows) == selection["origin_signature_collisions"]
    actual_collisions = sorted((len(g), len({x["date"] for x in g}), min(float(x["actual"]) for x in g),
        max(float(x["actual"]) for x in g), len({x["power_transition"] for x in g})) for g in conflicting)
    saved_collisions = sorted((int(r["hours"]), int(r["dates"]), float(r["actual_min"]),
        float(r["actual_max"]), int(r["transitions"])) for r in collision_rows)
    assert actual_collisions == saved_collisions
    assert selection["days_improved"] == sum(float(r["delta_MAE"]) < -tol for r in daily)
    assert selection["days_worsened"] == sum(float(r["delta_MAE"]) > tol for r in daily)
    assert selection["days_unchanged"] == sum(abs(float(r["delta_MAE"])) <= tol for r in daily)
    output = {"status": "passed", "checked_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "run_sha256": sha(OUT / "run.json"), "script_sha256": sha(Path(__file__)),
        "raw_origin_values_checked": origin_values_checked, "prediction_rows": len(pred),
        "metric_rows": len(metric_rows), "paired_rows": len(pairs), "daily_rows": len(daily),
        "rule_applied_hours": changed, "identical_evaluation_hours": len(times),
        "daily_maximum_hours": peak_hours, "daily_maximum_dates": len(dates),
        "origin_signature_collisions_checked": len(conflicting),
        "selection_checked": True, "no_fitting": True, "no_july_august_evaluation": True}
    (OUT / "independent_verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


def documents():
    verified = json.loads((OUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"] == "passed" and verified["run_sha256"] == sha(OUT / "run.json")
    assert verified["script_sha256"] == sha(Path(__file__))
    manuscript_path = ROOT / "Modeling/04_Modeling_원고.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    section = manuscript.split("## 4.6 M04:")[1]
    metrics = rows(OUT / "metrics.csv")
    checks = [("전체 MAE", "all", "all", "MAE"), ("전체 RMSE", "all", "all", "RMSE"),
        ("실제 일별 최대 시간 MAE", "daily_maximum", "all", "MAE"),
        ("실제 일별 최대 시간 평균 과소예측", "daily_maximum", "all", "mean_under"),
        ("낮은 구간 유지 MAE", "power_transition", "low->low", "MAE"),
        ("상승 전환 평균 과소예측", "power_transition", "low->above26", "mean_under"),
        ("낮은 유지 중 일별 최대 시간 MAE", "peak_power_transition", "low->low", "MAE"),
        ("전체 평균 과소예측", "all", "all", "mean_under")]
    for label, kind, value, field in checks:
        numbers = []
        for group in ["B", "B_prior_low_persistence"]:
            found = [r for r in metrics if (r["group"], r["kind"], r["value"]) == (group, kind, value)]
            assert len(found) == 1
            numbers.append(f"{float(found[0][field]):.3f}")
        assert f"| {label} | {numbers[0]} | {numbers[1]} |" in section, label
    contribution_checks = [("26초과 상태 유지", "above26->above26", 1),
        ("낮은 구간 유지", "low->low", 1), ("낮은 구간→26초과", "low->above26", 1),
        ("26초과→낮은 구간", "above26->low", 1), ("20미만→낮은 구간", "below20->low", 2),
        ("낮은 구간→20미만", "low->below20", 2)]
    for label, value, places in contribution_checks:
        r = next(r for r in metrics if (r["group"], r["kind"], r["value"]) == ("B", "power_transition", value))
        expected = f'| {label} | {int(r["hours"]):,} | {float(r["MAE"]):.3f} | {100 * float(r["absolute_error_share"]):.{places}f}% |'
        assert expected in section, expected
    docs = [manuscript_path, ROOT / "Modeling/README.md", ROOT / "README.md",
            ROOT / "Modeling/10.05_M04_조건별오류와낮은전력규칙_실행기록.md"]
    links = 0
    for path in docs:
        text = path.read_text(encoding="utf-8")
        if path == ROOT / "README.md":
            text = text.split("## 이전 jsw 작업 단계 기록")[0]
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if target.startswith(("http:", "https:", "#", "app:")):
                continue
            linked = (path.parent / target.split("#")[0].strip("<>")).resolve()
            assert linked.exists(), (path, target)
            links += 1
    result = {"status": "passed", "checked_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "numeric_rows": len(checks) + len(contribution_checks), "local_links": links, "manuscript_sha256": sha(manuscript_path),
        "script_sha256": sha(Path(__file__))}
    (OUT / "document_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", action="store_true")
    args = parser.parse_args()
    documents() if args.documents else main()
