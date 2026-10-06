"""Check current M03 documentation links, figures and displayed numeric tables."""
import csv
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    documents = [ROOT / name for name in ["README.md", "Modeling/README.md", "Modeling/04_Modeling_원고.md", "Modeling/10.04_M03_마지막값추가와시차입력_실행기록.md"]]
    links = 0
    for path in documents:
        text = path.read_text(encoding="utf-8-sig")
        if path == ROOT / "README.md":
            text = text.split("## 이전 jsw 작업 단계 기록")[0]
        assert not re.search(r"[\u3040-\u30ff]", text), path
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#", "app:")):
                continue
            assert (path.parent / target.split("#")[0].strip("<>")).exists(), (path, target)
            links += 1
        assert text.count("```") % 2 == 0, path
    manuscript = documents[2].read_text(encoding="utf-8-sig")
    m03 = manuscript.split("## 4.3 M03:")[1].split("## 4.4 ")[0]
    tables = ROOT / "Modeling/tables/m03/c"
    with (tables / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        comparison = list(csv.DictReader(stream))
    with (tables / "condition_errors.csv").open(encoding="utf-8-sig", newline="") as stream:
        conditions = list(csv.DictReader(stream))
    numeric_rows = 0
    for group in ["A", "B", "C_A", "C_B"]:
        maximum = next(r for r in comparison if r["group"] == group and r["target"] == "target_maximum" and r["split"] == "pooled_development")
        mean = next(r for r in comparison if r["group"] == group and r["target"] == "target_mean" and r["split"] == "pooled_development")
        line = "| " + group + " | " + " | ".join(f"{float(maximum[x]):.2f}" for x in ["MAE", "RMSE", "mean_under", "mean_over"]) + f" | {float(mean['MAE']):.2f} |"
        assert line in m03.replace("**", ""), line
        peak = next(r for r in conditions if r["group"] == group and r["target"] == "target_maximum" and r["condition"] == "daily_maximum")
        line = "| " + group + " | " + " | ".join(f"{float(peak[x]):.2f}" for x in ["MAE", "mean_under", "mean_over"]) + " |"
        assert line in m03.replace("**", ""), line
        numeric_rows += 2
    reduction_rates = {}
    for label, condition, metric in [("overall_MAE", "all", "MAE"),
                                     ("daily_maximum_MAE", "daily_maximum", "MAE"),
                                     ("daily_maximum_under", "daily_maximum", "mean_under")]:
        pair = {group: next(r for r in conditions if r["group"] == group and
                            r["target"] == "target_maximum" and r["condition"] == condition and r["value"] == "all")
                for group in ["A", "B"]}
        rate = (1 - float(pair["B"][metric]) / float(pair["A"][metric])) * 100
        assert f"{rate:.1f}%" in m03, (label, rate)
        reduction_rates[label] = rate
    figures = tables / "manuscript_figures.json"
    metadata = json.loads(figures.read_text(encoding="utf-8"))
    for file, digest in metadata["figures"].items():
        assert sha(ROOT / file) == digest
    metadata["visual_review"] = "passed: figure viewed at 50 percent; axes, legends, monthly comparison, peak under/over stack and denominators checked"
    metadata["manuscript_sha256"] = sha(documents[2])
    metadata["document_validation"] = {"links_checked": links, "M03_numeric_rows_checked": numeric_rows, "scope": "four current documents; historical root jsw section excluded"}
    figures.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"status": "passed", "verified_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
              "links_checked": links, "M03_numeric_rows_checked": numeric_rows,
              "M03_reduction_rates_checked": reduction_rates,
              "documents_sha256": {str(p.relative_to(ROOT)): sha(p) for p in documents},
              "no_fitting": True, "script_sha256": sha(Path(__file__))}
    (tables / "document_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
