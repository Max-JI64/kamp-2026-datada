"""M03 manuscript figure from independently verified C results, without fitting."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "Modeling/tables/m03/c"
OUT = ROOT / "Modeling/figures/m03"
COLORS = {"A": "#919ba4", "B": "#2463a0", "C_A": "#c69b73", "C_B": "#398a70"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified():
    run = json.loads((TABLES / "run.json").read_text(encoding="utf-8"))
    verification = json.loads((TABLES / "independent_verification.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and verification["status"] == "passed"
    assert verification["run_sha256"] == sha(TABLES / "run.json")
    for name in ["comparison.csv", "condition_errors.csv"]:
        assert sha(TABLES / name) == run["outputs_sha256"][name]
    return [pd.read_csv(TABLES / name, encoding="utf-8-sig").query("target == 'target_maximum'") for name in ["comparison.csv", "condition_errors.csv"]]


def plot_ablation(comparison, conditions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for group, color in COLORS.items():
        part = comparison.loc[comparison.group.eq(group)].set_index("split")
        values = [float(part.loc[s, "MAE"]) for s in ["dev_apr", "dev_may", "dev_jun"]]
        axes[0].plot([4, 5, 6], values, marker="o", color=color, label=group,
                     linestyle="--" if group.startswith("C_") else "-")
    axes[0].set_xticks([4, 5, 6], ["4월", "5월", "6월"])
    axes[0].set_ylim(0, 11)
    axes[0].set_ylabel("최대전력 예측 MAE")
    axes[0].legend(frameon=False, ncol=2)
    peak = conditions.loc[conditions.condition.eq("daily_maximum")].set_index("group").loc[list(COLORS)]
    y = np.arange(4)
    axes[1].bar(y, peak.mean_under, color="#2463a0", label="평균 과소예측량")
    axes[1].bar(y, peak.mean_over, bottom=peak.mean_under, color="#b55432", label="평균 과대예측량")
    for i, (u, o) in enumerate(zip(peak.mean_under, peak.mean_over)):
        axes[1].text(i, u/2, f"{u:.2f}", ha="center", va="center", color="white", fontsize=10)
        axes[1].text(i, u+o+.25, f"{u+o:.2f}", ha="center", fontsize=9)
    axes[1].set_xticks(y, list(COLORS))
    axes[1].set_ylim(0, 15)
    axes[1].set_ylabel("실제 일별 최대 시간의 예측오차")
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "last_slot_ablation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    comparison, conditions = load_verified()
    path = plot_ablation(comparison, conditions)
    preview = ROOT / "tmp/modeling_visual_review/m03_contact_50.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as im:
        im.resize((im.width//2, im.height//2), Image.Resampling.LANCZOS).save(preview)
    result = {"status": "generated", "visual_review": "pending", "no_fitting": True,
              "input_sha256": {name: sha(TABLES/name) for name in ["comparison.csv", "condition_errors.csv"]},
              "figures": {str(path.relative_to(ROOT)): sha(path)}, "script_sha256": sha(Path(__file__))}
    (TABLES / "manuscript_figures.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
