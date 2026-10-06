"""Manuscript figures from verified M02 tables; no fitting or new selection."""
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
TABLES = ROOT / "Modeling/tables/m02"
OUT = ROOT / "Modeling/figures/m02"
LABELS = {"previous_hour": "직전 시간", "previous_day": "전날", "previous_week": "전주",
          "ridge_00": "Ridge", "elasticnet_04": "Elastic Net", "svr_03": "SVR", "hgb_01": "HGB"}


def load_verified():
    run = json.loads((TABLES / "run.json").read_text(encoding="utf-8"))
    check = json.loads((TABLES / "independent_verification.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and check["status"] == "passed"
    for name in ["comparison.csv", "condition_errors.csv"]:
        assert hashlib.sha256((TABLES / name).read_bytes()).hexdigest() == run["outputs_sha256"][name]
    comparison = pd.read_csv(TABLES / "comparison.csv", encoding="utf-8-sig")
    conditions = pd.read_csv(TABLES / "condition_errors.csv", encoding="utf-8-sig")
    return comparison.loc[comparison.target.eq("target_maximum")], conditions.loc[conditions.target.eq("target_maximum")]


def plot_comparison(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    overall = comparison.loc[comparison.split.eq("pooled_development")].set_index("model").loc[list(LABELS)]
    colors = ["#a7b2ba"] * 6 + ["#2463a0"]
    bars = axes[0].barh(list(LABELS.values()), overall.MAE, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 49)
    axes[0].set_xlabel("최대전력 예측 MAE")
    for bar, value in zip(bars, overall.MAE):
        axes[0].text(value + .5, bar.get_y() + bar.get_height()/2, f"{value:.2f}", va="center", fontsize=9)
    for model, color, marker in [("previous_hour", "#767f86", "o"), ("previous_week", "#b55432", "s"), ("hgb_01", "#2463a0", "o")]:
        values = [float(comparison.loc[comparison.model.eq(model) & comparison.split.eq(split), "MAE"].iloc[0]) for split in ["dev_apr", "dev_may", "dev_jun"]]
        axes[1].plot([4, 5, 6], values, marker=marker, color=color, label=LABELS[model])
        for month, value in zip([4, 5, 6], values):
            axes[1].annotate(f"{value:.2f}", (month, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, color=color)
    axes[1].set_xticks([4, 5, 6], ["4월", "5월", "6월"])
    axes[1].set_ylim(0, 31)
    axes[1].set_ylabel("최대전력 예측 MAE")
    axes[1].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.grid(axis="x" if ax is axes[0] else "y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "model_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_conditions(conditions):
    wanted = [("low->above26", "낮은 구간→26 초과", 20), ("above26->low", "26 초과→낮은 구간", 19),
              ("low->low", "낮은 구간 유지", 625), ("above26->above26", "26 초과 유지", 1510)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(4)
    for model, shift, color in [("previous_hour", -.18, "#a7b2ba"), ("hgb_01", .18, "#2463a0")]:
        values = [float(conditions.loc[conditions.model.eq(model) & conditions.condition.eq("power_transition") & conditions.value.eq(state), "MAE"].iloc[0]) for state, _, _ in wanted]
        bars = axes[0].barh(y + shift, values, height=.34, label=LABELS[model], color=color)
        for bar, value in zip(bars, values):
            axes[0].text(value+.8, bar.get_y()+bar.get_height()/2, f"{value:.2f}", va="center", fontsize=9)
    axes[0].set_yticks(y, [f"{label} ({n}시간)" for _, label, n in wanted])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 89)
    axes[0].set_xlabel("최대전력 예측 MAE")
    axes[0].legend(frameon=False, loc="lower right")
    peak = conditions.loc[conditions.condition.eq("daily_maximum")].set_index("model")
    models = ["previous_hour", "previous_week", "hgb_01"]
    under = peak.loc[models, "mean_under"].to_numpy()
    over = peak.loc[models, "mean_over"].to_numpy()
    axes[1].bar(np.arange(3), under, label="평균 과소예측량", color="#2463a0")
    axes[1].bar(np.arange(3), over, bottom=under, label="평균 과대예측량", color="#b55432")
    for i, (u, o) in enumerate(zip(under, over)):
        axes[1].text(i, u+o+.8, f"{u+o:.2f}", ha="center", fontsize=9)
    axes[1].set_xticks(np.arange(3), [LABELS[m] for m in models])
    axes[1].set_ylabel("실제 일별 최대 시간의 예측오차")
    axes[1].set_ylim(0, 45)
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    for ax in axes:
        ax.grid(axis="x" if ax is axes[0] else "y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "condition_errors.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    comparison, conditions = load_verified()
    paths = [plot_comparison(comparison), plot_conditions(conditions)]
    qa = ROOT / "tmp/modeling_visual_review"
    qa.mkdir(parents=True, exist_ok=True)
    previews = []
    for path in paths:
        with Image.open(path) as im:
            previews.append(im.convert("RGB").resize((im.width//2, im.height//2), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (max(im.width for im in previews), sum(im.height for im in previews)), "white")
    y = 0
    for im in previews:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(qa / "m02_contact_50.png")
    result = {"status": "generated", "visual_review": "pending", "input_sha256": {name: hashlib.sha256((TABLES/name).read_bytes()).hexdigest() for name in ["comparison.csv", "condition_errors.csv"]},
              "figures": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, "no_fitting": True}
    (TABLES / "manuscript_figures.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
