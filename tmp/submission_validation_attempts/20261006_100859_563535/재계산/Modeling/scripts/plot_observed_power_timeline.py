"""Observed timeline from verified M01 rows; no forecasts or event thresholds."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Modeling/tables/m01/hourly_frame.csv"
OUT = ROOT / "Modeling/figures"
TABLES = ROOT / "Modeling/tables/observed_timeline"


def main():
    expected = json.loads((SOURCE.parent / "verification.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == expected["outputs_sha256"][SOURCE.name]
    data = pd.read_csv(SOURCE, encoding="utf-8-sig", parse_dates=["timestamp"], float_precision="round_trip")
    assert len(data) == 5784
    data["maximum_delta"] = data.target_maximum - data.lag1_maximum
    data["mean_delta"] = data.target_mean - data.lag1_mean
    zero = data.loc[data.target_maximum == 0]
    assert len(zero) == 17
    assert (zero.timestamp.diff().dropna() == pd.Timedelta(hours=1)).all()
    first, last = zero.timestamp.min(), zero.timestamp.max()
    assert data.maximum_delta.notna().sum() == 5781
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    # Reindex gaps to NaN: excluded dates never become a fabricated connecting line.
    indexed = data.set_index("timestamp")
    timeline = indexed.reindex(pd.date_range(indexed.index.min(), indexed.index.max(), freq="h"))
    figures = []
    for kind, cols, ylabel, filename in (
        ("level", ["target_mean", "target_maximum"], "전력값", "observed_power_monthly_2021_01_08.png"),
        ("delta", ["mean_delta", "maximum_delta"], "직전 시간 대비 차이", "observed_power_delta_monthly_2021_01_08.png"),
    ):
        fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharey=True)
        for month, ax in enumerate(axes.flat, 1):
            part = timeline.loc[timeline.index.month == month]
            for col, color, label in zip(cols, ["#2463a0", "#b55432"],
                                         ["시간 평균", "시간 최대"] if kind == "level" else ["평균 변화", "최대값 변화"]):
                ax.plot(part.index, part[col], linewidth=0.75, color=color, alpha=0.85, label=label)
            if kind == "delta":
                ax.axhline(0, color="#555555", linewidth=0.5)
                ax.set_ylim(-230, 230)
            else:
                ax.set_ylim(-5, 235)
            ax.set_title(f"2021년 {month}월", loc="left", fontsize=11)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(axis="y", alpha=0.2)
            if month % 2 == 1:
                ax.set_ylabel(ylabel)
        axes[0, 0].legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
        fig.tight_layout()
        path = OUT / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    zoom = timeline.loc[first - pd.Timedelta(hours=24):last + pd.Timedelta(hours=24)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, cols, ylabel in zip(axes, [["target_mean", "target_maximum"], ["mean_delta", "maximum_delta"]],
                                ["전력값", "직전 시간 대비 차이"]):
        for col, color, label in zip(cols, ["#2463a0", "#b55432"], ["평균", "최대"]):
            ax.plot(zoom.index, zoom[col], linewidth=1.5, marker=".", markersize=3, color=color, label=label)
        ax.axvspan(first, last + pd.Timedelta(hours=1), color="#999999", alpha=0.15, label="전력 0인 17시간")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    axes[0].set_ylim(-5, 240)
    axes[1].axhline(0, color="#555555", linewidth=0.6)
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H시"))
    axes[1].set_xlabel("관측 시각")
    fig.tight_layout()
    path = OUT / "observed_power_zero_interval.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path)
    cols = ["timestamp", "target_mean", "target_maximum", "lag1_mean", "lag1_maximum", "mean_delta", "maximum_delta"]
    pd.concat([data.nsmallest(5, "maximum_delta").assign(direction="largest_drop"),
               data.nlargest(5, "maximum_delta").assign(direction="largest_rise")])[cols + ["direction"]].to_csv(
                   TABLES / "largest_changes.csv", index=False, encoding="utf-8-sig")
    boundary = data.loc[data.timestamp.between(first - pd.Timedelta(hours=1), last + pd.Timedelta(hours=1)), cols]
    boundary.to_csv(TABLES / "zero_interval.csv", index=False, encoding="utf-8-sig")
    summary = {"source_frame_sha256": digest, "rows": len(data), "exact_hour_changes": 5781,
               "zero_hours": len(zero), "zero_start": str(first), "zero_end": str(last),
               "before_zero": boundary.iloc[0].astype(str).to_dict(), "after_zero": boundary.iloc[-1].astype(str).to_dict(),
               "maximum_drop": float(data.maximum_delta.min()), "maximum_rise": float(data.maximum_delta.max()),
               "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in figures},
               "gaps_not_connected": True, "no_smoothing": True, "no_model_fitting": True,
               "visual_review": "pending"}
    (TABLES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Authorized visual review: contact sheet at exactly 50 percent, at most four images.
    previews = []
    for p in figures:
        with Image.open(p) as img:
            previews.append(img.resize((img.width // 2, img.height // 2)))
    sheet = Image.new("RGB", (max(im.width for im in previews), sum(im.height for im in previews)), "white")
    y = 0
    for im in previews:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(TABLES / "contact_sheet_50.png")
    print(json.dumps({key: summary[key] for key in ("zero_hours", "zero_start", "zero_end", "before_zero", "after_zero", "maximum_drop", "maximum_rise")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
