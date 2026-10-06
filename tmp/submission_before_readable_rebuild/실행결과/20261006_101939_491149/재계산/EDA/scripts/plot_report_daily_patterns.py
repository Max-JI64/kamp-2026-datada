"""Combine validated daily means and date ranges for the compact EDA manuscript.

Run from the project root with regular CPython 3.13. Existing tables, figures,
and the detailed manuscript are read-only. Review thumbnails go to tmp.
"""
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import eda_figures as style

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "EDA"
DETAIL = BASE / "02_EDA_원고.md"
TARGET = BASE / "figures/report_compact/daily_pattern_distribution.png"
TABLES = [BASE / "tables/daily_pattern_restart.csv",
          BASE / "tables/daily_repetition/hourly_distribution.csv"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validated_tables():
    means = pd.read_csv(TABLES[0], encoding="utf-8-sig")
    ranges = pd.read_csv(TABLES[1], encoding="utf-8-sig")
    assert len(means) == len(ranges) == 48
    for group, days in [("weekday", 171), ("weekend", 70)]:
        m = means.loc[means.population.eq(group)].sort_values("hour")
        r = ranges.loc[ranges.group.eq(group)].sort_values("hour")
        np.testing.assert_array_equal(m.hour, np.arange(24))
        np.testing.assert_array_equal(r.hour, np.arange(24))
        assert m.n.eq(days).all() and r.n.eq(days).all()
        np.testing.assert_allclose(m.power_mean, r["mean"], rtol=1e-12)
        assert (r["min"] <= r.p10).all()
        assert (r.p10 <= r.p90).all() and (r.p90 <= r["max"]).all()
    return means, ranges


def plot_combined(means, ranges):
    """Edit panel geometry, lines, ranges, labels, and legends here."""
    style.setup()
    fig = style.plt.figure(figsize=(11.4, 6.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1])
    production = fig.add_subplot(grid[0, :])
    weekday = fig.add_subplot(grid[1, 0])
    weekend = fig.add_subplot(grid[1, 1], sharey=weekday)
    specs = [("weekday", "평일 (171일)", style.BLUE, "-"),
             ("weekend", "주말 (70일)", style.ORANGE, "--")]
    for group, label, color, line in specs:
        m = means.loc[means.population.eq(group)].sort_values("hour")
        production.plot(m.hour, m.production_mean, color=color, linestyle=line,
                        linewidth=2.2, label=label)
    production.set(ylabel="평균 생산량", ylim=(0, None))
    production.legend(loc="upper left", ncol=2, frameon=False)
    for ax, (group, label, color, line) in zip([weekday, weekend], specs):
        r = ranges.loc[ranges.group.eq(group)].sort_values("hour")
        ax.fill_between(r.hour, r.p10, r.p90, color=color, alpha=0.16,
                        label="날짜별 10~90백분위")
        ax.plot(r.hour, r["mean"], color=color, linestyle=line,
                linewidth=2.2, label="평균")
        ax.set(title=label, ylabel="시간 평균 전력", ylim=(0, 225),
               yticks=[0, 50, 100, 150, 200])
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    for ax in [production, weekday, weekend]:
        ax.set(xlim=(0, 23), xticks=[0, 3, 6, 9, 12, 15, 18, 21, 23],
               xlabel="시간대 (시)")
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    assert fig._suptitle is None and not fig.texts
    assert all(not ax.texts for ax in fig.axes)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(TARGET, dpi=180)
    style.plt.close(fig)
    bad = [str(w.message) for w in caught
           if "Glyph" in str(w.message) or "layout" in str(w.message).lower()]
    assert not bad, bad


def main():
    protected = [DETAIL, *TABLES]
    before = {str(p): sha(p) for p in protected}
    means, ranges = load_validated_tables()
    plot_combined(means, ranges)
    qa = ROOT / "tmp/eda_report_compact"
    qa.mkdir(parents=True, exist_ok=True)
    with Image.open(TARGET) as image:
        image.convert("RGB").resize((image.width // 2, image.height // 2),
                                    Image.Resampling.LANCZOS).save(qa / "daily_50.png")
    assert before == {str(p): sha(p) for p in protected}
    result = dict(status="passed", checked_hour_group_cells=48,
                  mean_series_match=True, input_tables_and_detailed_manuscript_unchanged=True,
                  input_sha256=before, figure=str(TARGET), figure_sha256=sha(TARGET),
                  visual_review=False,
                  display="Production means; separate weekday/weekend power means and P10-P90; no median lines")
    output = BASE / "tables/report_compact"
    output.mkdir(parents=True, exist_ok=True)
    (output / "daily_pattern_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
