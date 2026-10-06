"""Shared plot styling for the manuscript: font, colors, and PNG export."""
# %% Plot settings
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

REPORT = Path(__file__).resolve().parents[1]
TABLES, FIGURES = REPORT/"tables", REPORT/"figures"
BLUE, ORANGE, TEAL, GRAY = "#285B8C", "#D07834", "#25857A", "#7B8794"
COLORS = [BLUE, ORANGE]
GENERATED = []
STYLE_CHECKS = []


def read(name):
    return pd.read_csv(TABLES/f"{name}.csv", encoding="utf-8-sig")


def setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    selected = next((x for x in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"] if x in names), None)
    if selected is None:
        raise RuntimeError("Korean font required: Malgun Gothic / Noto Sans CJK KR / NanumGothic")
    plt.rcParams.update({"font.family": selected, "axes.unicode_minus": False,
                         "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "axes.labelcolor": "#283747",
                         "xtick.color": "#455563", "ytick.color": "#455563"})
    FIGURES.mkdir(parents=True, exist_ok=True)
    return selected


def finish(fig, name, title, note):
    # Report-wide user preference: titles/captions belong in Markdown.
    # Preserve panel titles, axes, legends and data annotations.
    assert fig._suptitle is None and not fig.texts, "No whole-figure title or footer"
    for ax in fig.axes:
        for label in [ax.get_xlabel(), ax.get_ylabel(), ax.get_title()]:
            assert "원자료 기록값" not in label and "원자료 단위" not in label
    fig.tight_layout(pad=1.5)
    path = FIGURES/f"{name}.png"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fig.savefig(path, dpi=180)
    missing = [str(w.message) for w in captured if "Glyph" in str(w.message)]
    plt.close(fig)
    if missing:
        raise RuntimeError(f"Missing font glyphs in {name}: {missing[:3]}")
    GENERATED.append(dict(file=path.name, bytes=path.stat().st_size,
                          title=title, caption=note, visual_review=False))
    STYLE_CHECKS.append(dict(path=str(path), whole_figure_title=False,
                             footer=False, original_value_label=False))


def base(ncols=1, nrows=1, size=(10, 5.4)):
    return plt.subplots(nrows, ncols, figsize=size)


