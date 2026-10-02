"""Generate two PNG candidates per EDA topic from report tables; never open images."""
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
TABLES, FIGURES = REPORT/"tables/eda", REPORT/"figures/eda"
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


# %% 01: Mean vs maximum
def mean_peak(hourly):
    fig, ax = base()
    for period, color, label in zip(["Jan-Jun", "Jul-Aug"], COLORS, ["1~6월", "7~8월"]):
        part = hourly.loc[hourly.period.eq(period)]
        ax.scatter(part.M, part.P, s=13, alpha=.3, c=color, label=f"{label} (n={len(part):,})", edgecolors="none")
    ax.plot([0, 225], [0, 225], color=GRAY, lw=1, label="평균 = 최대")
    ax.axhline(182, color=ORANGE, ls="--", lw=1)
    ax.axvline(182, color=GRAY, ls="--", lw=1)
    ax.set(xlim=(0, 225), ylim=(0, 230), xlabel="시간별 평균 전력", ylabel="시간별 최대 전력")
    ax.legend(loc="lower right", fontsize=9)
    finish(fig, "01A_mean_peak_scatter", "같은 평균 수준에서도 최대 전력은 다르다",
           "1~8월 정상 5,784시간 · 점이 겹칠 수 있음 · 점선 182는 분석 기준이며 공식 한계가 아님")
    t = read("mean_peak_visibility").query("mean_definition == 'M'")
    fig, ax = base()
    hidden, total = t.hidden_hours.to_numpy(), t.high_hours.to_numpy()
    x = np.arange(2)
    ax.bar(x, hidden, color=ORANGE, label="저장 평균 < 182")
    ax.bar(x, total-hidden, bottom=hidden, color=BLUE, label="저장 평균 ≥ 182")
    for i, (h, n) in enumerate(zip(hidden, total)):
        ax.text(i, h/2, f"{h}시간\n{h/n:.1%}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, n+4, f"최대 ≥ 182: {n}시간", ha="center")
    ax.set(xticks=x, xticklabels=["1~6월", "7~8월"], ylabel="시간 수", ylim=(0, 280))
    ax.legend(loc="upper right", fontsize=9)
    finish(fig, "01B_mean_peak_counts", "높은 최대값이 평균 기준에서는 드러나지 않는 시간",
           "각 기간의 최대 ≥ 182인 시간만 비교 · 분모 235 / 220시간 · 예측 모델의 미탐지율이 아님")


# %% 02 and appendix: calendar comparisons
def calendar(daily, holiday=False):
    key = "holiday" if holiday else "weekend"
    labels = ["비공휴일", "공휴일"] if holiday else ["평일", "주말"]
    prefix = "07" if holiday else "02"
    groups = [daily.loc[daily[key].eq(flag)] for flag in [False, True]]
    metrics = ["daily_mean", "daily_peak"]
    titles = ["일평균 전력", "일최대 전력"]
    fig, axes = base(ncols=2, size=(10, 5.4))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [g[metric].mean() for g in groups]
        bars = ax.bar(labels, vals, color=COLORS, width=.55)
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set(title=title, ylabel="날짜별 전력의 평균", ylim=(0, 220))
    scope = f"1~8월 · {labels[0]} {len(groups[0])}일 / {labels[1]} {len(groups[1])}일"
    caveat = "공휴일 비교는 보정 후 비유의 · 차이 없음의 증거는 아님" if holiday else "공휴일 포함 · 생산량 양수 시간 비율: 평일 74.61%, 주말 12.80%"
    finish(fig, f"{prefix}A_calendar_means", f"{labels[0]}과 {labels[1]}의 평균 전력 수준", scope+"\n"+caveat)
    fig, axes = base(ncols=2, size=(10, 5.4))
    rng = np.random.default_rng(20261002)
    for ax, metric, title in zip(axes, metrics, titles):
        values = [g[metric].to_numpy() for g in groups]
        boxes = ax.boxplot(values, patch_artist=True, showfliers=False,
                           medianprops={"color": "black", "linewidth": 1.5})
        for box, color in zip(boxes["boxes"], COLORS):
            box.set_facecolor(color); box.set_alpha(.20)
        for i, (v, color) in enumerate(zip(values, COLORS), 1):
            ax.scatter(i+rng.uniform(-.13, .13, len(v)), v, s=13, alpha=.5, c=color, edgecolors="none")
        ax.set(xticks=[1, 2], xticklabels=labels, title=title, ylabel="전력", ylim=(-5, 240))
    finish(fig, f"{prefix}B_calendar_distribution", f"{labels[0]}·{labels[1]}의 날짜별 분포", scope+
           "\n모든 날짜를 점으로 표시 · 상자는 25~75백분위, 중앙선은 중앙값 · 0 기록 포함")


# %% 03: hourly peak frequency and monthly maxima
def hours():
    data = read("hour_patterns")
    fig, axes = base(nrows=2, size=(10, 6.3))
    for ax, period, color, label in zip(axes, ["Jan-Jun", "Jul-Aug"], COLORS, ["1~6월", "7~8월"]):
        part = data.loc[data.period.eq(period)]
        ax.bar(part.hour, part.high_rate, color=color, width=.8)
        ax.set(xticks=range(0, 24, 2), ylim=(0, .65), ylabel="182 이상 비율", title=label)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("기록 시간대")
    finish(fig, "03A_hourly_frequency", "고전력이 자주 기록된 시간대는 기간에 따라 달랐다",
           "각 시각의 유효 기록이 분모: 1~6월 181 / 7~8월 60시간 · 시간별 최대 ≥ 182")
    maxima = read("monthly_maxima")
    fig, ax = base()
    ax.scatter(maxima.month, maxima.hour, s=110, c=TEAL)
    for row in maxima.itertuples():
        ax.annotate(f"{int(row.power)}", (row.month, row.hour), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set(xticks=range(1, 9), xticklabels=[f"{i}월" for i in range(1, 9)],
           yticks=range(0, 24, 2), ylim=(0, 23), ylabel="월최대가 발생한 기록 시간대")
    ax.grid(axis="y", alpha=.2)
    finish(fig, "03B_monthly_peak_timing", "월최대는 7월을 제외한 일곱 달에서 8시에 발생했다",
           "각 점은 월최대 발생 시간, 숫자는 전력 기록값 · 원본 15분 구간 위치는 결과표에 보존")


# %% 04: correlations and exact same-sample plots
def relations():
    data = read("conditional_correlations").query("population == 'positive'")
    data = data.loc[data.variable.isin(["생산량", "기온"])]
    fig, ax = base()
    for i, row in enumerate(data.itertuples()):
        ax.plot([row.raw_same_sample, row.centered_pearson], [i, i], color=GRAY, lw=3)
        ax.scatter(row.raw_same_sample, i, color=BLUE, s=100, label="원값" if i == 0 else None)
        ax.scatter(row.centered_pearson, i, color=ORANGE, s=100, label="집단 평균 제거 후" if i == 0 else None)
        ax.text(row.raw_same_sample, i+.10, f"{row.raw_same_sample:.3f}", ha="center")
        ax.text(row.centered_pearson, i+.10, f"{row.centered_pearson:.3f}", ha="center")
    ax.axvline(0, lw=1, color=GRAY, ls="--")
    ax.set(yticks=[0, 1], yticklabels=data.variable, ylim=(-.5, 1.5), xlim=(-.15, .60), xlabel="평균 전력과의 Pearson 상관")
    ax.legend(loc="upper right")
    finish(fig, "04A_conditional_correlations", "생산량·기온의 단순 상관은 같은 달력 조건에서 약해졌다",
           "양수 생산 중 동일 3,072시간 · 월×시간×평일/주말, 집단당 5개 이상 · 인과효과 추정 아님")
    points = read("conditional_points")
    fig, axes = base(nrows=2, ncols=2, size=(11, 8))
    for i, variable in enumerate(["생산량", "기온"]):
        part = points.loc[points.variable.eq(variable)]
        for j, (x, y, title) in enumerate([("x", "y", "원값"), ("x_centered", "y_centered", "집단 평균 제거 후")]):
            ax = axes[i, j]
            ax.scatter(part[x], part[y], s=8, alpha=.22, color=COLORS[j], edgecolors="none")
            ax.set(title=f"{variable}: {title}", xlabel=variable+("의 집단 평균 대비 차이" if j else ""),
                   ylabel="평균 전력"+("의 집단 평균 대비 차이" if j else ""))
    finish(fig, "04B_conditional_scatter", "동일 표본에서 원값과 조건 내 편차 비교",
           "양수 생산 3,072시간 · 모든 점 포함, 표본 추출·이상치 삭제 없음 · 선형 관계 외의 효과를 배제하지 않음")


# %% 05: lag persistence
def lag_figures():
    table = read("lag_correlations")
    total = table.loc[table.month.eq(0)]
    fig, ax = base()
    bars = ax.bar(["1시간 전", "24시간 전", "168시간 전"], total.pearson, color=[BLUE, GRAY, TEAL], width=.55)
    ax.bar_label(bars, labels=[f"r={r.pearson:.3f}\nn={r.n:,}" for r in total.itertuples()], padding=4)
    ax.set(ylabel="현재 평균 전력과의 Pearson 상관", ylim=(0, 1.15))
    finish(fig, "05A_lag_correlations", "직전 전력과의 관계가 가장 강하게 나타났다",
           "1~8월 · 정확한 날짜·시간 차이로 연결 · 시차마다 이용 가능한 쌍의 수가 다름 · 예측 성능 아님")
    fig, ax = base()
    for lag, color in [(1, BLUE), (24, GRAY), (168, ORANGE)]:
        part = table.loc[table.month.gt(0) & table.lag_hours.eq(lag)]
        ax.plot(part.month, part.pearson, marker="o", color=color, label=f"{lag}시간 전")
    ax.axhline(0, color=GRAY, lw=.7)
    ax.set(xticks=range(1, 9), xticklabels=[f"{i}월" for i in range(1, 9)], ylim=(-1.05, 1.05),
           ylabel="현재 평균 전력과의 Pearson 상관")
    ax.legend(ncol=3, loc="lower left")
    finish(fig, "05B_monthly_lag", "주간 반복의 강도는 월별로 일정하지 않았다",
           "예측 대상 시각이 속한 월로 집계 · 과거값은 이전 달에서 올 수 있음 · 정확 시차 연결")


# %% 06: statistical low load, not operationally removable load
def low_load(daily):
    fig, ax = base()
    values = [daily.loc[daily.month.eq(m), "q10"] for m in range(1, 9)]
    boxes = ax.boxplot(values, patch_artist=True, showfliers=True,
                       medianprops={"color": ORANGE, "linewidth": 2})
    for box in boxes["boxes"]:
        box.set_facecolor("#DCE7F1")
    ax.set(xticks=range(1, 9), xticklabels=[f"{i}월" for i in range(1, 9)],
           ylabel="하루 96개 전력값의 10백분위 (q10)", ylim=(-5, 115))
    finish(fig, "06A_low_load_monthly", "날짜별 낮은 전력 수준도 월 안에서 넓게 분포했다",
           "1~8월 정상 241일 · q10은 통계적 하위 수준이며 필수 기저부하·절감 가능량이 아님 · 0 포함")
    low = read("low_load_monthly")
    fig, ax = base()
    ax.plot(low.month, low.q10_median, marker="o", color=BLUE, label="날짜별 동일 가중")
    ax.plot(low.month, low.q10_unique_profile_median, marker="s", color=ORANGE, label="월 안 동일 전력 배열을 한 번씩 반영")
    ax.set(xticks=range(1, 9), xticklabels=[f"{i}월" for i in range(1, 9)],
           ylabel="날짜별 q10의 월 중앙값", ylim=(0, 115))
    ax.legend(loc="lower right", fontsize=9)
    finish(fig, "06B_low_load_weighting", "반복 기록의 가중 방식에 따라 저부하 요약값이 달라진다",
           "원본 삭제 없이 두 가중 방식 비교 · 1월 22→84.5, 4월 81.5→22 · 단일 기저값 가정에 주의")


# %% Execute generation only
def main():
    font = setup()
    hourly, daily = read("hourly_records"), read("daily_records")
    mean_peak(hourly)
    calendar(daily)
    hours()
    relations()
    lag_figures()
    low_load(daily)
    calendar(daily, holiday=True)
    assert len(GENERATED) == 14 and all(x["bytes"] > 10000 for x in GENERATED)
    (FIGURES/"manifest.json").write_text(json.dumps(dict(font=font, matplotlib=matplotlib.__version__,
        status="generated_not_visually_reviewed", images=GENERATED), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="generated", count=len(GENERATED), font=font,
                         image_analysis_performed=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
