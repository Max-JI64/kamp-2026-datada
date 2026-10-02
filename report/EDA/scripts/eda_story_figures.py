"""EDA narrative figures: generate only, never read or visually inspect images.

Run after eda_analysis.py. eda_figures.py provides the existing report style.
All plotted data come from report/EDA/tables; original and legacy results are preserved.
"""
# %% Imports and shared typography
import json
import argparse
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import eda_figures as style

plt, read, finish = style.plt, style.read, style.finish
BLUE, ORANGE, TEAL, GRAY = style.BLUE, style.ORANGE, style.TEAL, style.GRAY
SLOTS = ["15분", "30분", "45분", "60분"]


def clean(ax, axis="y"):
    ax.grid(axis=axis, alpha=.15, linewidth=.7)
    ax.set_axisbelow(True)


# %% 1. Whole-day rhythm and month-by-hour heterogeneity
def rhythm():
    table = read("hourly_overview")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, variable, label in zip(axes, ["power", "production"], ["평균 전력", "생산량"]):
        overall = table.query("population == 'all'").sort_values("hour")
        ax.fill_between(overall.hour, overall[f"{variable}_q25"], overall[f"{variable}_q75"],
                        color=BLUE, alpha=.13, label="전체 기록의 25~75백분위")
        for population, color, ls, name in [("all", "#202E39", "-", "전체 평균"),
                                           ("weekday", BLUE, "--", "평일 평균"),
                                           ("weekend", ORANGE, "--", "주말 평균")]:
            part = table.loc[table.population.eq(population)].sort_values("hour")
            ax.plot(part.hour, part[f"{variable}_mean"], color=color, ls=ls, lw=2,
                    marker="o" if population == "all" else None, ms=3, label=name)
        peak = overall.loc[overall[f"{variable}_mean"].idxmax()]
        ax.annotate(f"전체 평균 최고: {int(peak.hour):02d}시 / {peak[f'{variable}_mean']:.2f}",
                    (peak.hour, peak[f"{variable}_mean"]), xytext=(8, 22), textcoords="offset points",
                    fontsize=10, arrowprops=dict(arrowstyle="-", color=GRAY))
        ax.axvspan(11.6, 12.4, color=GRAY, alpha=.08)
        ax.set(ylabel=label, ylim=(0, None))
        ax.margins(y=.22); clean(ax)
    axes[0].legend(ncol=2, loc="upper left", fontsize=9)
    axes[1].set(xticks=range(24), xlim=(-.4, 23.4), xlabel="CSV 기록 시간대")
    finish(fig, "01A_daily_rhythm", "생산량과 전력의 하루 패턴은 어디에서 일치하고 달라지는가",
           "1~8월 정상 241일 · 시각별 전체 241 / 평일 171 / 주말 70시간 · 두 패널의 세로축 단위와 범위는 다름\n"
           "음영은 전체 관측값의 가운데 50% 범위이며 평균의 신뢰구간이 아님 · 12시 위치를 함께 표시")
    monthly = read("monthly_hour_overview")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    for ax, metric, title, cmap in zip(axes, ["power_mean", "production_mean"],
                                      ["평균 전력", "생산량"], ["YlGnBu", "YlOrBr"]):
        values = monthly.pivot(index="month", columns="hour", values=metric)
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, interpolation="nearest")
        peaks = values.to_numpy().argmax(axis=1)
        ax.scatter(peaks, np.arange(8), s=65, facecolors="none", edgecolors="black", lw=1.1)
        ax.set(yticks=range(8), yticklabels=[f"{m}월" for m in range(1, 9)], title=title)
        fig.colorbar(im, ax=ax, pad=.015, label="시간대별 평균")
    axes[-1].set(xticks=range(24), xlabel="CSV 기록 시간대")
    finish(fig, "01B_month_hour_rhythm", "전체 평균 뒤에 있는 월별 하루 패턴",
           "각 칸은 월×시각의 평균 · 원은 해당 월에서 평균이 가장 높은 시각(동률 시 첫 시각)\n"
           "7월은 정상 29일, 나머지는 각 월의 모든 날짜 · 전력과 생산량은 각각의 색 척도를 사용")


# %% 2. Daily distributions with production composition
def calendar(daily, holiday=False):
    key, prefix = ("holiday", "07") if holiday else ("weekend", "02")
    labels = ["비공휴일", "공휴일"] if holiday else ["평일", "주말"]
    groups = [daily.loc[daily[key].eq(flag)] for flag in (False, True)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), sharex=True, sharey=True)
    for ax, part, label in zip(axes, groups, labels):
        points = ax.scatter(part.daily_mean, part.daily_peak, c=part.producing_hours/24,
                            cmap="viridis", vmin=0, vmax=1, s=38, alpha=.75, edgecolors="white", lw=.3)
        ax.plot([0, 225], [0, 225], ls="--", color=GRAY, lw=.8)
        ax.set(title=f"{label} · {len(part)}일", xlabel="일평균 전력", xlim=(0, 225), ylim=(0, 235))
        clean(ax, "both")
    axes[0].set_ylabel("일최대 전력")
    fig.colorbar(points, ax=axes[1], pad=.025, label="그날 생산량이 양수인 시간 비율")
    scope = f"정상 241일 · {labels[0]} {len(groups[0])}일 / {labels[1]} {len(groups[1])}일"
    caveat = "공휴일 비교는 보정 후 비유의, 표본 10일" if holiday else "공휴일 포함 · 생산량 구성의 차이를 함께 표시"
    finish(fig, f"{prefix}A_calendar_joint", f"{labels[0]}·{labels[1]}의 전력 분포와 생산 기록 구성",
           scope+" · 한 점은 하루(겹칠 수 있음) · 점선은 일평균=일최대\n"+caveat)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, metric, title in zip(axes, ["daily_mean", "daily_peak"], ["일평균 전력", "일최대 전력"]):
        for part, label, color in zip(groups, labels, [BLUE, ORANGE]):
            vals = np.sort(part[metric].to_numpy())
            ax.step(np.r_[0, vals], np.r_[0, np.arange(1, len(vals)+1)/len(vals)],
                    where="post", lw=2.2, color=color, label=f"{label} (n={len(vals)})")
        ax.axhline(.5, color=GRAY, ls=":", lw=.9)
        ax.set(title=title, xlabel="전력", xlim=(0, 235), ylim=(0, 1.02))
        ax.yaxis.set_major_formatter(PercentFormatter(1)); clean(ax, "both")
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("해당 전력값 이하인 날짜의 누적 비율")
    finish(fig, f"{prefix}B_calendar_ecdf", f"평균 한 개 대신 {labels[0]}·{labels[1]}의 전체 분포를 비교",
           scope+" · 계단은 관측값의 누적 분포, 가로 점선은 50% 위치\n"+caveat)


# %% 3. Frequency and monthly maximum shown in one coordinate system
def peaks(hourly):
    table, frequency, maxima = read("monthly_hour_overview"), read("hour_frequency_187"), read("monthly_maxima")
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, .025], height_ratios=[3, 1])
    top = fig.add_subplot(grid[0, 0])
    bottom = fig.add_subplot(grid[1, 0], sharex=top)
    axes = [top, bottom]
    color_axis = fig.add_subplot(grid[0, 1])
    values = table.pivot(index="month", columns="hour", values="high_rate")
    im = axes[0].imshow(values, aspect="auto", cmap="YlOrRd", vmin=0,
                       vmax=values.to_numpy().max(), interpolation="nearest", extent=(-.5, 23.5, 8.5, .5))
    axes[0].scatter(maxima.hour, maxima.month, marker="*", s=165, c="#193649", edgecolors="white", lw=.7,
                    label="해당 월의 관측 최대")
    axes[0].set(yticks=range(1, 9), yticklabels=[f"{i}월" for i in range(1, 9)])
    axes[0].legend(loc="upper right", fontsize=9)
    fig.colorbar(im, cax=color_axis, format=PercentFormatter(1), label="시간별 최대 ≥ 187 비율")
    axes[1].plot(frequency.hour, frequency.high_rate, color=ORANGE, marker="o", lw=2)
    axes[1].set(xticks=range(24), xlim=(-.5, 23.5), xlabel="CSV 기록 시간대", ylabel="전체 빈도")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1)); clean(axes[1])
    finish(fig, "03A_month_hour_peaks", "고전력의 빈도와 월최대의 위치를 함께 본다",
           "색: 월×시각에서 최대 ≥ 187인 시간의 비율 / 별: 그 달의 최고값 위치 / 아래: 1~8월 전체 빈도\n"
           "187은 E004의 1~8월 전체 5,832행 기준을 재사용 · 시간 비교는 정상 5,784행 · 공식 한계 아님")
    rates = read("external_band_frequency")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, variable in zip(axes.flat, ["기온", "풍속", "습도", "강수량"]):
        selected = rates.loc[rates.variable.eq(variable)]
        for month, color in zip([6,7,8], [BLUE,ORANGE,TEAL]):
            part = selected.loc[selected.month.eq(month)].sort_values("band")
            x = part.band.to_numpy()+(month-7)*.06
            ax.plot(x,part.high_rate,color=color,lw=1.7,label=f"{month}월")
            ax.scatter(x,part.high_rate,s=part.n*1.1+12,color=color,alpha=.55,
                       edgecolors="white",linewidths=.6)
            for xx,row in zip(x,part.itertuples()):
                ax.annotate(f"n={row.n}",(xx,row.high_rate),xytext=(0,8 if month!=6 else -17),
                            textcoords="offset points",ha="center",fontsize=8,color=color)
        ticks = selected.drop_duplicates("band").sort_values("band")
        ax.set(xticks=ticks.band,xticklabels=ticks.band_label,title=variable,
               xlabel="6~8월 통합 분포로 고정한 구간",ylabel="최대 ≥ 187 비율",ylim=(-.035,.7))
        ax.yaxis.set_major_formatter(PercentFormatter(1)); clean(ax)
    axes[0,0].legend(fontsize=9,loc="upper left")
    finish(fig, "03B_weather_peak_frequency", "7월의 고전력 빈도: 기상 구간별로 6·8월과 비교",
           "생산량 양수인 평일만 사용 · 점 크기와 n은 시간 수 · 기온·풍속·습도는 통합 사분위 구간, 강수는 0/양수\n"
           "각 점은 관측 빈도 · 시간대와 다른 기상 변수의 구성을 아직 맞추지 않은 비교 · 연결선은 구간 비교용")


# %% 4. Full correlation context and the same-sample conditional comparison
def relations():
    overall = read("overall_correlations")
    variables = ["M", "P", "생산량", "기온", "풍속", "습도", "강수량"]
    labels = ["평균 전력", "최대 전력", "생산량", "기온", "풍속", "습도", "강수량"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, method, title in zip(axes, ["pearson", "spearman"], ["Pearson · 값의 선형 관계", "Spearman · 순위의 단조 관계"]):
        matrix = overall.pivot(index="x", columns="y", values=method).loc[variables, variables].to_numpy()
        im = ax.imshow(np.ma.array(matrix, mask=np.triu(np.ones_like(matrix), 1).astype(bool)),
                       cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(7):
            for j in range(i+1):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(matrix[i,j]) > .65 else "#263747")
        ax.set(xticks=range(7), xticklabels=labels, yticks=range(7), yticklabels=labels, title=title)
        ax.tick_params(axis="x", rotation=45)
        # Outline the full predictor-predictor block, including ALL weather pairs.
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((1.5,1.5),5,5,fill=False,edgecolor="#263747",linewidth=1.5))
    fig.colorbar(im, ax=axes[1], shrink=.72, pad=.025, label="공통 상관 색척도 (-1~1)")
    finish(fig, "04A_correlation_context", "독립변수끼리의 관계와 전력과의 관계를 두 상관으로 비교",
           "생산량·기온·풍속·습도·강수량의 모든 쌍을 포함 · 테두리는 독립변수 영역 · 두 패널은 동일한 변수 쌍별 표본\n"
           "전체 정상 5,784시간, 쌍별 결측 제외(n=5,780~5,784) · 관계의 방향·크기이며 인과효과나 변수 제거 기준은 아님")
    points = pd.read_csv(style.TABLES/"conditional_points.csv", encoding="utf-8-sig", float_precision="round_trip")
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    for i, variable in enumerate(["기온", "풍속", "습도", "강수량"]):
        part = points.loc[points.variable.eq(variable)]
        for j, (x, y, title) in enumerate([("x", "y", "원값"), ("x_centered", "y_centered", "조건 내 편차")]):
            ax = axes[i, j]
            hb = ax.hexbin(part[x], part[y], gridsize=36, bins="log", mincnt=1, cmap="YlGnBu", linewidths=0)
            assert int(hb.get_array().sum()) == len(part), "Every plotted point must be counted"
            ax.set(title=f"{variable} · {title} · r={part[x].corr(part[y]):.3f}, ρ={part[x].corr(part[y], method='spearman'):.3f} · n={len(part):,}",
                   xlabel=variable+("의 조건 평균 대비 편차" if j else ""),
                   ylabel="평균 전력"+("의 조건 평균 대비 편차" if j else ""))
            if j:
                ax.axhline(0, c=GRAY, ls=":", lw=.8); ax.axvline(0, c=GRAY, ls=":", lw=.8)
            fig.colorbar(hb, ax=ax, pad=.015, label="칸 안 기록 수 (로그 색척도)")
    finish(fig, "04B_conditional_density", "기상 네 변수 모두: 전력과의 분포 및 조건 안의 관계",
           "양수 생산 기록 · 각 행의 좌우는 정확히 같은 표본 · 월×시각×평일/주말 집단당 5개 이상\n"
           "조건 내 편차는 각 집단 평균 제거 후의 값 · 모든 점을 빈도에 반영 · r=Pearson, ρ=Spearman")


# %% 5. Month-by-lag structure and an observed weekly change
def lag_figures(hourly):
    table = read("lag_correlations").query("month > 0")
    values = table.pivot(index="month", columns="lag_hours", values="pearson")[[1, 24, 168]]
    counts = table.pivot(index="month", columns="lag_hours", values="n")[[1, 24, 168]]
    fig, ax = plt.subplots(figsize=(10, 7.8))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(8):
        for j in range(3):
            r = values.iloc[i,j]
            ax.text(j, i, f"r={r:.3f}\nn={counts.iloc[i,j]:,}", ha="center", va="center",
                    fontsize=10, color="white" if abs(r) > .6 else "#263747")
    ax.set(xticks=range(3), xticklabels=["1시간 전", "24시간 전", "168시간 전"],
           yticks=range(8), yticklabels=[f"{i}월" for i in range(1, 9)])
    fig.colorbar(im, ax=ax, pad=.02, label="현재 평균 전력과의 Pearson 상관")
    finish(fig, "05A_monthly_lag_matrix", "직전 관계는 유지되지만 주간 반복은 월마다 달라진다",
           "정확히 1·24·168시간 전 기록과 연결 · 월은 현재 기록 기준 · 각 칸에 유효 연결 쌍 수 병기\n"
           "시간 오류 구간을 압축하지 않으며 월별 상관은 예측 정확도가 아님")
    series = hourly.set_index(pd.to_datetime(hourly.timestamp)).M.sort_index()
    current = series.loc["2021-08-02":"2021-08-15 23:00"]
    prior = series.reindex(current.index-pd.Timedelta(hours=168)).to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(current.index, current, color=BLUE, lw=1.8, label="현재 평균 전력")
    axes[0].plot(current.index, prior, color=ORANGE, lw=1.5, ls="--", label="정확히 168시간 전 평균 전력")
    axes[0].set(ylabel="전력", ylim=(0, None)); axes[0].legend(loc="upper left", fontsize=9)
    axes[1].plot(current.index, current.to_numpy()-prior, color=TEAL, lw=1.3)
    axes[1].axhline(0, color=GRAY, lw=.8); axes[1].set_ylabel("현재 - 전주\n전력 차이")
    for ax in axes:
        ax.axvspan(pd.Timestamp("2021-08-02"), pd.Timestamp("2021-08-09"), color=GRAY, alpha=.10)
        clean(ax)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    finish(fig, "05B_weekly_change_trace", "전주와 어긋난 실제 기록: 8월 2~15일",
           "E003에서 확인한 변화 구간의 사례 · 회색 구간 8월 2~8일에는 생산량 기록이 모두 0\n"
           "과거 기록을 현재 시각에 맞춰 표시 · 시계열의 관측 비교이며 실제 휴무·설비 상태는 미식별")


# %% 6. All high-hour records, not selected exemplar peaks
def slot_shape():
    data = read("high_slot_records")
    summary = read("high_slot_summary")
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [1, 1.25]}, sharey=True)
    values = data[SLOTS].to_numpy()
    im = axes[0].imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=230, interpolation="nearest")
    yy, xx = np.where(values >= 187)
    axes[0].scatter(xx, yy, s=3, c="#16374B", marker="s")
    centers, labels, offset = [], [], 0
    for row in summary.itertuples():
        centers.append(offset+(row.hours-1)/2)
        labels.append(f"{row.high_slots}개 구간 높음\n{row.hours}시간 ({row.share:.1%})")
        if offset:
            for ax in axes: ax.axhline(offset-.5, color="#263747", lw=1)
        offset += row.hours
    axes[0].set(xticks=range(4), xticklabels=SLOTS, yticks=centers, yticklabels=labels,
                title="시간마다 네 구간의 전력 배열")
    fig.colorbar(im, ax=axes[0], pad=.015, shrink=.65, label="전력 · 작은 점은 187 이상")
    rows = np.arange(len(data))
    axes[1].hlines(rows, data.M, data.P, color=GRAY, lw=.5, alpha=.35)
    axes[1].scatter(data.M, rows, s=9, c=BLUE, label="평균 전력", alpha=.7)
    axes[1].scatter(data.P, rows, s=9, c=ORANGE, label="최대", alpha=.7)
    axes[1].axvline(187, color="#263747", ls="--", lw=1)
    axes[1].set(title="같은 기록의 평균과 최대", xlabel="전력", xlim=(0, 235))
    axes[1].legend(loc="lower left", fontsize=9)
    finish(fig, "06A_high_slot_map", "높은 값은 한 구간에만 나타나는가, 여러 구간에 걸치는가",
           "최대 ≥ 187인 287시간 전부 · 각 행은 한 시간, 높은 구간 수→배열→최대값 순으로 정렬(시간순 아님)\n"
           "같은 행을 양쪽에서 비교 · 평균 <187인 시간 194/287 · 구간 수를 실제 초과 지속시간으로 환산하지 않음")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for count, ax in enumerate(axes.flat, 1):
        part = data.loc[data.high_slots.eq(count), SLOTS]
        ax.plot(range(4), part.to_numpy().T, color=BLUE, alpha=.12, lw=.8)
        ax.plot(range(4), part.median().to_numpy(), color=ORANGE, marker="o", lw=2.5,
                label="구간별 중앙값")
        ax.axhline(187, color=GRAY, ls="--", lw=1)
        ax.set(title=f"187 이상 {count}개 구간 · {len(part)}시간", xticks=range(4), xticklabels=SLOTS,
               ylim=(0, 235), ylabel="전력")
        clean(ax)
    axes[0,0].legend(fontsize=9, loc="lower right")
    finish(fig, "06B_high_slot_profiles", "높은 구간 수에 따라 달라지는 시간 안의 전력 형태",
           "가는 선: 해당 집단의 모든 실제 네 구간 배열 / 굵은 선: 구간별 중앙값 / 점선: 탐색 기준 187\n"
           "중앙값 선은 개별 실제 사례가 아닐 수 있음 · 반복 날짜와 극단값을 모두 유지")


# %% Generate and record files only
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = ["rhythm", "calendar", "peaks", "relations", "lags", "slots", "low_load", "holiday"]
    parser.add_argument("--topics", nargs="+", choices=choices, default=choices,
                        help="Regenerate only affected topic pairs; preserve the other current figures")
    selected = set(parser.parse_args().topics)
    manifest_path = style.FIGURES/"story_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if selected != set(choices) and previous is None:
        raise RuntimeError("Generate all topics before using partial regeneration")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        font = style.setup()
        hourly, daily = read("hourly_records"), read("daily_records")
        if "rhythm" in selected: rhythm()
        if "calendar" in selected: calendar(daily)
        if "peaks" in selected: peaks(hourly)
        if "relations" in selected: relations()
        if "lags" in selected: lag_figures(hourly)
        if "slots" in selected: slot_shape()
        if "low_load" in selected: style.low_load(daily)
        if "holiday" in selected: calendar(daily, holiday=True)
    important = [str(w.message) for w in captured if "Glyph" in str(w.message) or "layout" in str(w.message).lower()]
    assert not important, important
    assert len(style.GENERATED) == 2*len(selected) and all(x["bytes"] > 10000 for x in style.GENERATED)
    for item in style.GENERATED:
        item["sha256"] = hashlib.sha256((style.FIGURES/item['file']).read_bytes()).hexdigest()
    images = {x['file']: x for x in previous['images']} if previous else {}
    if "peaks" in selected:
        images.pop("03B_peak_timeline.png", None)
    images.update({x['file']: x for x in style.GENERATED})
    assert len(images) == 16
    manifest = dict(status="generated_not_visually_reviewed", font=font,
                    images=list(images.values()), image_reading=False, visual_review=False,
                    last_changed_topics=sorted(selected), generated_in_last_run=len(style.GENERATED),
                    generation_warnings=[str(w.message) for w in captured],
                    script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="generated", images=len(style.GENERATED), current_images=16, font=font, visual_review=False,
                         warnings=manifest['generation_warnings']), ensure_ascii=False))


if __name__ == "__main__":
    main()
