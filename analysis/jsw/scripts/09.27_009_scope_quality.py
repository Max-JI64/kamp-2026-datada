"""Audit rounding, source redundancy, excluded records and descriptive September."""
import numpy as np
import pandas as pd
from analysis_common import load, events, SLOTS, SOURCE, save, finish

PREFIX = "09.27_009"


def main():
    raw, data = load(include_september=True)
    exact = raw[SLOTS].mean(axis=1)
    diff = raw["평균"] - exact
    assert np.floor(exact + .5).eq(raw["평균"]).all()
    denom = raw[SLOTS].sum(axis=1).to_numpy(float)
    ratio = np.divide(raw["생산량"].to_numpy(float), denom, out=np.full(len(raw), np.nan), where=denom > 0)
    observed = raw["공장인원"].notna().to_numpy()
    # Source decimal precision, not an assertion-driven arbitrary tolerance.
    lexical = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["공장인원"], dtype="string")["공장인원"]
    places = lexical.loc[observed].map(lambda s: len(s.split(".")[1]) if "." in s else 0).to_numpy()
    tolerance = .5 * 10. ** (-places) + 10 * np.finfo(float).eps * np.maximum(1, np.abs(ratio[observed]))
    staff_error = np.abs(raw.loc[observed, "공장인원"].to_numpy() - ratio[observed])
    staff_equal = staff_error <= tolerance
    assert staff_equal.all()
    dates = pd.to_datetime(raw["날짜"].astype(str), format="%Y%m%d")
    assert dates.dt.day.eq(raw["d"]).all() and dates.dt.month.eq(raw["m"]).all()
    assert (dates.dt.dayofweek + 1).eq(raw["day"]).all()
    rounding = {"all_rows": len(raw), "stored_minus_exact_min": diff.min(), "stored_minus_exact_max": diff.max(),
                "nonzero_rounding_rows": int(diff.ne(0).sum()),
                "stored_mean_ge182": int(raw["평균"].ge(182).sum()), "exact_mean_ge182": int(exact.ge(182).sum()),
                "classification_disagreement": int(raw["평균"].ge(182).ne(exact.ge(182)).sum()),
                "staff_nonmissing": int(observed.sum()), "staff_formula_matches": int(staff_equal.sum()),
                "staff_max_absolute_error": float(staff_error.max()),
                "staff_match_rule": "half of source last decimal place plus floating arithmetic epsilon",
                "staff_missing": int(raw["공장인원"].isna().sum()), "zero_denominator": int((denom == 0).sum())}
    save(pd.DataFrame([rounding]), PREFIX, "rounding_redundancy")
    invalid = raw.loc[~raw["시간"].between(0, 23)].copy()
    invalid["peak"] = invalid[SLOTS].max(axis=1)
    normal_july = data.loc[data["month"].eq(7)]
    normal_jan_aug = data.loc[data["period"].ne("Sep")]
    comparison = []
    for name, part in (("invalid_hour", invalid), ("valid_July", normal_july), ("valid_Jan-Aug", normal_jan_aug)):
        for variable in ["peak", "생산량", "기온", "풍속", "습도", "강수량"]:
            x = part[variable].dropna()
            comparison.append({"group": name, "variable": variable, "rows": len(part), "nonmissing": len(x),
                               "mean": x.mean(), "q50": x.quantile(.5), "q95": x.quantile(.95), "zero_fraction": x.eq(0).mean()})
    save(pd.DataFrame(comparison), PREFIX, "excluded_record_distributions")
    excluded_dates = invalid.groupby("날짜").agg(rows=("시간", "size"), high_records=("peak", lambda x: x.ge(182).sum()), max_peak=("peak", "max"), min_reported_hour=("시간", "min"), max_reported_hour=("시간", "max")).reset_index()
    assert invalid.shape[0] == 48
    save(excluded_dates, PREFIX, "invalid_dates")
    september_summary, september_hours = [], []
    train = data.loc[data["period"].eq("Jan-Jun")]
    for q in (.9, .95, .975):
        threshold = float(train["peak"].quantile(q))
        high, eligible, onset, event_table = events(data, threshold)
        for period, part in data.groupby("period"):
            idx = part.index
            es = event_table.loc[event_table["period"].eq(period)]
            september_summary.append({"period": period, "quantile": q, "threshold": threshold,
                                      "hours": len(part), "days": part["date"].nunique(), "high_hours": int(high.loc[idx].sum()),
                                      "eligible": int(eligible.loc[idx].sum()), "onsets": int(onset.loc[idx].sum()),
                                      "onset_rate": onset.loc[idx].sum() / eligible.loc[idx].sum(),
                                      "events_starting": len(es), "multi_hour_fraction": es["hours"].gt(1).mean(),
                                      "median_event_hours": es["hours"].median(), "median_peak": part["peak"].median(),
                                      "q95_peak": part["peak"].quantile(.95), "cross_period_events": int(es["crosses_period"].sum()),
                                      "right_censored": int(es["right_censored"].sum())})
            if q == .95:
                for hour, keep in (("08", part["hour"].eq(8)), ("13", part["hour"].eq(13)), ("other", ~part["hour"].isin([8, 13]))):
                    keys = part.index[keep]
                    september_hours.append({"period": period, "hour_group": hour, "eligible": int(eligible.loc[keys].sum()),
                                            "onsets": int(onset.loc[keys].sum()), "rate": onset.loc[keys].sum() / eligible.loc[keys].sum()})
    save(pd.DataFrame(september_summary), PREFIX, "extended_periods")
    save(pd.DataFrame(september_hours), PREFIX, "extended_hours")
    save(raw.groupby("m")["전기요금(계절)"].agg(["nunique", "min", "max"]).reset_index(), PREFIX, "calendar_tariff_redundancy")
    finish(PREFIX, {"rounding_redundancy": rounding, "excluded_dates": excluded_dates.to_dict("records"),
                    "extended_periods_182": [r for r in september_summary if r["quantile"] == .95],
                    "extended_hours": september_hours})


if __name__ == "__main__":
    main()
