"""Restart EDA 2.2: distribution/method diagnostics, then observational relations.

Supplied Jan-Aug valid-hour records only. No image reading or model evaluation.
Run --diagnose before --figures. All hours and extreme values remain in outputs.
"""
# %% Fixed scope and diagnostics chosen before examining new results
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import eda_figures as style

REPORT=Path(__file__).resolve().parents[1]
ROOT=REPORT.parent.parent
OUT=REPORT/'tables'
SOURCE=ROOT/'data/origin/okm_augumented_2021.csv'
SHA='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
INPUTS=['생산량','기온','풍속','습도','강수량']
VARIABLES=['평균 전력',*INPUTS]


def save(table,name):
    table.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')


def source_frame():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    raw=pd.read_csv(SOURCE,encoding='utf-8-sig',usecols=['날짜','시간','평균',*INPUTS])
    data=raw.loc[raw['날짜'].lt(20210901) & raw['시간'].between(0,23)].copy()
    assert len(data)==5784 and data['날짜'].nunique()==241
    data=data.rename(columns={'평균':'평균 전력'})
    dates=pd.to_datetime(data['날짜'].astype(str),format='%Y%m%d')
    data['month']=dates.dt.month
    data['weekend']=dates.dt.dayofweek.ge(5)
    return data


def diagnose(data):
    distributions=[]
    for variable in VARIABLES:
        values=data[variable].dropna()
        normal=stats.normaltest(values)
        distributions.append(dict(variable=variable,n=len(values),missing=len(data)-len(values),
                                  zero_n=int(values.eq(0).sum()),zero_share=values.eq(0).mean(),
                                  unique=values.nunique(),mode_share=values.value_counts().max()/len(values),
                                  skew=values.skew(),excess_kurtosis=values.kurt(),
                                  min=values.min(),q25=values.quantile(.25),median=values.median(),
                                  q75=values.quantile(.75),q99=values.quantile(.99),max=values.max(),
                                  normaltest_statistic=normal.statistic,normaltest_naive_iid_p=normal.pvalue))
    save(pd.DataFrame(distributions),'restart_relations_distribution')
    summaries=[]
    for variable in INPUTS:
        part=data[[variable,'평균 전력']].dropna()
        # A predeclared sensitivity ONLY, never a cleaned analysis population.
        cutoff=part[variable].quantile(.99)
        inner=part.loc[part[variable].le(cutoff)]
        summaries.append(dict(variable=variable,n=len(part),x_p99=cutoff,
                              diagnostic_inner_n=len(inner),diagnostic_omitted_n=len(part)-len(inner),
                              pearson_all=part[variable].corr(part['평균 전력']),
                              spearman_all=part[variable].corr(part['평균 전력'],method='spearman'),
                              pearson_below_x_p99=inner[variable].corr(inner['평균 전력']),
                              spearman_below_x_p99=inner[variable].corr(inner['평균 전력'],method='spearman')))
    save(pd.DataFrame(summaries),'restart_relations_sensitivity')
    # Bin summaries diagnose shape numerically; these are not regression curves.
    bins=[]
    for variable in INPUTS:
        part=data[[variable,'평균 전력']].dropna().copy()
        if variable in ['생산량','강수량']:
            part['band']='0'
            positive=part[variable].gt(0)
            part.loc[positive,'band']=pd.qcut(part.loc[positive,variable],q=5,duplicates='drop').astype(str)
        else:
            part['band']=pd.qcut(part[variable],q=5,duplicates='drop').astype(str)
        for band,cell in part.groupby('band'):
            bins.append(dict(variable=variable,band=band,n=len(cell),
                             x_min=cell[variable].min(),x_max=cell[variable].max(),
                             x_median=cell[variable].median(),power_mean=cell['평균 전력'].mean(),
                             power_median=cell['평균 전력'].median()))
    save(pd.DataFrame(bins).sort_values(['variable','x_min']),'restart_relations_bins')
    print(pd.DataFrame(distributions)[['variable','n','zero_share','skew','excess_kurtosis','normaltest_naive_iid_p']].to_json(orient='records',force_ascii=True))
    print(pd.DataFrame(summaries).to_json(orient='records',force_ascii=True))


def relations(data):
    # The main estimand is ordinal co-movement, not proportional/linear effect.
    # This choice is supported by zero masses, skew and extreme-value sensitivity,
    # not by an automatic normality-test switch. Conventional p-values are unused.
    rows=[]
    for x in VARIABLES:
        for y in VARIABLES:
            part=data[[x,y]].dropna() if x!=y else data[[x]].dropna()
            coefficient=part[x].corr(part[y],method='spearman') if x!=y else 1.
            rows.append(dict(x=x,y=y,n=len(part),spearman=coefficient))
    overall=pd.DataFrame(rows)
    save(overall,'restart_relations_spearman')
    monthly=[]
    for variable in INPUTS:
        for month,cell in data.groupby('month'):
            part=cell[[variable,'평균 전력']].dropna()
            monthly.append(dict(variable=variable,month=month,n=len(part),
                                days=cell['날짜'].nunique(),spearman=part[variable].corr(part['평균 전력'],method='spearman')))
    monthly=pd.DataFrame(monthly)
    save(monthly,'restart_relations_monthly')
    # Existing evidence audit and independent average-tie rank correlation.
    old=pd.read_csv(OUT/'overall_correlations.csv',encoding='utf-8-sig',float_precision='round_trip')
    for row in overall.itertuples():
        x='M' if row.x=='평균 전력' else row.x
        y='M' if row.y=='평균 전력' else row.y
        reference=old.loc[old.x.eq(x)&old.y.eq(y)].iloc[0]
        assert row.n==reference.n
        np.testing.assert_allclose(row.spearman,reference.spearman,atol=1e-12)
        if x!=y:
            part=data[[row.x,row.y]].dropna()
            independent=np.corrcoef(stats.rankdata(part[row.x]),stats.rankdata(part[row.y]))[0,1]
            np.testing.assert_allclose(row.spearman,independent,atol=1e-12)
    old=pd.read_csv(ROOT/'EDA/jsw/tables/09.23_007_monthly_correlations.csv',encoding='utf-8-sig')
    old=old.loc[old.population.eq('all'),['variable','month','n','days','spearman']]
    pd.testing.assert_frame_equal(monthly.sort_values(['variable','month']).reset_index(drop=True),
                                  old.sort_values(['variable','month']).reset_index(drop=True),
                                  check_dtype=False,rtol=1e-10,atol=1e-12)
    # Follow the observed month-dependent temperature sign with the E007 August
    # production-state example. Same method; no centered residuals or new model.
    august=data.loc[data.month.eq(8)]
    contexts=[]
    for population,part in [('all',august),('positive',august.loc[august['생산량'].gt(0)]),
                            ('zero',august.loc[august['생산량'].eq(0)])]:
        pair=part[['기온','평균 전력']].dropna()
        coefficient=pair['기온'].corr(pair['평균 전력'],method='spearman')
        independent=np.corrcoef(stats.rankdata(pair['기온']),stats.rankdata(pair['평균 전력']))[0,1]
        np.testing.assert_allclose(coefficient,independent,atol=1e-12)
        if population in ['all','positive']:
            reference=pd.read_csv(ROOT/'EDA/jsw/tables/09.23_007_monthly_correlations.csv',encoding='utf-8-sig')
            row=reference.loc[reference.population.eq(population)&reference.variable.eq('기온')&reference.month.eq(8)].iloc[0]
            assert row.n==len(pair)
            np.testing.assert_allclose(coefficient,row.spearman,atol=1e-12)
        contexts.append(dict(population=population,n=len(pair),days=part['날짜'].nunique(),
                             spearman=coefficient,power_mean=pair['평균 전력'].mean(),temperature_mean=pair['기온'].mean()))
    save(pd.DataFrame(contexts),'restart_relations_august_context')
    week=august.loc[august['날짜'].between(20210802,20210808)]
    assert len(week)==168 and week['생산량'].eq(0).all()
    week_row=dict(start=20210802,end=20210808,n=len(week),days=week['날짜'].nunique(),
                  power_mean=week['평균 전력'].mean(),temperature_mean=week['기온'].mean())
    reference=pd.read_csv(ROOT/'EDA/jsw/tables/09.23_007_august_state_weeks.csv',encoding='utf-8-sig')
    row=reference.loc[reference.week_start.eq('2021-08-02')&reference.producing.eq(False)].iloc[0]
    assert row.n==week_row['n'] and row.days==week_row['days']
    np.testing.assert_allclose([week_row['power_mean'],week_row['temperature_mean']],
                               [row.mean_power,row.temperature_mean],atol=1e-12)
    save(pd.DataFrame([week_row]),'restart_relations_august_week')
    return overall,monthly


def figures(data):
    assert (OUT/'restart_relations_distribution.csv').exists(),'Run diagnostics first'
    overall,monthly=relations(data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        font=style.setup()
        matrix=overall.pivot(index='x',columns='y',values='spearman').loc[VARIABLES,VARIABLES].to_numpy()
        fig,ax=style.plt.subplots(figsize=(8.8,7.2))
        masked=np.ma.array(matrix,mask=np.triu(np.ones_like(matrix),0).astype(bool))
        image=ax.imshow(masked,cmap='RdBu_r',vmin=-1,vmax=1)
        for i in range(len(VARIABLES)):
            for j in range(i):
                ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=12,
                        color='white' if abs(matrix[i,j])>.65 else '#283747')
        ax.set(xticks=range(6),xticklabels=VARIABLES,yticks=range(6),yticklabels=VARIABLES)
        ax.tick_params(axis='x',rotation=35)
        fig.colorbar(image,ax=ax,pad=.03,label='Spearman 상관계수')
        style.finish(fig,'restart_02_variable_relations','생산량·날씨·전력의 관계',
                     '정상 5,784시간, 쌍별 결측 제외 · Spearman, 동률 평균 순위 · 유의성·인과효과 검정 아님')
        # Same color scale, one coefficient per month, independent variables included.
        values=monthly.pivot(index='variable',columns='month',values='spearman').loc[INPUTS]
        fig,ax=style.plt.subplots(figsize=(11,5.4))
        image=ax.imshow(values,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
        for i in range(5):
            for j in range(8):
                value=values.iloc[i,j]
                ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=11,
                        color='white' if abs(value)>.65 else '#283747')
        ax.set(xticks=range(8),xticklabels=[f'{m}월' for m in range(1,9)],
               yticks=range(5),yticklabels=INPUTS,xlabel='월')
        fig.colorbar(image,ax=ax,pad=.02,label='평균 전력과의 Spearman 상관계수')
        style.finish(fig,'restart_02_monthly_relations','월에 따라 달라지는 전력 관계',
                     '전체 정상 기록의 월별 상관 · 생산량 양수만의 결과와 구분 · n과 날짜 수는 월별 표에 보존')
    important=[str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not important,important
    manifest=dict(status='passed',rows=len(data),scope='Restarted EDA 2.2 only',method='Spearman',
                  estimand='ordinal monotonic co-movement; no proportional or causal effect',
                  method_basis=['distribution: zero masses/ties and skewed production/rainfall','purpose: ordinal co-movement without proportional effect','sensitivity comparison is diagnostic only, not a method-selection rule'],
                  normality_diagnostic='D Agostino K2; naive iid p-values only, not calibrated for time dependence; not a selection switch',
                  main_population='All valid Jan-Aug hours; no tail exclusions',
                  checked_existing_rows=len(overall)+len(monthly)+3,
                  checked_independent_august_groups=3,pvalue_inference=False,
                  source_sha256=SHA,font=font,warnings=important,image_analysis_performed=False,
                  images=[dict(file=item['file'],bytes=item['bytes'],sha256=hashlib.sha256((style.FIGURES/item['file']).read_bytes()).hexdigest()) for item in style.GENERATED],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'restart_relations_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in manifest.items() if key in ['status','method','checked_existing_rows','warnings','image_analysis_performed']}))
    print(monthly.loc[monthly.variable.eq('기온')].to_json(orient='records',force_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnose',action='store_true')
    parser.add_argument('--figures',action='store_true')
    args=parser.parse_args()
    if not args.diagnose and not args.figures: parser.error('Choose --diagnose and/or --figures')
    frame=source_frame()
    if args.diagnose: diagnose(frame)
    if args.figures: figures(frame)
