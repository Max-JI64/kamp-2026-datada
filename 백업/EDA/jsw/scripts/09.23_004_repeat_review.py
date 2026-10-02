"""Bounded follow-up of identical daily profiles; no model or data removal."""
from pathlib import Path
import pandas as pd
BASE=Path(__file__).parent.parent
cols=['날짜','시간','15분','30분','45분','60분','평균','생산량']
df=pd.read_csv(BASE/'../../data/origin/okm_augumented_2021.csv',encoding='utf-8-sig',usecols=cols)
df=df.loc[df['날짜'].lt(20210901)&df['시간'].between(0,23)]
rows=[]
for label,values in [('rounded_mean',['평균']),('four_quarters',['15분','30분','45분','60분']),('production',['생산량'])]:
    wide=df.pivot(index='날짜',columns='시간',values=values).dropna()
    signatures=wide.apply(lambda row:tuple(row),axis=1)
    groups=signatures.value_counts()
    rows.append({'profile':label,'days':len(wide),'unique_profiles':len(groups),
                 'days_in_repeated_profiles':int(groups.loc[groups.gt(1)].sum()),
                 'largest_group':int(groups.max())})
    if label=='four_quarters':
        examples=[]
        for number,(signature,count) in enumerate(groups.loc[groups.gt(1)].head(5).items(),1):
            dates=signatures.index[signatures.map(lambda x:x==signature)].tolist()
            examples.append({'group':number,'days':int(count),'dates':';'.join(map(str,dates)),
                             'min_value':min(signature),'max_value':max(signature)})
        pd.DataFrame(examples).to_csv(BASE/'tables/09.23_004_repeated_profile_examples.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(rows).to_csv(BASE/'tables/09.23_004_profile_review.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string(index=False))
