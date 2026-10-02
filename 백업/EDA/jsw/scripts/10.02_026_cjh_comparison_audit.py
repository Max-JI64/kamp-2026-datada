"""Read-only review of teammate artifacts; no teammate code is executed."""
from pathlib import Path
import base64
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

EDA = Path(__file__).resolve().parent.parent
ROOT = EDA.parent.parent
CJH = ROOT / 'EDA/cjh'
OUT = EDA / 'tables/10.02_026_cjh_audit'
FIG = EDA / 'figures/10.02_026_cjh_audit'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
inventory = []
for p in sorted(CJH.rglob('*')):
    if p.is_file():
        raw = p.read_bytes()
        item = {'path': p.relative_to(ROOT).as_posix(), 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest()}
        if p.parent.name == 'scripts':
            try:
                compile(raw.decode('utf-8-sig'), str(p), 'exec')
                item['compile'] = 'ok'
            except SyntaxError as exc:
                item['compile'] = f'SyntaxError line {exc.lineno}: {exc.msg}'
        inventory.append(item)
pd.DataFrame(inventory).to_csv(OUT / 'inventory.csv', index=False, encoding='utf-8-sig')

nb = json.loads((CJH / 'code/안티그래프티1.ipynb').read_text(encoding='utf-8'))
text_parts, image_items = [], []
for ci, cell in enumerate(nb['cells']):
    text_parts.append(f"## cell {ci} ({cell['cell_type']}, execution={cell.get('execution_count')})\n")
    text_parts.append(''.join(cell.get('source', [])))
    for oi, output in enumerate(cell.get('outputs', [])):
        if output.get('output_type') == 'stream' and output.get('name') != 'stderr':
            text_parts.append(''.join(output.get('text', [])))
        plain = output.get('data', {}).get('text/plain')
        if plain:
            text_parts.append(''.join(plain))
        png = output.get('data', {}).get('image/png')
        if png:
            if isinstance(png, list):
                png = ''.join(png)
            p = FIG / f'cell_{ci:02d}_output_{oi:02d}.png'
            p.write_bytes(base64.b64decode(png))
            with Image.open(p) as im:
                image_items.append({'cell': ci, 'output': oi, 'file': p.name,
                                    'width': im.width, 'height': im.height})
    text_parts.append('')
(OUT / 'notebook_text.md').write_text('\n\n'.join(text_parts), encoding='utf-8')
# Original images at exactly 50% width/height, at most four per sheet.
for start in range(0, len(image_items), 4):
    chunk = image_items[start:start + 4]
    thumbs = []
    for item in chunk:
        with Image.open(FIG / item['file']) as im:
            thumbs.append(im.convert('RGB').resize((im.width // 2, im.height // 2), Image.Resampling.LANCZOS))
    cols = 2 if len(chunk) > 1 else 1
    widths = [max(t.width for t in thumbs[j::cols]) for j in range(cols)]
    heights = [max(t.height for t in thumbs[j:j + cols]) + 28 for j in range(0, len(thumbs), cols)]
    sheet = Image.new('RGB', (sum(widths), sum(heights)), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, (item, thumb) in enumerate(zip(chunk, thumbs)):
        x, y = sum(widths[:j % cols]), sum(heights[:j // cols])
        draw.text((x + 5, y + 5), f"Cell {item['cell']} | {item['file']}", fill='black')
        sheet.paste(thumb, (x, y + 28))
    sheet.save(FIG / f'contact_{start // 4 + 1:02d}.png')

src = ROOT / 'data/origin/okm_augumented_2021.csv'
df = pd.read_csv(src, encoding='utf-8-sig')
valid = df['시간'].between(0, 23)
dates = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d')
df['month_audit'] = dates.dt.month
stats = {'python': sys.version, 'pandas': pd.__version__,
         'run_at': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='minutes'),
         'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
         'rows': len(df), 'columns_original': 18, 'original_missing': int(df.drop(columns='month_audit').isna().sum().sum()),
         'invalid_hour_rows': int((~valid).sum()),
         'invalid_dates': sorted(df.loc[~valid, '날짜'].unique().tolist()),
         'valid_hour_matches_row_order': int((df.loc[valid, '시간'] == df.groupby('날짜').cumcount()[valid]).sum()),
         'valid_hour_rows': int(valid.sum()), 'images': image_items}
median_df = df.copy()
median_df.loc[~valid, '시간'] = np.nan
for col in ['시간', '공장인원', '풍속', '강수량']:
    median_df[col] = median_df[col].fillna(median_df[col].median())
stats['hour_median'] = float(median_df.loc[~valid, '시간'].iloc[0])
dt_imputed = dates + pd.to_timedelta(median_df['시간'], unit='h')
stats['imputed_datetime_duplicate_excess'] = int(dt_imputed.duplicated().sum())
stats['personnel_positive_original'] = int((df['공장인원'] > 0).sum())
stats['personnel_positive_imputed'] = int((median_df['공장인원'] > 0).sum())
power_sum = df[['15분', '30분', '45분', '60분']].sum(axis=1)
ratio = df['생산량'] / power_sum.replace(0, np.nan)
stats['personnel_ratio_max_abs_error'] = float((df['공장인원'] - ratio).abs().max())
corcols = ['시간', '15분', '30분', '45분', '60분', '평균', '생산량', '기온', '풍속', '습도', '강수량', '전기요금(계절)', '공장인원', '인건비']
median_df[corcols].corr().to_csv(OUT / 'notebook_correlations_recomputed.csv', encoding='utf-8-sig')
monthly = df.groupby('month_audit').agg(rows=('생산량', 'size'), mean_production=('생산량', 'mean'),
                                        sum_production=('생산량', 'sum'), mean_power=('평균', 'mean'))
monthly.to_csv(OUT / 'monthly.csv', encoding='utf-8-sig')
hour_rows = []
for name, frame in [('raw_valid_all', df.loc[valid]), ('notebook_median_all', median_df),
                    ('jsw_valid_jan_aug', df.loc[valid & (df['month_audit'] <= 8)])]:
    tab = frame.groupby('시간').agg(rows=('생산량', 'size'),
                                  mean_production=('생산량', 'mean'), mean_power=('평균', 'mean')).reset_index()
    tab['policy'] = name
    hour_rows.append(tab)
pd.concat(hour_rows).to_csv(OUT / 'hourly_policy_comparison.csv', index=False, encoding='utf-8-sig')
op_tables = []
for name, frame in [('raw_all', df), ('notebook_median_all', median_df), ('jsw_valid_jan_aug', df.loc[valid & (df['month_audit'] <= 8)])]:
    conditions = [frame['공장인원'].isna(), (frame['생산량'] == 0) & (frame['공장인원'] == 0),
                  (frame['생산량'] > 0) & (frame['공장인원'] > 0),
                  (frame['생산량'] > 0) & (frame['공장인원'] == 0),
                  (frame['생산량'] == 0) & (frame['공장인원'] > 0)]
    group = np.select(conditions, ['missing_personnel', 'zero_both', 'positive_both', 'production_only', 'personnel_only'], default='other')
    tab = frame.groupby(group).agg(rows=('평균', 'size'), mean_power=('평균', 'mean')).reset_index(names='record_condition')
    tab['policy'] = name
    op_tables.append(tab)
pd.concat(op_tables).to_csv(OUT / 'operation_conditions.csv', index=False, encoding='utf-8-sig')
prod = df['생산량']; q1, q3 = prod.quantile([.25, .75]); fence = q3 + 1.5 * (q3-q1)
stats['production_iqr_upper_all'] = float(fence)
stats['production_iqr_flagged_all'] = int((prod > fence).sum())
stats['hourly_production_max_valid'] = float(df.loc[valid].groupby('시간')['생산량'].mean().idxmax())
stats['monthly_production_max'] = int(monthly['mean_production'].idxmax())
(OUT / 'facts.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(stats, ensure_ascii=True, indent=2))
