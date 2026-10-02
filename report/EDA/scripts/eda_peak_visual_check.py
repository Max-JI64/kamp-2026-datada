from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
root=Path(__file__).resolve().parents[1]
names=['restart_03_threshold_timing.png','restart_03_condition_ranges.png','restart_03_hour_shape.png']
panels=[]
for name in names:
    with Image.open(root/'figures/eda'/name) as source:
        panel=source.convert('RGB').resize((source.width//2,source.height//2),Image.Resampling.LANCZOS)
    panels.append((name,panel))
width=max(p.width for _,p in panels)+32
height=sum(p.height+32 for _,p in panels)+16
sheet=Image.new('RGB',(width,height),'#ffffff')
draw=ImageDraw.Draw(sheet)
y=16
for name,panel in panels:
    draw.text((16,y),name,fill='#202020')
    sheet.paste(panel,(16,y+18))
    y+=panel.height+32
out=root/'figures/eda/qa'
out.mkdir(exist_ok=True)
sheet.save(out/'restart_03_contact_50.png')
print(str(out/'restart_03_contact_50.png'))