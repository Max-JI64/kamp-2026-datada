"""Check local manuscript links, selected PNGs, and relocated Python syntax."""
from pathlib import Path
from urllib.parse import unquote
import ast
import argparse
import hashlib
import json
import re
from PIL import Image

BASE=Path(__file__).resolve().parents[1]

def main(contact_sheets=False):
    checked=[]
    for doc in BASE.glob('*.md'):
        if not doc.name.startswith('10.02_'):continue
        text=doc.read_text(encoding='utf-8')
        for value in re.findall(r'!?\[[^\]]*\]\(([^\n]+?)\)',text):
            value=unquote(value.strip('<>')).split('#')[0]
            if value.startswith(('https://','http://')):continue
            path=Path(value)
            if not path.is_absolute():path=doc.parent/path
            assert path.is_file(),f'{doc.name}: broken link {value}'
            checked.append(str(path.resolve()))
    for path in (BASE/'scripts').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    doc=(BASE/'10.02_002_EDA_새원고.md').read_text(encoding='utf-8')
    assert re.findall(r'^## (2\.\d+) ',doc,re.M)==['2.1','2.2','2.3']
    for section in re.split(r'(?=^## 2\.)',doc,flags=re.M)[1:]:
        assert '**관련 Python 코드**' in section
    figures=[]
    for value in re.findall(r'!\[[^\]]*\]\(<([^>]+)>\)',doc):
        path=Path(value)
        with Image.open(path) as image:
            image.verify()
        figures.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(figures)==5
    contacts=[]
    # Visual QA is opt-in and goes outside the manuscript's EDA image folder.
    qa=BASE.parent/'tmp/eda_visual_review'
    if contact_sheets:qa.mkdir(parents=True,exist_ok=True)
    for offset in (range(0,len(figures),2) if contact_sheets else []):
        thumbnails=[]
        for item in figures[offset:offset+2]:
            with Image.open(item['path']) as image:
                thumbnails.append(image.convert('RGB').resize(
                    (image.width//2,image.height//2),Image.Resampling.LANCZOS))
        contact=Image.new('RGB',(max(x.width for x in thumbnails),sum(x.height for x in thumbnails)+16*(len(thumbnails)-1)),'white')
        y=0
        for thumbnail in thumbnails:
            contact.paste(thumbnail,(0,y));y+=thumbnail.height+16
        target=qa/f'manuscript_contact_{offset//2+1:02d}_50.png'
        contact.save(target)
        contacts.append(str(target))
    result=dict(status='passed',checked_links=len(checked),manuscript_sections=['2.1','2.2','2.3'],
        figures=figures,python_syntax='passed',contact_sheets=contacts,
        image_render_check='PNG decoded; visual inspection recorded separately')
    result_path=BASE/'tables/manuscript_asset_verification.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text(encoding='utf-8-sig'))
        if previous.get('figures')==figures and previous.get('visual_review'):
            result['visual_review']=previous['visual_review']
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='figures'},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact-sheets',action='store_true',help='Write 50%% review sheets to project tmp/eda_visual_review')
    main(contact_sheets=parser.parse_args().contact_sheets)
