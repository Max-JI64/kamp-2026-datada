"""Check all EDA Markdown links, visible figure paths, PNGs, and Python syntax."""
from pathlib import Path
from urllib.parse import unquote
import ast
import argparse
import hashlib
import json
import re
from PIL import Image

BASE=Path(__file__).resolve().parents[1]

def linked_path(document, value):
    """Resolve Markdown links from the document directory, including URL escapes."""
    target=Path(unquote(value.strip('<>')).split('#')[0])
    return (document.parent/target).resolve()

def main(contact_sheets=False):
    checked=[]
    documents=sorted(BASE.rglob('*.md'))
    for doc in documents:
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
    manuscript=BASE/'02_EDA_원고.md'
    doc=manuscript.read_text(encoding='utf-8')
    assert re.findall(r'^## (2\.\d+) ',doc,re.M)==['2.1','2.2','2.3','2.4','2.5']
    for section in re.split(r'(?=^## 2\.)',doc,flags=re.M)[1:]:
        assert '**관련 Python 코드**' in section
        image_paths=re.findall(r'!\[[^\]]*\]\(<([^>]+)>\)',section)
        displayed_paths=re.findall(r'이미지 파일: \[([^\]]+)\]\(<([^>]+)>\)',section)
        assert len(displayed_paths)==len(image_paths), 'Every figure needs a visible file path'
        for value,(label,target) in zip(image_paths,displayed_paths):
            assert linked_path(manuscript,value)==linked_path(manuscript,target), 'Displayed path must match embedded figure'
            assert label==linked_path(manuscript,target).relative_to(BASE.parent).as_posix(), 'Label must show project folder and filename'
        assert section.count('생성 코드:')==len(image_paths)
    data_doc=(BASE/'01_데이터_원고.md').read_text(encoding='utf-8')
    assert not re.search(r'!\[[^\]]*\]\(',data_doc), 'Data manuscript contains no plots'
    figures=[]
    for value in re.findall(r'!\[[^\]]*\]\(<([^>]+)>\)',doc):
        path=linked_path(manuscript,value)
        with Image.open(path) as image:
            image.verify()
        figures.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(figures)==9
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
    result=dict(status='passed',checked_links=len(checked),checked_documents=len(documents),
        visible_figure_paths=len(figures),manuscript_sections=['2.1','2.2','2.3','2.4','2.5'],
        figures=figures,python_syntax='passed',contact_sheets=contacts,
        image_render_check='PNG decoded; visual inspection recorded separately')
    result_path=BASE/'tables/manuscript_asset_verification.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text(encoding='utf-8-sig'))
        if previous.get('figures')==figures and previous.get('visual_review'):
            result['visual_review']=previous['visual_review']
        if previous.get('figures')==figures:
            for key,value in previous.items():
                if key.endswith('_visual_review'):
                    result[key]=value
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='figures'},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact-sheets',action='store_true',help='Write 50%% review sheets to project tmp/eda_visual_review')
    main(contact_sheets=parser.parse_args().contact_sheets)
