"""Render every manuscript page and validate text bounds for visual review."""
import json
import pymupdf
from scripts.resources import ROOT,sha256

def main():
    pdf=pymupdf.open(ROOT/'paper/paper.pdf')
    output=ROOT/'tmp/pdfs'
    output.mkdir(parents=True,exist_ok=True)
    evidence=[]
    for i,page in enumerate(pdf):
        page.get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(output/f'page-{i+1}.png')
        outside=[]
        for block in page.get_text('dict')['blocks']:
            if block['type']!=0:continue
            box=pymupdf.Rect(block['bbox'])
            if not page.rect.contains(box):outside.append(list(box))
        assert not outside, f'Page {i+1}: text extends outside paper'
        evidence.append(dict(page=i+1,text_characters=len(page.get_text()),text_outside_page=outside))
    assert 4<=len(pdf)<=6
    record=dict(renderer='PyMuPDF 1.26.1',pages=len(pdf),bounds_checks='passed',
                pdf_sha256=sha256(ROOT/'paper/paper.pdf'),
                visual_review='Page PNGs require visual inspection after each render',page_evidence=evidence)
    (ROOT/'results/pdf_checks.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    main()
