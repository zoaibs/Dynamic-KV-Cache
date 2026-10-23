"""Render the GPU manuscript to a visually checked PDF."""
import html
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from scripts.resources import ROOT

def inline(text):
    text=html.escape(text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`(.*?)`',r'<font name="Courier">\1</font>',text)
    text=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#315D7B">\1</link>',text)
    return text

def main():
    styles={
        'title':ParagraphStyle('Title',fontName='Helvetica-Bold',fontSize=17,leading=20,spaceAfter=10,textColor=colors.HexColor('#233E50')),
        'h2':ParagraphStyle('H2',fontName='Helvetica-Bold',fontSize=11.5,leading=14,spaceBefore=10,spaceAfter=5,textColor=colors.HexColor('#233E50')),
        'h3':ParagraphStyle('H3',fontName='Helvetica-Bold',fontSize=9.8,leading=12,spaceBefore=7,spaceAfter=4),
        'body':ParagraphStyle('Body',fontName='Helvetica',fontSize=9.4,leading=12.3,spaceAfter=6),
        'caption':ParagraphStyle('Caption',fontName='Helvetica',fontSize=8,leading=10.4,spaceAfter=7,textColor=colors.HexColor('#46535D')),
        'cell':ParagraphStyle('Cell',fontName='Helvetica',fontSize=7.9,leading=10),
        'table_head':ParagraphStyle('TableHead',fontName='Helvetica-Bold',fontSize=7.7,leading=10,textColor=colors.white)
    }
    text=(ROOT/'paper/paper.md').read_text(encoding='utf-8')
    story=[]
    lines=text.splitlines()
    i=0
    width=7.0*inch
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1;continue
        if line=='<!-- PAGE BREAK -->': story.append(PageBreak());i+=1;continue
        if line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            path=(ROOT/'paper'/match.group(2)).resolve()
            image=Image(str(path))
            max_height=2.75*inch if 'precision_timeline' in str(path) else 2.5*inch
            if 'architecture' in str(path): max_height=1.55*inch
            factor=min(width/image.imageWidth,max_height/image.imageHeight)
            image.drawWidth=image.imageWidth*factor;image.drawHeight=image.imageHeight*factor
            story.extend([Spacer(1,4),image,Spacer(1,5)]);i+=1;continue
        if line.startswith('|'):
            raw=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-:]+',x) for x in row): raw.append(row)
                i+=1
            cells=[[Paragraph(inline(c),styles['table_head' if j==0 else 'cell']) for c in row]
                   for j,row in enumerate(raw)]
            columns=len(raw[0]);colwidths=[width*.24]+[width*.76/(columns-1)]*(columns-1)
            table=Table(cells,colWidths=colwidths,hAlign='LEFT',repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#36566B')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F3F5'),colors.white]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),
                ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([Spacer(1,4),table,Spacer(1,6)]);continue
        key='title' if line.startswith('# ') else 'h2' if line.startswith('## ') else 'h3' if line.startswith('### ') else None
        if key:
            story.append(Paragraph(inline(line.lstrip('#').strip()),styles[key]));i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','<!--')):
            paragraph.append(lines[i].strip());i+=1
        paragraph=' '.join(paragraph)
        key='caption' if paragraph.startswith(('Figure ','Table ')) else 'body'
        story.append(Paragraph(inline(paragraph),styles[key]))
    def page(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#D4DBDF'))
        canvas.line(.75*inch,.48*inch,7.75*inch,.48*inch)
        canvas.setFont('Helvetica',7.5);canvas.setFillColor(colors.HexColor('#56616B'))
        canvas.drawString(.75*inch,.31*inch,'Dynamic KV Cache Scheduler | Local GPU study')
        canvas.drawRightString(7.75*inch,.31*inch,str(doc.page));canvas.restoreState()
    output=ROOT/'paper/paper.pdf'
    doc=SimpleDocTemplate(str(output),pagesize=letter,rightMargin=.75*inch,leftMargin=.75*inch,
                          topMargin=.52*inch,bottomMargin=.65*inch,
                          title='GPU Frame-Adaptive Packed KV Storage for Local Video Question Answering',
                          author='Dynamic KV Cache Scheduler project')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(output)

if __name__=='__main__':
    main()
