"""Score-blind natural spatial test preparation; official labels unchanged."""
import argparse
from collections import Counter
import os
from pathlib import Path
import tarfile
import urllib.request
import urllib.parse
import re
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA')
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.strengthening_cpu import OUT as ROOTOUT
from mirror.cases.color_binding.strengthening_spatial import OUT as SPATIAL
from mirror.cases.color_binding.repair_trainbank import COCO; from mirror.cases.color_binding.repair_trainbank import lines

OUT=ROOTOUT/'whatsup'


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    src=ROOTOUT/'spatial';coco=read(src/'whatsup_coco_two_obj.json');controlled=read(src/'whatsup_controlled_a.json')
    occupied={i for r in lines(SPATIAL/'rows.jsonl') for i in r['source_ids']}
    assert not occupied&{int(r[0]) for r in coco}
    rows=[]
    for i,(iid,pos,neg) in enumerate(coco):
        path=COCO/'val2017'/f'{iid:012d}.jpg'
        assert path.is_file()
        lr=(' to the left of ' in pos.lower() or ' to the right of ' in pos.lower())
        rows.append(dict(benchmark='coco_two_object',example_id=str(i),source_id=str(iid),image=str(path),
            captions=[pos,neg],correct_index=0,left_right=lr,official_image_sha256=sha(path)))
    for i,r in enumerate(controlled):
        name=Path(r['image_path']).name
        lr=('_left_of_' in name or '_right_of_' in name)
        lridx=[j for j,s in enumerate(r['caption_options']) if ' to the left of ' in s or ' to the right of ' in s]
        assert len(lridx)==2
        key=name.replace('_left_of_','_POSITION_').replace('_right_of_','_POSITION_') if lr else name
        rows.append(dict(benchmark='controlled_a',example_id=str(i),source_id=key,image=str(OUT/'images/controlled_images'/name),
            captions=r['caption_options'],correct_index=0,left_right=lr,left_right_caption_indices=lridx,
            annotation_issue_known=name in ('pillow_right_of_chair.jpeg','pillow_left_of_chair.jpeg')))
    jsonl(OUT/'rows.jsonl',rows)
    dump(OUT/'protocol.json',dict(manifest_sha256=sha(OUT/'rows.jsonl'),source_metadata_sha256={str(src/n):sha(src/n) for n in ['whatsup_coco_two_obj.json','whatsup_controlled_a.json']},
        script_sha256=sha(Path(__file__)),official_repository_commit='7c1f2550eace32e7b8c77de5a792347c402960d1',
        label_orientation='Official first caption is correct; unchanged labels',
        primary='All COCO two-object items whose correct caption is left/right; one image per item, behavioral test only, not an intervention interaction measurement',
        secondary='Controlled-A left/right photos, both official4way and predeclared2way left-vs-right; matched left/right image pairs only for response/preference',
        preservation='All440 official COCO two-object items and all412 Controlled-A items',
        ambiguity='Official issue4 reports two reversed pillow/chair controlled labels; retain official result primary and separately report fixed, score-independent exclude-known-issue sensitivity',
        official_issue='https://github.com/amitakamath/whatsup_vlms/issues/4',
        no_scores=True,training_overlap_sources=0,counts=dict(Counter((r['benchmark']+'/left_right' if r['left_right'] else r['benchmark']+'/other') for r in rows))))
    dump(OUT/'metadata_complete.json',dict(files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()},gpu=False,scoring=False))


def download(retry=False):
    verify_files(read(OUT/'metadata_complete.json')['files'])
    url='https://drive.google.com/uc?export=download&id=19KGYVQjrV3syb00GgcavB2nZTW5NXX0H'
    path=OUT/('controlled_images_confirmed.tar.gz' if retry else 'controlled_images.tar.gz');maximum=700*1024**2
    if retry:
        html=(OUT/'controlled_images.tar.gz').read_text()
        values=dict(re.findall(r'type="hidden" name="([^"]+)" value="([^"]+)"',html))
        assert values.get('id')=='19KGYVQjrV3syb00GgcavB2nZTW5NXX0H'
        url='https://drive.usercontent.google.com/download?'+urllib.parse.urlencode(values)
    log(ROOTOUT,'download_start',url=url,maximum_bytes=maximum)
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'InterBind-research/1.0'}),timeout=60) as res:
        if int(res.headers.get('Content-Length',0))>maximum:raise ValueError('Dataset exceeds declared download bound')
        size=0
        with path.open('xb') as f:
            while True:
                chunk=res.read(1024**2)
                if not chunk:break
                size+=len(chunk)
                if size>maximum:raise ValueError('Download exceeds bound')
                f.write(chunk)
    dest=OUT/'images';dest.mkdir(exist_ok=True)
    # Extract only regular image files after full path/type validation.
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            if not member.isfile() or Path(member.name).suffix.lower() not in ('.jpg','.jpeg','.png'):continue
            relative=Path(member.name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe archive path')
            target=(dest/relative).resolve()
            if dest.resolve() not in target.parents:raise ValueError('Unsafe archive target')
            target.parent.mkdir(parents=True,exist_ok=True)
            with tf.extractfile(member) as src,target.open('xb') as out:
                while True:
                    chunk=src.read(1024**2)
                    if not chunk:break
                    out.write(chunk)
    records=[]
    for r in lines(OUT/'rows.jsonl'):
        p=Path(r['image'])
        if not p.is_file():raise FileNotFoundError(p)
        records.append(dict(benchmark=r['benchmark'],example_id=r['example_id'],path=str(p),sha256=sha(p)))
    jsonl(OUT/'image_hashes.jsonl',records)
    dump(OUT/'download_complete.json',dict(archive_sha256=sha(path),archive_bytes=size,images_sha256=sha(OUT/'image_hashes.jsonl'),gpu=False,no_scores=True))
    log(ROOTOUT,'download_complete',url=url,bytes=size,sha256=sha(path))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','download','download_retry']);a=p.parse_args()
    log(ROOTOUT,'start',stage='whatsup_'+a.action)
    try:
        if a.action=='download_retry':download(retry=True)
        else:globals()[a.action]()
    except BaseException as e:log(ROOTOUT,'failed',stage='whatsup_'+a.action,error=repr(e));raise
    log(ROOTOUT,'complete',stage='whatsup_'+a.action)


if __name__=='__main__':main()
