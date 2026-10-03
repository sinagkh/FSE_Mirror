"""Score-blind, source-disjoint same-class test bank; no fitting or score selection."""
import argparse
from collections import Counter; from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image; from PIL import ImageDraw
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import COCO; from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for; from mirror.cases.color_binding.repair_trainbank import source
from mirror.cases.color_binding.banks import decode
from mirror.cases.color_binding.routing_relative_data import render
from mirror.cases.color_binding.rendering import captions

OUT=ROOT/'clip/interbind_routing_same_class_20260923'
TRAIN=ROOT/'clip/interbind_routing_repair_pilot_20260923'
MODELS=ROOT/'clip/interbind_routing_relative_pilot_20260923'
PLAN=ROOT/'FSE_VLM/plan/22_same_class_routing_retest.md'
PAIRS=(('bus','airplane'),('car','boat'),('cat','couch'),('dog','sports ball'),('person','bicycle'))
COLORS=(('red','blue'),('green','yellow'))
NAMESPACE='routing-same-class-new-sources-v1-20260923'


def order(*args):
    return hashlib.sha256(json.dumps([NAMESPACE,*args],sort_keys=True).encode()).hexdigest()


def eligible(a,im):
    h,w=im['height'],im['width'];x,y,bw,bh=a['bbox']
    return (not a.get('iscrowd',0) and .02<=a['area']/(h*w)<=.60
            and min(bw,bh)>=24 and min(x,y,w-x-bw,h-y-bh)>=1)


def freeze():
    paths=[PLAN,Path(__file__),TRAIN/'training_rows.jsonl',MODELS/'models.json',MODELS/'training_complete.json',
        ROOT/'data/color_binding/train/excluded_source_ids.json',
        *[COCO/f'annotations/instances_{s}.json' for s in ('train2017','val2017')]]
    paths += [Path(__file__).with_name(n+'.py') for n in ('routing_relative_data','repair_trainbank','banks','banks_v2',
        'rendering','rendering_v3','feature_cache','scorers','routing_adapter_reaudit_v2','routing_adequacy','requirements')]
    train=lines(TRAIN/'training_rows.jsonl');forbidden=set(read(paths[5]));hashes=set()
    for r in train:
        forbidden.update(r['source_ids']);forbidden.add(r['natural_guard']['image_id'])
        hashes.update(r['source_image_sha256'].values())
    for name in sorted({r['natural_guard']['file'] for r in train}):
        hashes.add(sha(COCO/'train2017'/name))
    assert sorted({tuple(r['objects']) for r in train})==list(PAIRS)
    models=read(MODELS/'models.json');verify_files({m['checkpoint']:m['sha256'] for m in models if m['checkpoint']})
    dump(OUT/'exclusions.json',dict(source_ids=sorted(forbidden),image_hashes=sorted(hashes)))
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in paths},exclusions_sha256=sha(OUT/'exclusions.json'),
        namespace=NAMESPACE,pairs=PAIRS,quota_per_pair=200,colors=COLORS,primary_color='red-blue',
        layouts=['canvas','swapped_canvas'],seed=42,models=models,plan_sha256=sha(PLAN),
        selection='Metadata and decoded-mask geometry only, fixed hash ordering, scarcity-first source allocation',
        source_reuse_allowed=False,training=False,post_specified=True,reserve_scores=False,
        gpu_batch=64,render_workers=8,cpu_threads=8,gpu_min_free_gib=12))


def verify():
    p=read(OUT/'protocol.json');verify_files(p['inputs'])
    assert sha(OUT/'exclusions.json')==p['exclusions_sha256'];return p


def build():
    p=verify();cv2.setNumThreads(1);exclude=read(OUT/'exclusions.json')
    blocked=set(exclude['source_ids']);forbidden_hashes=set(exclude['image_hashes']);nouns={n for pair in PAIRS for n in pair}
    pools=defaultdict(list)
    for split in ('train2017','val2017'):
        images,anns,cats,_=annotations_for(split);best={}
        for a in anns.values():
            noun=cats[a['category_id']];iid=a['image_id']
            if noun not in nouns or iid in blocked or not eligible(a,images[iid]):continue
            key=(noun,iid)
            if key not in best or (a['area'],-a['id'])>(best[key]['area'],-best[key]['id']):best[key]=a
        for (noun,iid),a in best.items():
            im=images[iid];path=COCO/split/im['file_name']
            if path.is_file():pools[noun].append(dict(noun=noun,image_id=iid,ann_id=a['id'],coco_split=split,file=im['file_name']))
    for noun in pools:pools[noun].sort(key=lambda r:order(noun,r['image_id'],r['ann_id']))
    counts={n:len(pools[n]) for n in sorted(nouns)};used_ids=set();used_hashes=set();rows=[];rejections=[]
    pair_order=sorted(PAIRS,key=lambda pair:(min(counts[n] for n in pair),pair))
    for pair in pair_order:
        selected={};pending_ids=set();pending_hashes=set()
        need=p['quota_per_pair']
        for noun in sorted(pair,key=lambda n:(counts[n],n)):
            selected[noun]=[]
            for r in pools[noun]:
                reason=None;path=COCO/r['coco_split']/r['file']
                if r['image_id'] in used_ids|pending_ids:reason='source_reuse'
                if reason is None:
                    h=sha(path)
                    if h in forbidden_hashes|used_hashes|pending_hashes:reason='identical_source_bytes'
                if reason is None:
                    rgb,mask,a=source(r)
                    if not .02<=float(mask.mean())<=.60:reason='decoded_mask_area'
                if reason:
                    rejections.append(dict(**r,reason=reason));continue
                rr=dict(r,image_sha256=h,mask_area_fraction=float(mask.mean()))
                selected[noun].append(rr);pending_ids.add(r['image_id']);pending_hashes.add(h)
                if len(selected[noun])==need:break
            need=min(need,len(selected[noun]))
        n=min(len(selected[k]) for k in pair)
        if n==0:raise ValueError(f'No independent eligible sources for {pair}; do not relax')
        for j in range(n):
            sources=[selected[k][j] for k in pair];ids=[s['image_id'] for s in sources]
            key=[(s['image_id'],s['ann_id']) for s in sources]
            row=dict(anchor_id='sameclass_'+order('anchor',key)[:20],family='routing',objects=pair,sources=sources,source_ids=ids,
                source_image_sha256={str(COCO/s['coco_split']/s['file']):s['image_sha256'] for s in sources})
            assert not set(ids)&(used_ids|blocked) and len(set(ids))==2
            used_ids.update(ids);used_hashes.update(s['image_sha256'] for s in sources);rows.append(row)
        print('BANK',pair,n,'independent source pairs',flush=True)
    rows.sort(key=lambda r:(r['objects'],r['anchor_id']))
    assert len(used_ids)==len(used_hashes)==2*len(rows)
    jsonl(OUT/'rows.jsonl',rows);jsonl(OUT/'rejections.jsonl',rejections)
    dump(OUT/'source_ids.json',sorted(used_ids))
    dump(OUT/'bank_complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),rows_sha256=sha(OUT/'rows.jsonl'),
        source_ids_sha256=sha(OUT/'source_ids.json'),rejections_sha256=sha(OUT/'rejections.jsonl'),
        candidate_counts=counts,pair_counts=dict(Counter('+'.join(r['objects']) for r in rows)),
        n_anchors=len(rows),n_unique_source_images=len(used_ids),n_unique_image_hashes=len(used_hashes),
        excluded_source_overlap=len(used_ids&blocked),source_reuse=0,pair_processing_order=pair_order,
        model_scores_loaded=False,training=False))


def bank():
    p=verify();done=read(OUT/'bank_complete.json')
    for name,key in [('rows.jsonl','rows_sha256'),('source_ids.json','source_ids_sha256'),('rejections.jsonl','rejections_sha256')]:
        assert sha(OUT/name)==done[key]
    rows=lines(OUT/'rows.jsonl')
    assert len({s for r in rows for s in r['source_ids']})==2*len(rows)
    verify_files({k:v for r in rows for k,v in r['source_image_sha256'].items()})
    return p,rows


def gallery():
    p,rows=bank();dest=OUT/'gallery';dest.mkdir(exist_ok=False);chosen=[]
    for pair in PAIRS:
        take=sorted([r for r in rows if tuple(r['objects'])==pair],key=lambda r:order('gallery',r['anchor_id']))[:2]
        page=Image.new('RGB',(8*128,2*168),'white');draw=ImageDraw.Draw(page)
        for j,r in enumerate(take):
            first=render(r,False);second=render(r,True)
            images=first[0][:4]+second[0][:4]
            assert all(c['edit']['outside_unchanged'] for c in first[3]+second[3])
            draw.text((4,j*168+2),r['anchor_id']+'  '+str(pair),fill='black')
            for k,im in enumerate(images):
                page.paste(Image.fromarray(im).resize((128,128)),(k*128,j*168+24))
                draw.text((k*128+2,j*168+153),('D ' if k<4 else 'S ')+['RR','RB','BR','BB'][k%4],fill='black')
            chosen.append(dict(anchor_id=r['anchor_id'],objects=pair,direct_pixels=first[2],swapped_pixels=second[2]))
        page.save(dest/('_'.join(pair).replace(' ','-')+'.png'))
    jsonl(dest/'selection.jsonl',chosen)


def encode():
    import torch
    from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
    from mirror.core.features import cache_bank
    from mirror.cases.color_binding.routing_repair_pilot import MODEL; from mirror.cases.color_binding.routing_repair_pilot import objects
    p,rows=bank();review=read(OUT/'visual_review.json')
    assert review['construction_accepted'] and review['selection_sha256']==sha(OUT/'gallery/selection.jsonl')
    torch.set_num_threads(p['cpu_threads']);cv2.setNumThreads(1)
    while torch.cuda.mem_get_info()[0]<p['gpu_min_free_gib']*1024**3:
        print('Waiting for GPU memory; existing jobs untouched',flush=True);time.sleep(45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    annotations_for('train2017');annotations_for('val2017')
    scorer=load_subject(MODEL,device='cuda');checks=[];vocab=sorted(PAIRS)
    def renderer(row):
        ims=[];names=[];hashes=[];cc=[]
        for swap in (False,True):
            ii,nn,hh,c=render(row,swap);ims+=ii;names+=nn;hashes+=hh;cc+=c
        assert all(c['edit']['outside_unchanged'] for c in cc)
        checks.append(dict(anchor_id=row['anchor_id'],checks=cc))
        return ims,names,hashes
    def prompts(family,pair):
        pair=tuple(pair);wrong=vocab[(vocab.index(pair)+1)%len(vocab)]
        return [g for colors in COLORS for g in captions(family,pair,colors)]+[objects(pair),objects(wrong)]
    cache_bank(OUT/'features',rows,scorer,renderer,prompts,registry_path=DEFAULT_REGISTRY,
        input_hashes={str(OUT/n):sha(OUT/n) for n in ('protocol.json','bank_complete.json','rows.jsonl','visual_review.json')},
        image_batch_size=p['gpu_batch'],text_batch_size=64,render_workers=p['render_workers'],
        details=dict(condition='same-class new-source composites',colors=COLORS,layouts=['canvas','swapped_canvas']))
    jsonl(OUT/'pixel_checks.jsonl',sorted(checks,key=lambda r:r['anchor_id']))
    cc=[c for r in checks for c in r['checks']]
    dump(OUT/'encoding_complete.json',dict(feature_complete_sha256=sha(OUT/'features/complete.json'),
        pixel_checks_sha256=sha(OUT/'pixel_checks.jsonl'),n_anchors=len(rows),n_images=16*len(rows),
        pixel_checks=len(cc),pixel_qualified=sum(c['edit']['passes'] for c in cc),outside_unchanged=True,
        score_based_exclusions=False))
    print('ENCODING COMPLETE',len(rows),'anchors',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','build','gallery','encode']);args=parser.parse_args()
    log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
