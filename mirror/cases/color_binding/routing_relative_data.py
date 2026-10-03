"""Position-balanced training features, preserving original noun-slot identity."""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding import routing_repair_pilot as prior
from mirror.cases.color_binding.repair_trainbank import source; from mirror.cases.color_binding.repair_trainbank import annotations_for; from mirror.cases.color_binding.repair_trainbank import lines
from mirror.cases.color_binding.rendering import canvas_pair; from mirror.cases.color_binding.rendering import captions
from mirror.cases.color_binding.rendering_v3 import recolor
from mirror.cases.color_binding.generators import edit_checks; from mirror.cases.color_binding.generators import pixel_hash
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache

OUT=ROOT/'clip/interbind_routing_relative_pilot_20260923'
PLAN=ROOT/'FSE_VLM/plan/21_relative_routing_repair_pilot.md'


def paste_states(packed,masks,colors,swapped=True):
    base,slots=canvas_pair(packed,masks,swapped=swapped)
    images=[];names=[];checks=[]
    for a in colors:
        for b in colors:
            image=base.copy()
            for j,(mask,color) in enumerate(zip(slots,(a,b))):
                before=image;image=recolor(image,mask,color)
                checks.append(dict(state=a+'_'+b,slot=j,color=color,edit=edit_checks(before,image,mask,color,'rgb_blend')))
            images.append(image);names.append(('swapped_canvas' if swapped else 'canvas')+'/'+a+'_'+b)
    return images,names,[pixel_hash(x) for x in images],checks


def render(row,swapped=True):
    sources=[source(r) for r in row['sources']]
    h=max(s[0].shape[0] for s in sources);w=sum(s[0].shape[1] for s in sources)
    packed=np.zeros((h,w,3),np.uint8);masks=[];left=0
    for rgb,mask,_ in sources:
        hh,ww=mask.shape;packed[:hh,left:left+ww]=rgb
        mm=np.zeros((h,w),bool);mm[:hh,left:left+ww]=mask;masks.append(mm);left+=ww
    ims=[];names=[];hashes=[];checks=[]
    for pair in prior.PAIRS:
        ii,nn,hh,cc=paste_states(packed,masks,pair,swapped)
        ims+=ii;names+=['-'.join(pair)+'/'+n for n in nn];hashes+=hh;checks+=cc
    return ims,names,hashes,checks


def freeze():
    prior.verify();verify_cache(prior.OUT/'features')
    paths=[Path(__file__),PLAN,prior.OUT/'protocol.json',prior.OUT/'encoding_complete.json',
           prior.OUT/'training_rows.jsonl',prior.OUT/'features/complete.json']
    checks={str(p):sha(p) for p in paths}
    checks.update({str(prior.OUT/'features'/k):v for k,v in read(prior.OUT/'features/complete.json')['files'].items()})
    checks.update({str(prior.OUT/k):v for k,v in read(prior.OUT/'encoding_complete.json')['files'].items()})
    dump(OUT/'data_protocol.json',dict(inputs=checks,seed=42,source_selection='All 640 original training pairs; no new examples or exclusions',
        image_states=8,colors=prior.PAIRS,layout='swapped_canvas',caption_slots='Original noun slots, never physical left/right positions',
        source_overlap_rule='Reuse the first pilot source firewall for both image sources and natural training captions',
        precision='FP32, TF32 off, pinned L/14 encoder',image_batch=16,render_workers=8,no_scores=True))
    old={r['anchor_id']:r for r in lines(prior.OUT/'features/index.jsonl')};smoke=[]
    for row in lines(prior.OUT/'training_rows.jsonl')[:4]:
        _,_,hashes,_=render(row,False);assert hashes==old[row['anchor_id']]['pixel_sha256']
        _,names,hashes,cc=render(row,True);assert all(c['edit']['outside_unchanged'] for c in cc)
        smoke.append(dict(anchor_id=row['anchor_id'],names=names,swapped_pixels=hashes,checks=cc,direct_replay_exact=True))
    jsonl(OUT/'construction_smoke.jsonl',smoke)


def verify():
    p=read(OUT/'data_protocol.json');verify_files(p['inputs']);prior.verify();return p


def encode():
    p=verify();torch.set_num_threads(4);cv2.setNumThreads(1)
    while torch.cuda.mem_get_info()[0]<12*1024**3:
        print('Waiting for >=12GiB CUDA memory; other jobs unchanged',flush=True);time.sleep(45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=lines(prior.OUT/'training_rows.jsonl');annotations_for('train2017')
    scorer=load_subject(prior.MODEL,device='cuda');all_checks=[]
    def renderer(row):
        images,names,hashes,checks=render(row,True)
        assert all(c['edit']['outside_unchanged'] for c in checks)
        all_checks.append(dict(anchor_id=row['anchor_id'],checks=checks))
        return images,names,hashes
    def prompts(family,objects):return [g for pair in prior.PAIRS for g in captions(family,objects,pair)]
    cache_bank(OUT/'swapped_features',rows,scorer,renderer,prompts,registry_path=DEFAULT_REGISTRY,
        input_hashes={str(OUT/'data_protocol.json'):sha(OUT/'data_protocol.json')},image_batch_size=16,text_batch_size=64,
        render_workers=8,details=dict(layout='swapped_canvas',caption_slots='original nouns',score_scale=1,score_bias=0))
    jsonl(OUT/'swapped_pixel_checks.jsonl',sorted(all_checks,key=lambda r:r['anchor_id']))
    cc=[c for r in all_checks for c in r['checks']]
    dump(OUT/'encoding_complete.json',dict(files={'swapped_features/complete.json':sha(OUT/'swapped_features/complete.json'),
        'swapped_pixel_checks.jsonl':sha(OUT/'swapped_pixel_checks.jsonl')},pixel_checks=len(cc),pixel_qualified=sum(c['edit']['passes'] for c in cc),
        direct_cache_reused=True,heldout_or_reserve_images_encoded=False))
    print('SWAPPED TRAINING CACHE COMPLETE',len(cc),'checks',sum(c['edit']['passes'] for c in cc),'qualified',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','encode']);args=parser.parse_args();log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
