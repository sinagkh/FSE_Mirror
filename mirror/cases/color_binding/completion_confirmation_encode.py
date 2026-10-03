"""Outcome-free confirmation rendering, including predeclared held-out colors."""
import argparse
import numpy as np
import torch
from mirror.core.io import sha; from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.completion_confirmation import OUT
from mirror.cases.color_binding.completion_data import configure
from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import source; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.cases.color_binding.routing_relative_data import paste_states
from mirror.cases.color_binding.rendering import captions
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache

COLORS=(('red','blue'),('green','yellow'),('purple','orange'))


def render(row):
    ss=[source(s) for s in row['sources']];h=max(s[0].shape[0] for s in ss);w=sum(s[0].shape[1] for s in ss)
    packed=np.zeros((h,w,3),np.uint8);masks=[];left=0
    for rgb,mask,_ in ss:
        hh,ww=mask.shape;packed[:hh,left:left+ww]=rgb;mm=np.zeros((h,w),bool);mm[:hh,left:left+ww]=mask;masks.append(mm);left+=ww
    ims=[];names=[];hashes=[];checks=[]
    for pair in COLORS:
        for swapped in (False,True):
            ii,nn,hh,cc=paste_states(packed,masks,pair,swapped)
            ims+=ii;names+=['-'.join(pair)+'/'+n for n in nn];hashes+=hh;checks+=cc
    assert all(c['edit']['outside_unchanged'] for c in checks)
    return ims,names,hashes,checks


def encode(model):
    configure();verify_files(read(OUT/'complete.json')['files']);rows=lines(OUT/'rows.jsonl')
    dest=OUT/'features'/model
    if (dest/'checks_complete.json').exists():verify_cache(dest);return
    if torch.cuda.mem_get_info()[0]<12*1024**3:raise RuntimeError('GPU busy')
    annotations_for('train2017');annotations_for('val2017');scorer=load_subject(model,device='cuda');checks=[]
    def renderer(row):
        ims,nn,hh,cc=render(row);checks.append(dict(anchor_id=row['anchor_id'],checks=cc));return ims,nn,hh
    def prompts(family,objects):return [g for pair in COLORS for g in captions(family,objects,pair)]
    cache_bank(dest,rows,scorer,renderer,prompts,registry_path=DEFAULT_REGISTRY,
        input_hashes={str(OUT/'complete.json'):sha(OUT/'complete.json')},image_batch_size=64,text_batch_size=64,render_workers=8,
        details=dict(colors=COLORS,layouts=['canvas','swapped_canvas'],score_scale=1.,score_bias=0.,no_outcomes=True))
    jsonl(dest/'pixel_checks.jsonl',sorted(checks,key=lambda r:r['anchor_id']));cc=[c for r in checks for c in r['checks']]
    dump(dest/'checks_complete.json',dict(n=len(cc),passed=sum(c['edit']['passes'] for c in cc),
        checks_sha256=sha(dest/'pixel_checks.jsonl'),score_dependent_exclusions=False))
    print('CONFIRMATION_ENCODING_COMPLETE',model,len(rows),len(cc),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='openclip_laion_l14');a=ap.parse_args();log(OUT,'start',stage='encode',model=a.model)
    try:encode(a.model)
    except BaseException as e:log(OUT,'failed',error=repr(e));raise
    log(OUT,'complete',stage='encode',model=a.model)


if __name__=='__main__':main()
