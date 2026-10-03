"""Prospective spatial study: immutable data, CPU encoding, then diagnosis.

Positions exchange the same object cutouts; objects are never mirrored. Caption
order is balanced by pooling logically equivalent paraphrases, not negatives.
"""
import argparse
from collections import Counter; from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError("Use CUDA_VISIBLE_DEVICES=''")
import cv2
import numpy as np
from PIL import Image; from PIL import ImageDraw
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.strengthening_cpu import OUT as ROOTOUT; from mirror.cases.color_binding.strengthening_cpu import clean; from mirror.cases.color_binding.strengthening_cpu import paired_intervals
from mirror.cases.color_binding.repair_trainbank import COCO; from mirror.cases.color_binding.repair_trainbank import source; from mirror.cases.color_binding.repair_trainbank import lines
from mirror.cases.color_binding.rendering import canvas_pair
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY

OUT=ROOTOUT/'spatial_v2'
NOUNS=('bicycle','bus','car','cat','chair','dog','person','truck')
TRAIN_PAIRS=tuple((NOUNS[i],NOUNS[(i+1)%8]) for i in range(8))
NEW_PAIRS=tuple((NOUNS[i],NOUNS[(i+2)%8]) for i in range(8))
MODEL='openclip_laion_l14'


def order(*v):
    return hashlib.sha256(json.dumps(['spatial-prospective-v1',*v],sort_keys=True).encode()).hexdigest()


def partition(i):
    h=int(order('global-source-partition',int(i))[:12],16)/16**12
    return 'train' if h<.40 else 'development' if h<.55 else 'confirmation'


def captions(nouns):
    a,b=nouns
    return [[f'a photo of a {a} to the left of a {b}', f'a photo of a {b} to the right of a {a}'],
            [f'a photo of a {a} to the right of a {b}', f'a photo of a {b} to the left of a {a}']]


def draw(row):
    ss=[source(s) for s in row['sources']]
    h=max(s[0].shape[0] for s in ss);w=sum(s[0].shape[1] for s in ss)
    packed=np.zeros((h,w,3),np.uint8);masks=[];left=0
    for rgb,m,_ in ss:
        hh,ww=m.shape;packed[:hh,left:left+ww]=rgb
        mask=np.zeros((h,w),bool);mask[:hh,left:left+ww]=m;masks.append(mask);left+=ww
    images=[];checks=[]
    for swapped in (False,True):
        image,mm=canvas_pair(packed,masks,swapped)
        centers=[float(np.where(m)[1].mean()) for m in mm]
        assert (centers[0]>centers[1])==swapped and not (mm[0]&mm[1]).any()
        images.append(image);checks.append(dict(swapped=swapped,centers_x=centers,nonoverlap=True,
            mask_pixels=[int(m.sum()) for m in mm]))
    assert checks[0]['mask_pixels']==checks[1]['mask_pixels']
    return images,checks


def build():
    src=ROOTOUT/'data_feasibility';verify_files(read(src/'complete.json')['files'])
    OUT.mkdir(parents=True,exist_ok=False)
    benchmark=ROOTOUT/'spatial/whatsup_coco_two_obj.json'
    benchmark_ids={int(r[0]) for r in read(benchmark)}
    dump(OUT/'data_protocol.json',dict(script_sha256=sha(Path(__file__)),inventory_sha256=sha(src/'complete.json'),
        model=MODEL,train_pairs=TRAIN_PAIRS,heldout_pairs=NEW_PAIRS,nouns=NOUNS,
        quotas=dict(train=80,development=25,confirmation=50,heldout_pairs=50),
        source_partition='Deterministic SHA256 source-ID 40% train,15% development,45% confirmation; old training/guard/evaluation and every official WhatsUp COCO two-object source excluded',
        external_benchmark_manifest_sha256=sha(benchmark), external_benchmark_source_count=len(benchmark_ids),
        revision='v1 metadata overlap check found3 training and1 development source in external benchmark before any model scoring or training; v2 rebuild excludes entire benchmark from every bank, not only observed overlaps',
        selection='Hash order, decoded-mask geometry, source and byte-hash uniqueness; no model outcomes; no quota relaxation',
        renderer='Existing 256px canvas,112px cutouts, same natural object pixels at exchanged positions; no recoloring or mirroring',
        semantics='Two mutually exclusive relational captions; equivalent reversed-noun paraphrases pooled within each class, never treated as negatives',
        independent_test='Official WhatsUp left/right test categories, untouched until recipe freeze',
        prediction='Frozen development diagnosis saved before repair launch; confirm outcomes not read during development',
        training_plan=dict(seed=42,arms=['F','R','IS'],rank=64,alpha=64,dropout=.05,lr=.0002,weight_decay=.01,
                           epochs=36,batch_anchors=24,selection='fixed_last',max_diagnosis_driven_revisions=1),
        bounded_validation='First 16 hash-ordered training examples, fixed 2-state gallery; visual check before encoder scoring',
        gpu=False))
    pools=defaultdict(list)
    for r in lines(src/'eligible_single_objects.jsonl'):
        if r['noun'] in NOUNS and r['image_id'] not in benchmark_ids:pools[(partition(r['image_id']),r['noun'])].append(r)
    for key in pools:pools[key].sort(key=lambda r:order('candidate',r['image_id'],r['ann_id']))
    used=set();hashes=set(read(src/'exclusions.json')['image_hashes']);rows=[];reject=Counter()
    tasks=[('train',TRAIN_PAIRS,80),('development',TRAIN_PAIRS,25),('confirmation',TRAIN_PAIRS,50),('heldout_pairs',NEW_PAIRS,50)]
    for bank,pairs,n in tasks:
        split='confirmation' if bank=='heldout_pairs' else bank
        # All pairs need identical quota. Shared-source uniqueness applies globally.
        for pair in pairs:
            selected={}
            for noun in pair:
                selected[noun]=[]
                for r in pools[(split,noun)]:
                    iid=r['image_id']
                    if iid in used:continue
                    path=COCO/r['coco_split']/r['file'];digest=sha(path)
                    if digest in hashes:reject['identical_bytes']+=1;continue
                    _,mask,_=source(r)
                    if not .02<=mask.mean()<=.60:reject['decoded_mask_area']+=1;continue
                    selected[noun].append(dict(r,image_sha256=digest));used.add(iid);hashes.add(digest)
                    if len(selected[noun])==n:break
                if len(selected[noun])!=n:
                    dump(OUT/'construction_shortage.json',dict(bank=bank,pair=pair,noun=noun,needed=n,found=len(selected[noun])))
                    raise ValueError('Insufficient disjoint spatial sources; inspect metadata before changing protocol')
            for j in range(n):
                ss=[selected[k][j] for k in pair]
                rows.append(dict(anchor_id='spatial_'+order(bank,[(s['image_id'],s['ann_id']) for s in ss])[:20],
                    bank=bank,family='spatial',objects=pair,sources=ss,source_ids=[s['image_id'] for s in ss],
                    source_image_sha256={str(COCO/s['coco_split']/s['file']):s['image_sha256'] for s in ss}))
        print('SPATIAL_CONSTRUCTED',bank,sum(r['bank']==bank for r in rows),flush=True)
    jsonl(OUT/'rows.jsonl',rows);dump(OUT/'rejections.json',dict(reject))
    preview=sorted((r for r in rows if r['bank']=='train'),key=lambda r:order('gallery',r['anchor_id']))[:16]
    for page in range(2):
        panel=Image.new('RGB',(1024,8*154),'white');painter=ImageDraw.Draw(panel)
        for j,r in enumerate(preview[page*8:(page+1)*8]):
            images,checks=draw(r)
            for k,image in enumerate(images):panel.paste(Image.fromarray(image).resize((128,128)),(k*140,j*154))
            painter.text((286,j*154+12),r['anchor_id']+'\n'+' / '.join(r['objects'])+'\nfirst noun LEFT | first noun RIGHT',fill='black')
        with (OUT/f'gallery_{page}.jpg').open('xb') as f:panel.save(f,quality=90)
    jsonl(OUT/'gallery_selection.jsonl',preview)
    dump(OUT/'data_complete.json',dict(files={str(OUT/n):sha(OUT/n) for n in ['data_protocol.json','rows.jsonl','rejections.json','gallery_selection.jsonl','gallery_0.jpg','gallery_1.jpg']},
        counts=dict(Counter(r['bank'] for r in rows)),sources=len(used),source_overlap=0,no_scores=True))


def encode(banks):
    verify_files(read(OUT/'data_complete.json')['files'])
    if not (OUT/'visual_check.json').is_file():raise RuntimeError('Bounded score-blind visual check required')
    torch.set_num_threads(8);cv2.setNumThreads(1)
    scorer=load_subject(MODEL,device='cpu');start=time.monotonic()
    assert scorer.device.type=='cpu' and next(scorer.model.parameters()).device.type=='cpu'
    rows=lines(OUT/'rows.jsonl')
    for bank in banks:
        target=OUT/'features'/bank;target.mkdir(parents=True,exist_ok=False)
        selected=[r for r in rows if r['bank']==bank];pairs=sorted(set(tuple(r['objects']) for r in selected))
        groups=[g for p in pairs for g in captions(p)];tt=scorer.encode_texts(groups,batch_size=32).numpy().reshape(len(pairs),2,-1)
        vv=[];texts=[];checks=[]
        for begin in range(0,len(selected),8):
            batch=selected[begin:begin+8];ims=[]
            for r in batch:
                verify_files(r['source_image_sha256']);im,ch=draw(r);ims+=im
                texts.append(tt[pairs.index(tuple(r['objects']))]);checks.append(dict(anchor_id=r['anchor_id'],checks=ch))
            vv.extend(scorer.encode_images(ims,batch_size=16).numpy().reshape(len(batch),2,-1))
            if begin%80==0:print('SPATIAL_CPU_ENCODE',bank,begin+len(batch),len(selected),'seconds',round(time.monotonic()-start,1),flush=True)
        for name,x in [('images',vv),('texts',texts)]:
            with (target/f'{name}.npy').open('xb') as f:np.save(f,np.asarray(x,np.float32))
        jsonl(target/'rows.jsonl',selected);jsonl(target/'pixel_checks.jsonl',checks)
        dump(target/'complete.json',dict(files={str(p):sha(p) for p in target.iterdir() if p.is_file()},bank=bank,model=MODEL,
            registry_sha256=sha(DEFAULT_REGISTRY),data_sha256=sha(OUT/'data_complete.json'),encoder_device='cpu',precision='fp32',
            n_anchors=len(selected),elapsed_seconds=time.monotonic()-start,model_scores_formed=False))


def measurements(v,t):
    scores=np.asarray(v,float)@np.asarray(t,float).transpose(0,2,1)
    margins=np.stack((scores[:,0,0]-scores[:,0,1],scores[:,1,1]-scores[:,1,0]),1)
    e=margins.mean(1);b=(margins[:,0]-margins[:,1])/2
    dv=v[:,0]-v[:,1];dt=t[:,0]-t[:,1];nv=np.linalg.norm(dv,axis=-1);nt=np.linalg.norm(dt,axis=-1)
    cos=np.divide(2*e,nv*nt,out=np.full_like(e,np.nan),where=nv*nt>1e-12)
    return dict(response_raw=e,preference_raw=b,absolute_preference_raw=abs(b),surplus_raw=e-abs(b),
        exchange_accuracy=(margins>0).mean(1),both_correct=(margins>0).all(1).astype(float),
        image_difference_norm=nv,text_difference_norm=nt,alignment=cos,
        regime=np.where(e<=0,'nonpositive_response',np.where(e<=abs(b),'preference_dominated','both_correct')))


def diagnose():
    dest=OUT/'frozen_diagnosis';dest.mkdir(parents=True,exist_ok=False)
    records=[]
    for bank in ('train','development'):
        path=OUT/'features'/bank;verify_files(read(path/'complete.json')['files'])
        v,t=[np.load(path/f'{k}.npy') for k in ('images','texts')];rr=lines(path/'rows.jsonl');met=measurements(v,t)
        for i,r in enumerate(rr):records.append(dict(anchor_id=r['anchor_id'],bank=bank,group='+'.join(r['objects']),
            source_ids=r['source_ids'],**{k:str(a[i]) if k=='regime' else float(a[i]) for k,a in met.items()}))
    jsonl(dest/'per_example.jsonl',clean(records))
    summary={}
    for bank in ('train','development'):
        rr=[r for r in records if r['bank']==bank]
        summary[bank]=dict(n=len(rr),means={k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ('anchor_id','bank','group','source_ids','regime')},
            regimes=dict(Counter(r['regime'] for r in rr)))
    dump(dest/'summary.json',clean(summary));dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},gpu=False,confirmation_scores=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','encode','diagnose']);p.add_argument('--banks',nargs='+',default=['train','development']);a=p.parse_args()
    cv2.setNumThreads(1);log(ROOTOUT,'start',stage='spatial_'+a.action)
    try:
        if a.action=='encode':encode(a.banks)
        else:globals()[a.action]()
    except BaseException as e:log(ROOTOUT,'failed',stage='spatial_'+a.action,error=repr(e));raise
    log(ROOTOUT,'complete',stage='spatial_'+a.action)


if __name__=='__main__':main()
