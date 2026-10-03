"""Plan 39: score-blind nuisance changes and a complete noun-pair gallery."""
import argparse
from collections import defaultdict
import itertools
from pathlib import Path

import numpy as np
from PIL import Image; from PIL import ImageDraw
import torch

from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding import targeted_suppression as train
from mirror.cases.color_binding import indirect_generalization as ind
from mirror.cases.color_binding import ranking_transfer_extension as previous
from mirror.cases.color_binding.rendering import canvas_pair; from mirror.cases.color_binding.rendering import _rgba; from mirror.cases.color_binding.rendering import captions
from mirror.cases.color_binding.transparency_development import recolor
from mirror.cases.color_binding.generators import pixel_hash; from mirror.cases.color_binding.generators import edit_checks
from mirror.cases.color_binding.repair_trainbank import source; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding.behavioral_pilot import lines

DEST=train.OUT/'transfer'
BACKGROUNDS={'gray':(128,128,128),'dark':(45,45,45),'red':(200,60,60),'blue':(60,90,200)}
GEOMETRY={'small80':(80,(128,128)),'small96':(96,(128,128)),
          'stagger_up':(96,(80,176)),'stagger_down':(96,(176,80))}
VIEWS=('canvas','swapped_canvas')
FAMILIES=('background_shift','geometry_shift','caption_gallery')


def packed_source(row):
    ss=[source(s) for s in row['sources']]
    h,w=max(s[0].shape[0] for s in ss),sum(s[0].shape[1] for s in ss)
    rgb=np.zeros((h,w,3),np.uint8);masks=[];left=0
    for image,mask,_ in ss:
        hh,ww=mask.shape;rgb[:hh,left:left+ww]=image
        m=np.zeros((h,w),bool);m[:hh,left:left+ww]=mask;masks.append(m);left+=ww
    return rgb,masks


def geometry_pair(rgb,masks,size,centers,swapped=False):
    canvas=Image.new('RGB',(256,256),(235,235,235));slots=[None,None]
    for pos,slot in enumerate((1,0) if swapped else (0,1)):
        obj=_rgba(rgb,masks[slot]);scale=size/max(obj.size)
        obj=obj.resize((max(1,int(obj.width*scale)),max(1,int(obj.height*scale))),Image.Resampling.LANCZOS)
        x=(64,192)[pos]-obj.width//2;y=centers[pos]-obj.height//2
        assert x>=0 and y>=0 and x+obj.width<=256 and y+obj.height<=256
        canvas.paste(obj,(x,y),obj)
        m=np.zeros((256,256),bool);m[y:y+obj.height,x:x+obj.width]=np.asarray(obj.getchannel('A'))>0
        slots[slot]=m
    assert all(m.any() for m in slots) and not np.any(slots[0]&slots[1])
    return np.asarray(canvas).copy(),slots


def conditions(family):
    if family=='background_shift':return list(BACKGROUNDS)
    if family=='geometry_shift':return list(GEOMETRY)
    raise ValueError(family)


def render(row,family):
    rgb,masks=packed_source(row);images=[];names=[];checks=[]
    for cond in conditions(family):
        for view in VIEWS:
            if family=='background_shift':
                original,slots=canvas_pair(rgb,masks,view=='swapped_canvas')
                union=slots[0]|slots[1];base=original.copy();base[~union]=BACKGROUNDS[cond]
                assert np.array_equal(base[union],original[union])
            else:
                size,centers=GEOMETRY[cond]
                base,slots=geometry_pair(rgb,masks,size,centers,view=='swapped_canvas')
            assert not np.any(slots[0]&slots[1]) and all(m.any() for m in slots)
            for colors in itertools.product(('red','blue'),repeat=2):
                image=base.copy();name=cond+'/'+view+'/'+'_'.join(colors)
                for slot,(mask,color) in enumerate(zip(slots,colors)):
                    before=image;image=recolor(image,mask,color,.9)
                    cc=edit_checks(before,image,mask,color,'rgb_blend')
                    assert cc['outside_unchanged']
                    checks.append(dict(state=name,slot=slot,area=int(mask.sum()),**cc))
                images.append(image);names.append(name)
    return images,names,[pixel_hash(im) for im in images],checks


def gallery_entries(rows):
    vocab=list(dict.fromkeys(n for r in rows for n in r['objects']))
    assert len(vocab)==8
    result=[]
    for pairid,pair in enumerate(itertools.combinations(vocab,2)):
        for state,group in enumerate(captions('routing',pair,('red','blue'))):
            result.append(dict(index=len(result),pair_index=pairid,objects=list(pair),state=state,templates=group))
    assert len(result)==112 and len({tuple(r['templates']) for r in result})==112
    return result


def freeze():
    rows=lines(ind.CONFIRM/'rows.jsonl');training=lines(ind.TRAIN/'training_rows.jsonl')
    assert len(rows)==400 and len({i for r in rows for i in r['source_ids']})==800
    used={s for r in training for s in r['source_ids']}|{r['natural_guard']['image_id'] for r in training}
    assert not used.intersection(s for r in rows for s in r['source_ids'])
    jsonl(DEST/'rows.jsonl',[dict(r,source_cluster=j) for j,r in enumerate(rows)])
    entries=gallery_entries(rows);dump(DEST/'caption_gallery.json',entries)
    paths=[train.PLAN,Path(__file__),Path(__file__).parent/'tests/test_targeted_transfer.py',
        DEST/'rows.jsonl',DEST/'caption_gallery.json',ind.CONFIRM/'rows.jsonl',ind.TRAIN/'training_rows.jsonl',
        DEFAULT_REGISTRY,DEFAULT_REGISTRY.with_name('activation_addendum.json')]
    paths += [Path(__file__).with_name(n+'.py') for n in ('rendering','repair_trainbank',
        'transparency_development','rendering_v2','feature_cache','scorers','generators')]
    hashes={p:h for r in rows for p,h in r['source_image_sha256'].items()};verify_files(hashes)
    dump(DEST/'protocol.json',dict(inputs={str(p):sha(p) for p in paths},source_hashes=hashes,
        backgrounds=BACKGROUNDS,geometry=GEOMETRY,families=FAMILIES,views=VIEWS,
        tint_strength=.9,anchors=400,source_images=800,source_clusters=400,
        new_states=25600,caption_gallery_size=112,draws=10000,
        primary_contrast='IS_templates minus development-selected plain template ranking',
        primary_metrics=dict(background_shift='exchange_accuracy',geometry_shift='exchange_accuracy',caption_gallery='gallery_top1'),
        multiple_family_interval=.9833333333333333,all_conditions_retained=True,
        test_outcomes_seen=False,rows_shared_with_prior_confirmation=True))
    print('TRANSFER_FROZEN',sha(DEST/'protocol.json'),flush=True)


def verify():
    p=read(DEST/'protocol.json');verify_files(p['inputs']);return p


def gallery():
    verify();ind.configure();annotations_for('train2017')
    rows=lines(DEST/'rows.jsonl')
    selected=[min([r for r in rows if tuple(r['objects'])==pair],key=lambda r:r['anchor_id'])
        for pair in sorted({tuple(r['objects']) for r in rows})]
    records=[]
    for family in FAMILIES[:2]:
        page=Image.new('RGB',(4*196,4*222),'white');draw=ImageDraw.Draw(page)
        for j,row in enumerate(selected):
            ii,nn,hh,cc=render(row,family)
            for col,i in enumerate((1,9,17,25)):
                page.paste(Image.fromarray(ii[i]).resize((196,196)),(196*col,222*j+14))
                draw.text((196*col+2,222*j+2),'/'.join(row['objects']),fill='black')
                draw.text((196*col+2,222*j+208),nn[i].split('/')[0]+' red/blue',fill='black')
            records.append(dict(family=family,anchor_id=row['anchor_id'],
                indices=[1,9,17,25],hashes=[hh[i] for i in (1,9,17,25)]))
        with (DEST/(family+'_gallery.png')).open('xb') as f:page.save(f,format='PNG')
    jsonl(DEST/'gallery_selection.jsonl',records)


def cached_gallery():
    """Reuse exact frozen pooled embeddings; do not invent text-side averaging."""
    source_paths=(ind.CONFIRM/'features'/ind.MODEL,previous.DEST/'noun_recombination')
    mapping={};inputs={}
    for path in source_paths:
        verify_cache(path)
        feature=np.load(path/'texts.npy')
        for r in lines(path/'text_groups.jsonl'):
            key=tuple(r['templates'])
            if key in mapping:assert np.max(abs(mapping[key]-feature[r['index']]))<2e-6
            mapping[key]=feature[r['index']]
        inputs[str(path/'texts.npy')]=sha(path/'texts.npy')
    entries=read(DEST/'caption_gallery.json')
    vectors=np.stack([mapping[tuple(r['templates'])] for r in entries])
    with (DEST/'gallery_features.npy').open('xb') as f:np.save(f,vectors)
    return dict(inputs=inputs,features_sha256=sha(DEST/'gallery_features.npy'))


def encode():
    verify();ind.configure();review=read(DEST/'visual_review.json');assert review['accepted']
    verify_files(review['files']);train.cn.available_gpu()
    if torch.cuda.mem_get_info()[0]<12*1024**3:raise RuntimeError('Not enough free GPU memory; no other jobs touched')
    scorer=load_subject(ind.MODEL,device='cuda');annotations_for('train2017')
    for family in FAMILIES[:2]:
        dest=DEST/family
        if (dest/'complete.json').exists():verify_cache(dest);continue
        rows=lines(DEST/'rows.jsonl');checks=[]
        def renderer(row):
            ii,nn,hh,cc=render(row,family)
            checks.append(dict(anchor_id=row['anchor_id'],outside_unchanged=all(c['outside_unchanged'] for c in cc),
                pixel_qualified=sum(c['passes'] for c in cc),n_checks=len(cc),
                min_mask_area=min(c['area'] for c in cc),disjoint_visible_masks=True))
            if len(checks)%100==0:print('RENDERED',family,len(checks),'/',len(rows),flush=True)
            return ii,nn,hh
        cache_bank(dest,rows,scorer,renderer,lambda f,n:captions(f,n,('red','blue')),
            registry_path=DEFAULT_REGISTRY,input_hashes={str(DEST/n):sha(DEST/n) for n in ('protocol.json','visual_review.json')},
            image_batch_size=64,text_batch_size=128,render_workers=8,
            details=dict(family=family,conditions=conditions(family),no_exclusions=True))
        jsonl(DEST/(family+'_pixel_checks.jsonl'),sorted(checks,key=lambda r:r['anchor_id']))
        print('ENCODED',family,flush=True)
    gallery=cached_gallery()
    dump(DEST/'encoding_complete.json',dict(files={str(DEST/f/'complete.json'):sha(DEST/f/'complete.json') for f in FAMILIES[:2]},
        gallery=gallery,protocol_sha256=sha(DEST/'protocol.json')))


def arrays(family,condition,view):
    path=DEST/family;idx=lines(path/'index.jsonl');images=np.load(path/'images.npy',mmap_mode='r');texts=np.load(path/'texts.npy')
    names=[condition+'/'+view+'/'+a+'_'+b for a,b in itertools.product(('red','blue'),repeat=2)]
    v=np.stack([images[r['image_offset']+np.array([r['state_names'].index(n) for n in names])] for r in idx])
    t=texts[np.array([r['text_indices'] for r in idx])]
    rows=lines(DEST/'rows.jsonl');assert [r['anchor_id'] for r in rows]==[r['anchor_id'] for r in idx]
    return v,t,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','gallery','encode'))
    args=p.parse_args();log(train.OUT,'start',stage='transfer_'+args.action)
    try:globals()[args.action]()
    except BaseException as e:log(train.OUT,'failed',stage='transfer_'+args.action,error=repr(e));raise
    log(train.OUT,'complete',stage='transfer_'+args.action)


if __name__=='__main__':main()
