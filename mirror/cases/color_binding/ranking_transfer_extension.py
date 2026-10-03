"""Three outcome-independent transfer families; no training or score-based subsets."""
import argparse
from collections import Counter; from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.ranking_followup_pilot import OUT; from mirror.cases.color_binding.ranking_followup_pilot import PLAN; from mirror.cases.color_binding.ranking_followup_pilot import old_registry
from mirror.cases.color_binding import indirect_generalization as previous
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.core.metrics import adapt; from mirror.core.metrics import routing
from mirror.cases.color_binding.strengthening_statistics import intervals

DEST=OUT/'transfer'
NEW_COLORS=(('red','green'),('red','yellow'),('blue','green'),('blue','yellow'))
VIEWS=('canvas','swapped_canvas')
METRICS=('exchange_accuracy','caption_accuracy','binding','cross','response','preference',
         'word1_accuracy','word2_accuracy','both_correct','reverse_image_choice')


def complete_recombination(rows):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r['objects'])].append(r)
    groups={k:sorted(v,key=lambda r:r['anchor_id']) for k,v in groups.items()}
    assert len(groups)==4 and {len(v) for v in groups.values()}=={100}
    vocab=list(dict.fromkeys(n for pair in groups for n in pair))
    assert len(vocab)==8
    sources={};hashes={}
    for pair,group in groups.items():
        for slot,noun in enumerate(pair):
            sources[noun]=[r['sources'][slot] for r in group]
        for r in group:hashes.update(r['source_image_sha256'])
    seen={frozenset(p) for p in groups}
    pairs=[p for p in itertools.combinations(vocab,2) if frozenset(p) not in seen]
    assert len(pairs)==24
    result=[]
    for pair in pairs:
        for j in range(100):
            ss=[sources[n][j] for n in pair]
            key=[(s['image_id'],s['ann_id']) for s in ss]
            aid=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
            hs={s['image_sha256'] for s in ss}
            result.append(dict(anchor_id='allpairs_'+aid,family='routing',objects=list(pair),
                sources=ss,source_ids=[s['image_id'] for s in ss],source_cluster=j,
                source_image_sha256={p:h for p,h in hashes.items() if h in hs}))
    assert len({r['anchor_id'] for r in result})==2400
    # Each source belongs to one and only one resampling block.
    ids=defaultdict(set)
    for r in result:
        for s in r['source_ids']:ids[s].add(r['source_cluster'])
    assert len(ids)==800 and all(len(v)==1 for v in ids.values())
    return result


def conditions(family):
    if family=='noun_recombination':return [('red-blue',.9)]
    if family=='color_recombination':return [('-'.join(c),.9) for c in NEW_COLORS]
    if family=='weaker_edits':return [('red-blue',s) for s in (.4,.65)]
    raise ValueError(family)


def condition_name(color,strength):return f'{color}_s{round(strength*100):03d}'


def freeze():
    rows=lines(previous.CONFIRM/'rows.jsonl');combined=complete_recombination(rows)
    train=lines(previous.TRAIN/'training_rows.jsonl')
    training_pairs={frozenset(r['objects']) for r in train}
    audit_pairs={frozenset(r['objects']) for r in rows}
    audit_nouns={n for r in rows for n in r['objects']}
    assert audit_pairs <= training_pairs
    assert {p for p in training_pairs if p <= audit_nouns}==audit_pairs
    assert {frozenset(r['objects']) for r in combined}.isdisjoint(training_pairs)
    used={s for r in train for s in r['source_ids']}|{r['natural_guard']['image_id'] for r in train}
    assert not used.intersection(s for r in rows for s in r['source_ids'])
    jsonl(DEST/'noun_rows.jsonl',combined)
    jsonl(DEST/'original_rows.jsonl',[dict(r,source_cluster=j) for j,r in enumerate(rows)])
    files=[PLAN,Path(__file__),Path(__file__).with_name('transparency_development.py'),
           previous.CONFIRM/'rows.jsonl',previous.TRAIN/'training_rows.jsonl',
           DEST/'noun_rows.jsonl',DEST/'original_rows.jsonl',DEFAULT_REGISTRY,
           DEFAULT_REGISTRY.with_name('activation_addendum.json')]
    files += [Path(__file__).with_name(n+'.py') for n in
              ('rendering','rendering_v2','repair_trainbank','feature_cache','scorers','generators')]
    source_hashes={p:h for r in rows for p,h in r['source_image_sha256'].items()}
    verify_files(source_hashes)
    dump(DEST/'protocol.json',dict(inputs={str(p):sha(p) for p in files},source_hashes=source_hashes,
        original_models=old_registry(),families={f:conditions(f) for f in
            ('noun_recombination','color_recombination','weaker_edits')},views=VIEWS,
        counts=dict(noun_composites=2400,noun_source_clusters=100,source_images=800,original_anchors=400),
        paired=True,source_shared_with_previous_tests=True,draws=10000,
        primary='IS minus R_s100, exchange accuracy, all conditions and layouts in each family',
        intervals='95% pointwise plus 98.3333% for three family contrasts',
        no_outcome_selection=True,new_outcomes_seen=False,new_training_uses_this_bank=False))
    print('TRANSFER_PROTOCOL_FROZEN',sha(DEST/'protocol.json'),flush=True)


def verify():
    p=read(DEST/'protocol.json');verify_files(p['inputs']);return p


def render(row,family):
    from mirror.cases.color_binding.repair_trainbank import source
    from mirror.cases.color_binding.rendering import canvas_pair
    from mirror.cases.color_binding.transparency_development import recolor
    from mirror.cases.color_binding.generators import edit_checks; from mirror.cases.color_binding.generators import pixel_hash
    ss=[source(s) for s in row['sources']]
    h,w=max(s[0].shape[0] for s in ss),sum(s[0].shape[1] for s in ss)
    packed=np.zeros((h,w,3),np.uint8);masks=[];left=0
    for rgb,mask,_ in ss:
        hh,ww=mask.shape;packed[:hh,left:left+ww]=rgb
        mm=np.zeros((h,w),bool);mm[:hh,left:left+ww]=mask;masks.append(mm);left+=ww
    images=[];names=[];checks=[]
    for color,strength in conditions(family):
        for view in VIEWS:
            base,slots=canvas_pair(packed,masks,view=='swapped_canvas')
            assert not np.any(slots[0]&slots[1])
            for cc in itertools.product(color.split('-'),repeat=2):
                image=base.copy();name=condition_name(color,strength)+'/'+view+'/'+('_'.join(cc))
                for slot,(mask,c) in enumerate(zip(slots,cc)):
                    before=image;image=recolor(image,mask,c,strength)
                    check=edit_checks(before,image,mask,c,'rgb_blend')
                    assert check['outside_unchanged']
                    checks.append(dict(state=name,slot=slot,color=c,**check))
                images.append(image);names.append(name)
    return images,names,[pixel_hash(im) for im in images],checks


def gallery():
    from PIL import Image; from PIL import ImageDraw
    from mirror.cases.color_binding.repair_trainbank import annotations_for
    verify();previous.configure();annotations_for('train2017')
    rows=lines(DEST/'original_rows.jsonl')
    chosen=[next(r for r in rows if tuple(r['objects'])==p) for p in sorted({tuple(r['objects']) for r in rows})]
    selected=[]
    for family in ('color_recombination','weaker_edits'):
        count=len(conditions(family))*2
        page=Image.new('RGB',(count*150,4*184),'white');draw=ImageDraw.Draw(page)
        for j,row in enumerate(chosen):
            images,names,hashes,_=render(row,family)
            indices=[8*k+i for k in range(len(conditions(family))) for i in (1,2)]
            draw.text((3,j*184+2),' / '.join(row['objects']),fill='black')
            for col,i in enumerate(indices):
                page.paste(Image.fromarray(images[i]).resize((150,150)),(col*150,j*184+18))
                draw.text((col*150+2,j*184+170),names[i].replace('/canvas/',' '),fill='black')
            selected.append(dict(family=family,anchor_id=row['anchor_id'],indices=indices,pixel_hashes=[hashes[i] for i in indices]))
        with (DEST/(family+'_gallery.png')).open('xb') as f:page.save(f,format='PNG')
    # First ordinal of every new pairing, selected without scores.
    new=[r for r in lines(DEST/'noun_rows.jsonl') if r['source_cluster']==0]
    page=Image.new('RGB',(6*170,4*190),'white');draw=ImageDraw.Draw(page)
    for j,row in enumerate(new):
        images,names,hashes,_=render(row,'noun_recombination');x=(j%6)*170;y=(j//6)*190
        page.paste(Image.fromarray(images[1]).resize((170,170)),(x,y+18))
        draw.text((x+2,y+2),'/'.join(row['objects']),fill='black')
        selected.append(dict(family='noun_recombination',anchor_id=row['anchor_id'],indices=[1],pixel_hashes=[hashes[1]]))
    with (DEST/'noun_gallery.png').open('xb') as f:page.save(f,format='PNG')
    jsonl(DEST/'gallery_selection.jsonl',selected)


def encode():
    from mirror.cases.color_binding.repair_trainbank import annotations_for
    from mirror.cases.color_binding.rendering import captions
    verify();previous.configure()
    review=read(DEST/'visual_review.json');assert review['accepted']
    verify_files(review['files'])
    if torch.cuda.mem_get_info()[0]<12*1024**3:raise RuntimeError('GPU busy; no other jobs interrupted')
    scorer=load_subject(previous.MODEL,device='cuda');annotations_for('train2017')
    for family in ('noun_recombination','color_recombination','weaker_edits'):
        dest=DEST/family
        if (dest/'complete.json').exists():verify_cache(dest);continue
        rows=lines(DEST/('noun_rows.jsonl' if family=='noun_recombination' else 'original_rows.jsonl'))
        checks=[]
        def renderer(row):
            ii,nn,hh,cc=render(row,family)
            checks.append(dict(anchor_id=row['anchor_id'],n_checks=len(cc),outside_unchanged=all(c['outside_unchanged'] for c in cc),
                pixel_qualified=sum(c['passes'] for c in cc),conditions={condition_name(color,s):dict(
                    checks=sum(c['state'].startswith(condition_name(color,s)+'/') for c in cc),
                    qualified=sum(c['passes'] and c['state'].startswith(condition_name(color,s)+'/') for c in cc))
                    for color,s in conditions(family)}))
            if len(checks)%100==0:print('RENDERED',family,len(checks),'/',len(rows),flush=True)
            return ii,nn,hh
        colors=list(dict.fromkeys(c for c,s in conditions(family)))
        def texts(f,nouns):return [g for color in colors for g in captions(f,nouns,tuple(color.split('-')))]
        cache_bank(dest,rows,scorer,renderer,texts,registry_path=DEFAULT_REGISTRY,
            input_hashes={str(DEST/n):sha(DEST/n) for n in ('protocol.json','visual_review.json')},
            image_batch_size=64,text_batch_size=128,render_workers=8,
            details=dict(family=family,conditions=conditions(family),no_score_dependent_exclusions=True))
        jsonl(DEST/(family+'_pixel_checks.jsonl'),sorted(checks,key=lambda r:r['anchor_id']))
        print('ENCODED',family,read(dest/'complete.json')['n_images'],flush=True)
    dump(DEST/'encoding_complete.json',dict(files={str(DEST/f/'complete.json'):sha(DEST/f/'complete.json') for f in
        ('noun_recombination','color_recombination','weaker_edits')},protocol_sha256=sha(DEST/'protocol.json')))


def arrays(family,color,strength,view):
    path=DEST/family;idx=lines(path/'index.jsonl');images=np.load(path/'images.npy',mmap_mode='r');texts=np.load(path/'texts.npy')
    colors=list(dict.fromkeys(c for c,s in conditions(family)));ci=colors.index(color)
    wanted=[condition_name(color,strength)+'/'+view+'/'+a+'_'+b for a in color.split('-') for b in color.split('-')]
    v=np.stack([images[r['image_offset']+np.array([r['state_names'].index(n) for n in wanted])] for r in idx])
    t=texts[np.array([r['text_indices'][ci*4:ci*4+4] for r in idx])]
    rows=lines(DEST/('noun_rows.jsonl' if family=='noun_recombination' else 'original_rows.jsonl'))
    assert [r['anchor_id'] for r in rows]==[r['anchor_id'] for r in idx]
    return v,t,rows


def measurements(x):
    m=routing(x)
    m['both_correct']=((x[:,1,1]>x[:,1,2])&(x[:,2,2]>x[:,2,1])).astype(float)
    m['reverse_image_choice']=np.stack((x[:,1,1]>x[:,2,1],x[:,2,2]>x[:,1,2]),1).mean(1)
    return np.stack([m[k] for k in METRICS],-1)


def comparisons(names):
    pairs=[('IS','F'),('IS','R_s100'),('IS','RG_s100_g0.25')]
    pairs += [('R_to_IS',b) for b in ('R_s100','R_to_R','R_to_G')]
    pairs += [('IS_templates',b) for b in ('R_templates','IS','R_s100')]
    return [(a,b) for a,b in pairs if a in names and b in names]


def summarize(values,metrics,meta,clusters):
    names=list(dict.fromkeys(k for k,s in values));out=[]
    for cohort,seeds in [('pilot_seed42',[42]),('three_seed',[42,43,44])]:
        available=[n for n in names if n=='F' or all((n,s) in values for s in seeds)]
        packed={n:np.stack([values[(n,0 if n=='F' else s)] for s in seeds]) for n in available}
        labels=list(packed.items())+[(a+' - '+b,packed[a]-packed[b]) for a,b in comparisons(available)]
        x=np.concatenate([a for _,a in labels],-1)
        stats=intervals(x,clusters,draws=10000)
        out += [dict(**meta,cohort=cohort,comparison=label,metric=metric,seed_ids=seeds,**stats[i*len(metrics)+j])
            for i,(label,_) in enumerate(labels) for j,metric in enumerate(metrics)]
    return out


def family_adjusted_ci(difference,clusters):
    x=np.asarray(difference,float);ns,n=x.shape;nc=int(max(clusters))+1
    counts=np.bincount(clusters,minlength=nc)
    sums=np.stack([np.bincount(clusters,weights=row,minlength=nc) for row in x])
    rng=np.random.default_rng(20260925);samples=[]
    for _ in range(100):
        w=rng.multinomial(nc,np.full(nc,1/nc),size=100)
        sw=rng.multinomial(ns,np.full(ns,1/ns),size=100)/ns
        per=(w@sums.T)/(w@counts)[:,None]
        samples.extend((per*sw).sum(1))
    return np.quantile(samples,[.05/6,1-.05/6]).tolist()


def score():
    verify();previous.configure();verify_files(read(DEST/'encoding_complete.json')['files'])
    models=read(OUT/'checkpoints_frozen.json');regs=models['original_models']+models['models']
    verify_files({r['checkpoint']:r['sha256'] for r in regs if r['checkpoint']})
    dest=DEST/'evaluation';dest.mkdir(exist_ok=False);all_summary=[];adjusted=[];index=[]
    for family in ('noun_recombination','color_recombination','weaker_edits'):
        accumulated=defaultdict(list);records=[]
        for color,strength in conditions(family):
            views=defaultdict(list)
            for view in VIEWS:
                v,t,rows=arrays(family,color,strength,view);values={};raw=[]
                for reg in regs:
                    x=np.einsum('nid,njd->nij',v,adapt(t,reg['checkpoint']),optimize=True)
                    a=measurements(x);key=(reg['name'],reg['seed']);values[key]=a
                    views[key].append(a);accumulated[key].append(a);raw.append(x)
                    records += [dict(name=reg['name'],seed=reg['seed'],condition=condition_name(color,strength),view=view,
                        anchor_id=r['anchor_id'],source_ids=r['source_ids'],source_cluster=r['source_cluster'],objects=r['objects'],
                        **dict(zip(METRICS,a[j].tolist()))) for j,r in enumerate(rows)]
                filename=family+'_'+condition_name(color,strength)+'_'+view+'.npz'
                with (dest/filename).open('xb') as f:np.savez_compressed(f,scores=np.stack(raw))
                index.append(dict(file=filename,models=[dict(name=r['name'],seed=r['seed']) for r in regs],
                    anchor_ids=[r['anchor_id'] for r in rows],axes=['model','anchor','image_state','caption_state']))
                print('SCORED',family,color,strength,view,flush=True)
            clusters=np.array([r['source_cluster'] for r in rows])
            meanviews={k:np.mean(a,axis=0) for k,a in views.items()}
            all_summary+=summarize(meanviews,METRICS,dict(family=family,condition=condition_name(color,strength)),clusters)
        values={k:np.mean(a,axis=0) for k,a in accumulated.items()}
        all_summary+=summarize(values,METRICS,dict(family=family,condition='all_conditions'),clusters)
        delta=np.stack([values[('IS',s)][:,0]-values[('R_s100',s)][:,0] for s in (42,43,44)])
        adjusted.append(dict(family=family,comparison='IS - R_s100',metric='exchange_accuracy',mean=float(delta.mean()),
            confidence=1-.05/3,ci=family_adjusted_ci(delta,clusters),family_size=3))
        jsonl(dest/(family+'_per_example.jsonl'),records)
        # Complete descriptive noun-pair table, not a search for favorable pairs.
        if family=='noun_recombination':
            pairrows=[]
            for pair in sorted({tuple(r['objects']) for r in rows}):
                mask=np.array([tuple(r['objects'])==pair for r in rows])
                for (name,seed),a in values.items():
                    pairrows.append(dict(objects=pair,name=name,seed=seed,anchors=int(mask.sum()),
                        **dict(zip(METRICS,a[mask].mean(0).tolist()))))
            jsonl(dest/'all_noun_pairs.jsonl',pairrows)
    jsonl(dest/'summary.jsonl',all_summary);dump(dest/'family_adjusted_intervals.json',adjusted);dump(dest/'score_index.json',index)
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},
        protocol_sha256=sha(DEST/'protocol.json'),checkpoint_registry_sha256=sha(OUT/'checkpoints_frozen.json')))
    print('TRANSFER_SCORING_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','gallery','encode','score'))
    args=p.parse_args();log(OUT,'start',stage='transfer_'+args.action)
    try:globals()[args.action]()
    except BaseException as e:log(OUT,'failed',stage='transfer_'+args.action,error=repr(e));raise
    log(OUT,'complete',stage='transfer_'+args.action)


if __name__=='__main__':main()
