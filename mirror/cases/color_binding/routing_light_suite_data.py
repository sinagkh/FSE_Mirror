"""Fixed paper-test grid at 40% tint; original renderers/data remain read-only."""
import argparse
from functools import partial
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image; from PIL import ImageDraw
from mirror.cases.color_binding import routing_light_tint as pilot
from mirror.cases.color_binding import ranking_transfer_extension as rt
from mirror.cases.color_binding import targeted_transfer as tt
from mirror.cases.color_binding import indirect_generalization as ind
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding.rendering import captions
from mirror.cases.color_binding.transparency_development import recolor as blend

OUT=pilot.OUT/'paper_retest_v1'
FAMILIES=('background_shift','geometry_shift','color_recombination','noun_recombination')
ORIGINAL_CONDITIONS=rt.conditions
ALPHA=.4
DELETIONS={'no_cross':('cross',),'no_response':('response',),'no_preference':('preference',),
           'no_interaction':('binding','cross','response')}


def rows_path(family):
    return rt.DEST/'noun_rows.jsonl' if family=='noun_recombination' else pilot.CONFIRM/'rows.jsonl'


def conditions(family):
    if family in ('background_shift','geometry_shift'):return tt.conditions(family)
    return [(c,ALPHA) for c,_ in ORIGINAL_CONDITIONS(family)]


def install_strength(alpha):
    # These are process-local renderer arguments; never edit old scripts/caches.
    rt.conditions=lambda family:[(c,alpha) for c,_ in ORIGINAL_CONDITIONS(family)]
    tt.recolor=lambda rgb,mask,color,strength:blend(rgb,mask,color,alpha)


def render(row,family):
    return tt.render(row,family) if family in ('background_shift','geometry_shift') else rt.render(row,family)


def freeze():
    pilot.verify()
    verify_files({str(pilot.OUT/k):v for k,v in read(pilot.OUT/'complete.json')['files'].items()})
    files=[Path(__file__),Path(pilot.__file__),Path(rt.__file__),Path(tt.__file__),Path(ind.__file__),
           pilot.OUT/'complete.json',pilot.OUT/'final_verification.json',pilot.OUT/'models.json',
           pilot.OUT/'protocol.json',pilot.CONFIRM/'rows.jsonl',rt.DEST/'noun_rows.jsonl',
           tt.DEST/'caption_gallery.json',tt.DEST/'gallery_features.npy',
           ind.OUT/'template_features.npy',ind.OUT/'template_index.json',
           rt.DEST/'weaker_edits/complete.json',pilot.OUT/'features/test/complete.json',
           pilot.OUT/'features/train/complete.json',pilot.prior.OUT/'normalization.json']
    for family in FAMILIES:
        path=(tt.DEST if family.endswith('_shift') else rt.DEST)/family
        files += [path/'complete.json',path/'index.jsonl']
    files += [Path(r['checkpoint']) for r in pilot.registry() if r['checkpoint']]
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in files},seed=42,alpha=.4,
        author_request='Complete the paper-facing indirect tests and required controls on this seed before considering three seeds/replacement',
        families={f:conditions(f) for f in FAMILIES},
        additional_tests=['400 original source pairs: six unseen caption wordings, red-blue and green-yellow',
            '112-caption gallery on 400 original source pairs','archived 65% tint intermediate transfer',
            'full 16-context interactions, continuous distributions, clause directions, fix/break',
            'single-word-correct decision cohort, with full-cohort results retained',
            'four matched component deletions on seed42, all three color pairs',
            'frozen e/b diagnosis and component-specific gains across opposite layouts',
            'source-paired opacity difference-in-differences, absolute and frozen-relative suppression',
            'paired source intervals for saved complete SugarCrepe, ARO and bidirectional COCO results'],
        deletion_targets=DELETIONS,ablation_updates=1944,ablation_selection='fixed last; same weights and all guards',
        original_checkpoint_scores_known=True,new_transfer_scores_seen=False,
        diagnostic_cohort='For each mixed-color image, both single-color-word alternatives score below its correct caption; report whether the assignment swap still wins. No any-corner gate.',
        diagnosis_prediction='no_response costs more for frozen nonpositive e; no_preference costs more for positive e dominated by abs(b); label one layout and measure other layout',
        comparisons='Frozen, existing 40% IS/Ranking, archived 90% IS/Ranking; ablations on primary banks only',
        statistics=dict(draws=5000,seed=20260930,unit='connected source-image component; all conditions/layouts retained within source',
                        seed_variability='not estimable from one seed',transfer_family_ci='95% and Bonferroni 99.1667% for six named indirect families'),
        decisions='No automatic extra seeds, no new loss selection, no manuscript replacement. Report each fixed family and material trade-off before author decision.',
        artifacts='Append-only separate subdirectory; preserve parent and all first-round results'))
    dump(OUT/'protocol_hash.json',dict(sha256=sha(OUT/'protocol.json')))
    print('SUITE_FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    assert sha(OUT/'protocol.json')==read(OUT/'protocol_hash.json')['sha256']
    verify_files(read(OUT/'protocol.json')['inputs'])


def preview():
    verify();cv2.setNumThreads(1);annotations_for('train2017');annotations_for('val2017')
    dest=OUT/'preview';dest.mkdir(exist_ok=False);replays=[];selection=[]
    for family in FAMILIES:
        row=lines(rows_path(family))[0]
        oldpath=(tt.DEST if family.endswith('_shift') else rt.DEST)/family
        old=next(r for r in lines(oldpath/'index.jsonl') if r['anchor_id']==row['anchor_id'])
        install_strength(.9);_,names,hashes,_=render(row,family)
        assert names==old['state_names'] and hashes==old['pixel_sha256'],family
        replays.append(dict(family=family,anchor_id=row['anchor_id'],states=len(names),old90_replay_exact=True))
        install_strength(.4);ims,names,hashes,checks=render(row,family)
        assert all(c['outside_unchanged'] for c in checks)
        cells=[i for i,n in enumerate(names) if '/canvas/' in n and n.endswith('_blue')][:4]
        if not cells:cells=list(range(min(4,len(ims))))
        page=Image.new('RGB',(256*len(cells),288),'white');draw=ImageDraw.Draw(page)
        for j,i in enumerate(cells):
            page.paste(Image.fromarray(ims[i]),(j*256,25));draw.text((j*256+4,4),names[i],fill='black')
        page.save(dest/(family+'.png'))
        selection.append(dict(family=family,anchor_id=row['anchor_id'],indices=cells,states=[names[i] for i in cells]))
    # Exact 40% corners, not an edited screenshot or generated illustration.
    selected=sorted([r for r in lines(pilot.TRAIN/'training_rows.jsonl') if r['objects']==['car','boat']],key=lambda r:r['anchor_id'])[:3]
    page=Image.new('RGB',(4*256,6*286),'white');draw=ImageDraw.Draw(page)
    for j,row in enumerate(selected):
        ims,names,_,=pilot.render(row,pilot.COLORS[:1],.4)
        for v,view in enumerate(pilot.VIEWS):
            iy=2*j+v
            for k in range(4):
                page.paste(Image.fromarray(ims[4*v+k]),(k*256,iy*286+24))
                draw.text((k*256+4,iy*286+3),f'Example {j+1} / {view} / '+['RR','RB','BR','BB'][k],fill='black')
        selection.append(dict(family='primary_car_boat',anchor_id=row['anchor_id'],all_eight_states=True))
    page.save(dest/'car_boat_40_only.png')
    jsonl(dest/'selection.jsonl',selection);dump(dest/'replay.json',replays)
    print('SUITE_PREVIEW_PASS',flush=True)


def encode():
    verify();install_strength(.4);torch.set_num_threads(4);cv2.setNumThreads(1)
    assert (OUT/'preview/review.json').exists()
    annotations_for('train2017');annotations_for('val2017')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info()[0]>12*1024**3,'GPU busy; no other jobs touched'
    scorer=load_subject(pilot.prior.MODEL,device='cuda')
    for family in FAMILIES:
        dest=OUT/'features'/family
        if (dest/'complete.json').exists():verify_cache(dest);continue
        checks=[];rows=lines(rows_path(family))
        def renderer(row):
            ims,names,hashes,cc=render(row,family)
            checks.append(dict(anchor_id=row['anchor_id'],outside_unchanged=all(c['outside_unchanged'] for c in cc),
                               hue_check_passes=sum(c['passes'] for c in cc),n_checks=len(cc)))
            assert checks[-1]['outside_unchanged']
            if len(checks)%100==0:print('RENDERED',family,len(checks),'/',len(rows),flush=True)
            return ims,names,hashes
        colors=[('red','blue')] if family.endswith('_shift') else [tuple(c.split('-')) for c,_ in conditions(family)]
        def prompts(f,objects):return [g for c in colors for g in captions(f,objects,c)]
        cache_bank(dest,rows,scorer,renderer,prompts,registry_path=DEFAULT_REGISTRY,
            input_hashes={str(OUT/'protocol.json'):sha(OUT/'protocol.json'),str(OUT/'preview/review.json'):sha(OUT/'preview/review.json'),
                          str(rows_path(family)):sha(rows_path(family))},image_batch_size=64,text_batch_size=128,render_workers=8,
            details=dict(family=family,opacity=.4,conditions=conditions(family),no_exclusions=True))
        jsonl(OUT/'features'/(family+'_pixel_checks.jsonl'),sorted(checks,key=lambda r:r['anchor_id']))
        print('ENCODED',family,read(dest/'complete.json')['elapsed_seconds'],flush=True)
    dump(OUT/'encoding_complete.json',dict(files={str(OUT/'features'/f/'complete.json'):sha(OUT/'features'/f/'complete.json') for f in FAMILIES}))


def arrays(family,condition,view):
    path=OUT/'features'/family;idx=lines(path/'index.jsonl');images=np.load(path/'images.npy',mmap_mode='r');texts=np.load(path/'texts.npy')
    if family.endswith('_shift'):
        colors=('red','blue');prefix=condition;ci=0
    else:
        color,strength=condition;colors=tuple(color.split('-'));prefix=rt.condition_name(color,strength)
        ci=[c for c,_ in conditions(family)].index(color)
    names=[prefix+'/'+view+'/'+a+'_'+b for a in colors for b in colors]
    v=np.stack([images[r['image_offset']+np.array([r['state_names'].index(n) for n in names])] for r in idx])
    t=texts[np.array([r['text_indices'][ci*4:ci*4+4] for r in idx])]
    rows=lines(rows_path(family));assert [r['anchor_id'] for r in rows]==[r['anchor_id'] for r in idx]
    return v,t,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','preview','encode']);a=ap.parse_args()
    log(OUT,'start',stage=a.action)
    try:globals()[a.action]()
    except BaseException as exc:log(OUT,'failed',stage=a.action,error=repr(exc));raise
    log(OUT,'complete',stage=a.action)


if __name__=='__main__':main()
