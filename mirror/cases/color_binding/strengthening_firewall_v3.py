"""Pre-benchmark source firewall correction; immutable test lists, cached reuse.

No model scores are read here. All COCO2017 validation sources and all mapped
ARO sources are excluded from new training/development/regularization sources.
"""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import shutil
import zipfile
import json
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA')
import numpy as np
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.strengthening_cpu import OUT
from mirror.cases.color_binding.behavioral_pilot import lines


def build():
    dest=OUT/'source_firewall_v3';dest.mkdir(exist_ok=False)
    val=ROOT/'clip/data/coco/annotations/instances_val2017.json'
    vg=ROOT/'clip/interbind_natural_20260922/source/image_data.json.zip'
    aro=ROOT/'clip/interbind_phase_bc_completion_20260923/preservation/manifests/aro/rows.jsonl'
    with zipfile.ZipFile(vg) as z:meta=json.loads(z.read('image_data.json'))
    mapping={int(r['image_id']):int(r['coco_id']) for r in meta if r.get('coco_id') is not None}
    vgids={int(Path(r['source_id']).stem) for r in lines(aro)}
    aro_ids={mapping[i] for i in vgids if i in mapping}
    val_ids={r['id'] for r in read(val)['images']};blocked=val_ids|aro_ids
    dump(dest/'protocol.json',dict(inputs={str(p):sha(p) for p in (val,vg,aro,Path(__file__))},
        no_scores_read=True,rule='Exclude every COCO2017 validation source plus every known COCO source mapped from full ARO; apply to training,development and natural-caption guards. Preserve all confirmation IDs.',
        coco_validation_sources=len(val_ids),aro_mapped_coco_sources=len(aro_ids),union_sources=len(blocked),
        unknown_cross_dataset_duplicates='Mapping is official VG-to-COCO metadata, not perceptual near-duplicate detection.'))
    dump(dest/'blocked_coco_ids.json',sorted(blocked))
    # Existing natural-caption cache: preserve only the rows whose source is clean.
    parent=ROOT/'clip/interbind_routing_repair_pilot_20260923'
    oldrows=lines(parent/'training_rows.jsonl');guard=np.load(parent/'features/natural_texts.npy')
    take=[j for j,r in enumerate(oldrows) if int(r['natural_guard']['image_id']) not in blocked]
    with (dest/'natural_texts_clean.npy').open('xb') as f:np.save(f,guard[take])
    jsonl(dest/'natural_guard_rows.jsonl',[dict(original_index=j,source_id=oldrows[j]['natural_guard']['image_id']) for j in take])
    summary={}
    for name in ('spatial','routing_transfer'):
        old=OUT/(name+'_v2');new=OUT/(name+'_v3');new.mkdir()
        rows=lines(old/'rows.jsonl');kept=[];removed=[]
        for r in rows:
            if r['bank'] in ('train','development') and set(r['source_ids'])&blocked:
                removed.append(dict(anchor_id=r['anchor_id'],bank=r['bank'],source_ids=r['source_ids'],overlap=sorted(set(r['source_ids'])&blocked)))
            else:kept.append(r)
        if name=='routing_transfer':
            # Match support-bank counts BEFORE any transfer model has been trained.
            original=sorted([r for r in kept if r['bank']=='train' and r['stratum']=='original_vocabulary'],key=lambda r:r['anchor_id'])
            added=sorted([r for r in kept if r['bank']=='train' and r['stratum']=='expanded_vocabulary'],key=lambda r:r['anchor_id'])
            n=min(len(original),2*len(added))//8*8
            assert n>=192, 'Insufficient meaningful matched support after strict firewall'
            narrow=original[:n];broad=original[:n//2]+added[:n//2]
            used={r['anchor_id'] for r in narrow+broad}
            kept=[r for r in kept if r['bank']!='train' or r['anchor_id'] in used]
            dump(new/'training_assignments.json',dict(narrow=[r['anchor_id'] for r in narrow],broad=[r['anchor_id'] for r in broad]))
            pp=read(old/'protocol.json');pp['training'].update(narrow_anchors=n,broad_anchors=n,broad_composition=f'{n//2} shared narrow +{n//2} added-vocabulary anchors',
                anchor_color_blocks=2*n,updates_per_arm=36*math.ceil(2*n/24),budget_reason='Largest multiple-of8 permitted by score-blind clean-source availability; same count and update budget in every cell')
            pp.update(source_firewall_sha256=sha(dest/'protocol.json'),supersedes=old.name,confirmation_ids_unchanged=True)
            dump(new/'protocol.json',pp)
            (new/'validation').mkdir();check=read(old/'validation/visual_check.json')
            check.update(reuse='Same already-checked renderer and subset of original source rows; no new images selected by scores',parent_check_sha256=sha(old/'validation/visual_check.json'))
            dump(new/'validation/visual_check.json',check)
        else:
            pp=read(old/'data_protocol.json');counts=Counter(r['bank'] for r in kept)
            assert counts['train']>=500 and counts['development']>=150
            pp.update(source_firewall_sha256=sha(dest/'protocol.json'),supersedes=old.name,confirmation_ids_unchanged=True,
                actual_counts=dict(counts),source_correction='Filter all protected sources without replacing/relaxing masks or changing any test IDs; same frozen recipe,36 epochs, matched arm budgets')
            dump(new/'data_protocol.json',pp);check=read(old/'visual_check.json')
            check.update(reuse='Unchanged rendering, subset of previously checked source bank',parent_check_sha256=sha(old/'visual_check.json'))
            dump(new/'visual_check.json',check)
        jsonl(new/'rows.jsonl',kept);jsonl(new/'source_exclusions.jsonl',removed)
        frozen=[r for r in rows if r['bank'] not in ('train','development')]
        assert frozen==[r for r in kept if r['bank'] not in ('train','development')]
        filename='data_complete.json' if name=='spatial' else 'complete.json'
        dump(new/filename,dict(files={str(p):sha(p) for p in new.rglob('*') if p.is_file()},
            counts=dict(Counter(r['bank'] for r in kept)),parent_manifest_sha256=sha(old/'rows.jsonl'),protected_train_dev_overlap=0,
            unchanged_confirmation_ids=True,no_scores_read=True))
        summary[name]=dict(counts=dict(Counter(r['bank'] for r in kept)),removed_by_bank=dict(Counter(r['bank'] for r in removed)))
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},banks=summary,clean_guard_rows=len(take),original_guard_rows=len(oldrows),gpu=False))
    print('CLEAN_SOURCE_BANKS',summary,'guard rows',len(take),flush=True)


def spatial_cache(banks):
    old=OUT/'spatial_v2';new=OUT/'spatial_v3';rows=lines(new/'rows.jsonl')
    for bank in banks:
        src=old/'features'/bank;dst=new/'features'/bank
        verify_files(read(src/'complete.json')['files']);dst.mkdir(parents=True,exist_ok=False)
        oldrows=lines(src/'rows.jsonl');wanted={r['anchor_id'] for r in rows if r['bank']==bank};take=[j for j,r in enumerate(oldrows) if r['anchor_id'] in wanted]
        assert len(take)==len(wanted)
        for name in ('images','texts'):
            with (dst/f'{name}.npy').open('xb') as f:np.save(f,np.load(src/f'{name}.npy')[take])
        jsonl(dst/'rows.jsonl',[oldrows[j] for j in take]);jsonl(dst/'pixel_checks.jsonl',[r for r in lines(src/'pixel_checks.jsonl') if r['anchor_id'] in wanted])
        dump(dst/'complete.json',dict(files={str(p):sha(p) for p in dst.iterdir() if p.is_file()},bank=bank,model='openclip_laion_l14',
            parent_complete_sha256=sha(src/'complete.json'),data_sha256=sha(new/'data_complete.json'),retained_rows=len(take),no_new_encoder_calls=True,no_scores=True))


def transfer_cache():
    from mirror.core.features import verify_cache
    old=OUT/'routing_transfer_v2';new=OUT/'routing_transfer_v3';rows=lines(new/'rows.jsonl')
    for bank in ('train','development'):
        src=old/'features'/bank;dst=new/'features'/bank;verify_cache(src);dst.mkdir(parents=True,exist_ok=False)
        index=lines(src/'index.jsonl');wanted={r['anchor_id'] for r in rows if r['bank']==bank};selected=[r for r in index if r['anchor_id'] in wanted]
        assert len(selected)==len(wanted)
        images=np.load(src/'images.npy',mmap_mode='r');vv=[];offset=0
        for r in selected:
            vv.append(images[r['image_offset']:r['image_offset']+r['image_count']]);r['image_offset']=offset;offset+=r['image_count']
        with (dst/'images.npy').open('xb') as f:np.save(f,np.concatenate(vv))
        for name in ('texts.npy','text_groups.jsonl'):shutil.copyfile(src/name,dst/name)
        jsonl(dst/'index.jsonl',selected);p=read(src/'protocol.json')
        p['input_hashes'].update({str(new/'complete.json'):sha(new/'complete.json'),str(OUT/'source_firewall_v3/complete.json'):sha(OUT/'source_firewall_v3/complete.json')})
        p['details'].update(source_firewall_correction='metadata-only retained source subset; raw features unchanged',parent_complete_sha256=sha(src/'complete.json'))
        dump(dst/'protocol.json',p);dump(dst/'complete.json',dict(files={p.name:sha(p) for p in dst.iterdir() if p.is_file()},n_images=offset,no_scores_or_training=True,peak_cuda_bytes=None))
        verify_cache(dst)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','spatial_cache','transfer_cache']);p.add_argument('--banks',nargs='+',default=['train','development']);a=p.parse_args()
    log(OUT,'start',stage='firewall_v3_'+a.action)
    try:
        if a.action=='spatial_cache':spatial_cache(a.banks)
        else:globals()[a.action]()
    except BaseException as e:log(OUT,'failed',stage='firewall_v3_'+a.action,error=repr(e));raise
    log(OUT,'complete',stage='firewall_v3_'+a.action)


if __name__=='__main__':main()
