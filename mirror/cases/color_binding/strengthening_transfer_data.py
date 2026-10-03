"""Outcome-independent feasible vocabulary/context factorial bank, CPU only."""
import argparse
from collections import Counter
import hashlib
import json
import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA')
from pathlib import Path
import cv2
import numpy as np
from PIL import Image; from PIL import ImageDraw
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.strengthening_cpu import OUT as ROOTOUT
from mirror.cases.color_binding.strengthening_spatial import partition
from mirror.cases.color_binding.repair_trainbank import COCO; from mirror.cases.color_binding.repair_trainbank import source; from mirror.cases.color_binding.repair_trainbank import lines
from mirror.cases.color_binding.banks_v2 import square_crop; from mirror.cases.color_binding.banks_v2 import crop_array
from mirror.cases.color_binding.rendering import canvas_pair
from mirror.cases.color_binding.rendering_v3 import recolor

OUT=ROOTOUT/'routing_transfer_v2'
NARROW={'bus','airplane','car','boat','cat','couch','dog','sports ball','person','bicycle'}
ADDED={'chair','cup','bowl','tv','umbrella','horse','potted plant','laptop','oven','refrigerator'}


def order(*args):return hashlib.sha256(json.dumps(['support-factorial-20260925',*args],sort_keys=True).encode()).hexdigest()


def stratum(objects):
    nouns=set(objects)
    if nouns<=NARROW:return 'original_vocabulary'
    if nouns<=NARROW|ADDED:return 'expanded_vocabulary'
    if not nouns&(NARROW|ADDED):return 'neither_in_training_vocabulary'
    return 'one_outside_training_vocabulary'


def build():
    cv2.setNumThreads(1);src=ROOTOUT/'data_feasibility';verify_files(read(src/'complete.json')['files'])
    OUT.mkdir(parents=True,exist_ok=False)
    counts=read(src/'counts.json');bench=ROOTOUT/'spatial/whatsup_coco_two_obj.json';excluded={int(r[0]) for r in read(bench)}
    dump(OUT/'protocol.json',dict(script_sha256=sha(Path(__file__)),inventory_sha256=sha(src/'complete.json'),
        original_exact_pair_counts=counts['narrow_pairs'],narrow_vocabulary=sorted(NARROW),added_vocabulary=sorted(ADDED),
        revision_before_scores='Original five pairs have84 eligible co-occurring images and one absent pair. The feasible new experiment varies vocabulary support, not fidelity to the five original pairs. New baseline and every cell retrained on this common protocol; no historical checkpoint substitutes for a cell.',
        construction='Same geometry as inventory; confirm decoded area2%-60%, non-overlapping masks; retain full natural scene with uniform235-gray letterboxing to square, preserving every source pixel and both complete masks. Fixed hash order; unique source image across banks',
        crop_revision_before_scores='The earlier no-padding1.25x square crop rejected294 of380 otherwise eligible original-vocabulary training sources, leaving81. Replace that lossy crop with deterministic full-scene letterboxing for every item; never relax masks or select based on scores.',
        partition='Shared deterministic40/15/45 train/development/confirmation source-ID split, with all prior study and WhatsUp sources excluded',
        training=dict(narrow_anchors=256,broad_anchors=256,broad_composition='128 fixed hash-selected narrow training anchors +128 new expanded-vocabulary training anchors',
            seed=42,colors=[['red','blue'],['green','yellow']],heldout_colors=['purple','orange'],
            cells=['narrow_canvas','narrow_mixed','broad_canvas','broad_mixed'],arms=['R','IS'],
            epochs=36,anchor_color_blocks=512,blocks_per_update=24,updates_per_arm=792,
            budget_reason='256 source-disjoint co-occurring anchors per cell is feasible under fixed geometry; same count, exposure budget and optimizer in all cells; not claimed equal to the earlier640-anchor experiment',
            mixed='Same source objects and captions. One canvas and one in-situ lattice per anchor/color block; canvas-only uses direct and swapped layouts. Two lattices per block in either setting.',
            ranking='Existing four-way CE plus0.2 embedding anchoring; no added IS guards',
            is_recipe='Existing final common-noise interaction+preference+preservation recipe; train-only gradient normalization for each common support bank, shared across its context cells; no public benchmark tuning'),
        development_quotas={'original_vocabulary':80,'expanded_vocabulary':80},
        confirmation_quotas={'original_vocabulary':200,'expanded_vocabulary':200,'neither_in_training_vocabulary':200},
        primary='Factorial one-seed development diagnosis; promote only after training recipe freeze, then three-seed confirmation with original grouping and whole natural benchmarks',
        no_model_scores=True,external_benchmark_sha256=sha(bench)))
    pool=[r for r in lines(src/'eligible_cooccurrences.jsonl') if r['image_id'] not in excluded]
    pool.sort(key=lambda r:order(r['image_id'],r['ann_ids']))
    used=set();usedhash=set(read(src/'exclusions.json')['image_hashes']);rows=[];rejections=Counter();available={}
    for split,quota in [('train',{'original_vocabulary':256,'expanded_vocabulary':128}),
                        ('development',{'original_vocabulary':80,'expanded_vocabulary':80}),
                        ('confirmation',{'original_vocabulary':200,'expanded_vocabulary':200,'neither_in_training_vocabulary':200})]:
        for group,n in quota.items():
            nfound=0;candidate=[r for r in pool if partition(r['image_id'])==split and stratum(r['objects'])==group]
            available[split+'/'+group]=len({r['image_id'] for r in candidate})
            for r in candidate:
                if r['image_id'] in used:continue
                path=COCO/r['coco_split']/r['file'];digest=sha(path)
                if digest in usedhash:continue
                ss=[dict(noun=noun,image_id=r['image_id'],ann_id=aid,file=r['file'],coco_split=r['coco_split']) for noun,aid in zip(r['objects'],r['ann_ids'])]
                data=[source(s) for s in ss];masks=[a[1] for a in data];rgb=data[0][0]
                if any(not .02<=m.mean()<=.60 for m in masks):rejections['decoded_area']+=1;continue
                if (masks[0]&masks[1]).any():rejections['mask_overlap']+=1;continue
                crop=[0,0,rgb.shape[1],rgb.shape[0]]
                used.add(r['image_id']);usedhash.add(digest);nfound+=1
                rows.append(dict(anchor_id='transfer_'+order(r['image_id'],r['ann_ids'])[:20],bank=split,stratum=group,family='routing',
                    objects=r['objects'],sources=ss,source_ids=[r['image_id']],crop_xyxy=crop,source_image_sha256={str(path):digest}))
                if nfound==n:break
            if nfound!=n:
                dump(OUT/'shortage.json',dict(bank=split,stratum=group,needed=n,found=nfound,metadata_unique_counts=available,decoded_rejections=dict(rejections)))
                jsonl(OUT/'partial_rows.jsonl',rows)
                raise ValueError('Insufficient score-blind data; do not relax after scores')
            print('TRANSFER_BANK',split,group,nfound,flush=True)
    jsonl(OUT/'rows.jsonl',rows);dump(OUT/'rejections.json',dict(rejections));dump(OUT/'metadata_availability.json',available)
    narrow=[r for r in rows if r['bank']=='train' and r['stratum']=='original_vocabulary']
    narrow.sort(key=lambda r:order('broad-shared-half',r['anchor_id']))
    broad=narrow[:128]+[r for r in rows if r['bank']=='train' and r['stratum']=='expanded_vocabulary']
    dump(OUT/'training_assignments.json',dict(narrow=[r['anchor_id'] for r in narrow],broad=[r['anchor_id'] for r in broad]))
    dump(OUT/'complete.json',dict(files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()},no_scores=True,no_gpu=True,
        banks=dict(Counter(r['bank'] for r in rows)),all_sources_unique=True))


def main():
    log(ROOTOUT,'start',stage='routing_transfer_construction')
    try:build()
    except BaseException as e:log(ROOTOUT,'failed',stage='routing_transfer_construction',error=repr(e));raise
    log(ROOTOUT,'complete',stage='routing_transfer_construction')


if __name__=='__main__':main()
