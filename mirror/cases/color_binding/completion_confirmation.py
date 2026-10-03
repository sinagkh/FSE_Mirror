"""Fresh same-rule confirmation sources: fixed geometry, no outcome selection."""
import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import cv2
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.completion_data import OUT as ROOTOUT; from mirror.cases.color_binding.completion_data import PLAN
from mirror.cases.color_binding.routing_same_class_data import PAIRS; from mirror.cases.color_binding.routing_same_class_data import eligible
from mirror.cases.color_binding.repair_trainbank import COCO; from mirror.cases.color_binding.repair_trainbank import source; from mirror.cases.color_binding.repair_trainbank import annotations_for; from mirror.cases.color_binding.repair_trainbank import lines

OUT=ROOTOUT/'same_rule_confirmation'
OLD=ROOT/'clip/interbind_routing_same_class_20260923'


def order(*v):return hashlib.sha256(json.dumps(['fresh-confirmation-v1-20260923',*v],sort_keys=True).encode()).hexdigest()


def build():
    files=[PLAN,Path(__file__),OLD/'rows.jsonl',OLD/'exclusions.json',
        ROOT/'data/color_binding/train/rows.jsonl',
        *[COCO/f'annotations/instances_{s}.json' for s in ('train2017','val2017')]]
    excluded=read(OLD/'exclusions.json');ids=set(excluded['source_ids']);hashes=set(excluded['image_hashes'])
    for r in lines(OLD/'rows.jsonl')+lines(files[4]):
        ids.update(r['source_ids']);hashes.update(r['source_image_sha256'].values())
        if r.get('natural_guard'):ids.add(r['natural_guard']['image_id'])
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in files},quota_per_pair=100,pairs=PAIRS,
        selection='fixed hash, scarcity-first, same original geometry; unique source images across all pairs',
        shortages='retain every eligible available source up to100, record zero support; never substitute classes or relax geometry',
        colors=[['red','blue'],['green','yellow'],['purple','orange']],
        layouts=['canvas','swapped_canvas'],no_scores=True,selection_from_metadata_only=True,
        exclusion_count=len(ids),interpretation='Confirmation for supported trained pairs on sources not used by prior development; absent pairs are not confirmed'))
    dump(OUT/'exclusions.json',dict(source_ids=sorted(ids),image_hashes=sorted(hashes)))
    pools=defaultdict(list);nouns={x for p in PAIRS for x in p};cv2.setNumThreads(1)
    for split in ('train2017','val2017'):
        ims,anns,cats,_=annotations_for(split);best={}
        for a in anns.values():
            noun=cats[a['category_id']];iid=a['image_id']
            if noun not in nouns or iid in ids or not eligible(a,ims[iid]):continue
            key=(noun,iid)
            if key not in best or (a['area'],-a['id'])>(best[key]['area'],-best[key]['id']):best[key]=a
        for (noun,iid),a in best.items():
            path=COCO/split/ims[iid]['file_name']
            if path.is_file():pools[noun].append(dict(noun=noun,image_id=iid,ann_id=a['id'],coco_split=split,file=ims[iid]['file_name']))
    for n in pools:pools[n].sort(key=lambda r:order(n,r['image_id'],r['ann_id']))
    used=set();used_hashes=set();rows=[];counts={};rejects=[]
    for pair in sorted(PAIRS,key=lambda p:(min(len(pools[n]) for n in p),p)):
        selected={};pending=set();pending_hashes=set();need=100
        for noun in sorted(pair,key=lambda n:len(pools[n])):
            selected[noun]=[]
            if not need:continue
            for r in pools[noun]:
                if r['image_id'] in used|pending:continue
                path=COCO/r['coco_split']/r['file'];h=sha(path)
                if h in hashes|used_hashes|pending_hashes:continue
                _,mask,_=source(r)
                if not .02<=mask.mean()<=.60:rejects.append(dict(**r,reason='decoded_mask_area'));continue
                selected[noun].append(dict(r,image_sha256=h));pending.add(r['image_id']);pending_hashes.add(h)
                if len(selected[noun])==need:break
            need=min(need,len(selected[noun]))
        n=min(len(selected[k]) for k in pair);counts['+'.join(pair)]=n
        for j in range(n):
            ss=[selected[k][j] for k in pair];ii=[s['image_id'] for s in ss]
            assert len(set(ii))==2 and not (set(ii)&(ids|used))
            used.update(ii);used_hashes.update(s['image_sha256'] for s in ss)
            rows.append(dict(anchor_id='confirm_'+order([(s['image_id'],s['ann_id']) for s in ss])[:20],
                family='routing',objects=pair,sources=ss,source_ids=ii,
                source_image_sha256={str(COCO/s['coco_split']/s['file']):s['image_sha256'] for s in ss}))
        print('CONFIRMATION_BANK',pair,n,flush=True)
    rows.sort(key=lambda r:(r['objects'],r['anchor_id']));jsonl(OUT/'rows.jsonl',rows);jsonl(OUT/'rejections.jsonl',rejects)
    dump(OUT/'complete.json',dict(files={str(OUT/n):sha(OUT/n) for n in ('protocol.json','exclusions.json','rows.jsonl','rejections.jsonl')},
        counts=counts,n_anchors=len(rows),n_images=len(used),source_overlap=0,model_scores_loaded=False,
        missing_pairs=[k for k,v in counts.items() if v==0]))


def main():
    log(OUT,'start',stage='construction')
    try:build()
    except BaseException as e:log(OUT,'failed',error=repr(e));raise
    log(OUT,'complete',stage='construction')


if __name__=='__main__':main()
