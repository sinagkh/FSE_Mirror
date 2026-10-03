"""Create-only feature preparation for the bounded B/C completion program."""
import argparse
import gc
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import OUT as TRAINBANK; from mirror.cases.color_binding.repair_trainbank import render; from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.cases.color_binding.routing_repair_pilot import objects
from mirror.cases.color_binding.routing_relative_data import render as routing_render
from mirror.cases.color_binding.rendering import captions
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache

OUT=ROOT/'clip/interbind_phase_bc_completion_20260923'
PLAN=ROOT/'FSE_VLM/plan/25_finish_diagnostics_and_repair.md'
MODELS=('openclip_laion_l14','openclip_laion_b32','openai_clip_l14')
COLORS=(('red','blue'),('green','yellow'))


def configure():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def freeze():
    paths=[PLAN,Path(__file__),TRAINBANK/'protocol.json',TRAINBANK/'complete.json',
           TRAINBANK/'rows.jsonl',TRAINBANK/'excluded_source_ids.json',DEFAULT_REGISTRY,
           DEFAULT_REGISTRY.with_name('activation_addendum.json')]
    paths += [Path(__file__).with_name(n+'.py') for n in
              ('repair_trainbank','routing_relative_data','rendering','rendering_v3','scorers','feature_cache')]
    rows=lines(TRAINBANK/'rows.jsonl')
    assert len(rows)==1408
    blocked=set(read(TRAINBANK/'excluded_source_ids.json'))
    assert not blocked.intersection(i for r in rows for i in r['source_ids'])
    assert not blocked.intersection(r['natural_guard']['image_id'] for r in rows)
    dump(OUT/'data_protocol.json',dict(inputs={str(p):sha(p) for p in paths},models=MODELS,
         colors=COLORS,renderer='unchanged blend90',batch_images=64,workers=8,
         no_training=True,no_outcome_selection=True))


def verify():
    p=read(OUT/'data_protocol.json');verify_files(p['inputs']);return p


def object_prompts(family,nouns):
    return objects(nouns) if family=='routing' else [f'a photo of a {nouns[0]}',f'a {nouns[0]}']


def encode(model,family):
    p=verify();configure()
    free,_=torch.cuda.mem_get_info()
    if free<12*1024**3:raise RuntimeError('GPU busy; no other job interrupted')
    dest=OUT/'training_features'/model/family
    if (dest/'extras_complete.json').exists():
        verify_cache(dest);verify_files(read(dest/'extras_complete.json')['files']);return
    if dest.exists():raise FileExistsError(f'Partial cache retained: {dest}')
    rows=[r for r in lines(TRAINBANK/'rows.jsonl') if r['family']==family]
    vocab=sorted({tuple(r['objects']) for r in rows})
    annotations_for('train2017')
    scorer=load_subject(model,device='cuda');qualified=[]
    def renderer(row):
        images=[];names=[];hashes=[];checks=[]
        if family=='routing':
            for swapped in (False,True):
                ims,nn,hh,cc=routing_render(row,swapped=swapped)
                images+=ims;names+=nn;hashes+=hh;checks+=cc
        else:
            for pair in COLORS:
                ims,nn,hh,cc=render(row,pair)
                images+=ims;names+=['-'.join(pair)+'/'+n for n in nn];hashes+=hh;checks+=cc
        assert all(c['edit']['outside_unchanged'] for c in checks)
        qualified.append(dict(anchor_id=row['anchor_id'],checks=checks))
        return images,names,hashes
    def prompts(fam,nouns):
        nouns=tuple(nouns);wrong=vocab[(vocab.index(nouns)+1)%len(vocab)]
        return [g for pair in COLORS for g in captions(fam,nouns,pair)]+[
            object_prompts(fam,nouns),object_prompts(fam,wrong)]
    cache_bank(dest,rows,scorer,renderer,prompts,registry_path=DEFAULT_REGISTRY,
        input_hashes={str(OUT/'data_protocol.json'):sha(OUT/'data_protocol.json')},
        image_batch_size=p['batch_images'],text_batch_size=64,render_workers=p['workers'],
        details=dict(family=family,colors=COLORS,score_scale=1.,score_bias=0.))
    natural=[[c['text'] for c in r['natural_guard']['captions']] for r in rows]
    unique=list(dict.fromkeys(t for group in natural for t in group));lookup={s:i for i,s in enumerate(unique)}
    t=scorer.encode_texts(unique,batch_size=64).numpy()
    with (dest/'natural_texts.npy').open('xb') as f:
        np.save(f,t[np.asarray([[lookup[s] for s in group] for group in natural])])
    jsonl(dest/'source_rows.jsonl',rows)
    jsonl(dest/'pixel_checks.jsonl',sorted(qualified,key=lambda r:r['anchor_id']))
    checks=[c for r in qualified for c in r['checks']]
    dump(dest/'extras_complete.json',dict(files={str(dest/n):sha(dest/n) for n in
        ('natural_texts.npy','source_rows.jsonl','pixel_checks.jsonl','complete.json')},
        n_checks=len(checks),passed=sum(c['edit']['passes'] for c in checks),
        no_score_filtering=True))
    del scorer;gc.collect();torch.cuda.empty_cache()
    print('ENCODED',model,family,len(rows),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','encode'])
    ap.add_argument('--model',choices=MODELS,default=MODELS[0]);ap.add_argument('--family',choices=['routing','background'],default='background')
    a=ap.parse_args();log(OUT,'start',stage='data_'+a.action,model=a.model,family=a.family)
    try:
        if a.action=='freeze':freeze()
        else:encode(a.model,a.family)
    except BaseException as e:log(OUT,'failed',error=repr(e));raise
    log(OUT,'complete',stage='data_'+a.action,model=a.model,family=a.family)


if __name__=='__main__':main()
