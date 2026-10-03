"""Exact 75% rendered inputs across backbones; no training or score selection."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from mirror.cases.color_binding import replicate as base
from mirror.cases.color_binding import prewriting_routing_breadth as breadth
from mirror.cases.color_binding import indirect_generalization as ind
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding.rendering import captions

OUT=ROOT/'clip/interbind_routing75_complete_20260930'
PLAN=ROOT/'FSE_VLM/plan/88_uniform_routing75_completion.md'
MODELS=('openai_clip_l14','google_siglip_b16_224')
p=base.p; olddata=base.ev.olddata
FAMILIES=('train','primary',*olddata.FAMILIES)

def rowpath(family):
    if family=='train':return p.TRAIN/'training_rows.jsonl'
    return olddata.rows_path(family)

def reference(family):
    return base.mid.OUT/'features/train' if family=='train' else base.OLD/'features'/family

def bank(model,family):
    return reference(family) if model=='openclip_laion_l14' else OUT/model/'features'/family

def freeze():
    paths=[PLAN,Path(__file__),base.OUT/'protocol.json',base.OUT/'verification.json',
           base.OUT/'results/complete.json',DEFAULT_REGISTRY,DEFAULT_REGISTRY.with_name('activation_addendum.json')]
    paths += [Path(m.__file__) for m in (p,olddata,base.mid,base.engine,ind)]
    paths += [reference(f)/'complete.json' for f in FAMILIES]
    paths += [rowpath(f) for f in FAMILIES]
    paths += [breadth.OUT/m/'normalization.json' for m in MODELS]
    dump(OUT/'data_protocol.json',dict(inputs={str(f):sha(f) for f in paths},models=MODELS,
        tint=.75,no_outcome_selection=True,render_workers=8,image_batch_size=64,
        required_pixel_identity='All states equal completed primary LAION75 study; backbone is only encoding difference'))
    dump(OUT/'data_protocol_hash.json',dict(sha256=sha(OUT/'data_protocol.json')))

def verify():
    assert sha(OUT/'data_protocol.json')==read(OUT/'data_protocol_hash.json')['sha256']
    verify_files(read(OUT/'data_protocol.json')['inputs'])

def encode(model):
    verify();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info()[0]>12*1024**3,'GPU busy; no jobs interrupted'
    annotations_for('train2017');annotations_for('val2017');olddata.install_strength(.75)
    scorer=load_subject(model,device='cuda')
    for family in FAMILIES:
        dest=bank(model,family)
        if (dest/'complete.json').exists():verify_cache(dest);continue
        rows=lines(rowpath(family));ref={r['anchor_id']:r for r in lines(reference(family)/'index.jsonl')}
        if family=='train':colors=p.COLORS[:2]
        elif family=='primary':colors=p.COLORS
        elif family.endswith('_shift'):colors=[('red','blue')]
        else:colors=[tuple(c.split('-')) for c,_ in olddata.ORIGINAL_CONDITIONS(family)]
        done=[0]
        def renderer(row):
            if family in ('train','primary'):ims,names,hashes=p.render(row,colors,.75)
            else:
                ims,names,hashes,checks=olddata.render(row,family)
                assert all(c['outside_unchanged'] for c in checks)
            expected=ref[row['anchor_id']]
            assert names==expected['state_names'] and hashes==expected['pixel_sha256'],(family,row['anchor_id'])
            done[0]+=1
            if done[0]%100==0:print('ENCODE_PROGRESS',model,family,done[0],len(rows),flush=True)
            return ims,names,hashes
        cache_bank(dest,rows,scorer,renderer,lambda f,n:[g for c in colors for g in captions(f,n,c)],
            registry_path=DEFAULT_REGISTRY,input_hashes={str(OUT/'data_protocol.json'):sha(OUT/'data_protocol.json'),str(rowpath(family)):sha(rowpath(family))},
            image_batch_size=64,text_batch_size=128,render_workers=8,
            details=dict(tint=.75,all_pixels_equal_reference=str(reference(family)),no_exclusions=True))
        print('ENCODE_COMPLETE',model,family,flush=True)
    # Exactly the original text strings, plus reversed training noun order.
    extra=OUT/model/'extra_text'
    if not (extra/'complete.json').exists():
        trainrows=lines(rowpath('train'));vocab=sorted({tuple(r['objects']) for r in trainrows})
        strings=[]
        for nouns in vocab:
            for colors in p.COLORS[:2]:
                for family in ('train_templates','reverse_order'):
                    strings += [s for forms in ind.prompts(nouns,colors,'canvas',family) for s in forms]
            wrong=vocab[(vocab.index(nouns)+1)%len(vocab)]
            strings += breadth.original.objects(nouns)+breadth.original.objects(wrong)
        strings += [c['text'] for r in trainrows for c in r['natural_guard']['captions']]
        strings += read(ind.OUT/'template_index.json')
        for entry in read(base.ev.tt.DEST/'caption_gallery.json'):strings += entry['templates']
        strings=list(dict.fromkeys(strings));extra.mkdir(parents=True,exist_ok=False)
        features=scorer.encode_texts(strings,batch_size=128).numpy()
        np.save(extra/'features.npy',features);dump(extra/'strings.json',strings)
        dump(extra/'complete.json',dict(files={str(f):sha(f) for f in extra.iterdir() if f.is_file()}))
    dump(OUT/model/'encoding_complete.json',dict(files={str(f):sha(f) for f in
        [*(bank(model,k)/'complete.json' for k in FAMILIES),extra/'complete.json']}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','encode']);ap.add_argument('--model',choices=MODELS)
    a=ap.parse_args();log(OUT,'command_start',**vars(a))
    try:freeze() if a.action=='freeze' else encode(a.model)
    except BaseException as exc:log(OUT,'failed',error=repr(exc),**vars(a));raise
    log(OUT,'command_complete',**vars(a))

if __name__=='__main__':main()
