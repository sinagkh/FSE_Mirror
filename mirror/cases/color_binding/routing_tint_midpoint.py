"""Single post-development75% midpoint; reuse the exact matched training engine."""
import argparse
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image; from PIL import ImageDraw
from mirror.cases.color_binding import routing_tint_balance_train as engine
from mirror.cases.color_binding import routing_tint_balance_data as data
from mirror.cases.color_binding import routing_tint_balance_retest as retest
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding.rendering import captions
from mirror.core.repair import cache_sha256; from mirror.core.repair import derive_scales

ORIGINAL=data.OUT
OUT=ORIGINAL/'midpoint75'
PLAN=engine.p.ROOT/'FSE_VLM/plan/85_routing_tint_midpoint75.md'
ORIGINAL_LOAD=data.load_training
ORIGINAL_DEVPATH=data.devpath
p=engine.p


def verify():
    data.verify();verify_files(read(OUT/'protocol.json')['inputs'])
    assert sha(OUT/'training_protocol.json')==read(OUT/'training_protocol_hash.json')['sha256']
    verify_files(read(OUT/'training_protocol.json')['inputs'])


def freeze():
    engine.verify();selection=read(ORIGINAL/'selection.json');assert selection['selected'] is None
    paths=[PLAN,Path(__file__),Path(engine.__file__),Path(retest.__file__),ORIGINAL/'selection.json',
           ORIGINAL/'training_protocol.json',ORIGINAL/'data_protocol.json',ORIGINAL/'implementation_checks.json']
    dump(OUT/'protocol.json',dict(inputs={str(f):sha(f) for f in paths},tint=75,seed=42,
        recipe='color_order',arms=['Ranking','IS','IS2'],new_heldout_scores_seen=False,
        previous_development_outcomes_known=True,criteria_unchanged=True,additional_opacities_after_this=False,
        implementation='Original training engine with explicit root, bank-loader and tint parameters rebound by this frozen wrapper; no objective changes',
        no_new_seeds=True,no_manuscript_edits=True))
    # Exact training engine expects this name in its checkpoint receipts.
    dump(OUT/'training_protocol.json',dict(inputs={str(OUT/'protocol.json'):sha(OUT/'protocol.json')},
        same_engine_sha256=sha(engine.__file__),tint=75,recipe='color_order',epochs=36,updates=1944,
        same_initialization_optimizer_guards_order_schedule=True,selection_rule=str(PLAN)))
    dump(OUT/'training_protocol_hash.json',dict(sha256=sha(OUT/'training_protocol.json')))


def encode():
    verify();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info()[0]>12*1024**3
    annotations_for('train2017');annotations_for('val2017')
    # Fixed first car/boat training ID: visualization before scores, no filtering.
    row=sorted([r for r in lines(p.TRAIN/'training_rows.jsonl') if r['objects']==['car','boat']],key=lambda r:r['anchor_id'])[0]
    ims,names,hashes=p.render(row,p.COLORS[:1],.75)
    panel=Image.new('RGB',(1024,288),'white');draw=ImageDraw.Draw(panel)
    for i,im in enumerate(ims[:4]):
        panel.paste(Image.fromarray(im),(256*i,24));draw.text((256*i+4,4),'75% '+['RR','RB','BR','BB'][i],fill='black')
    panel.save(OUT/'preview75.png');dump(OUT/'preview75.json',dict(anchor_id=row['anchor_id'],states=names[:4],hashes=hashes[:4],no_model_score_selection=True))
    scorer=load_subject(p.prior.MODEL,device='cuda')
    for bank,rowpath in [('train',p.TRAIN/'training_rows.jsonl'),('development',data.oldsearch.DEV/'rows.jsonl')]:
        dest=OUT/'features'/bank;rows=lines(rowpath);counter=[0]
        if (dest/'complete.json').exists():verify_cache(dest);continue
        def renderer(row):
            x=p.render(row,data.COLORS,.75);counter[0]+=1
            if counter[0]%100==0:print('MIDPOINT_RENDER',bank,counter[0],len(rows),flush=True)
            return x
        cache_bank(dest,rows,scorer,renderer,lambda f,n:[g for c in data.COLORS for g in captions(f,n,c)],
            registry_path=DEFAULT_REGISTRY,input_hashes={str(OUT/'protocol.json'):sha(OUT/'protocol.json'),str(rowpath):sha(rowpath)},
            image_batch_size=64,text_batch_size=128,render_workers=8,details=dict(tint=75,bank=bank,no_exclusions=True))
        print('MIDPOINT_ENCODED',bank,flush=True)
    dump(OUT/'encoding_complete.json',dict(files={str(OUT/'features'/f/'complete.json'):sha(OUT/'features'/f/'complete.json') for f in ('train','development')}))


def load_training(tint,device='cuda'):
    assert tint==75
    # Only images differ from65%; template tensors and source order are identical.
    loaded,forms=ORIGINAL_LOAD(65,device);cache,spec,contexts,cfg=loaded
    folder=OUT/'features/train';verify_cache(folder);idx=lines(folder/'index.jsonl');v=np.load(folder/'images.npy');values=[]
    assert [r['anchor_id'] for r in idx]==[r['anchor_id'] for r in lines(p.TRAIN/'training_rows.jsonl')]
    for row in idx:
        for colors in data.COLORS:
            for view in p.VIEWS:
                names=['-'.join(colors)+'/'+view+'/'+a+'_'+b for a in colors for b in colors]
                values.append(v[row['image_offset']+np.array([row['state_names'].index(n) for n in names])])
    cache=replace(cache,images=torch.tensor(np.stack(values),device=device),bank_id='routing_tint_balance_75')
    cfg=replace(cfg,bank_id=cache.bank_id,expected_cache_sha256=cache_sha256(cache),scales=derive_scales(cache,spec,(0,1,2,3)))
    return (cache,spec,contexts,cfg),forms


def configure_wrapper():
    # Rebinding is process-local. Original scripts, protocols and artifacts are immutable.
    engine.OUT=OUT;engine.verify=verify;engine.RECIPES=('color_order',)
    data.TINTS=(75,);data.PLAN=PLAN;data.load_training=load_training
    data.devpath=lambda tint:OUT/'features/development' if tint==75 else ORIGINAL_DEVPATH(tint)
    retest.OUT=OUT/'retest'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','encode','train','score','select','retest']);args=ap.parse_args()
    log(OUT,'start',stage=args.action)
    try:
        if args.action=='freeze':freeze()
        elif args.action=='encode':encode()
        else:
            verify();configure_wrapper()
            if args.action=='retest':
                if read(OUT/'selection.json')['selected'] is None:
                    print('NO_APPROVED_MIDPOINT_CANDIDATE',flush=True)
                else:
                    for action in ('freeze','ablations','encode','primary','transfer','natural','diagnostics','report'):
                        log(OUT,'retest_stage',stage=action);getattr(retest,action)()
            else:getattr(engine,args.action)()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
