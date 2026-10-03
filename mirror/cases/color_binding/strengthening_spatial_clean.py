"""Clean-source spatial rerun: frozen recipe, cached subset, untouched test IDs."""
import argparse
import math
import os
from pathlib import Path
import shutil
import time
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA')
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.strengthening_cpu import OUT as ROOTOUT
from mirror.cases.color_binding import strengthening_spatial as data
from mirror.cases.color_binding import strengthening_spatial_repair as base
from mirror.cases.color_binding import strengthening_spatial_confirmation as confirm
from mirror.cases.color_binding.strengthening_firewall_v3 import spatial_cache
from mirror.cases.color_binding.behavioral_pilot import lines; from mirror.cases.color_binding.behavioral_pilot import CAL
from mirror.core.metrics import bank_arrays as raw_bank_arrays
from mirror.core.repair import _state_hash

DATA=ROOTOUT/'spatial_v3';OUT=DATA/'repair_v1';OLD=ROOTOUT/'spatial_v2/repair_v1'
BLOCKED=set(read(ROOTOUT/'source_firewall_v3/blocked_coco_ids.json'))


def utility_bank(path,*args,**kwargs):
    rows=lines(ROOT/'data/color_binding/object_pairs/rows.jsonl')
    kwargs['allowed']={r['anchor_id'] for r in rows if not set(r['source_ids'])&BLOCKED}
    return raw_bank_arrays(path,*args,**kwargs)


def setup():
    data.OUT=DATA;base.DATA=DATA;base.OUT=OUT;base.GUARD=ROOTOUT/'source_firewall_v3/natural_texts_clean.npy'
    base.bank_arrays=utility_bank;confirm.DATA=DATA;confirm.OUT=DATA/'confirmation_results'


def prepare():
    data.diagnose();OUT.mkdir(exist_ok=False)
    p=read(OLD/'protocol.json');n=read(DATA/'data_complete.json')['counts']['train']
    p.update(data_sha256=sha(DATA/'data_complete.json'),diagnosis_sha256=sha(DATA/'frozen_diagnosis/complete.json'),
        frozen_diagnosis=read(DATA/'frozen_diagnosis/summary.json'),source_firewall_sha256=sha(ROOTOUT/'source_firewall_v3/complete.json'),
        clean_rerun_script_sha256=sha(Path(__file__)),guard_cache_sha256=sha(base.GUARD),
        source_integrity_correction='Remove all COCO2017 validation and mapped ARO sources from training,development,guards and internal-utility selection. Keep prior frozen v1 loss/optimizer/epoch recipe; test IDs unchanged.',
        parent_recipe_selection_sha256=sha(ROOTOUT/'spatial_v2/recipe_selection.json'),no_recipe_search_after_correction=True)
    p['optimizer']['updates']=36*math.ceil(n/24);p['actual_training_anchors']=n
    dump(OUT/'protocol.json',p)
    for name in ('object_texts.npy','object_prompts.json'):shutil.copyfile(OLD/name,OUT/name)
    dump(OUT/'preparation_complete.json',dict(files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()},training=False,gpu=False))
    dump(DATA/'recipe_selection.json',dict(selected_version='repair_v1',confirmation_seen=False,benchmark_seen=False,
        further_revisions_allowed=0,reason='Same frozen v1 recipe; metadata-only source correction, not a new efficacy search',
        parent_selection_sha256=sha(ROOTOUT/'spatial_v2/recipe_selection.json')))


def train(seed):
    verify_files(read(OUT/'preparation_complete.json')['files']);v,t,rows=base.load('train');unit=read(CAL)['units'][base.MODEL]['unit']
    weights=read(OUT/'normalization.json')['weights'] if (OUT/'normalization.json').exists() else base.normalize(v,t,unit)
    if seed!=42:assert read(OUT/'seed42/development/gate.json')['passed']
    dest=OUT/f'seed{seed}';dest.mkdir(exist_ok=False);rng=np.random.default_rng(seed);schedule=[rng.permutation(len(v)).tolist() for _ in range(36)]
    dump(dest/'schedule.json',schedule);regs=[];initials=[];states=[]
    for arm in base.ARMS:
        folder=dest/arm;folder.mkdir();m=base.model(seed);initials.append(_state_hash(m.state_dict()))
        opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.01);history=[];started=time.monotonic()
        for epoch,order in enumerate(schedule,1):
            m.train();torch.manual_seed(seed*10000+epoch)
            for start in range(0,len(v),24):
                ix=order[start:start+24];p=base.parts(m,v[ix],t[ix],unit);loss=base.objective(p,weights,arm)
                opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                assert torch.isfinite(loss)&torch.isfinite(gn);opt.step()
                history.append(dict(epoch=epoch,loss=float(loss.detach()),gradient_norm=float(gn),**{k:float(x.detach()) for k,x in p.items()}))
            if epoch%12==0:print('CLEAN_SPATIAL_CPU_TRAIN',seed,arm,epoch,'seconds',round(time.monotonic()-started,1),flush=True)
        assert len(history)==read(OUT/'protocol.json')['optimizer']['updates'];states.append(torch.get_rng_state().numpy().tobytes())
        torch.save(dict(state_dict=m.state_dict(),seed=seed,arm=arm,selection='fixed_last',protocol_sha256=sha(OUT/'protocol.json')),folder/'last.pt')
        jsonl(folder/'history.jsonl',history);regs.append(dict(arm=arm,seed=seed,checkpoint=str(folder/'last.pt'),sha256=sha(folder/'last.pt'),seconds=time.monotonic()-started))
    assert len(set(initials))==len(set(states))==1
    dump(dest/'models.json',regs);dump(dest/'complete.json',dict(files={str(dest/'models.json'):sha(dest/'models.json')},matched_initialization=True,matched_rng=True,gpu=False))


def seeds():
    for seed in (43,44):train(seed)
    selection=DATA/'final_recipe_selection.json' if (DATA/'final_recipe_selection.json').exists() else DATA/'recipe_selection.json'
    dump(DATA/'three_seed_checkpoints_frozen.json',dict(selected_version=OUT.name,checkpoints=[m for seed in (42,43,44) for m in read(OUT/f'seed{seed}/models.json')],
        selection_sha256=sha(selection),confirmation_not_scored=True,benchmarks_not_scored=True,clean_source_firewall_sha256=sha(ROOTOUT/'source_firewall_v3/complete.json')))


def features():
    # The old process is an encoder only; matrices are never scored with v2 weights.
    paths=[ROOTOUT/f'spatial_v2/features/{b}/complete.json' for b in ('confirmation','heldout_pairs')]+[ROOTOUT/'whatsup/features/complete.json']
    start=time.monotonic()
    while not all(p.exists() for p in paths):
        if time.monotonic()-start>3*3600:raise RuntimeError('Parent test encoding not complete; inspect log before retry')
        time.sleep(10)
    spatial_cache(['confirmation','heldout_pairs'])


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','evaluate_dev','seeds','features','confirm']);a=p.parse_args()
    setup();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);log(ROOTOUT,'start',stage='clean_spatial_'+a.action)
    try:
        if a.action=='train':train(42)
        elif a.action=='evaluate_dev':base.evaluate(42)
        elif a.action=='confirm':confirm.evaluate()
        else:globals()[a.action]()
    except BaseException as e:log(ROOTOUT,'failed',stage='clean_spatial_'+a.action,error=repr(e));raise
    log(ROOTOUT,'complete',stage='clean_spatial_'+a.action)


if __name__=='__main__':main()
