"""Final-budget routing component controls; main ranking/IS checkpoints unchanged."""
import argparse
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.completion_data import OUT as ROOTOUT; from mirror.cases.color_binding.completion_data import PLAN
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.cases.color_binding import routing_common_noise_pilot as cn
from mirror.cases.color_binding import routing_budget_replication as final
from mirror.core.repair import _state_hash

OUT=ROOTOUT/'routing_controls'
ARMS=('G','no_cross','no_response','no_preference','no_interaction')


def objective(parts,weights,arm):
    full=cn.objective(parts,weights)
    if arm=='G':return prior.objective(parts,weights,'G')
    if arm=='no_cross':return full-4*weights['cross']*parts['cross']
    if arm=='no_response':return full-weights['response']*parts['response']
    if arm=='no_preference':return full-4*weights['preference']*parts['preference']
    if arm=='no_interaction':return prior.objective(parts,weights,'P')
    raise ValueError(arm)


def freeze():
    assert (ROOTOUT/'phase_b_complete.json').exists(),'Finish B first'
    final.verify()
    paths=[PLAN,Path(__file__),ROOTOUT/'phase_b_complete.json',final.OUT/'protocol.json',
        final.OUT/'complete.json',prior.OUT/'normalization.json']
    paths += [final.OUT/f'seed{s}/models.json' for s in final.SEEDS]
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in paths},seeds=final.SEEDS,arms=ARMS,
        epochs=36,updates=1944,configuration='identical to final plan24 including common dropout and original fixed normalizers',
        selection='fixed_last',main_baseline='Original ranking already complete; no endpoint control',
        no_interaction='remove binding, cross, response steering; keep preference penalty and every guard',
        no_cross='remove only cross hinge, retain binding/response/preference and all guards',
        no_response='remove only response steering; response_keep guard remains',
        preserve_all_forwards_and_rng=True,no_score_based_selection=True))


def train():
    p=read(OUT/'protocol.json');verify_files(p['inputs']);prior.configure();cn.available_gpu()
    cache,spec,contexts,cfg=prior.load_training('cuda');weights=read(prior.OUT/'normalization.json')['weights']
    registry=[dict(arm='F',seed=0,checkpoint=None)]
    for seed in final.SEEDS:
        refs=read(final.OUT/f'seed{seed}/models.json');registry += [r for r in refs if r['arm']!='F']
        original=next(r for r in refs if r['arm']=='IS');schedule=final.schedule_for(seed)
        for arm in ARMS:
            dest=OUT/f'seed{seed}'/arm
            if (dest/'complete.json').exists():
                r=read(dest/'complete.json');assert sha(r['checkpoint'])==r['sha256'];registry.append(r);continue
            dest.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=cn.make_model(replace(cfg,seed=seed,epochs=36),'cuda')
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};ih=_state_hash(state)
            assert ih==original['initial_state_hash']
            torch.save(dict(state_dict=state,state_hash=ih),dest/'initial.pt')
            opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);history=[];start_time=time.monotonic()
            for epoch,order in enumerate(schedule,1):
                model.train();torch.manual_seed(seed*10000+epoch);torch.cuda.manual_seed_all(seed*10000+epoch)
                for start in range(0,1280,24):
                    rows=prior.paired_rows(torch.tensor(order[start:start+24],device='cuda'))
                    parts=prior.components(model,cache,spec,contexts,cfg,rows);loss=objective(parts,weights,arm)
                    opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                    assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
                    history.append(dict(epoch=epoch,rows=rows.cpu().tolist(),loss=float(loss.detach()),gradient_norm=float(gn),components={k:float(v.detach()) for k,v in parts.items()}))
            rh=final.sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes());assert rh==original['final_rng_sha256']
            assert len(history)==1944
            # Full-IS first-batch components must be identical before any optimization.
            old=prior.lines(final.OUT/f'seed{seed}/IS/history.jsonl')
            assert history[0]['components']==old[0]['components']
            assert [r['rows'] for r in history]==[r['rows'] for r in old]
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,
                protocol_sha256=sha(OUT/'protocol.json'),selection='fixed_last',epochs=36,updates=1944),dest/'last.pt')
            jsonl(dest/'history.jsonl',history)
            r=dict(arm=arm,seed=seed,checkpoint=str(dest/'last.pt'),sha256=sha(dest/'last.pt'),initial_state_hash=ih,
                final_rng_sha256=rh,matched_forwards_batches_rng=True,seconds=time.monotonic()-start_time)
            dump(dest/'complete.json',r);registry.append(r);print('CONTROL_COMPLETE',r,flush=True)
            del model,opt
    dump(OUT/'models.json',registry);dump(OUT/'complete.json',dict(models_sha256=sha(OUT/'models.json'),n_new_training=15,
        n_total_trained=21,frozen_models=1,epochs=36,all_main_models_unchanged=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','train']);a=ap.parse_args();log(OUT,'start',stage=a.action)
    try:globals()[a.action]()
    except BaseException as e:log(OUT,'failed',error=repr(e));raise
    log(OUT,'complete',stage=a.action)


if __name__=='__main__':main()
