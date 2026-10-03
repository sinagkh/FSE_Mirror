"""Fixed moderate routing recipe: three seeds, matched baseline and deletions."""
from mirror.cases.color_binding import routing_constraint_pilot as parent
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import argparse
import time
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files

base=parent.base
OUT=ROOT/'clip/interbind_routing_moderate_replication_20260929'
PLAN=ROOT/'FSE_VLM/plan/80_routing_moderate_replication_20260929.md'
SEEDS=(42,43,44)
DELETIONS={'no_cross':('cross',),'no_response':('response',),'no_preference':('preference',),
           'no_interaction':('binding','cross','response')}
ARMS=('IS','Ranking',*DELETIONS,'Previous_IS')


def initial_path(seed):return base.old.OUT/f'seed{seed}/IS/initial.pt'


def freeze():
    parent.verify()
    refs={name:read(path/'complete.json') for name,path in {
        'IS':parent.OUT/'runs/IS8_Dual','Ranking':parent.OUT/'runs/Ranking_Dual',
        'Previous_IS':base.OUT/'runs/IS4'}.items()}
    files=[PLAN,Path(__file__),parent.OUT/'protocol.json',parent.OUT/'preflight.json',
           parent.OUT/'selection.json',base.prior.OUT/'normalization.json']
    files += [Path(m.__file__) for m in (parent,base,base.prior,base.cn,base.prev)]
    files += [initial_path(s) for s in SEEDS]
    for r in refs.values():files += [Path(r['checkpoint']),Path(r['checkpoint']).with_name('history.jsonl')]
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in files},seeds=SEEDS,arms=ARMS,
        deletions=DELETIONS,seed42_reused=refs,epochs=36,updates=1944,device='cpu',
        dual_step=20.,dual_cap=100.,binding_buffer_fraction=.02,binding_buffer_min_scale=.1,
        selection='fixed_last; author-authorized seed expansion; no recipe selection',
        known_seed42_retained_test_outcomes=True,new_seed_outcomes_seen=False))
    print('PROTOCOL_FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    p=read(OUT/'protocol.json');verify_files(p['inputs']);parent.verify();return p


def make(seed,cfg):
    torch.manual_seed(seed);model=base.cn.make_model(cfg,'cpu')
    model.load_state_dict(torch.load(initial_path(seed),map_location='cpu',weights_only=False)['state_dict'])
    return model


def objective(parts,weights,arm,ce):
    if arm=='Ranking':return parent.objective(parts,weights,'Ranking_Dual',ce)
    if arm=='Previous_IS':return base.objective(parts,weights,'IS4',ce)
    w=dict(weights)
    for k in DELETIONS.get(arm,()):w[k]=0.
    return parent.objective(parts,w,'IS8_Dual',ce)


def preflight():
    p=verify();base.configure();loaded=base.prior.load_training('cpu');torch.set_num_threads(2)
    cache,spec,ctx,cfg=loaded;cfg=replace(cfg,epochs=36,seed=42,grad_clip=5.)
    forms=base.prev.template_cache(cache);reps=base.prev.representation_schedule(42)
    current=replace(cache,texts=forms[torch.arange(2560),torch.as_tensor(np.repeat(reps[0],2))])
    weights=read(base.prior.OUT/'normalization.json')['weights'];model=make(42,cfg).train()
    old=base.prev.lines(Path(p['seed42_reused']['IS']['checkpoint']).with_name('history.jsonl'))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    order=base.old.schedule_for(42)[0];torch.manual_seed(420001);checks=[]
    for j in range(3):
        ids=base.prior.paired_rows(torch.tensor(order[j*24:(j+1)*24]))
        parts,ce,d,df=parent.components(model,current,spec,ctx,cfg,ids)
        loss=objective(parts,weights,'IS',ce)
        assert {k:float(v.detach()) for k,v in parts.items()}==old[j]['components']
        assert float(loss.detach())==old[j]['loss']
        if j==0:
            params=tuple(model.parameters())
            def grad(z):return torch.cat([g.flatten() for g in torch.autograd.grad(z,params,retain_graph=True)])
            for arm,keys in DELETIONS.items():
                removed=sum((8 if k=='cross' else 4 if k=='preference' else 1)*weights[k]*parts[k] for k in keys)
                actual=objective(parts,weights,arm,ce)
                torch.testing.assert_close(actual,loss-removed,rtol=1e-6,atol=1e-6)
                torch.testing.assert_close(grad(actual),grad(loss-removed),rtol=1e-4,atol=2e-6)
        opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        assert float(gn)==old[j]['gradient_norm'];opt.step();checks.append(dict(update=j+1,exact_replay=True))
    dump(OUT/'preflight.json',dict(seed42_first_three_updates=checks,all_deletion_gradients_verified=True,
        device='cpu',protocol_sha256=sha(OUT/'protocol.json')))
    print('PREFLIGHT_PASS',flush=True)


def train_one(seed,arm,loaded,forms,p):
    dest=OUT/'runs'/f'seed{seed}'/arm
    if (dest/'complete.json').exists():
        r=read(dest/'complete.json');verify_files({r['checkpoint']:r['sha256']});return r
    dest.mkdir(parents=True,exist_ok=False)
    cache,spec,ctx,cfg=loaded;cfg=replace(cfg,epochs=36,seed=seed,grad_clip=1. if arm=='Previous_IS' else 5.)
    weights=read(base.prior.OUT/'normalization.json')['weights']
    model=make(seed,cfg);initial=base._state_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    schedule=base.old.schedule_for(seed);reps=base.prev.representation_schedule(seed)
    frozen=torch.tensor(read(parent.OUT/'preflight.json')['frozen_training_means'])
    buffer=p['binding_buffer_fraction']*frozen.abs().clamp_min(p['binding_buffer_min_scale'])
    lambdas=torch.zeros(2,8);history=[];diagnostics=[];start=time.monotonic()
    for ei,order in enumerate(schedule):
        current=replace(cache,texts=forms[torch.arange(2560),torch.as_tensor(np.repeat(reps[ei],2))])
        model.train();torch.manual_seed(seed*10000+ei+1)
        for j in range(0,1280,24):
            ids=base.prior.paired_rows(torch.tensor(order[j:j+24]))
            parts,ce,d,df=parent.components(model,current,spec,ctx,cfg,ids)
            penalty=parent.constraint_penalty(d,df,ids,lambdas,buffer)
            loss=objective(parts,weights,arm,ce)+penalty
            opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
            history.append(dict(epoch=ei+1,rows=ids.tolist(),loss=float(loss.detach()),gradient_norm=float(gn),
                adaptive_penalty=float(penalty.detach()),ce=float(ce.detach()),components={k:float(v.detach()) for k,v in parts.items()}))
        dm,fm=parent.monitor(model,cache,forms,spec,ctx);torch.testing.assert_close(fm,frozen,rtol=0,atol=0)
        g=frozen+buffer-dm;before=lambdas.clone()
        if arm!='Previous_IS':lambdas=parent.update_dual(lambdas,g,p)
        diagnostics.append(dict(epoch=ei+1,binding_mean=dm.tolist(),constraint=g.tolist(),lambda_used=before.tolist(),lambda_next=lambdas.tolist()))
        if (ei+1)%12==0:print('TRAIN',seed,arm,ei+1,'seconds',round(time.monotonic()-start,1),flush=True)
    assert len(history)==1944 and not torch.cuda.is_initialized()
    ckpt=dest/'last.pt';torch.save(dict(state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},
        configuration=asdict(cfg),optimizer=opt.state_dict(),seed=seed,arm=arm,selection='fixed_last',
        epochs=36,updates=1944,weights=weights,target_deletions=DELETIONS.get(arm,()),multipliers=lambdas,
        initial_state_hash=initial,protocol_sha256=sha(OUT/'protocol.json')),ckpt)
    jsonl(dest/'history.jsonl',history);jsonl(dest/'constraints.jsonl',diagnostics)
    r=dict(name=arm,seed=seed,checkpoint=str(ckpt),sha256=sha(ckpt),initial_state_hash=initial,
        updates=1944,schedule_sha256=base.hash_array(np.asarray(schedule)),representation_sha256=base.hash_array(reps),
        final_rng_sha256=base.hash_array(torch.get_rng_state().numpy()),first_components=history[0]['components'],
        seconds=time.monotonic()-start,grad_clip=cfg.grad_clip,device='cpu',initial_path=str(initial_path(seed)))
    dump(dest/'complete.json',r);print('TRAIN_COMPLETE',seed,arm,r['seconds'],flush=True);return r


def run_seed(seed):
    p=verify();base.configure();assert (OUT/'preflight.json').exists()
    loaded=base.prior.load_training('cpu');torch.set_num_threads(2);forms=base.prev.template_cache(loaded[0])
    regs=[]
    for arm in ARMS:
        if seed==42 and arm in p['seed42_reused']:
            r=dict(p['seed42_reused'][arm],name=arm,reused=True)
        else:
            log(OUT,'train_start',seed=seed,arm=arm)
            r=train_one(seed,arm,loaded,forms,p);log(OUT,'train_complete',seed=seed,arm=arm)
        regs.append(r)
    for key in ('initial_state_hash','updates','schedule_sha256','representation_sha256','final_rng_sha256'):
        assert len({r[key] for r in regs})==1,(seed,key)
    assert all(r['first_components']==regs[0]['first_components'] for r in regs)
    dump(OUT/f'seed{seed}_complete.json',dict(models=regs,matched_init_stream_rng_updates=True))


def register():
    verify();models=[dict(name='F',seed=0,checkpoint=None)]
    for seed in SEEDS:models+=read(OUT/f'seed{seed}_complete.json')['models']
    verify_files({r['checkpoint']:r['sha256'] for r in models if r['checkpoint']})
    dump(OUT/'models.json',models)
    dump(OUT/'training_complete.json',dict(registry_sha256=sha(OUT/'models.json'),
        protocol_sha256=sha(OUT/'protocol.json'),all_three_seeds=True,all_component_deletions=True,
        matched_comparisons=True,device='cpu',new_evaluation_started=False))
    print('TRAINING_COMPLETE',len(models),'models including frozen',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('freeze','preflight','run_seed','register'))
    ap.add_argument('--seed',type=int,choices=SEEDS);a=ap.parse_args();log(OUT,'start',**vars(a))
    try:
        if a.action=='run_seed':run_seed(a.seed)
        else:globals()[a.action]()
    except BaseException as e:log(OUT,'failed',error=repr(e),**vars(a));raise
    log(OUT,'complete',**vars(a))


if __name__=='__main__':main()
