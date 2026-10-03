"""Fixed joint output-adapter recipe, matched ranking and three-seed deletions."""
from mirror.cases.color_binding import routing_image_joint_pilot as parent
from mirror.cases.color_binding import routing_moderate_replication as previous
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import argparse
import time
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files

base=parent.base
OUT=ROOT/'clip/interbind_routing_joint_replication_20260929'
PLAN=ROOT/'FSE_VLM/plan/83_routing_joint_replication_20260929.md'
SEEDS=(42,43,44)
DELETIONS={'no_cross':('cross',),'no_response':('response',),'no_preference':('preference',),
           'no_interaction':('binding','cross','response')}
ARMS=('IS','Ranking',*DELETIONS)
PRIORITIES=dict(binding=1,cross=4,response=1,preference=4)
CAL=parent.OUT/'calibration_joint.json'


def initial_path(seed):return base.old.OUT/f'seed{seed}/IS/initial.pt'


def freeze():
    parent.verify()
    refs={a:read(parent.OUT/'runs'/('Joint_'+a)/'complete.json') for a in ('IS','Ranking')}
    old=[r for r in read(previous.OUT/'models.json') if r['name']=='Previous_IS']
    assert {r['seed'] for r in old}==set(SEEDS)
    files=[PLAN,Path(__file__),CAL,parent.OUT/'protocol.json',parent.OUT/'preflight.json',
           parent.OUT/'selection.json',parent.OUT/'complete.json',base.prior.OUT/'normalization.json']
    files += [Path(m.__file__) for m in (parent,base,base.prior,base.cn,base.prev,base.old)]
    files += [initial_path(s) for s in SEEDS]
    for r in [*refs.values(),*old]:files += [Path(r['checkpoint']),Path(r['checkpoint']).with_name('history.jsonl')]
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in files},seeds=SEEDS,arms=ARMS,
        deletions=DELETIONS,seed42_reused=refs,previous_text_IS=old,epochs=36,updates=1944,
        device='cpu',parameters=98304,calibration_sha256=sha(CAL),grad_clip=1.,
        selection='fixed_last; author-authorized replication despite missed pilot suppression screen',
        historical_test_outcomes_known=True,new_joint_test_outcomes_seen=False))
    print('PROTOCOL_FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    p=read(OUT/'protocol.json');verify_files(p['inputs']);parent.verify();return p


def make(seed):
    torch.manual_seed(seed);model=parent.SideMaps('joint')
    state=torch.load(initial_path(seed),map_location='cpu',weights_only=False)['state_dict']
    assert torch.count_nonzero(state['B.weight'])==0
    with torch.no_grad():
        model.image_map.A.weight.copy_(state['A.weight'][:32])
        model.text_map.A.weight.copy_(state['A.weight'][32:])
        model.image_map.B.weight.zero_();model.text_map.B.weight.zero_()
    return model


def objective(parts,weights,arm):
    w=dict(weights)
    for k in DELETIONS.get(arm,()):w[k]=0.
    return parent.objective(parts,w,'Joint_Ranking' if arm=='Ranking' else 'Joint_IS')


def preflight():
    p=verify();base.configure();loaded=base.prior.load_training('cpu');torch.set_num_threads(2)
    cache,spec,ctx,cfg=loaded
    forms=base.prev.template_cache(cache);reps=base.prev.representation_schedule(42)
    current=replace(cache,texts=forms[torch.arange(2560),torch.as_tensor(np.repeat(reps[0],2))])
    weights=read(CAL)['weights'];model=make(42).train()
    assert base._state_hash(model.state_dict())==p['seed42_reused']['IS']['initial_state_hash']
    old=base.prev.lines(Path(p['seed42_reused']['IS']['checkpoint']).with_name('history.jsonl'))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    order=base.old.schedule_for(42)[0];torch.manual_seed(420001);checks=[]
    for j in range(3):
        ids=base.prior.paired_rows(torch.tensor(order[j*24:(j+1)*24]))
        parts,_,_=parent.components(model,current,spec,ctx,ids);loss=objective(parts,weights,'IS')
        assert {k:float(v.detach()) for k,v in parts.items()}==old[j]['components']
        assert float(loss.detach())==old[j]['loss']
        if j==0:
            for arm,keys in DELETIONS.items():
                removed=sum(PRIORITIES[k]*weights[k]*parts[k] for k in keys)
                actual=objective(parts,weights,arm)
                torch.testing.assert_close(actual,loss-removed,rtol=1e-6,atol=1e-6)
                torch.testing.assert_close(parent.gradient(actual,model),parent.gradient(loss-removed,model),rtol=1e-4,atol=2e-6)
        opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert float(gn)==old[j]['gradient_norm'];opt.step();checks.append(dict(update=j+1,exact_replay=True))
    dump(OUT/'preflight.json',dict(seed42_first_three_updates=checks,all_deletion_gradients_verified=True,
        device='cpu',protocol_sha256=sha(OUT/'protocol.json')))
    print('PREFLIGHT_PASS',flush=True)


def train_one(seed,arm,loaded,forms,p):
    dest=OUT/'runs'/f'seed{seed}'/arm
    if (dest/'complete.json').exists():
        r=read(dest/'complete.json');verify_files({r['checkpoint']:r['sha256']});return r
    dest.mkdir(parents=True,exist_ok=False)
    cache,spec,ctx,cfg=loaded;cfg=replace(cfg,epochs=36,seed=seed,grad_clip=1.)
    weights=read(CAL)['weights'];model=make(seed);initial=base._state_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    schedule=base.old.schedule_for(seed);reps=base.prev.representation_schedule(seed)
    history=[];start=time.monotonic()
    for ei,order in enumerate(schedule):
        current=replace(cache,texts=forms[torch.arange(2560),torch.as_tensor(np.repeat(reps[ei],2))])
        model.train();torch.manual_seed(seed*10000+ei+1)
        for j in range(0,1280,24):
            ids=base.prior.paired_rows(torch.tensor(order[j:j+24]))
            parts,_,_=parent.components(model,current,spec,ctx,ids);loss=objective(parts,weights,arm)
            opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
            history.append(dict(epoch=ei+1,rows=ids.tolist(),loss=float(loss.detach()),gradient_norm=float(gn),
                components={k:float(v.detach()) for k,v in parts.items()}))
        if (ei+1)%12==0:print('TRAIN',seed,arm,ei+1,'seconds',round(time.monotonic()-start,1),flush=True)
    assert len(history)==1944 and not torch.cuda.is_initialized()
    ckpt=dest/'last.pt';torch.save(dict(state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},
        mode='joint',configuration=asdict(cfg),optimizer=opt.state_dict(),seed=seed,arm=arm,selection='fixed_last',
        epochs=36,updates=1944,weights=weights,target_deletions=DELETIONS.get(arm,()),
        initial_state_hash=initial,protocol_sha256=sha(OUT/'protocol.json'),calibration_sha256=sha(CAL)),ckpt)
    jsonl(dest/'history.jsonl',history)
    r=dict(name=arm,seed=seed,mode='joint',checkpoint=str(ckpt),sha256=sha(ckpt),initial_state_hash=initial,
        updates=1944,parameters=sum(x.numel() for x in model.parameters()),
        schedule_sha256=base.hash_array(np.asarray(schedule)),representation_sha256=base.hash_array(reps),
        final_rng_sha256=base.hash_array(torch.get_rng_state().numpy()),first_components=history[0]['components'],
        seconds=time.monotonic()-start,grad_clip=1.,device='cpu',initial_path=str(initial_path(seed)),
        calibration_sha256=sha(CAL))
    dump(dest/'complete.json',r);print('TRAIN_COMPLETE',seed,arm,r['seconds'],flush=True);return r


def run_seed(seed):
    p=verify();base.configure();assert (OUT/'preflight.json').exists()
    loaded=base.prior.load_training('cpu');torch.set_num_threads(2);forms=base.prev.template_cache(loaded[0]);regs=[]
    for arm in ARMS:
        if seed==42 and arm in p['seed42_reused']:r=dict(p['seed42_reused'][arm],name=arm,reused=True)
        else:
            log(OUT,'train_start',seed=seed,arm=arm);r=train_one(seed,arm,loaded,forms,p);log(OUT,'train_complete',seed=seed,arm=arm)
        regs.append(r)
    for key in ('initial_state_hash','updates','parameters','schedule_sha256','representation_sha256','final_rng_sha256','calibration_sha256'):
        assert len({r[key] for r in regs})==1,(seed,key)
    assert all(r['first_components']==regs[0]['first_components'] for r in regs)
    dump(OUT/f'seed{seed}_complete.json',dict(models=regs,matched_init_stream_rng_updates=True))


def register():
    p=verify();models=[dict(name='F',seed=0,checkpoint=None)]
    for seed in SEEDS:models+=read(OUT/f'seed{seed}_complete.json')['models']
    models+=p['previous_text_IS'];verify_files({r['checkpoint']:r['sha256'] for r in models if r['checkpoint']})
    dump(OUT/'models.json',models)
    dump(OUT/'training_complete.json',dict(registry_sha256=sha(OUT/'models.json'),protocol_sha256=sha(OUT/'protocol.json'),
        all_three_seeds=True,all_component_deletions=True,matched_comparisons=True,device='cpu',new_evaluation_started=False))
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
