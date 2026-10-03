"""Fixed seeds43/44 replication; existing tint engines and outputs are immutable."""
import argparse
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from mirror.cases.color_binding import routing_tint_midpoint as mid
from mirror.cases.color_binding import routing_tint_balance_retest as ev
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.core.repair import _state_hash
from mirror.cases.color_binding.repair_trainbank import lines

p=mid.p; engine=mid.engine
OUT=p.ROOT/'clip/interbind_routing75_replication_20260930'
OLD=mid.OUT/'directional_retest'
PLAN=p.ROOT/'FSE_VLM/plan/87_routing75_three_seed_replication.md'
ARMS=('Ranking','IS',*ev.olddata.DELETIONS)


def initial(seed):return p.old.OUT/f'seed{seed}/IS/initial.pt'


def representations(seed):
    reps=p.templates.representation_schedule(seed)
    offset=np.random.default_rng(840000+seed).integers(0,2,size=1280)
    order=(np.arange(36)[:,None]//4+offset[None])%2
    order[32:]=(np.array([0,0,1,1])[:,None]+offset[None])%2
    assert np.all(order.sum(0)==18)
    assert all(np.all((reps==k).sum(0)==9) for k in range(4))
    return reps+4*order


def freeze():
    mid.verify();verify_files(read(OLD/'wrapper_protocol.json')['inputs'])
    for family in ('primary','transfer','natural','diagnostics'):
        verify_files(read(OLD/'analysis'/family/'complete.json')['files'])
    paths=[PLAN,Path(__file__),Path(engine.__file__),Path(mid.__file__),Path(ev.__file__),
           Path(ev.analysis.__file__),Path(p.cn.__file__),Path(p.prior.__file__),
           Path(p.templates.__file__),Path(p.metrics.__file__),
           mid.OUT/'calibration/75.json',p.prior.OUT/'normalization.json',
           OLD/'choice.json',OLD/'complete.json',OLD/'ablations.json',mid.OUT/'models.json',
           *[initial(s) for s in (42,43,44)],
           *[OLD/'features'/f/'complete.json' for f in ('primary',*ev.olddata.FAMILIES)],
           *[OLD/'analysis'/f/'complete.json' for f in ('primary','transfer','natural','diagnostics')]]
    assert np.array_equal(representations(42),engine.representation_schedule('color_order'))
    dump(OUT/'protocol.json',dict(inputs={str(f):sha(f) for f in paths},seeds=[42,43,44],new_seeds=[43,44],
        arms=ARMS,tint=75,recipe='color_order',target_strength='original standard, not IS2',
        epochs=36,updates=1944,selection='fixed_last',calibration='reuse seed42 training-only values',
        seed42_existing_output=str(OLD),new_recipe_selection=False,no_manuscript_change=True,
        outcome_knowledge='Seed42 full retest observed; new seeds fixed-recipe replications on the same previously observed banks',
        statistics='5000 paired connected-source draws, plus paired seed and connected-source draws',
        implementations='Exact existing parts/objective/evaluation functions; seed and input/output paths parameterized'))
    dump(OUT/'protocol_hash.json',dict(sha256=sha(OUT/'protocol.json')))
    print('REPLICATION_FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    assert sha(OUT/'protocol.json')==read(OUT/'protocol_hash.json')['sha256']
    verify_files(read(OUT/'protocol.json')['inputs'])


def train_one(seed,arm,loaded,forms,cal,dest):
    if (dest/'complete.json').exists():
        result=read(dest/'complete.json');assert sha(result['checkpoint'])==result['sha256'];return result
    dest.mkdir(parents=True,exist_ok=False)
    cache,spec,contexts,cfg=loaded;cfg=replace(cfg,seed=seed)
    weights=dict(read(p.prior.OUT/'normalization.json')['weights'])
    deletions=ev.olddata.DELETIONS.get(arm,())
    for k in deletions:weights[k]=0.
    schedule=p.old.schedule_for(seed);reps=representations(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=p.cn.make_model(cfg,'cuda')
    saved=torch.load(initial(seed),map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict']);ih=_state_hash(model.state_dict())
    assert ih==saved['state_hash']
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    history=[];started=time.monotonic()
    for epoch,order in enumerate(schedule):
        current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),torch.tensor(np.repeat(reps[epoch],2),device='cuda')])
        model.train();torch.manual_seed(seed*10000+epoch+1);torch.cuda.manual_seed_all(seed*10000+epoch+1)
        for start in range(0,1280,24):
            ids=p.prior.paired_rows(torch.tensor(order[start:start+24],device='cuda'))
            parts,ce=engine.parts(model,current,spec,contexts,cfg,ids,cal['m0'])
            loss=engine.objective(parts,ce,weights,'color_order','Ranking' if arm=='Ranking' else 'IS',cal['color_weight'])
            opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
            history.append(dict(epoch=epoch+1,rows=ids.cpu().tolist(),loss=float(loss.detach()),ce=float(ce.detach()),
                gradient_norm=float(gn),components={k:float(v.detach()) for k,v in parts.items()}))
        if (epoch+1)%12==0:print('TRAIN',seed,arm,epoch+1,round(time.monotonic()-started,1),flush=True)
    assert len(history)==1944
    path=dest/'last.pt'
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuration=asdict(cfg),
        seed=seed,arm=arm,tint=75,recipe='color_order',weights=weights,color_calibration=cal,deletions=deletions,
        updates=1944,selection='fixed_last',protocol_sha256=sha(OUT/'protocol.json')),path)
    jsonl(dest/'history.jsonl',history)
    result=dict(name=arm,seed=seed,checkpoint=str(path),sha256=sha(path),updates=1944,
        initial_state_hash=ih,schedule_sha256=p.array_hash(schedule),representation_sha256=p.array_hash(reps),
        final_rng_sha256=p.array_hash(torch.cuda.get_rng_state().cpu().numpy()),first_components=history[0]['components'],
        seconds=time.monotonic()-started,deletions=deletions)
    dump(dest/'complete.json',result);print('TRAINED',seed,arm,flush=True);return result


def replay():
    verify();p.prior.configure();loaded,forms=mid.load_training(75)
    result=train_one(42,'IS',loaded,forms,read(mid.OUT/'calibration/75.json'),OUT/'replay42/IS')
    original=read(OLD/'choice.json')['candidate']['model']
    for k in ('initial_state_hash','schedule_sha256','representation_sha256','final_rng_sha256','first_components','updates'):
        assert result[k]==original[k],k
    new=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    old=torch.load(original['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    error=max(float((new[k]-old[k]).abs().max()) for k in old)
    assert error==0,error
    dump(OUT/'replay42_complete.json',dict(max_parameter_error=error,all_1944_updates_replayed=True,
         original_checkpoint=original['checkpoint'],replay_checkpoint=result['checkpoint']))
    print('EXACT_SEED42_REPLAY',error,flush=True)


def train(seed):
    verify();assert seed in (43,44);assert read(OUT/'replay42_complete.json')['max_parameter_error']==0
    p.prior.configure();loaded,forms=mid.load_training(75);dest=OUT/f'seed{seed}'
    cal=read(mid.OUT/'calibration/75.json')
    results=[train_one(seed,arm,loaded,forms,cal,dest/'runs'/arm) for arm in ARMS]
    for key in ('initial_state_hash','schedule_sha256','representation_sha256','final_rng_sha256','updates','first_components'):
        assert all(r[key]==results[0][key] for r in results),key
    dump(dest/'models.json',results)
    dump(dest/'training_complete.json',dict(seed=seed,matched_initialization_schedule_rng_and_guards=True,
        models_sha256=sha(dest/'models.json'),arms=ARMS))


def evaluate(seed):
    verify();assert seed in (43,44);dest=OUT/f'seed{seed}'
    regs=read(dest/'models.json');verify_files({r['checkpoint']:r['sha256'] for r in regs})
    ev.OUT=dest;ev.analysis.OUT=dest/'analysis'
    ev.selected=lambda:dict(tint=75,recipe='color_order',name='IS')
    ev.registry=lambda ablations=False:[dict(name='Frozen',checkpoint=None,seed=seed),
        *[r for r in regs if ablations or r['name'] in ('Ranking','IS')]]
    ev.primary_path=lambda:OLD/'features/primary'
    original_arrays=ev.arrays
    def arrays(*args):
        # Original reader uses OUT only to locate immutable feature banks.
        ev.OUT=OLD
        try:return original_arrays(*args)
        finally:ev.OUT=dest
    ev.arrays=arrays
    def check():verify();ev.analysis.OUT=dest/'analysis'
    ev.verify=check
    OriginalRecorder=ev.analysis.Recorder
    class SeedRecorder(OriginalRecorder):
        def emit(self,*args,**kwargs):
            before=len(self.summary);super().emit(*args,**kwargs)
            for r in self.summary[before:]:r['seed']=seed
        def finish(self):
            dump(self.dest/'seed_metadata.json',dict(seed=seed,protocol_sha256=sha(OUT/'protocol.json')))
            super().finish()
    ev.analysis.Recorder=SeedRecorder
    for stage in ('primary','transfer','natural','diagnostics'):
        complete=dest/'analysis'/stage/'complete.json'
        if complete.exists():verify_files(read(complete)['files']);continue
        log(dest,'evaluation_start',stage=stage,seed=seed);getattr(ev,stage)()
        log(dest,'evaluation_complete',stage=stage,seed=seed)
    dump(dest/'complete.json',dict(seed=seed,training_and_all_evaluations_complete=True,
        artifacts={str(dest/'analysis'/s/'complete.json'):sha(dest/'analysis'/s/'complete.json') for s in ('primary','transfer','natural','diagnostics')}))
    print('SEED_COMPLETE',seed,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','replay','train','evaluate','run']);ap.add_argument('--seed',type=int)
    a=ap.parse_args();log(OUT,'start',action=a.action,seed=a.seed)
    try:
        if a.action=='freeze':freeze()
        elif a.action=='replay':replay()
        elif a.action=='train':train(a.seed)
        elif a.action=='evaluate':evaluate(a.seed)
        else:
            if not (OUT/f'seed{a.seed}/training_complete.json').exists():train(a.seed)
            evaluate(a.seed)
    except BaseException as exc:log(OUT,'failed',action=a.action,seed=a.seed,error=repr(exc));raise
    log(OUT,'complete',action=a.action,seed=a.seed)


if __name__=='__main__':main()
