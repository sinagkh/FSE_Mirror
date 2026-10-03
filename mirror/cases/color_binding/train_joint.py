"""Fixed 75%-tint joint extension; original joint implementations are read-only."""
from mirror.cases.color_binding import routing_joint_replication as old
from mirror.cases.color_binding import replicate as text
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files

OUT=ROOT/'clip/interbind_routing75_complete_20260930/joint'
PLAN=ROOT/'FSE_VLM/plan/88_uniform_routing75_completion.md'
p=text.p

def parts(model,cache,spec,ctx,ids,m0):
    terms,scores,frozen=old.parent.components(model,cache,spec,ctx,ids)
    a=text.engine.color_margins(scores[:,:,:4],spec.unit)
    f=text.engine.color_margins(frozen[:,:,:4],spec.unit)
    terms['color_floor']=p.prior.worse_layout(F.relu(torch.maximum(f,f.new_tensor(m0))-a-p.prior.EPS))
    return terms

def objective(terms,weights,arm,cal):
    return old.objective(terms,weights,arm)+4*cal['color_weight']*terms['color_floor']

def freeze():
    paths=[PLAN,Path(__file__),Path(old.__file__),Path(old.parent.__file__),old.CAL,
           Path(text.__file__),Path(text.engine.__file__),text.OUT/'protocol.json',
           text.mid.OUT/'calibration/75.json',*[text.initial(s) for s in (42,43,44)]]
    dump(OUT/'protocol.json',dict(inputs={str(f):sha(f) for f in paths},tint=.75,seeds=[42,43,44],
        arms=old.ARMS,selection='fixed_last',updates=1944,epochs=36,rank_per_side=32,
        original_joint_weights=True,shared_new_color_floor=True,schedule='exact text75 color_order schedule',
        no_new_search=True,no_encoder_training=True,device='cpu'))
    dump(OUT/'protocol_hash.json',dict(sha256=sha(OUT/'protocol.json')))

def verify():
    assert sha(OUT/'protocol.json')==read(OUT/'protocol_hash.json')['sha256']
    verify_files(read(OUT/'protocol.json')['inputs'])

def calibrate(loaded,forms):
    dest=OUT/'calibration.json'
    if dest.exists():return read(dest)
    cache,spec,ctx,cfg=loaded
    with torch.no_grad():
        mm=torch.cat([text.engine.color_margins(cache.images@forms[:,i,:4].transpose(1,2),spec.unit).flatten() for i in range(4)])
        m0=float(.25*mm[mm>0].median())
    assert abs(m0-read(text.mid.OUT/'calibration/75.json')['m0'])<1e-6
    model=old.make(42).eval();gen=torch.Generator().manual_seed(20260923)
    with torch.no_grad():
        for name,value in model.named_parameters():
            if name.endswith('B.weight'):value.copy_(torch.randn(value.shape,generator=gen)*.001)
    order=np.random.default_rng(20260923).permutation(1280);records=[]
    for j in range(8):
        ids=p.prior.paired_rows(torch.tensor(order[j*24:(j+1)*24]))
        term=parts(model,cache,spec,ctx,ids,m0)['color_floor']
        records.append(dict(rows=ids.tolist(),gradient_norm=float(old.parent.gradient(term,model).norm())))
    reference=read(old.CAL)['reference_norm'];norm=float(np.mean([r['gradient_norm'] for r in records]))
    cal=dict(m0=m0,color_weight=float(np.clip(reference/max(norm,reference/10),.1,10)),
             reference_norm=reference,gradient_norm=norm,batches=records,training_only=True,shared_across_seeds=True)
    dump(dest,cal);return cal

def preflight(loaded,forms,cal):
    cache,spec,ctx,cfg=loaded;weights=read(old.CAL)['weights'];checks=[]
    for seed in (42,43,44):
        reps=text.representations(seed);schedule=p.old.schedule_for(seed)
        current=replace(cache,texts=forms[torch.arange(2560),torch.tensor(np.repeat(reps[0],2))])
        ids=p.prior.paired_rows(torch.tensor(schedule[0][:24]));reference=None
        for arm in old.ARMS:
            model=old.make(seed).train();torch.manual_seed(seed*10000+1)
            terms=parts(model,current,spec,ctx,ids,cal['m0'])
            record=({k:float(v.detach()) for k,v in terms.items()},p.array_hash(torch.get_rng_state().numpy()))
            if reference is None:reference=record
            assert record==reference,(seed,arm)
            loss=objective(terms,weights,arm,cal)
            if arm in old.DELETIONS:
                full=objective(terms,weights,'IS',cal)
                removed=sum(old.PRIORITIES[k]*weights[k]*terms[k] for k in old.DELETIONS[arm])
                torch.testing.assert_close(loss,full-removed,rtol=1e-6,atol=1e-6)
                torch.testing.assert_close(old.parent.gradient(loss,model),old.parent.gradient(full-removed,model),rtol=1e-4,atol=2e-6)
            opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step()
            assert all(torch.isfinite(v).all() for v in model.parameters())
            checks.append(dict(seed=seed,arm=arm,loss=float(loss.detach()),shared_components_and_rng=True))
    dump(OUT/'preflight.json',dict(checks=checks,all_deletion_gradients_verified=True,
        calibration_sha256=sha(OUT/'calibration.json'),no_gpu_training=True))

def train_one(seed,arm,loaded,forms,cal):
    dest=OUT/f'seed{seed}'/arm
    if (dest/'complete.json').exists():
        r=read(dest/'complete.json');assert sha(r['checkpoint'])==r['sha256'];return r
    dest.mkdir(parents=True,exist_ok=False)
    cache,spec,ctx,cfg=loaded;cfg=replace(cfg,seed=seed)
    weights=read(old.CAL)['weights'];model=old.make(seed);initial=old.base._state_hash(model.state_dict())
    schedule=p.old.schedule_for(seed);reps=text.representations(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    history=[];started=time.monotonic()
    for epoch,order in enumerate(schedule):
        current=replace(cache,texts=forms[torch.arange(2560),torch.tensor(np.repeat(reps[epoch],2))])
        model.train();torch.manual_seed(seed*10000+epoch+1)
        for start in range(0,1280,24):
            ids=p.prior.paired_rows(torch.tensor(order[start:start+24]))
            terms=parts(model,current,spec,ctx,ids,cal['m0']);loss=objective(terms,weights,arm,cal)
            opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
            history.append(dict(epoch=epoch+1,rows=ids.tolist(),loss=float(loss.detach()),gradient_norm=float(gn),
                                components={k:float(v.detach()) for k,v in terms.items()}))
        if (epoch+1)%12==0:print('JOINT_TRAIN',seed,arm,epoch+1,round(time.monotonic()-started,1),flush=True)
    assert len(history)==1944 and not torch.cuda.is_initialized()
    path=dest/'last.pt'
    torch.save(dict(state_dict=model.state_dict(),mode='joint',configuration=asdict(cfg),seed=seed,arm=arm,
        weights=weights,color_calibration=cal,target_deletions=old.DELETIONS.get(arm,()),
        tint=75,recipe='color_order',updates=1944,selection='fixed_last',protocol_sha256=sha(OUT/'protocol.json')),path)
    jsonl(dest/'history.jsonl',history)
    result=dict(name=arm,seed=seed,mode='joint',checkpoint=str(path),sha256=sha(path),initial_state_hash=initial,
        schedule_sha256=p.array_hash(schedule),representation_sha256=p.array_hash(reps),
        final_rng_sha256=p.array_hash(torch.get_rng_state().numpy()),first_components=history[0]['components'],
        seconds=time.monotonic()-started,updates=1944)
    dump(dest/'complete.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train']);ap.add_argument('--seed',type=int)
    a=ap.parse_args();log(OUT,'start',**vars(a));torch.set_num_threads(2)
    if a.action=='prepare' and not (OUT/'protocol.json').exists():freeze()
    verify();loaded,forms=text.mid.load_training(75,'cpu');cal=calibrate(loaded,forms)
    if a.action=='prepare':preflight(loaded,forms,cal)
    else:
        assert a.seed in (42,43,44) and (OUT/'preflight.json').exists()
        results=[train_one(a.seed,arm,loaded,forms,cal) for arm in old.ARMS]
        for key in ('initial_state_hash','schedule_sha256','representation_sha256','final_rng_sha256','first_components','updates'):
            assert all(r[key]==results[0][key] for r in results),key
        dump(OUT/f'seed{a.seed}/models.json',results)
        dump(OUT/f'seed{a.seed}/training_complete.json',dict(models_sha256=sha(OUT/f'seed{a.seed}/models.json'),all_matched=True))
    log(OUT,'complete',**vars(a))

if __name__=='__main__':main()
