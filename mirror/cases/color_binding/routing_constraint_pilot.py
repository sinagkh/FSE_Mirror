"""Bounded CPU routing experiment; immutable originals and no GPU encoding."""
from mirror.cases.color_binding import routing_suppression_strength as base  # hides CUDA before torch import
from dataclasses import asdict; from dataclasses import replace
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files

OUT = ROOT/'clip/interbind_routing_constraint_pilot_20260929'
PLAN = ROOT/'FSE_VLM/plan/77_routing_context_constraints_20260929.md'
ARMS = {
    'Ranking_C5': dict(ranking=True, cross=0, binding=0, adaptive=False),
    'IS4_C5': dict(ranking=False, cross=4, binding=1, adaptive=False),
    'IS8B4_C5': dict(ranking=False, cross=8, binding=4, adaptive=False),
    'Ranking_Dual': dict(ranking=True, cross=0, binding=0, adaptive=True),
    'IS8_Dual': dict(ranking=False, cross=8, binding=1, adaptive=True),
    'IS16_Dual': dict(ranking=False, cross=16, binding=1, adaptive=True),
}
VIEWS = ('canvas', 'swapped_canvas')


def freeze():
    base.verify()
    paths = [PLAN, Path(__file__), base.OUT/'protocol.json', base.INITIAL,
             OUT/'training_only_diagnosis.json', base.prior.OUT/'normalization.json']
    paths += [Path(m.__file__) for m in (base, base.prior, base.cn, base.prev, base.metrics)]
    dump(OUT/'protocol.json', dict(inputs={str(p):sha(p) for p in paths}, arms=ARMS,
         device='cpu', seed=42, epochs=36, updates=1944, clip=5., dual_step=20.,
         dual_cap=100., binding_buffer_fraction=.02, binding_buffer_min_scale=.1,
         selection='fixed_last_then_frozen_development_gate', test_scoring_started=False))
    print('FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    p=read(OUT/'protocol.json');verify_files(p['inputs']);base.verify();return p


def components(model,cache,spec,contexts,cfg,ids):
    capture=[]
    hook=model.register_forward_hook(lambda _m,_a,z:capture.append(z))
    try: parts=base.prior.components(model,cache,spec,contexts,cfg,ids)
    finally: hook.remove()
    assert len(capture)==1
    scores=cache.images[ids]@capture[0][:,:4].transpose(1,2)
    frozen=cache.images[ids]@cache.texts[ids,:4].transpose(1,2)
    d=base.prior.measurements(scores,spec,contexts)[0]
    df=base.prior.measurements(frozen,spec,contexts)[0]
    ce=F.cross_entropy(100*scores.reshape(-1,4),torch.arange(4).repeat(len(ids)))
    return parts,ce,d,df


def objective(parts,weights,arm,ce):
    a=ARMS[arm]
    if a['ranking']: return base.objective(parts,weights,'Ranking',ce)
    return (base.objective(parts,weights,'IS'+str(a['cross']),ce)
            +(a['binding']-1)*weights['binding']*parts['binding'])


def constraint_penalty(d,df,ids,multipliers,buffer):
    colors=(ids//2)%2
    return (multipliers[colors]*(df+buffer[colors]-d)).mean()


def update_dual(lambdas,g,p):
    return torch.clamp(lambdas+p['dual_step']*g,min=0,max=p['dual_cap'])


@torch.inference_mode()
def monitor(model,cache,forms,spec,contexts):
    """Full training-only context means, no RNG use; same calls in all arms."""
    model.eval(); ds=[]; fs=[]
    for start in range(0,2560,128):
        v=cache.images[start:start+128];t=forms[start:start+128,:,:4]
        pred=[];ref=[]
        for form in range(4):
            text=t[:,form]
            pred.append(base.prior.measurements(v@model(text).transpose(1,2),spec,contexts)[0])
            ref.append(base.prior.measurements(v@text.transpose(1,2),spec,contexts)[0])
        ds.append(torch.stack(pred,1).mean(1));fs.append(torch.stack(ref,1).mean(1))
    return (torch.cat(ds).reshape(640,2,2,8).mean((0,2)),
            torch.cat(fs).reshape(640,2,2,8).mean((0,2)))


def preflight():
    p=verify();base.configure()
    cache,spec,ctx,cfg=base.prior.load_training('cpu');torch.set_num_threads(2)
    forms=base.prev.template_cache(cache);model=base.new_model(cfg).train()
    current=replace(cache,texts=forms[:,1]);ids=base.prior.paired_rows(torch.arange(24))
    torch.manual_seed(420001)
    parts,ce,d,df=components(model,current,spec,ctx,cfg,ids)
    weights=read(base.prior.OUT/'normalization.json')['weights']
    params=tuple(model.parameters())
    def gradients(loss):
        return torch.cat([g.flatten() for g in torch.autograd.grad(loss,params,retain_graph=True)])
    expected=base.objective(parts,weights,'IS4',ce)
    assert torch.equal(expected,objective(parts,weights,'IS4_C5',ce))
    torch.testing.assert_close(gradients(expected),gradients(objective(parts,weights,'IS4_C5',ce)),rtol=0,atol=0)
    extra=constraint_penalty(d,df,ids,torch.zeros(2,8),torch.ones(2,8)*.02)
    torch.testing.assert_close(gradients(expected+extra),gradients(expected),rtol=0,atol=0)
    synthetic=torch.zeros(8,8,requires_grad=True);row=torch.arange(8)
    penalty=constraint_penalty(synthetic,torch.ones_like(synthetic),row,torch.ones(2,8),torch.zeros(2,8))
    assert (torch.autograd.grad(penalty,synthetic)[0]<0).all()
    assert ((torch.arange(2560)//2)%2).reshape(640,2,2)[0].tolist()==[[0,0],[1,1]]
    assert update_dual(torch.zeros(2),torch.tensor([1.,-1.]),p).tolist()==[20.,0.]
    assert update_dual(torch.ones(2)*99,torch.ones(2),p).tolist()==[100.,100.]
    rng=torch.get_rng_state().clone();dm,fm=monitor(model,cache,forms,spec,ctx)
    assert torch.equal(rng,torch.get_rng_state())
    torch.testing.assert_close(dm,fm,atol=2e-6,rtol=2e-6)
    dump(OUT/'preflight.json',dict(single_forward=True,original_loss_and_gradient_exact=True,
         zero_dual_gradient_exact=True,binding_gradient_sign=True,color_indexing=True,
         projected_dual_update=True,monitor_no_rng_consumption=True,identity_monitor_matches=True,
         frozen_training_means=fm.tolist(),device='cpu'))
    print('PREFLIGHT_PASS',flush=True)


def train(arm):
    p=verify();base.configure();assert (OUT/'preflight.json').exists()
    dest=OUT/'runs'/arm;dest.mkdir(parents=True,exist_ok=False)
    cache,spec,ctx,cfg=base.prior.load_training('cpu');torch.set_num_threads(2)
    cfg=replace(cfg,epochs=36,seed=42,grad_clip=5.)
    forms=base.prev.template_cache(cache);weights=read(base.prior.OUT/'normalization.json')['weights']
    model=base.new_model(cfg);initial=base._state_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    schedule=base.old.schedule_for(42);reps=base.prev.representation_schedule(42)
    frozen=torch.tensor(read(OUT/'preflight.json')['frozen_training_means'])
    buffer=p['binding_buffer_fraction']*frozen.abs().clamp_min(p['binding_buffer_min_scale'])
    lambdas=torch.zeros(2,8);history=[];diagnostics=[];start=time.monotonic()
    for ei,order in enumerate(schedule):
        current=replace(cache,texts=forms[torch.arange(2560),torch.as_tensor(np.repeat(reps[ei],2))])
        model.train();torch.manual_seed(420000+ei+1)
        for j in range(0,1280,24):
            ids=base.prior.paired_rows(torch.tensor(order[j:j+24]))
            parts,ce,d,df=components(model,current,spec,ctx,cfg,ids)
            main=objective(parts,weights,arm,ce)
            penalty=constraint_penalty(d,df,ids,lambdas,buffer)
            loss=main+penalty
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
            history.append(dict(epoch=ei+1,rows=ids.tolist(),loss=float(loss.detach()),
                 adaptive_penalty=float(penalty.detach()),gradient_norm=float(grad),
                 ce=float(ce.detach()),components={k:float(v.detach()) for k,v in parts.items()}))
        dm,fm=monitor(model,cache,forms,spec,ctx)
        torch.testing.assert_close(fm,frozen,rtol=0,atol=0)
        g=frozen+buffer-dm
        before=lambdas.clone()
        if ARMS[arm]['adaptive']: lambdas=update_dual(lambdas,g,p)
        diagnostics.append(dict(epoch=ei+1,binding_mean=dm.tolist(),constraint=g.tolist(),
                          lambda_used=before.tolist(),lambda_next=lambdas.tolist()))
        if (ei+1)%6==0:
            print('TRAIN',arm,ei+1,'seconds',round(time.monotonic()-start,1),
                  'max_constraint',round(float(g.max()),5),'max_dual',round(float(lambdas.max()),3),flush=True)
    assert len(history)==1944 and not torch.cuda.is_initialized()
    ckpt=dest/'last.pt'
    torch.save(dict(state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},
        configuration=asdict(cfg),optimizer=opt.state_dict(),seed=42,arm=arm,selection='fixed_last',
        epochs=36,updates=1944,device='cpu',weights=weights,multipliers=lambdas,
        initial_state_hash=initial,protocol_sha256=sha(OUT/'protocol.json')),ckpt)
    jsonl(dest/'history.jsonl',history);jsonl(dest/'constraints.jsonl',diagnostics)
    receipt=dict(name=arm,seed=42,checkpoint=str(ckpt),sha256=sha(ckpt),initial_state_hash=initial,
        updates=1944,device='cpu',schedule_sha256=base.hash_array(np.asarray(schedule)),
        representation_sha256=base.hash_array(reps),final_rng_sha256=base.hash_array(torch.get_rng_state().numpy()),
        first_components=history[0]['components'],seconds=time.monotonic()-start,
        monitor_passes=36*4,grad_clip=5.,adaptive=ARMS[arm]['adaptive'])
    dump(dest/'complete.json',receipt);print('TRAIN_COMPLETE',arm,receipt['seconds'],flush=True)


def evaluate():
    verify();base.configure()
    regs=[read(base.OUT/'runs'/a/'complete.json') for a in ('IS4','Ranking')]
    regs += [read(base.OUT/'binding_balance/runs/IS8B4/complete.json')]
    regs += [read(OUT/'runs'/a/'complete.json') for a in ARMS]
    for key in ('initial_state_hash','updates','schedule_sha256','representation_sha256','final_rng_sha256'):
        assert len({r[key] for r in regs})==1,key
    assert all(r['first_components']==regs[0]['first_components'] for r in regs)
    regs=[dict(name='F',seed=0,checkpoint=None),*regs]
    rows=base.prev.development(regs,OUT/'development.jsonl');means=base.prev.means(rows)
    arrays={};ctx={}
    features=np.load(base.prev.OUT/'training_template_features.npy')
    lookup={s:i for i,s in enumerate(read(base.prev.OUT/'training_template_strings.json'))}
    groups=base.prev.lines(base.prev.DEV/'features/text_groups.jsonl')
    for view in VIEWS:
        v,t,idx=base.metrics.bank_arrays(base.prev.DEV/'features','routing','red-blue',view)
        arrays[view+'/ids']=np.asarray([r['anchor_id'] for r in idx])
        forms=np.stack([features[[lookup[s] for ti in r['text_indices'][:4]
            for s in groups[ti]['templates']]].reshape(4,3,768).transpose(1,0,2) for r in idx])
        for reg in regs:
            x=base.metrics.score_arrays(v,t,reg['checkpoint']);name=reg['name']
            arrays[view+'/'+name]=x
            arrays[view+'/'+name+'/templates']=np.einsum('nid,nkjd->nkij',v,base.metrics.adapt(forms,reg['checkpoint']))
            ctx[view+'/'+name]={k:float(a.mean()) for k,a in base.metrics.routing(x).items() if k.startswith('contrast/')}
    with (OUT/'development_scores.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    f=means['F'];old=means['IS4'];candidates=[]
    for arm,config in ARMS.items():
        if config['ranking']:continue
        a=means[arm]
        differences={r['name']:float(np.mean([ctx[v+'/'+arm]['contrast/'+r['name']]-ctx[v+'/F']['contrast/'+r['name']] for v in VIEWS]))
                     for r in base.metrics.all_contexts() if r['kind']=='binding'}
        checks=dict(cross_20percent=a['cross']<=.8*f['cross'],binding_contexts=min(differences.values())>=0,
                    pooled_accuracy=a['exchange_accuracy']>=old['exchange_accuracy']-.005,
                    individual_accuracy=a['individual_exchange']>=old['individual_exchange']-.01)
        for view in VIEWS:
            rr=next(r for r in rows if r['name']==arm and r['view']==view)
            ff=next(r for r in rows if r['name']=='F' and r['view']==view)
            checks[view+'/words']=min(rr['word1_accuracy'],rr['word2_accuracy'])>=.99
            checks[view+'/object']=rr['object_guard']>=ff['object_guard']-.01
        candidates.append(dict(name=arm,eligible=all(checks.values()),checks=checks,binding_changes=differences))
    eligible=[r['name'] for r in candidates if r['eligible']]
    selected=min(eligible,key=lambda n:(means[n]['cross'],-means[n]['exchange_accuracy'],n)) if eligible else None
    ranking=('Ranking_Dual' if ARMS[selected]['adaptive'] else 'Ranking_C5') if selected else None
    dump(OUT/'development_means.json',means);dump(OUT/'development_contexts.json',ctx)
    dump(OUT/'selection.json',dict(selected=selected,matched_ranking=ranking,candidates=candidates,
         test_scoring_started=False,matched_initialization_schedule_rng=True,source='888 development anchors',
         protocol_sha256=sha(OUT/'protocol.json')))
    for arm,a in means.items():
        print('DEVELOPMENT',arm,{k:round(a[k],6) for k in ('binding','cross','exchange_accuracy','individual_exchange')},flush=True)
    print('SELECTION',selected,ranking,candidates,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('freeze','preflight','train','evaluate'))
    ap.add_argument('--arm',choices=ARMS);args=ap.parse_args();log(OUT,'start',action=args.action,arm=args.arm)
    try:
        if args.action=='train': train(args.arm)
        else: globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',error=repr(exc),action=args.action,arm=args.arm);raise
    log(OUT,'complete',action=args.action,arm=args.arm)


if __name__=='__main__':main()
