"""Matched seed42 tint/guard/order study; no held-out model selection."""
import argparse
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mirror.cases.color_binding import routing_tint_balance_data as data
from mirror.cases.color_binding import routing_light_tint as p
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import lines
from mirror.core.repair import _state_hash

OUT=data.OUT
RECIPES=('base','color_floor','color_order')
ARMS=('Ranking','IS','IS2')


def representation_schedule(recipe):
    rep=p.templates.representation_schedule(42)
    if recipe=='color_order':
        offset=np.random.default_rng(840042).integers(0,2,size=1280)
        order=(np.arange(36)[:,None]//4+offset[None])%2
        # Eight full alternating blocks balance the first 32 epochs; balance
        # the final four explicitly (nine alternating blocks would be 20/16).
        order[32:]=(np.array([0,0,1,1])[:,None]+offset[None])%2
        assert np.all(order.sum(0)==18)
        rep=rep+4*order
    assert all(np.all((rep%4==k).sum(0)==9) for k in range(4))
    return rep


def freeze():
    data.verify();verify_files(read(OUT/'encoding_complete.json')['files'])
    paths=[Path(__file__),Path(data.__file__),data.PLAN,OUT/'data_protocol.json',OUT/'encoding_complete.json',
           p.prior.OUT/'normalization.json',p.OUT/'training_complete.json']
    dump(OUT/'training_protocol.json',dict(inputs={str(f):sha(f) for f in paths},tints=data.TINTS,recipes=RECIPES,
        arms={'base':['Ranking','IS'],'color_floor':['Ranking','IS'],'color_order':list(ARMS)},
        seed=42,epochs=36,updates=1944,fixed_last=True,train_from_original_identity=True,
        within_setting_comparisons=True,shared_new_guard=True,shared_new_order_schedule=True,
        new_guard='mean hinge(max(frozen single-word margin,m0)-adapted margin-EPS); original guards retained',
        m0='0.25 times median positive frozen single-word training margin, both colors/layouts/four canonical representations',
        guard_weight='original reference gradient norm / new guard norm on same eight training batches; clip [.1,10]; priority4',
        existing_weights='unchanged; IS2 changes only cross target coefficient by2',
        order_schedule='Eight alternating four-epoch blocks, then a balanced 2+2 final block; exactly18 epochs per order/source',
        selection_rule='Complete plan84; same frozen criteria for every tint; no new-checkpoint heldout scores consulted',
        old_test_outcomes_known=True,no_paper_change=True,no_additional_seeds=True))
    dump(OUT/'training_protocol_hash.json',dict(sha256=sha(OUT/'training_protocol.json')))
    print('TRAINING_FROZEN',flush=True)


def verify():
    data.verify();assert sha(OUT/'training_protocol.json')==read(OUT/'training_protocol_hash.json')['sha256']
    verify_files(read(OUT/'training_protocol.json')['inputs'])


def color_margins(scores,unit):
    diagonal=scores.diagonal(dim1=1,dim2=2)
    ids=torch.arange(4,device=scores.device)
    return torch.stack((diagonal-scores[:,ids,ids^2],diagonal-scores[:,ids,ids^1]),dim=-1)/unit


def parts(model,cache,spec,contexts,cfg,ids,m0):
    capture=[];hook=model.register_forward_hook(lambda _m,_args,out:capture.append(out))
    try:result=p.prior.components(model,cache,spec,contexts,cfg,ids)
    finally:hook.remove()
    assert len(capture)==1
    v=cache.images[ids];x=v@capture[0][:,:4].transpose(1,2)
    frozen=v@cache.texts[ids,:4].transpose(1,2)
    old=color_margins(frozen,spec.unit);new=color_margins(x,spec.unit)
    floor=torch.maximum(old,old.new_tensor(m0))
    result['color_floor']=p.prior.worse_layout(F.relu(floor-new-p.prior.EPS))
    ce=F.cross_entropy(100*x.reshape(-1,4),torch.arange(4,device=x.device).repeat(len(ids)))
    return result,ce


def objective(terms,ce,weights,recipe,arm,color_weight):
    value=ce+p.prior.objective(terms,weights,'G') if arm=='Ranking' else p.cn.objective(terms,weights)
    if recipe!='base':value=value+p.prior.PRIORITY*color_weight*terms['color_floor']
    return value


def calibrate(tint,loaded,forms):
    cache,spec,contexts,cfg=loaded;dest=OUT/'calibration'/f'{tint}.json'
    if dest.exists():return read(dest)
    with torch.inference_mode():
        margins=[color_margins(cache.images@forms[:,i,:4].transpose(1,2),spec.unit).flatten() for i in range(4)]
        allm=torch.cat(margins);positive=allm[allm>0]
        m0=float(.25*positive.median())
    original=read(p.prior.OUT/'normalization.json');model=p.make(cfg).eval()
    gen=torch.Generator(device='cuda').manual_seed(original['seed'])
    with torch.no_grad():model.B.weight.copy_(torch.randn(model.B.weight.shape,device='cuda',generator=gen)*.001)
    records=[]
    for row in original['batches']:
        ids=p.prior.paired_rows(torch.tensor(row['blocks'],device='cuda'))
        result,ce=parts(model,cache,spec,contexts,cfg,ids,m0)
        gradients=torch.autograd.grad(result['color_floor'],tuple(model.parameters()))
        norm=float(torch.sqrt(sum(g.square().sum() for g in gradients)))
        records.append(dict(blocks=row['blocks'],gradient_norm=norm,loss=float(result['color_floor'].detach())))
    norm=float(np.mean([r['gradient_norm'] for r in records]));ref=original['reference_norm']
    weight=float(np.clip(ref/max(norm,ref/10),.1,10))
    assert m0>0 and norm>0 and np.isfinite(weight)
    record=dict(tint=tint,m0=m0,color_weight=weight,original_reference_norm=ref,new_gradient_norm=norm,
        calibration=records,training_only=True,optimizer_updates=0,
        frozen_single_word_accuracy=float((allm>0).float().mean()),floor_unit=spec.unit,
        original_normalization_sha256=sha(p.prior.OUT/'normalization.json'))
    dump(dest,record);print('CALIBRATED',tint,'m0',m0,'weight',weight,flush=True);return record


def train_one(tint,recipe,arm,loaded,forms,cal):
    cache,spec,contexts,cfg=loaded;dest=OUT/'runs'/str(tint)/recipe/arm
    if (dest/'complete.json').exists():
        r=read(dest/'complete.json');assert sha(r['checkpoint'])==r['sha256'];return r
    dest.mkdir(parents=True,exist_ok=False)
    weights=dict(read(p.prior.OUT/'normalization.json')['weights'])
    if arm=='IS2':weights['cross']*=2
    schedule=p.old.schedule_for(42);reps=representation_schedule(recipe)
    model=p.make(cfg);initial=_state_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    history=[];start=time.monotonic()
    for epoch,order in enumerate(schedule):
        current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),torch.tensor(np.repeat(reps[epoch],2),device='cuda')])
        model.train();torch.manual_seed(420001+epoch);torch.cuda.manual_seed_all(420001+epoch)
        for j in range(0,1280,24):
            ids=p.prior.paired_rows(torch.tensor(order[j:j+24],device='cuda'))
            components,ce=parts(model,current,spec,contexts,cfg,ids,cal['m0'])
            loss=objective(components,ce,weights,recipe,arm,cal['color_weight'])
            opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(grad)
            opt.step()
            history.append(dict(epoch=epoch+1,rows=ids.cpu().tolist(),loss=float(loss.detach()),ce=float(ce.detach()),
                gradient_norm=float(grad),components={k:float(v.detach()) for k,v in components.items()}))
        if (epoch+1)%12==0:print('TRAIN',tint,recipe,arm,epoch+1,round(time.monotonic()-start,1),flush=True)
    assert len(history)==1944
    path=dest/'last.pt'
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuration=asdict(cfg),
        seed=42,tint=tint,recipe=recipe,arm=arm,weights=weights,color_calibration=cal,updates=1944,selection='fixed_last',
        protocol_sha256=sha(OUT/'training_protocol.json')),path)
    jsonl(dest/'history.jsonl',history)
    result=dict(name=arm,seed=42,tint=tint,recipe=recipe,checkpoint=str(path),sha256=sha(path),
        initial_state_hash=initial,schedule_sha256=p.array_hash(schedule),representation_sha256=p.array_hash(reps),
        final_rng_sha256=p.array_hash(torch.cuda.get_rng_state().cpu().numpy()),updates=1944,
        seconds=time.monotonic()-start,first_components=history[0]['components'])
    dump(dest/'complete.json',result);return result


def preflight(tint,loaded,forms,cal):
    cache,spec,contexts,cfg=loaded;reps=representation_schedule('color_order')
    current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),torch.tensor(np.repeat(reps[0],2),device='cuda')])
    ids=p.prior.paired_rows(torch.tensor(p.old.schedule_for(42)[0][:24],device='cuda'))
    model=p.make(cfg).train();weights=read(p.prior.OUT/'normalization.json')['weights'];records=[]
    for arm in ('Ranking','IS'):
        torch.manual_seed(420001);torch.cuda.manual_seed_all(420001)
        result,ce=parts(model,current,spec,contexts,cfg,ids,cal['m0'])
        loss=objective(result,ce,weights,'color_order',arm,cal['color_weight'])
        g=torch.autograd.grad(loss,tuple(model.parameters()))
        assert all(torch.isfinite(v).all() for v in g)
        records.append(dict(arm=arm,first_parts={k:float(v.detach()) for k,v in result.items()},
            rng=p.array_hash(torch.cuda.get_rng_state().cpu().numpy()),loss=float(loss.detach())))
    assert records[0]['first_parts']==records[1]['first_parts'] and records[0]['rng']==records[1]['rng']
    # The additional guard is exactly zero when every single-word margin exceeds its floor.
    gold=torch.eye(4,device='cuda')[None].repeat(2,1,1)*(cal['m0']+1)
    assert bool((color_margins(gold,1)>=cal['m0']).all())
    # Each diagonal faces exactly its two Hamming-distance-one negatives.
    probe=torch.arange(16,device='cuda',dtype=torch.float32).reshape(1,4,4)
    expected=torch.tensor([[[-2,-1],[-2,1],[2,-1],[2,1]]],device='cuda')
    assert torch.equal(color_margins(probe,1),expected)
    dump(OUT/'preflight'/f'{tint}.json',dict(shared_forward_and_rng=True,finite_gradients=True,checks=records,
        both_layouts_same_caption_order=True,order_count_per_source=18,canonical_representation_count=9))


def train():
    verify();p.prior.configure();records=[]
    assert torch.cuda.mem_get_info()[0]>12*1024**3,'GPU busy; no jobs interrupted'
    for tint in data.TINTS:
        loaded,forms=data.load_training(tint);cal=calibrate(tint,loaded,forms)
        if not (OUT/'preflight'/f'{tint}.json').exists():preflight(tint,loaded,forms,cal)
        for recipe in RECIPES:
            group=[]
            for arm in ARMS if recipe=='color_order' else ARMS[:2]:group.append(train_one(tint,recipe,arm,loaded,forms,cal))
            for key in ('initial_state_hash','schedule_sha256','representation_sha256','final_rng_sha256','updates','first_components'):
                assert all(g[key]==group[0][key] for g in group),key
            records.extend(group)
        del loaded,forms;torch.cuda.empty_cache()
    dump(OUT/'models.json',records)
    dump(OUT/'training_complete.json',dict(models_sha256=sha(OUT/'models.json'),matched_within_each_setting=True,seed42_only=True))
    print('ALL_TRAINING_COMPLETE',flush=True)


def score():
    verify();torch.set_num_threads(4);regs=read(OUT/'models.json');rows=lines(data.oldsearch.DEV/'rows.jsonl')
    groups=np.array(['+'.join(r['objects']) for r in rows]);records=[];raw={}
    for tint in data.TINTS:
        rr=[dict(name='Frozen',checkpoint=None,recipe='frozen'),*[r for r in regs if r['tint']==tint]]
        for colors in data.COLORS:
            color='-'.join(colors)
            for order in ('canonical','reversed'):
                tt=data.text_for(rows,color,order)
                parts={r['recipe']+'/'+r['name']:[] for r in rr}
                for view in p.VIEWS:
                    v,original,idx=p.metrics.bank_arrays(data.devpath(tint),'routing',color,view)
                    assert [r['anchor_id'] for r in idx]==[r['anchor_id'] for r in rows]
                    if order=='canonical':assert np.max(abs(tt-original))<2e-6
                    for r in rr:
                        name=r['recipe']+'/'+r['name'];x=p.metrics.score_arrays(v,tt,r['checkpoint'])
                        m=p.metrics.routing(x)
                        for c in p.metrics.all_contexts():
                            if c['kind']=='unwanted':m['absolute/'+c['name']]=abs(m['contrast/'+c['name']])
                        parts[name].append(m);raw[f'{tint}/{color}/{order}/{view}/{name}']=x
                for r in rr:
                    name=r['recipe']+'/'+r['name'];mm={k:np.mean([a[k] for a in parts[name]],axis=0) for k in parts[name][0]}
                    mean={k:float(np.mean([a[groups==g].mean() for g in sorted(set(groups))])) for k,a in mm.items()}
                    records.append(dict(tint=tint,color=color,order=order,recipe=r['recipe'],name=r['name'],n=888,**mean))
        print('DEVELOPMENT_SCORED',tint,flush=True)
    jsonl(OUT/'development.jsonl',records);np.savez_compressed(OUT/'development_scores.npz',**raw)
    dump(OUT/'development_index.json',dict(anchor_ids=[r['anchor_id'] for r in rows],colors=data.COLORS,orders=['canonical','reversed'],views=p.VIEWS))


def select():
    verify();records=lines(OUT/'development.jsonl');regs=read(OUT/'models.json');candidates=[]
    def values(tint,recipe,name,color,order=None):
        rr=[r for r in records if r['tint']==tint and r['recipe']==recipe and r['name']==name and r['color']==color and (order is None or r['order']==order)]
        assert len(rr)==(2 if order is None else 1)
        return {k:float(np.mean([r[k] for r in rr])) for k in rr[0] if isinstance(rr[0][k],(float,int))}
    for reg in regs:
        if reg['name']=='Ranking' or reg['recipe']=='base':continue
        tint,recipe,name=reg['tint'],reg['recipe'],reg['name'];checks={};acc=[];red=[];summ=[]
        for color in map('-'.join,data.COLORS):
            v=values(tint,recipe,name,color);f=values(tint,'frozen','Frozen',color);r=values(tint,recipe,'Ranking',color)
            checks[color+'/assignment']=v['exchange_accuracy']>=max(.75,f['exchange_accuracy']+.10,r['exchange_accuracy']-.01)
            for metric in ('word1_accuracy','word2_accuracy'):
                checks[color+'/'+metric]=v[metric]>=max(.90,f[metric]-.01,r[metric]-.02)
            checks[color+'/caption']=v['caption_accuracy']>=r['caption_accuracy']-.02
            checks[color+'/binding']=v['binding']>=f['binding']
            checks[color+'/cross']=v['cross']<=.9*f['cross']
            red.append(1-v['cross']/f['cross'])
            for order in ('canonical','reversed'):
                current=values(tint,recipe,name,color,order)['exchange_accuracy'];acc.append(current)
                checks[color+'/'+order]=current>=values(tint,'frozen','Frozen',color,order)['exchange_accuracy']
            summ.append(dict(color=color,Frozen=f,Ranking=r,IS=v))
        candidates.append(dict(tint=tint,recipe=recipe,name=name,eligible=all(checks.values()),checks=checks,
            worst_assignment=min(acc),minimum_cross_reduction=min(red),color_summaries=summ,model=reg))
    eligible=[r for r in candidates if r['eligible']]
    selected=None
    if eligible:
        peak=max(r['worst_assignment'] for r in eligible)
        eligible=[r for r in eligible if r['worst_assignment']>=peak-.01]
        selected=min(eligible,key=lambda r:(-r['minimum_cross_reduction'],r['name']=='IS2',r['tint']))
    dump(OUT/'selection.json',dict(selected=selected,candidates=candidates,development_only=True,seed=42,
        rule_sha256=sha(data.PLAN),new_test_scores_used=False))
    print('SELECTED',None if selected is None else (selected['tint'],selected['recipe'],selected['name']),flush=True)
    text=['# Routing tint/guard/order development — seed42','',
        'All comparisons are within a tint and a matched training recipe. Different tints are different input conditions.',
        'Complete 888-source development bank, equal weight across five noun pairs, both color pairs, both layouts and both caption orders.',
        '', '| Tint | Recipe | Method | Assignment | Four-caption | Word1 | Word2 | Binding | Cross |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for tint in data.TINTS:
        methods=[('frozen','Frozen')]+[(r,a) for r in RECIPES for a in (ARMS if r=='color_order' else ARMS[:2])]
        for recipe,name in methods:
            z=[values(tint,recipe,name,c) for c in map('-'.join,data.COLORS)]
            av={k:np.mean([v[k] for v in z]) for k in ('exchange_accuracy','caption_accuracy','word1_accuracy','word2_accuracy','binding','cross')}
            text.append(f'| {tint}% | {recipe} | {name} | '+' | '.join(f'{av[k]*(100 if "accuracy" in k else 1):.3f}' for k in av)+' |')
    text+=['','## Selection','',str(None if selected is None else {k:selected[k] for k in ('tint','recipe','name','worst_assignment','minimum_cross_reduction')}),'',
           'Every failed/passed screen and per-color result is in selection.json. A failed screen is not a per-example guarantee or a reason to hide the candidate.',
           'Reversed-order training phrases are now exposed in color_order; do not label them unseen. No manuscript changes or additional seeds.']
    with (OUT/'DEVELOPMENT_REPORT.md').open('x') as stream:stream.write('\n'.join(text)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','train','score','select']);args=ap.parse_args()
    log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
