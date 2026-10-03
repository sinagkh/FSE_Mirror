"""Bounded seed42 cross-target weight search, using only the disjoint development bank."""
import argparse
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from mirror.cases.color_binding import routing_light_tint as p
from mirror.cases.color_binding import routing_light_suite_data as suite
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding.rendering import captions
from mirror.core.repair import _state_hash

OUT=suite.OUT/'weight_search'
DEV=p.ROOT/'clip/interbind_routing_same_class_20260923'
GRID={'IS_half':.5,'IS40':1.,'IS_double':2.,'IS_quadruple':4.}


def freeze():
    suite.verify();rows=lines(DEV/'rows.jsonl');assert len(rows)==888
    train=lines(p.TRAIN/'training_rows.jsonl');test=lines(p.CONFIRM/'rows.jsonl')
    source=lambda rr:{i for r in rr for i in r['source_ids']}
    assert not source(rows)&(source(train)|source(test)|{r['natural_guard']['image_id'] for r in train})
    files=[Path(__file__),Path(p.__file__),Path(suite.__file__),suite.OUT/'protocol.json',DEV/'rows.jsonl',
           p.OUT/'training_complete.json',p.OUT/'runs/IS40/history.jsonl',p.OUT/'features/train/complete.json',
           p.prior.OUT/'normalization.json',p.INITIAL]
    files += [Path(r['checkpoint']) for r in p.registry() if r['checkpoint']]
    inputs={str(f):sha(f) for f in files}
    inputs.update({f:h for r in rows for f,h in r['source_image_sha256'].items()});verify_files(inputs)
    dump(OUT/'protocol.json',dict(inputs=inputs,author_addendum='2026-09-30: one seed only; test different suppression weights',
        supersedes_only='The no-new-loss-selection provision of paper_retest_v1/protocol.json; all frozen test definitions retained',
        seed=42,grid=GRID,weight_meaning='Multiply only the calibrated cross-target coefficient: current coefficient is 4*w_cross. Binding/response/preference/guards unchanged.',
        development_anchors=888,development_source_overlap=0,development_colors=p.COLORS[:2],development_views=p.VIEWS,
        retraining='3 new 1944-update runs from identical original initialization; reuse current IS40 and Ranking40; fixed final checkpoints',
        selection='Use development only. Equal weight across five noun pairs, two training color pairs, both layouts. Retain complete grid/Pareto record.',
        eligibility='All eight binding means >= frozen and all eight absolute cross means <= frozen, separately for both training color pairs. Per-color exchange, four-caption and each one-word accuracy at most 1pp below current IS40.',
        tie_rule='Among eligible candidates within 1pp of highest eligible macro exchange accuracy, choose lowest macro absolute cross-effect; ties prefer lower multiplier. If none eligible, retain IS40 as unapproved fallback, not a passed gate.',
        heldout='No new transfer, preservation, or ablation outcomes used to select. Earlier pilot primary/public outcomes already known.',
        guards='Identical natural-caption and finite-difference retention in every grid arm and ranking',
        after_selection='Freeze selected multiplier, then run component deletions with selected weights and complete fixed paper retests. No extra seeds or manuscript edits.',
        ablation_targets=suite.DELETIONS))
    dump(OUT/'protocol_hash.json',dict(sha256=sha(OUT/'protocol.json')))
    print('SEARCH_FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    assert sha(OUT/'protocol.json')==read(OUT/'protocol_hash.json')['sha256']
    verify_files(read(OUT/'protocol.json')['inputs'])


def encode():
    verify();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info()[0]>12*1024**3
    annotations_for('train2017');annotations_for('val2017')
    scorer=load_subject(p.prior.MODEL,device='cuda')
    cache_bank(OUT/'features',lines(DEV/'rows.jsonl'),scorer,lambda row:p.render(row,p.COLORS[:2],.4),
        lambda f,n:[g for c in p.COLORS[:2] for g in captions(f,n,c)],registry_path=DEFAULT_REGISTRY,
        input_hashes={str(OUT/'protocol.json'):sha(OUT/'protocol.json'),str(DEV/'rows.jsonl'):sha(DEV/'rows.jsonl')},
        image_batch_size=64,text_batch_size=128,render_workers=8,details=dict(development_only=True,alpha=.4))
    print('DEVELOPMENT_ENCODING_COMPLETE',flush=True)


def train_one(name,multiplier,loaded,forms,deletions=()):
    cache,spec,contexts,cfg=loaded
    cfg=replace(cfg,epochs=36,seed=42)
    dest=OUT/'runs'/name
    if (dest/'complete.json').exists():
        record=read(dest/'complete.json');assert sha(record['checkpoint'])==record['sha256'];return record
    dest.mkdir(parents=True,exist_ok=False)
    weights=dict(read(p.prior.OUT/'normalization.json')['weights']);weights['cross']*=multiplier
    for key in deletions:weights[key]=0.
    schedule=p.old.schedule_for(42);reps=p.templates.representation_schedule(42)
    model=p.make(cfg);initial=_state_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    history=[];start=time.monotonic()
    for ei,order in enumerate(schedule):
        current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),torch.tensor(np.repeat(reps[ei],2),device='cuda')])
        model.train();torch.manual_seed(420001+ei);torch.cuda.manual_seed_all(420001+ei)
        for j in range(0,1280,24):
            ids=p.prior.paired_rows(torch.tensor(order[j:j+24],device='cuda'))
            parts,ce=p.parts_ce(model,current,spec,contexts,cfg,ids)
            loss=p.cn.objective(parts,weights)
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(grad)
            opt.step()
            history.append(dict(epoch=ei+1,rows=ids.cpu().tolist(),loss=float(loss.detach()),gradient_norm=float(grad),
                components={k:float(v.detach()) for k,v in parts.items()}))
        if (ei+1)%12==0:print('TRAIN',name,ei+1,round(time.monotonic()-start,1),flush=True)
    assert len(history)==1944
    path=dest/'last.pt'
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuration=asdict(cfg),
        weights=weights,seed=42,arm=name,multiplier=multiplier,deletions=deletions,updates=1944,selection='fixed_last',
        protocol_sha256=sha(OUT/'protocol.json')),path)
    jsonl(dest/'history.jsonl',history)
    record=dict(name=name,seed=42,checkpoint=str(path),sha256=sha(path),initial_state_hash=initial,
        schedule_sha256=p.array_hash(schedule),representation_sha256=p.array_hash(reps),
        final_rng_sha256=p.array_hash(torch.cuda.get_rng_state().cpu().numpy()),updates=1944,
        multiplier=multiplier,deletions=deletions,seconds=time.monotonic()-start,first_components=history[0]['components'])
    reference=next(r for r in read(p.OUT/'training_complete.json')['models'] if r['name']=='IS40')
    for key in ('initial_state_hash','schedule_sha256','representation_sha256','final_rng_sha256','updates','first_components'):
        assert record[key]==reference[key],key
    dump(dest/'complete.json',record);return record


def preflight(loaded,forms):
    cache,spec,contexts,cfg=loaded;model=p.make(cfg).train()
    reps=p.templates.representation_schedule(42)
    current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),torch.tensor(np.repeat(reps[0],2),device='cuda')])
    ids=p.prior.paired_rows(torch.tensor(p.old.schedule_for(42)[0][:24],device='cuda'))
    torch.manual_seed(420001);torch.cuda.manual_seed_all(420001)
    parts,ce=p.parts_ce(model,current,spec,contexts,cfg,ids)
    expected=lines(p.OUT/'runs/IS40/history.jsonl')[0]
    assert max(abs(float(v.detach())-expected['components'][k]) for k,v in parts.items())==0
    weights=read(p.prior.OUT/'normalization.json')['weights'];original=p.cn.objective(parts,weights)
    params=tuple(model.parameters())
    def grad(z):return torch.cat([g.flatten() for g in torch.autograd.grad(z,params,retain_graph=True)])
    checks={}
    for name,mult in GRID.items():
        w=dict(weights);w['cross']*=mult
        actual=grad(p.cn.objective(parts,w))-grad(original)
        expected=(mult-1)*grad(4*weights['cross']*parts['cross'])
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-4)
        checks[name]=float((actual-expected).abs().max())
    dump(OUT/'preflight.json',dict(original_first_batch_exact=True,only_cross_gradient_changes=checks,optimizer_updates=0))


def train():
    verify();p.prior.configure();loaded=p.light_training();forms=p.templates.template_cache(loaded[0]);preflight(loaded,forms)
    regs=[r for r in p.registry() if r['name'] in ('Frozen','IS40','Ranking40')]
    for name,mult in GRID.items():
        if name!='IS40':regs.append(train_one(name,mult,loaded,forms))
    dump(OUT/'models.json',regs);print('GRID_TRAINING_COMPLETE',flush=True)


def select():
    verify();torch.set_num_threads(4);verify_cache(OUT/'features')
    regs=read(OUT/'models.json');rows=lines(DEV/'rows.jsonl');groups=np.array(['+'.join(r['objects']) for r in rows])
    aggregates={};raw={};records=[]
    for colors in p.COLORS[:2]:
        color='-'.join(colors);parts={r['name']:[] for r in regs}
        for view in p.VIEWS:
            v,t,idx=p.metrics.bank_arrays(OUT/'features','routing',color,view)
            assert [r['anchor_id'] for r in idx]==[r['anchor_id'] for r in rows]
            for reg in regs:
                x=p.metrics.score_arrays(v,t,reg['checkpoint']);raw[color+'/'+view+'/'+reg['name']]=x
                m=p.metrics.routing(x)
                for ctx in p.metrics.all_contexts():
                    if ctx['kind']=='unwanted':m['absolute/'+ctx['name']]=abs(m['contrast/'+ctx['name']])
                parts[reg['name']].append(m)
        for name,pp in parts.items():
            m={k:np.mean([z[k] for z in pp],axis=0) for k in pp[0]}
            mean={k:float(np.mean([a[groups==g].mean() for g in sorted(set(groups))])) for k,a in m.items()}
            aggregates[color,name]=mean
            records.append(dict(color=color,name=name,n=888,macro_over_five_pairs=True,**mean))
    checks=[]
    for name,mult in GRID.items():
        guard={}
        for color in map('-'.join,p.COLORS[:2]):
            a=aggregates[color,name];f=aggregates[color,'Frozen'];current=aggregates[color,'IS40']
            for ctx in p.metrics.all_contexts():
                k=('contrast/' if ctx['kind']=='binding' else 'absolute/')+ctx['name']
                guard[color+'/'+ctx['name']]=(a[k]>=f[k]-1e-5) if ctx['kind']=='binding' else (a[k]<=f[k]+1e-5)
            for k in ('exchange_accuracy','caption_accuracy','word1_accuracy','word2_accuracy'):
                guard[color+'/'+k]=a[k]>=current[k]-.01
        macro={k:float(np.mean([aggregates[c,name][k] for c in map('-'.join,p.COLORS[:2])]))
               for k in ('exchange_accuracy','caption_accuracy','cross','binding','word1_accuracy','word2_accuracy')}
        checks.append(dict(name=name,multiplier=mult,eligible=all(guard.values()),checks=guard,macro=macro))
    eligible=[r for r in checks if r['eligible']]
    if eligible:
        peak=max(r['macro']['exchange_accuracy'] for r in eligible)
        acceptable=[r for r in eligible if r['macro']['exchange_accuracy']>=peak-.01]
        chosen=min(acceptable,key=lambda r:(r['macro']['cross'],r['multiplier']))
        selected=chosen['name'];status='development_rule_selected'
    else:selected='IS40';status='unapproved_current_recipe_fallback_no_eligible_candidate'
    record=next(r for r in regs if r['name']==selected)
    dump(OUT/'selection.json',dict(selected=selected,multiplier=GRID[selected],status=status,candidates=checks,model=record,
         protocol_sha256=sha(OUT/'protocol.json'),heldout_scores_used=False,additional_seeds=False))
    jsonl(OUT/'development_summary.jsonl',records);np.savez_compressed(OUT/'development_scores.npz',**raw)
    dump(OUT/'development_score_index.json',dict(anchor_ids=[r['anchor_id'] for r in rows],views=p.VIEWS))
    print('SELECTION',selected,status,flush=True)
    for r in checks:print('DEVELOPMENT',r['name'],r['eligible'],r['macro'],flush=True)


def ablations():
    verify();selected=read(OUT/'selection.json');p.prior.configure()
    loaded=p.light_training();forms=p.templates.template_cache(loaded[0]);regs=[]
    for name,deleted in suite.DELETIONS.items():regs.append(train_one(name,selected['multiplier'],loaded,forms,deleted))
    dump(OUT/'ablations.json',dict(models=regs,selection_sha256=sha(OUT/'selection.json')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','encode','train','select','ablations']);a=ap.parse_args()
    log(OUT,'start',stage=a.action)
    try:globals()[a.action]()
    except BaseException as exc:log(OUT,'failed',stage=a.action,error=repr(exc));raise
    log(OUT,'complete',stage=a.action)


if __name__=='__main__':main()
