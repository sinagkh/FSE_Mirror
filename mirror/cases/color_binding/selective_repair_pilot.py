"""One-seed shared-endpoint/selective-interaction development, plan 40.

All outputs are create-only. Existing caches and experiments are read-only.
The executable is CPU-only so the four cached-head studies can run together.
"""
import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import time

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError("Launch this cached pilot with CUDA_VISIBLE_DEVICES=''")

import numpy as np
import torch
from torch.nn import functional as F

from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.behavioral_pilot import CAL; from mirror.cases.color_binding.behavioral_pilot import CACHE; from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.core.metrics import adapt; from mirror.core.metrics import bank_arrays; from mirror.core.metrics import routing; from mirror.core.metrics import background
from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import _state_hash; from mirror.core.repair import historical_ranking
from mirror.cases.color_binding.routing_context_coverage import all_contexts
from mirror.cases.color_binding.routing_common_noise_pilot import PairedCaptionDropout
from mirror.cases.color_binding.completion_breadth import CommonCaptionDropout

OUT = ROOT/'clip/interbind_selective_repair_pilot_20260925'
PLAN = ROOT/'FSE_VLM/plan/40_selective_repair_one_seed.md'
MODEL = 'openclip_laion_l14'
STRENGTH = ROOT/'clip/interbind_strengthening_cpu_20260925'
OLD = ROOT/'clip/interbind_phase_bc_completion_20260923'
TEMPERATURES = (1, 3, 10, 30, 60, 100)
MULTIPLIERS = (.1, .3, 1.)
FAMILIES = ('routing', 'full_scene', 'spatial', 'background')
UNIT = read(CAL)['units'][MODEL]['unit']
K, TAU, BETA = .12, .025, .025


def digest(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


@dataclass
class Bundle:
    family: str
    images: torch.Tensor
    texts: torch.Tensor
    forms: torch.Tensor | None
    paired: bool
    ncap: int
    anchors: list
    inputs: list
    anchor_weight: float = .2
    agreement_weight: float = 0.
    agreement_margin: float = .045

    @property
    def nblocks(self):
        return len(self.images)//(2 if self.paired else 1)


def load_training(family):
    if family == 'routing':
        from mirror.cases.color_binding import routing_relative_pilot as prior
        from mirror.cases.color_binding import ranking_followup_pilot as fp
        cache, _, _, _ = prior.load_training('cpu')
        return Bundle(family, cache.images, cache.texts, fp.template_cache(cache), True, 4,
            lines(fp.TRAIN/'training_rows.jsonl'),
            [prior.OUT/'training_config.json', fp.OUT/'training_encoding.json',
             fp.TRAIN/'training_rows.jsonl', prior.OUT/'encoding_complete.json'])
    if family == 'full_scene':
        from mirror.cases.color_binding import strengthening_transfer_repair as tr
        cache, _, _, _ = tr.load('broad_mixed')
        ids=set(read(tr.DATA/'training_assignments.json')['broad'])
        rows=[r for r in lines(tr.DATA/'rows.jsonl') if r['anchor_id'] in ids]
        return Bundle(family, cache.images, cache.texts, None, True, 4, rows,
            [tr.DATA/'complete.json', tr.DATA/'training_assignments.json',
             tr.DATA/'features/train/complete.json', tr.OUT/'protocol.json'])
    if family == 'spatial':
        from mirror.cases.color_binding import strengthening_spatial_clean as clean
        from mirror.cases.color_binding import strengthening_spatial_repair as sp
        clean.setup(); sp.OUT=clean.DATA/'repair_response_balance'
        v,t,rows=sp.load('train')
        return Bundle(family, v,t[:,:4],None,False,2,rows,
            [clean.DATA/'features/train/complete.json',sp.OUT/'object_prompts.json',
             sp.OUT/'protocol.json',sp.OUT/'object_texts.npy'])
    from mirror.cases.color_binding import completion_breadth as cb
    cache,_,_,cfg=cb.load(MODEL,'background','cpu')
    path=OLD/'training_features'/MODEL/'background'
    return Bundle(family,cache.images,cache.texts,None,False,2,
        lines(path/'source_rows.jsonl'),[path/'complete.json',path/'extras_complete.json'],
        cfg.legacy_ranking.anchor_weight,cfg.legacy_ranking.agreement_weight,
        cfg.legacy_ranking.agreement_margin)


def make_model(bundle, seed=42):
    torch.manual_seed(seed)
    m=TextLowRankAdapter(bundle.texts.shape[-1],64,64,.05)
    m.drop=PairedCaptionDropout(.05) if bundle.paired else CommonCaptionDropout(.05)
    return m


def schedule(bundle):
    rng=np.random.default_rng(42)
    return np.stack([rng.permutation(bundle.nblocks) for _ in range(36)])


def representation(bundle):
    if bundle.forms is None:return np.zeros((36,bundle.nblocks),dtype=np.int64)
    offset=np.random.default_rng(380042).integers(0,4,size=bundle.nblocks)
    return np.stack([(offset+epoch)%4 for epoch in range(36)])


def batch(bundle, blocks, forms):
    ids=(blocks[:,None]*2+torch.arange(2)).flatten() if bundle.paired else blocks
    text=bundle.texts[ids] if bundle.forms is None else bundle.forms[ids,forms[ids]]
    return bundle.images[ids],text,ids


def contexts_tensor(x):
    return torch.as_tensor(np.stack([c['weights'] for c in all_contexts()]),dtype=x.dtype,device=x.device)


def quantities(x, family):
    z=x/UNIT
    if family in ('routing','full_scene'):
        contexts=all_contexts();v=torch.einsum('nij,kij->nk',z,contexts_tensor(z))
        d=v[:,[r['kind']=='binding' for r in contexts]]
        c=v[:,[r['kind']=='unwanted' for r in contexts]]
        m=torch.stack((z[:,1,1]-z[:,1,2],z[:,2,2]-z[:,2,1]),1)
        return dict(binding=d,cross=c,response=m.mean(1),preference=(m[:,0]-m[:,1])/2)
    if family=='spatial':
        m=torch.stack((z[:,0,0]-z[:,0,1],z[:,1,1]-z[:,1,0]),1)
        return dict(response=m.mean(1),preference=(m[:,0]-m[:,1])/2)
    q=z[:,:,0]-z[:,:,1];d=torch.stack((q[:,0]-q[:,1],q[:,2]-q[:,3]),1)
    return dict(binding=d,gap=d[:,1]-d[:,0])


def violation_vectors(q, f, family):
    """Each value is N x clauses; no target depends on candidate/test outcomes."""
    out={}
    if 'binding' in q:
        d,df=q['binding'],f['binding']
        # Permit redistribution across contexts while retaining factor-wise mean.
        ng=2 if family in ('routing','full_scene') else 1
        dmean=d.reshape(len(d),ng,-1).mean(-1)
        fmean=df.reshape(len(df),ng,-1).mean(-1)
        out['binding']=torch.cat((F.relu(K-d),F.relu(torch.maximum(fmean,fmean.new_tensor(K))-dmean)),1).square()
    if 'cross' in q:
        cap=torch.minimum(f['cross'].abs(),q['cross'].new_tensor(TAU))
        out['cross']=F.relu(q['cross'].abs()-cap).square()
    if 'gap' in q:
        cap=torch.minimum(f['gap'].abs(),q['gap'].new_tensor(TAU))
        out['gap']=F.relu(q['gap'].abs()-cap).square()[:,None]
    if 'response' in q:
        floor=.1 if family=='spatial' else K-TAU
        out['response']=F.relu(torch.maximum(f['response'],q['response'].new_tensor(floor))-q['response']).square()[:,None]
        cap=torch.minimum(f['preference'].abs(),q['preference'].new_tensor(BETA))
        out['preference']=F.relu(q['preference'].abs()-cap).square()[:,None]
    return out


def reduce_violation(value, paired=False, mode='mean'):
    per=value.mean(1)
    if paired:per=per.reshape(-1,2).mean(1)
    if mode=='tail':return per.topk(max(1,math.ceil(len(per)*.25))).values.mean()
    if mode!='mean':raise ValueError(mode)
    return per.mean()


def objective(model, bundle, v, t, temperature, mode, multiplier, weights):
    z=model(t);x=v@z[:,:bundle.ncap].transpose(1,2);f=v@t[:,:bundle.ncap].transpose(1,2)
    labels=torch.arange(x.shape[1])%bundle.ncap
    ce=F.cross_entropy(temperature*x.reshape(-1,bundle.ncap),labels.repeat(len(v)))
    anchor=(1-F.cosine_similarity(z[:,:bundle.ncap],t[:,:bundle.ncap],dim=-1)).mean()
    anchor=anchor+(1-F.cosine_similarity(z[:,bundle.ncap],t[:,bundle.ncap],dim=-1)).mean()
    agreement=x.new_zeros(())
    if bundle.family=='background':
        margin=(x[:,:,0]-x[:,:,1])*x.new_tensor([1,-1,1,-1])
        agreement=F.relu(bundle.agreement_margin-margin).square().mean()
    base=ce+bundle.anchor_weight*anchor+bundle.agreement_weight*agreement
    raw=violation_vectors(quantities(x,bundle.family),quantities(f,bundle.family),bundle.family)
    terms={k:reduce_violation(val,bundle.paired,mode) for k,val in raw.items()}
    penalty=sum(weights.get(k,1.)*val for k,val in terms.items())
    return base+multiplier*penalty,dict(base=base,ce=ce,anchor=anchor,agreement=agreement,**terms)


def gradient_norm(loss,model):
    gs=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True,allow_unused=False)
    return float(torch.sqrt(sum(g.square().sum() for g in gs)))


def normalization(bundle, dest):
    m=make_model(bundle,20260925).eval()
    with torch.no_grad():m.B.weight.normal_(std=.001)
    perm=np.random.default_rng(20260925).permutation(bundle.nblocks);records=[]
    zero=torch.zeros(len(bundle.images),dtype=torch.long)
    for j in range(8):
        ids=torch.as_tensor(np.resize(perm,(8*24,))[j*24:(j+1)*24])
        v,t,_=batch(bundle,ids,zero)
        for mode in ('mean','tail'):
            _,p=objective(m,bundle,v,t,100,mode,0.,{})
            records.append(dict(mode=mode,base=gradient_norm(p['base'],m),
                **{k:gradient_norm(val,m) for k,val in p.items() if k not in ('base','ce','anchor','agreement')}))
    weights={}
    for mode in ('mean','tail'):
        rr=[r for r in records if r['mode']==mode]
        base=float(np.median([r['base'] for r in rr]))
        weights[mode]={k:float(np.clip(base/max(float(np.median([r[k] for r in rr])),1e-8),.01,100))
            for k in rr[0] if k not in ('mode','base')}
    dump(dest/'normalization.json',dict(weights=weights,records=records,training_only=True))
    return weights


def numeric_metrics(x, family):
    if family in ('routing','full_scene'):
        m=routing(x);q=quantities(torch.from_numpy(x),family)
        d=q['binding'].numpy();c=abs(q['cross'].numpy())
        m.update(binding_context_sd=d.std(1),binding_min=d.min(1),cross_context_sd=c.std(1),cross_max=c.max(1))
        m['both_correct']=((x[:,1,1]>x[:,1,2])&(x[:,2,2]>x[:,2,1])).astype(float)
        m['reverse_image_choice']=np.stack((x[:,1,1]>x[:,2,1],x[:,2,2]>x[:,1,2]),1).mean(1)
        m={k:a for k,a in m.items() if not k.startswith('contrast/')}
    elif family=='spatial':
        q=quantities(torch.from_numpy(x),family);e=q['response'].numpy();b=q['preference'].numpy()
        m=dict(response=e,preference=abs(b),surplus=e-abs(b),exchange_accuracy=((e+b>0).astype(float)+(e-b>0))/2,
            both_correct=((e+b>0)&(e-b>0)).astype(float),
            reverse_image_choice=np.stack((x[:,0,0]>x[:,1,0],x[:,1,1]>x[:,0,1]),1).mean(1))
    else:
        m=background(x)
        m['exchange_accuracy']=m['caption_accuracy']
        within=np.stack((x[:,0,0]>x[:,1,0],x[:,1,1]>x[:,0,1],x[:,2,0]>x[:,3,0],x[:,3,1]>x[:,2,1]),1)
        across=np.stack((x[:,0,0]>x[:,3,0],x[:,2,0]>x[:,1,0],x[:,1,1]>x[:,2,1],x[:,3,1]>x[:,0,1]),1)
        m['reverse_image_choice']=within.mean(1);m['cross_background_image_choice']=across.mean(1)
        d=quantities(torch.from_numpy(x),family)['binding'].numpy()
        m['binding_min']=d.min(1);m['binding_context_sd']=d.std(1)
    return m


def dev_banks(family):
    if family=='routing':
        from mirror.cases.color_binding.ranking_followup_pilot import DEV
        path=DEV/'features';meta={r['anchor_id']:r for r in lines(DEV/'rows.jsonl')}
        for view in ('canvas','swapped_canvas'):
            v,t,idx=bank_arrays(path,'routing','red-blue',view)
            obj=np.load(path/'texts.npy')[np.array([r['text_indices'][-2:] for r in idx])]
            yield view,v,t,[meta[r['anchor_id']] for r in idx],obj
    elif family=='full_scene':
        path=STRENGTH/'routing_transfer_v3/features/development'
        meta={r['anchor_id']:r for r in lines(STRENGTH/'routing_transfer_v3/rows.jsonl')}
        v,t,idx=bank_arrays(path,'routing','red-blue','in_situ')
        obj=np.load(path/'texts.npy')[np.array([r['text_indices'][-2:] for r in idx])]
        yield 'in_situ',v,t,[meta[r['anchor_id']] for r in idx],obj
    elif family=='spatial':
        path=STRENGTH/'spatial_v3/features/development'
        rows=lines(path/'rows.jsonl');sp=STRENGTH/'spatial_v3/repair_response_balance'
        pairs=read(sp/'object_prompts.json')['pairs'];objects=np.load(sp/'object_texts.npy')
        obj=np.stack([objects[pairs.index(r['objects'])] for r in rows])
        yield 'canvas',np.load(path/'images.npy'),np.load(path/'texts.npy'),rows,obj
    else:
        path=CACHE/'features'/MODEL/'pilot/background'
        v,t,idx=bank_arrays(path,'background','red-blue','audit','blend90_luminance/')
        metadata={r['anchor_id']:r for r in lines(ROOT/'clip/interbind_source_quality_20260922/pilot/accepted_rows.jsonl')}
        # This older diagnostic cache has only the task captions, no wrong-noun bank.
        yield 'audit',v,t,[metadata[r['anchor_id']] for r in idx],None


def score_development(regs,family,dest):
    results=[];raw=[]
    banks=list(dev_banks(family))
    for view,v,t,rows,objects in banks:
        for reg in regs:
            x=np.einsum('nid,njd->nij',v,adapt(t,reg.get('checkpoint')),optimize=True).astype(float)
            m=numeric_metrics(x,family)
            if objects is not None:
                om=np.einsum('nid,njd->nij',v,adapt(objects,reg.get('checkpoint')),optimize=True)
                m['object_guard']=(om[:,:,0]>om[:,:,1]).mean(1)
            summary={k:float(a.mean()) for k,a in m.items()}
            target='cross' if family in ('routing','full_scene') else 'gap' if family=='background' else 'preference'
            summary['unwanted_p90']=float(np.quantile(m[target],.9))
            # Same unscaled CE for a tie-break, never compare differently scaled likelihoods.
            labels=np.arange(x.shape[1])%x.shape[2]
            summary['endpoint_error']=float((np.logaddexp.reduce(x,axis=2)-x[:,np.arange(x.shape[1]),labels]).mean())
            results.append(dict(name=reg['name'],view=view,n=len(rows),**summary))
            for i,r in enumerate(rows):
                raw.append(dict(name=reg['name'],view=view,anchor_id=r['anchor_id'],
                    source_ids=r.get('source_ids',[r.get('source_id',r['anchor_id'])]),
                    scores=x[i].tolist(),**{k:float(a[i]) for k,a in m.items()}))
    jsonl(dest/'development_summary.jsonl',results);jsonl(dest/'development_per_example.jsonl',raw)
    means={}
    for reg in regs:
        rr=[r for r in results if r['name']==reg['name']]
        means[reg['name']]={k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ('name','view')}
    dump(dest/'development_means.json',means)
    return means


def choose(regs,means,family):
    ranking=[r for r in regs if r.get('method')=='ranking']
    def utility(a):
        return all(a[k]>=means['F'][k]-.01 for k in ('word1_accuracy','word2_accuracy','object_guard') if k in a)
    eligible_r=[r for r in ranking if utility(means[r['name']])]
    rbest=min(eligible_r or ranking,key=lambda r:(-means[r['name']]['exchange_accuracy'],means[r['name']]['endpoint_error'],r['temperature']))
    f=means['F'];r=means[rbest['name']];eligibility=[]
    for cand in [q for q in regs if q.get('method')=='is']:
        a=means[cand['name']]
        checks=dict(direct=a['exchange_accuracy']>=r['exchange_accuracy']-.01,utility=utility(a))
        if family in ('routing','full_scene'):
            checks.update(binding=a['binding']>=.99*f['binding'],cross_frozen=a['cross']<=.85*f['cross'],cross_ranking=a['cross']<=.85*r['cross'],
                word1=a['word1_accuracy']>=f['word1_accuracy']-.01,word2=a['word2_accuracy']>=f['word2_accuracy']-.01)
        elif family=='background':
            checks.update(binding=a['binding']>=.99*f['binding'],gap_frozen=a['gap']<=.75*f['gap'],gap_ranking=a['gap']<r['gap'])
        else:checks.update(response=a['response']>f['response'],preference=a['preference']<f['preference'])
        eligibility.append(dict(name=cand['name'],passed=all(checks.values()),checks=checks))
    good={x['name'] for x in eligibility if x['passed']}
    pool=[q for q in regs if q.get('method')=='is' and (not good or q['name'] in good)]
    best=min(pool,key=lambda q:(-means[q['name']]['exchange_accuracy'],means[q['name']]['unwanted_p90'],q['multiplier'],q['mode']))
    return dict(ranking=rbest,is_candidate=best,adopted=bool(good),eligibility=eligibility,
        ranking_utility_gate_passed=bool(eligible_r),ranking_fallback=not bool(eligible_r),
        selected_on_development_only=True,test_scores_for_new_checkpoints_seen=False,
        other_seeds_authorized_here=False,scope='one-seed development; fallback is diagnostic only')


def run(family):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    dest=OUT/family;dest.mkdir(parents=True,exist_ok=False)
    b=load_training(family);sch=schedule(b);rep=representation(b)
    train_sources={str(x) for r in b.anchors for x in r['source_ids']}
    source_checks=[]
    for view,_,_,rows,_ in dev_banks(family):
        development_sources={str(x) for r in rows for x in r['source_ids']}
        overlap=train_sources&development_sources
        assert not overlap,(family,'training/development source overlap',sorted(overlap))
        source_checks.append(dict(view=view,development_sources=len(development_sources),overlap=0))
    inputs=[PLAN,Path(__file__),CAL,*b.inputs]
    inputs += [Path(__file__).with_name(n+'.py') for n in ('repair','completion_metrics','routing_context_coverage')]
    weights=normalization(b,dest)
    configurations=[dict(name=f'R_t{t}',method='ranking',temperature=t,mode='mean',multiplier=0.) for t in TEMPERATURES]
    configurations += [dict(name=f'IS_{mode}_w{w:g}',method='is',temperature=100,mode=mode,multiplier=w) for mode in ('mean','tail') for w in MULTIPLIERS]
    dump(dest/'protocol.json',dict(inputs={str(p):sha(p) for p in inputs},family=family,seed=42,
        candidates=configurations,normalization_sha256=sha(dest/'normalization.json'),epochs=36,
        updates=36*math.ceil(b.nblocks/24),training_lattices=len(b.images),training_anchors=len(b.anchors),
        schedule_sha256=digest(sch),representation_sha256=digest(rep),same_loss_information=True,
        common_anchor_weight=b.anchor_weight,common_agreement_weight=b.agreement_weight,
        extra_natural_supervision=False,all_prior_results_preserved=True,
        source_checks=source_checks,training_source_ids=sorted(train_sources)))
    np.save(dest/'schedule.npy',sch);np.save(dest/'representation.npy',rep)
    references={
        'routing':ROOT/'clip/interbind_ranking_followup_20260925/runs/seed42/IS_templates/last.pt',
        'full_scene':STRENGTH/'routing_transfer_v3/repair_v1/broad_mixed/seed42/IS/last.pt',
        'spatial':STRENGTH/'spatial_v3/repair_response_balance/seed42/IS/last.pt',
        'background':OLD/'breadth'/MODEL/'background/seed42/IS/last.pt'}
    previous=references[family];assert previous.exists(),previous
    regs=[dict(name='F',checkpoint=None,seed=0),dict(name='previous_IS',checkpoint=str(previous),seed=42,
        sha256=sha(previous),role='historical reference, not a newly matched objective ablation')];initials=[];rngs=[]
    for cand in configurations:
        folder=dest/'runs'/cand['name'];folder.mkdir(parents=True)
        m=make_model(b);initials.append(_state_hash(m.state_dict()))
        opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.01)
        history=[];started=time.monotonic()
        for epoch,order in enumerate(sch,1):
            form=torch.as_tensor(np.repeat(rep[epoch-1],2) if b.paired else rep[epoch-1])
            m.train();torch.manual_seed(420000+epoch)
            aggregate=[]
            for start in range(0,b.nblocks,24):
                v,t,ids=batch(b,torch.as_tensor(order[start:start+24]),form)
                loss,parts=objective(m,b,v,t,cand['temperature'],cand['mode'],cand['multiplier'],weights[cand['mode']])
                opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
                aggregate.append(dict(loss=float(loss.detach()),grad=float(gn),**{k:float(val.detach()) for k,val in parts.items()}))
            history.append(dict(epoch=epoch,updates=len(aggregate),**{k:float(np.mean([r[k] for r in aggregate])) for k in aggregate[0]}))
            if epoch%12==0:print('PILOT',family,cand['name'],epoch,round(time.monotonic()-started,1),history[-1]['loss'],flush=True)
        rh=digest(torch.get_rng_state().numpy());rngs.append(rh)
        torch.save(dict(state_dict=m.state_dict(),seed=42,candidate=cand,epochs=36,updates=sum(r['updates'] for r in history),
            protocol_sha256=sha(dest/'protocol.json')),folder/'last.pt')
        jsonl(folder/'history.jsonl',history)
        reg=dict(**cand,seed=42,checkpoint=str(folder/'last.pt'),sha256=sha(folder/'last.pt'),
            initial_hash=initials[-1],final_rng_hash=rh,seconds=time.monotonic()-started)
        dump(folder/'complete.json',reg);regs.append(reg)
    assert len(set(initials))==len(set(rngs))==1
    dump(dest/'models.json',regs)
    means=score_development(regs,family,dest);selection=choose(regs,means,family)
    dump(dest/'selection.json',selection)
    dump(dest/'complete.json',dict(files={str(dest/p):sha(dest/p) for p in ('protocol.json','models.json','selection.json','development_summary.jsonl')},
        matched_initialization=True,matched_rng=True,matched_information=True,adopted=selection['adopted']))
    print('SELECTION',family,selection['adopted'],selection['ranking']['name'],selection['is_candidate']['name'],flush=True)
    for name in ['F',selection['ranking']['name'],selection['is_candidate']['name']]:print(name,means[name],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=FAMILIES);a=ap.parse_args()
    log(OUT,'start',family=a.family)
    try:run(a.family)
    except BaseException as e:log(OUT,'failed',family=a.family,error=repr(e));raise
    log(OUT,'complete',family=a.family)


if __name__=='__main__':main()
