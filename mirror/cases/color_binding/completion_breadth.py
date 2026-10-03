"""Fixed default routing/background recipes across bases, with matched controls."""
import argparse
from dataclasses import replace; from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.completion_data import OUT as ROOTOUT; from mirror.cases.color_binding.completion_data import configure
from mirror.cases.color_binding.behavioral_pilot import CAL; from mirror.cases.color_binding.behavioral_pilot import make_spec; from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.cases.color_binding.routing_adequacy import preference_spec
from mirror.core.specifications import compile_requirement
from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import RepairCache; from mirror.core.repair import historical_ranking; from mirror.core.repair import legacy_ranking_loss; from mirror.core.repair import _state_hash
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.cases.color_binding.routing_common_noise_pilot import PairedCaptionDropout
from mirror.core.metrics import bank_arrays
from mirror.core.features import verify_cache

OUT=ROOTOUT/'breadth'
PLAN=ROOT/'FSE_VLM/plan/26_fixed_recipe_breadth_and_background.md'
SEEDS=(42,43,44)


class CommonCaptionDropout(nn.Module):
    def __init__(self,p=.05):super().__init__();self.p=p
    def forward(self,x):
        if not self.training or self.p==0:return x
        return x*x.new_empty((len(x),1,x.shape[-1])).bernoulli_(1-self.p)/(1-self.p)


def load(model,family,device='cpu'):
    path=ROOTOUT/'training_features'/model/family;meta,_=verify_cache(path)
    verify_files(read(path/'extras_complete.json')['files'])
    idx=lines(path/'index.jsonl');rows=lines(path/'source_rows.jsonl')
    images=np.load(path/'images.npy');texts=np.load(path/'texts.npy');natural=np.load(path/'natural_texts.npy')
    vv=[];tt=[];nn=[];sources=[];ncap=4 if family=='routing' else 2
    for j,r in enumerate(idx):
        for ci,color in enumerate(('red-blue','green-yellow')):
            a,b=color.split('-')
            for view in (('canvas','swapped_canvas') if family=='routing' else ('audit',)):
                names=[color+'/'+view+'/'+x+'_'+y for x in (a,b) for y in (a,b)] if family=='routing' else [color+'/'+ctx+'/'+c for ctx in ('gray','blue') for c in (a,b)]
                ii=[r['state_names'].index(s) for s in names]
                vv.append(images[r['image_offset']+np.asarray(ii)])
                ti=r['text_indices'][ci*ncap:(ci+1)*ncap]+r['text_indices'][-2:]
                tt.append(texts[ti]);nn.append(natural[j]);sources.append(rows[j]['source_ids'][0])
    cache=RepairCache(torch.tensor(np.stack(vv),device=device),torch.tensor(np.stack(tt),device=device),torch.tensor(np.stack(nn),device=device),
        tuple(sources),model,meta['model_binding']['checkpoint_files'][0]['sha256'],'completion_'+family)
    cal=read(CAL)
    if family=='routing':
        raw,contexts=preference_spec(cal)
        spec=compile_requirement(raw,calibration_unit=cal['units'][model]['unit'],base_model_id=model,calibration_bank_id='natural_calibration_20260922')
    else:spec=make_spec(family,('red','blue'),'blend90_luminance',model,cal);contexts=[]
    # Only legacy recipe and clip/LR fields are used by the reused components.
    from types import SimpleNamespace
    cfg=SimpleNamespace(legacy_ranking=historical_ranking(family,(ncap,)*len(vv)),lr=.0002,weight_decay=.01,grad_clip=1.)
    return cache,spec,contexts,cfg


def model_for(cache,family):
    m=TextLowRankAdapter(cache.texts.shape[-1],64,64,.05).to(cache.texts.device)
    m.drop=PairedCaptionDropout(.05) if family=='routing' else CommonCaptionDropout(.05)
    return m


def bg_components(adapter,cache,spec,cfg,rows):
    v=cache.images[rows];base=cache.texts[rows];nat=cache.natural_texts[rows]
    a=adapter(torch.cat((base,nat),1));x=v@a[:,:4].transpose(1,2);f=v@base.transpose(1,2);u=spec.unit
    q=(x[:,:,0]-x[:,:,1])/u;qf=(f[:,:,0]-f[:,:,1])/u
    d=torch.stack((q[:,0]-q[:,1],q[:,2]-q[:,3]),1);df=torch.stack((qf[:,0]-qf[:,1],qf[:,2]-qf[:,3]),1)
    g=d[:,1]-d[:,0];gf=df[:,1]-df[:,0];bounds=spec.spec['thresholds']['fractions'];eps=prior.EPS
    margins=q*q.new_tensor([1,-1,1,-1]);mf=qf*qf.new_tensor([1,-1,1,-1])
    om=(x[:,:,2]-x[:,:,3])/u;of=(f[:,:,2]-f[:,:,3])/u
    raw=dict(binding=F.relu(torch.maximum(df,df.new_tensor(bounds['K']))-d-eps),
        gap=F.relu(g.abs()-torch.minimum(gf.abs(),gf.new_tensor(bounds['tau']))-eps),
        caption_guard=F.relu(mf-margins-eps)*(mf>0),binding_keep=F.relu(df-d-eps)*(df>0),
        object_guard=F.relu(of-om-eps)*(of>0),natural=(a[:,4:]-F.normalize(nat,dim=-1)).square().sum(-1),
        drift=(a[:,:4]-F.normalize(base,dim=-1)).square().sum(-1))
    parts={k:v.mean() for k,v in raw.items()}
    parts['ranking'],_=legacy_ranking_loss(x,a[:,:4],base,cfg.legacy_ranking,rows)
    return parts


def objective(parts,w,arm,family):
    if family=='routing':
        full=prior.objective(parts,w,'IP')+3*w['cross']*parts['cross']
        return {'IS':full,'R':parts['ranking'],'G':prior.objective(parts,w,'G'),'no_cross':full-4*w['cross']*parts['cross']}[arm]
    guards=4*(sum(w[k]*parts[k] for k in ('caption_guard','binding_keep','object_guard'))+parts['natural']+parts['drift'])
    return {'R':parts['ranking'],'G':guards,'IS':guards+w['binding']*parts['binding']+4*w['gap']*parts['gap'],
            'no_gap':guards+w['binding']*parts['binding']}[arm]


def normalization(cache,spec,contexts,cfg,family):
    torch.manual_seed(42);m=model_for(cache,family).eval()
    gen=torch.Generator(device=cache.images.device).manual_seed(20260923)
    with torch.no_grad():m.B.weight.copy_(torch.randn(m.B.weight.shape,device=cache.images.device,generator=gen)*.001)
    names=prior.SCORE_PARTS if family=='routing' else ('binding','gap','caption_guard','binding_keep','object_guard')
    n=len(cache.images)//2 if family=='routing' else len(cache.images)
    order=np.random.default_rng(20260923).permutation(n);records=[]
    for j in range(8):
        ids=torch.tensor(order[j*24:(j+1)*24],device=cache.images.device)
        if family=='routing':ids=prior.paired_rows(ids)
        parts=prior.components(m,cache,spec,contexts,cfg,ids) if family=='routing' else bg_components(m,cache,spec,cfg,ids)
        norms={k:prior.norm_grads(parts[k],m,True) for k in names}
        assert all(np.isfinite(v) for v in norms.values());records.append(norms)
    means={k:float(np.mean([r[k] for r in records])) for k in names};ref=float(np.median([v for v in means.values() if v>0]))
    weights={k:float(np.clip(ref/max(v,ref/10),.1,10)) for k,v in means.items()}
    return dict(weights=weights,mean_norms=means,reference=ref,batches=records,training_only=True,rule='same eight-batch perturbation rule as plan21')


def run(model,family):
    assert (ROOTOUT/'phase_b_finalized.json').exists();configure();prior.configure()
    if torch.cuda.mem_get_info()[0]<6*1024**3:raise RuntimeError('GPU busy')
    dest=OUT/model/family
    if (dest/'complete.json').exists():return
    cache,spec,contexts,cfg=load(model,family,'cuda')
    arms=('R','IS','G','no_cross' if family=='routing' else 'no_gap')
    if not (dest/'protocol.json').exists():
        norm=normalization(cache,spec,contexts,cfg,family)
        dump(dest/'normalization.json',norm)
        paths=[PLAN,Path(__file__),CAL,ROOTOUT/'training_features'/model/family/'extras_complete.json']
        dump(dest/'protocol.json',dict(inputs={str(p):sha(p) for p in paths},model=model,family=family,seeds=SEEDS,arms=arms,
            epochs=36,lr=.0002,weight_decay=.01,rank=64,alpha=64,dropout=.05,selection='fixed_last',
            normalization_sha256=sha(dest/'normalization.json'),benchmark_selection=False))
    p=read(dest/'protocol.json');verify_files(p['inputs']);assert sha(dest/'normalization.json')==p['normalization_sha256']
    weights=read(dest/'normalization.json')['weights'];n=len(cache.images)//2 if family=='routing' else len(cache.images)
    models=[dict(arm='F',seed=0,checkpoint=None,model=model,family=family)]
    for seed in SEEDS:
        rng=np.random.default_rng(seed);schedule=[rng.permutation(n).tolist() for _ in range(36)];initials=[];ends=[]
        for arm in arms:
            folder=dest/f'seed{seed}'/arm
            if (folder/'complete.json').exists():
                entry=read(folder/'complete.json');assert sha(entry['checkpoint'])==entry['sha256'];models.append(entry)
                initials.append(entry['initial_state_hash']);ends.append(entry['final_rng_hash']);continue
            folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            m=model_for(cache,family);ih=_state_hash({k:v.cpu().detach() for k,v in m.state_dict().items()});initials.append(ih)
            torch.save(dict(state_dict={k:v.cpu().detach().clone() for k,v in m.state_dict().items()},state_hash=ih),folder/'initial.pt')
            opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.01);history=[];started=time.monotonic()
            for epoch,order in enumerate(schedule,1):
                m.train();torch.manual_seed(seed*10000+epoch);torch.cuda.manual_seed_all(seed*10000+epoch)
                for start in range(0,n,24):
                    ids=torch.tensor(order[start:start+24],device='cuda')
                    if family=='routing':ids=prior.paired_rows(ids)
                    parts=prior.components(m,cache,spec,contexts,cfg,ids) if family=='routing' else bg_components(m,cache,spec,cfg,ids)
                    loss=objective(parts,weights,arm,family);opt.zero_grad(set_to_none=True);loss.backward()
                    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
                    history.append(dict(epoch=epoch,loss=float(loss.detach()),parts={k:float(v.detach()) for k,v in parts.items()}))
            from mirror.cases.color_binding.routing_budget_replication import sha_bytes
            rh=sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes());ends.append(rh)
            torch.save(dict(state_dict={k:v.cpu().detach() for k,v in m.state_dict().items()},model=model,family=family,arm=arm,seed=seed,
                epochs=36,updates=len(history),protocol_sha256=sha(dest/'protocol.json')),folder/'last.pt')
            jsonl(folder/'history.jsonl',history)
            entry=dict(model=model,family=family,arm=arm,seed=seed,checkpoint=str(folder/'last.pt'),sha256=sha(folder/'last.pt'),
                initial_state_hash=ih,final_rng_hash=rh,seconds=time.monotonic()-started)
            dump(folder/'complete.json',entry);models.append(entry);print('BREADTH_TRAINED',entry,flush=True);del m,opt
        assert len(set(initials))==len(set(ends))==1
    dump(dest/'models.json',models);dump(dest/'complete.json',dict(models_sha256=sha(dest/'models.json'),n_trained=12,
        protocol_sha256=sha(dest/'protocol.json'),matched_initialization_and_rng=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--family',choices=['routing','background'],required=True);a=ap.parse_args()
    log(OUT,'start',model=a.model,family=a.family)
    try:run(a.model,a.family)
    except BaseException as e:log(OUT,'failed',error=repr(e));raise
    log(OUT,'complete',model=a.model,family=a.family)


if __name__=='__main__':main()
