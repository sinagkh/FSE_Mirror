"""One-seed, CPU-only prospective spatial repair and diagnosis-guided controls."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA explicitly')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.strengthening_spatial import OUT as DATA; from mirror.cases.color_binding.strengthening_spatial import MODEL; from mirror.cases.color_binding.strengthening_spatial import NOUNS; from mirror.cases.color_binding.strengthening_spatial import captions; from mirror.cases.color_binding.strengthening_spatial import measurements; from mirror.cases.color_binding.strengthening_spatial import order
from mirror.cases.color_binding.strengthening_cpu import OUT as ROOTOUT; from mirror.cases.color_binding.strengthening_cpu import clean
from mirror.cases.color_binding.strengthening_labclip import intervals
from mirror.cases.color_binding.behavioral_pilot import CAL; from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.core.metrics import adapt; from mirror.core.metrics import bank_arrays; from mirror.core.metrics import routing
from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import _state_hash
from mirror.core.encoders import load_subject

OUT=DATA/'repair_v1'
ARMS=('IS','R','no_response','no_preference')
GUARD=ROOT/'clip/interbind_routing_repair_pilot_20260923/features/natural_texts.npy'


class SharedCaptionDropout(nn.Module):
    def __init__(self,p=.05):super().__init__();self.p=p
    def forward(self,x):
        if not self.training:return x
        mask=x.new_empty((len(x),1,x.shape[-1])).bernoulli_(1-self.p)/(1-self.p)
        return x*mask


def model(seed):
    torch.manual_seed(seed)
    m=TextLowRankAdapter(768,64,64,.05);m.drop=SharedCaptionDropout(.05)
    return m


def prepare():
    verify_files(read(DATA/'frozen_diagnosis/complete.json')['files'])
    OUT.mkdir(parents=True,exist_ok=False)
    # Record prediction and gate before constructing any optimized checkpoint.
    summary=read(DATA/'frozen_diagnosis/summary.json')
    dump(OUT/'protocol.json',dict(script_sha256=sha(Path(__file__)),data_sha256=sha(DATA/'data_complete.json'),
        diagnosis_sha256=sha(DATA/'frozen_diagnosis/complete.json'),frozen_diagnosis=summary,
        model=MODEL,seed=42,arms=['F',*ARMS],device='cpu',threads=4,
        prediction='Response term should increase e especially among frozen nonpositive-response development anchors; preference term should reduce |b| among frozen preference-dominated anchors. Test these paired ablation contrasts without redefining groups.',
        feasibility='Only necessary image-difference bound is tested. A nonzero feature difference does not guarantee a shared low-rank patch can repair it.',
        target=dict(response_floor_fraction_of_fixed_base_unit=.1,preference_cap_fraction=.025,
            response='max(frozen_e,response_floor); positive response never deliberately suppressed',
            preference='min(abs(frozen_b),preference_cap); no enlargement rewarded'),
        objective='Normalized response hinge +4*normalized preference hinge +4*(caption-sign retention + object-margin retention + natural embedding consistency + task embedding drift)',
        guard_detail='Preserve frozen-correct caption decisions at a small floor .025*unit, not the original biased margin magnitude. Object-margin guard retains positive frozen correct-vs-wrong noun margin.',
        ranking='Two-way cross-entropy on actual left/right labels plus0.2*(relation-embedding cosine drift + correct noun-pair embedding cosine drift); no IS guards added',
        normalization='8 fixed training-only batches, eval-mode adapter with B std.001; normalize term gradients to response gradient with weights clipped[.1,10]. No development or benchmark outcomes.',
        optimizer=dict(name='AdamW',lr=.0002,weight_decay=.01,epochs=36,batch_anchors=24,updates=972,grad_clip=1.,rank=64,alpha=64,dropout=.05),
        selection='fixed_last; same initialization,batches,dropout draws,forwards for all arms',
        promotion_gate=dict(response_change_min=0,exchange_accuracy_gain_min=.05,existing_development_word_accuracy_drop_max=.01,
            bothword_metrics_required=True,finite_metrics=True),
        max_diagnosis_driven_revisions=1,confirmation_scores_allowed=False,
        guard_cache_sha256=sha(GUARD),no_public_benchmark_selection=True))
    scorer=load_subject(MODEL,device='cpu');rows=lines(DATA/'rows.jsonl')
    pairs=sorted(set(tuple(r['objects']) for r in rows));objects={}
    for pair in pairs:
        wrong=sorted(set(NOUNS)-set(pair),key=lambda n:order('wrong-noun',pair,n))[:2]
        objects['+'.join(pair)]=[[f'a photo of a {a} and a {b}',f'a {a} next to a {b}'] for a,b in (pair,wrong)]
    groups=[g for p in pairs for g in objects['+'.join(p)]]
    t=scorer.encode_texts(groups,batch_size=32).numpy().reshape(len(pairs),2,-1)
    with (OUT/'object_texts.npy').open('xb') as f:np.save(f,t)
    dump(OUT/'object_prompts.json',dict(pairs=pairs,groups=objects))
    dump(OUT/'preparation_complete.json',dict(files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()},training=False,gpu=False))


def load(bank):
    path=DATA/'features'/bank;verify_files(read(path/'complete.json')['files'])
    rows=lines(path/'rows.jsonl');v=np.load(path/'images.npy');t=np.load(path/'texts.npy')
    ot=np.load(OUT/'object_texts.npy');idx=read(OUT/'object_prompts.json')['pairs']
    extra=np.stack([ot[idx.index(r['objects'])] for r in rows])
    natural=np.load(GUARD);rng=np.random.default_rng(20260925)
    natural=natural[rng.permutation(len(natural))[:len(rows)]]
    return torch.tensor(v),torch.tensor(np.concatenate([t,extra,natural],axis=1)),rows


def parts(m,v,t,unit):
    z=m(t);s=v@z[:,:4].transpose(1,2);f=v@t[:,:4].transpose(1,2)
    def eb(x):
        margins=torch.stack((x[:,0,0]-x[:,0,1],x[:,1,1]-x[:,1,0]),1)/unit
        return margins.mean(1),(margins[:,0]-margins[:,1])/2,margins
    e,b,margin=eb(s);ef,bf,mf=eb(f)
    response=F.relu(torch.maximum(ef,ef.new_tensor(.1))-e-1e-5).mean()
    preference=F.relu(b.abs()-torch.minimum(bf.abs(),bf.new_tensor(.025))-1e-5).mean()
    cap=(F.relu(.025-margin)*(mf>0)).mean()
    old=(f[:,:,2]-f[:,:,3])/unit;new=(s[:,:,2]-s[:,:,3])/unit
    obj=(F.relu(old-new-1e-5)*(old>0)).mean()
    natural=(z[:,4:]-t[:,4:]).square().sum(-1).mean();drift=(z[:,:4]-t[:,:4]).square().sum(-1).mean()
    ce=F.cross_entropy(s[:,:,:2].reshape(-1,2),torch.arange(2).repeat(len(v)))
    anchor=(1-F.cosine_similarity(z[:,:2],t[:,:2],dim=-1)).mean()+(1-F.cosine_similarity(z[:,2],t[:,2],dim=-1)).mean()
    return dict(response=response,preference=preference,caption_guard=cap,object_guard=obj,natural=natural,drift=drift,ranking=ce+.2*anchor)


def objective(p,w,arm):
    if arm=='R':return p['ranking']
    guard=4*(w['caption_guard']*p['caption_guard']+w['object_guard']*p['object_guard']+p['natural']+p['drift'])
    return guard+(0 if arm=='no_response' else w['response']*p['response'])+(0 if arm=='no_preference' else 4*w['preference']*p['preference'])


def normalize(v,t,unit):
    m=model(20260925).eval();torch.manual_seed(20260925)
    with torch.no_grad():m.B.weight.normal_(std=.001)
    keys=('response','preference','caption_guard','object_guard');records=[]
    order=np.random.default_rng(20260925).permutation(len(v))
    for i in range(8):
        ix=order[i*24:(i+1)*24];p=parts(m,v[ix],t[ix],unit);norm={}
        for k in keys:
            g=torch.autograd.grad(p[k],tuple(m.parameters()),retain_graph=True)
            norm[k]=float(torch.sqrt(sum(x.square().sum() for x in g)))
        records.append(norm)
    med={k:float(np.median([r[k] for r in records])) for k in keys}
    weights={k:float(np.clip(med['response']/max(med[k],1e-10),.1,10)) for k in keys}
    dump(OUT/'normalization.json',dict(weights=weights,gradient_norms=records,training_only=True))
    return weights


def train(seed):
    verify_files(read(OUT/'preparation_complete.json')['files']);cfg=read(OUT/'protocol.json')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    v,t,rows=load('train');unit=read(CAL)['units'][MODEL]['unit']
    w=read(OUT/'normalization.json')['weights'] if (OUT/'normalization.json').exists() else normalize(v,t,unit)
    if seed!=42:
        if not read(OUT/'seed42/development/gate.json')['passed']:raise ValueError('Seed42 promotion gate not passed')
    dest=OUT/f'seed{seed}';dest.mkdir(parents=True,exist_ok=False)
    schedule=[r.tolist() for r in np.random.default_rng(seed).permuted(np.tile(np.arange(len(v)),(36,1)),axis=1)]
    dump(dest/'schedule.json',schedule);regs=[];initials=[];rngs=[]
    for arm in ARMS:
        folder=dest/arm;folder.mkdir();m=model(seed);initials.append(_state_hash(m.state_dict()))
        opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.01);history=[];started=time.monotonic()
        for epoch,order in enumerate(schedule,1):
            m.train();torch.manual_seed(seed*10000+epoch)
            for start in range(0,len(v),24):
                ix=order[start:start+24];p=parts(m,v[ix],t[ix],unit);loss=objective(p,w,arm)
                opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                assert torch.isfinite(loss)&torch.isfinite(gn);opt.step()
                history.append(dict(epoch=epoch,loss=float(loss.detach()),gradient_norm=float(gn),**{k:float(x.detach()) for k,x in p.items()}))
            if epoch%12==0:print('SPATIAL_CPU_TRAIN',seed,arm,epoch,'seconds',round(time.monotonic()-started,1),flush=True)
        assert len(history)==972
        rngs.append(torch.get_rng_state().numpy().tobytes())
        torch.save(dict(state_dict=m.state_dict(),seed=seed,arm=arm,selection='fixed_last',protocol_sha256=sha(OUT/'protocol.json')),folder/'last.pt')
        jsonl(folder/'history.jsonl',history);regs.append(dict(arm=arm,seed=seed,checkpoint=str(folder/'last.pt'),sha256=sha(folder/'last.pt'),seconds=time.monotonic()-started))
    assert len(set(initials))==len(set(rngs))==1
    dump(dest/'models.json',regs);dump(dest/'complete.json',dict(files={str(dest/'models.json'):sha(dest/'models.json')},matched_initialization=True,matched_rng=True,device='cpu'))


def evaluate(seed):
    dest=OUT/f'seed{seed}/development';dest.mkdir(parents=True,exist_ok=False)
    regs=[dict(arm='F',seed=0,checkpoint=None),*read(OUT/f'seed{seed}/models.json')]
    v,t,rows=load('development');v=v.numpy();t=t.numpy();records=[];values={}
    for r in regs:
        met=measurements(v,adapt(t[:,:2],r['checkpoint']));values[r['arm']]=met
        for i,row in enumerate(rows):records.append(dict(arm=r['arm'],seed=r['seed'],anchor_id=row['anchor_id'],source_ids=row['source_ids'],
            frozen_regime=values['F']['regime'][i],**{k:str(x[i]) if k=='regime' else float(x[i]) for k,x in met.items()}))
    jsonl(dest/'per_example.jsonl',clean(records));stats=[]
    metrics=[k for k in values['F'] if k!='regime']
    for cohort in ['all',*sorted(set(values['F']['regime']))]:
        mask=np.ones(len(rows),bool) if cohort=='all' else values['F']['regime']==cohort
        if not mask.any():continue
        for label in ['F',*ARMS,'IS-F','IS-R','IS-no_response','IS-no_preference']:
            if label.startswith('IS-'):
                other=label[3:];x=np.stack([values['IS'][k][mask]-values[other][k][mask] for k in metrics],1)
            else:x=np.stack([values[label][k][mask] for k in metrics],1)
            for metric,s in zip(metrics,intervals(x[None],np.arange(mask.sum()))):stats.append(dict(cohort=cohort,comparison=label,metric=metric,**s))
    jsonl(dest/'summary.jsonl',clean(stats))
    # Internal utility is old development, never SugarCrepe/ARO/confirmation.
    bank=ROOT/'clip/interbind_routing_same_class_20260923/features';vv,tt,_=bank_arrays(bank,'routing','red-blue','canvas')
    utility=[]
    for r in regs:
        m=routing(np.einsum('nid,njd->nij',vv,adapt(tt,r['checkpoint'])),MODEL)
        utility.append(dict(arm=r['arm'],word1=float(m['word1_accuracy'].mean()),word2=float(m['word2_accuracy'].mean()),exchange=float(m['exchange_accuracy'].mean())))
    dump(dest/'internal_utility.json',utility)
    gain=float(np.mean(values['IS']['exchange_accuracy']-values['F']['exchange_accuracy']));change=float(np.mean(values['IS']['response_raw']-values['F']['response_raw']))
    uf=next(r for r in utility if r['arm']=='F');ui=next(r for r in utility if r['arm']=='IS')
    checks=dict(response_increases=change>0,exchange_gain_at_least5pp=gain>=.05,
        word1_preserved=ui['word1']>=uf['word1']-.01,word2_preserved=ui['word2']>=uf['word2']-.01,
        finite=all(np.isfinite(values[a]['response_raw']).all() for a in values))
    dump(dest/'gate.json',dict(passed=all(checks.values()),checks=checks,exchange_gain=gain,response_change_raw=change,confirmation_opened=False))
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},gpu=False))
    print('SPATIAL_DEVELOPMENT_GATE',checks,'exchange gain',gain,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','evaluate']);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    torch.set_num_threads(4);log(ROOTOUT,'start',stage='spatial_repair_'+a.action,seed=a.seed)
    try:
        if a.action=='prepare':prepare()
        elif a.action=='train':train(a.seed)
        else:evaluate(a.seed)
    except BaseException as e:log(ROOTOUT,'failed',stage='spatial_repair_'+a.action,error=repr(e));raise
    log(ROOTOUT,'complete',stage='spatial_repair_'+a.action,seed=a.seed)


if __name__=='__main__':main()
