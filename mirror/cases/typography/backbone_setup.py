"""Plan49 B1 fixed-recipe typography ports; no outcome-dependent selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY

CODE=ROOT/'mirror/cases/typography'
sys.path.insert(0,str(CODE))
import mirror.cases.typography.diagnose as data
import mirror.cases.typography.objective as objective
import mirror.cases.typography.broad_support_filtered as original

OUT=ROOT/'clip/fse_pre_writing_20260926/B1_typography'
CACHE=Path('/external-cache/fse_plan49_typography')
MODELS=('openai_clip_l14','openclip_laion_b32')
SEEDS=(42,43,44)
ARMS={'ranking':dict(scale=100,weight=0.),'IS':dict(scale=100,weight=1.)}

def ah(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def stream(y,seed):
    labels=np.asarray(y);pools=[np.flatnonzero(labels==i) for i in range(int(labels.max())+1)]
    assert all(len(p)>0 for p in pools);seq=[];cycle=1
    while len(seq)<2752*32:
        rng=np.random.default_rng(seed*100+cycle)
        entries=np.concatenate([np.resize(rng.permutation(p),64) for p in pools])
        seq.extend(rng.permutation(entries).tolist());cycle+=1
    return np.array(seq[:2752*32],dtype=np.int64)

def configuration():return read(original.DEST/'filtered_data_protocol.json')

def freeze(model):
    dest=OUT/model;assert not (dest/'protocol.json').exists();cfg=original.verify_data()
    paths=[Path(__file__),CODE/'diagnose.py',CODE/'objective.py',CODE/'train.py',
        CODE/'replicate.py',CODE/'broad_support_filtered.py',
        original.DEST/'filtered_data_protocol.json',data.OUT/'protocol.json',
        DEFAULT_REGISTRY,DEFAULT_REGISTRY.with_name('activation_addendum.json'),
        ROOT/'FSE_VLM/plan/49_pre_writing_experiments.md',ROOT/'FSE_VLM/plan/49a_execution_review.md',
        *[Path(v['manifest']) for v in cfg['sources'].values()],
        *[data.OUT/(b+'.json') for b in ('train','development','test_seen','test_heldout')]]
    row=next(r for r in read(DEFAULT_REGISTRY)['subjects'] if r['id']==model)
    paths.extend(Path(r['path']) for r in row['files'])
    dump(dest/'protocol.json',dict(item='B1 fixed typography backbone replication',model=model,
        seeds=SEEDS,arms=ARMS,updates=2752,batch_size=32,training_classes=cfg['training_classes'],
        training_manifest=cfg['sources']['train']['manifest'],training_data_unchanged=True,
        optimizer='SGD',lr=.002,eta_min=.00005,gradient_clipping=False,checkpoint='fixed_last',
        prompt='One shared trainable vector after BOS, random .02 Gaussian with actual token width.',
        objective='Unmodified prompt_unclipped.loss; cosine CE scale100; native logit scale only in frozen-reference KL.',
        kappa='Ratio of median CE to median nuisance prefix-gradient norms on first16 fixed training batches at seed42 initialization; freeze across all seeds.',
        evaluation=dict(primary='A3 original-32-label fresh bank: IS minus ranking attack pairwise accuracy',
            other=['original held-out','new font','new placement','38 held-out labels','added-label test',
                   'SCAM/NoSCAM/SynthSCAM','RTA100','SugarCrepe','retained repair/regressions','interaction','occlusion/word flips'],
            seeds=SEEDS,draws=5000,selection='none',all_registered_results=True),
        runtime=dict(image_batch_sources=16,image_loader_workers=8,torch_threads=2,
            image_precision='FP16 autocast, normalized FP32 features',prompt_training='FP32',
            tf32=False,feature_cache=str(CACHE/model)),
        inputs={str(p):sha(p) for p in paths}))
    with (dest/'protocol.sha256').open('x') as f:f.write(sha(dest/'protocol.json')+'\n')
    log(dest,'freeze')

def verify(model):
    dest=OUT/model;p=read(dest/'protocol.json')
    assert sha(dest/'protocol.json')==(dest/'protocol.sha256').read_text().strip()
    verify_files(p['inputs']);return p

def prefix(model,seed):
    g=torch.Generator(device='cpu').manual_seed(seed)
    return torch.nn.Parameter((.02*torch.randn(model.token_embedding.weight.shape[1],generator=g)).cuda())

def feature_cache(subject,bank,rows):
    dest=OUT/subject.subject['id'];cached=CACHE/subject.subject['id'];cached.mkdir(parents=True,exist_ok=True)
    meta=dest/(bank+'_features.json');target=cached/(bank+'.npy')
    if meta.exists():
        m=read(meta);assert sha(target)==m['sha256'];return np.load(target)
    assert not target.exists();values=[];start=time.monotonic()
    loader=DataLoader(data.Images(rows,subject.preprocess,'standard'),batch_size=16,num_workers=8,
        pin_memory=True,worker_init_fn=data.worker_init)
    with torch.no_grad():
        for batch in loader:
            with torch.autocast('cuda',dtype=torch.float16):
                encoded=subject.model.encode_image(batch.flatten(0,1).cuda(non_blocking=True))
            values.append(data.norm(encoded.float()).reshape(len(batch),5,-1).cpu().numpy())
    arr=np.concatenate(values);assert np.isfinite(arr).all()
    with target.open('xb') as f:np.save(f,arr)
    dump(meta,dict(path=str(target),sha256=sha(target),shape=list(arr.shape),
        ids_sha256=ah(np.array([r['image_id'] for r in rows],dtype=np.int64)),
        rows=len(rows),seconds=time.monotonic()-start,renderer_checks='Unchanged per-state outside-note pixel assertions'))
    print('FEATURES',subject.subject['id'],bank,arr.shape,flush=True);return arr

def load_training(subject):
    cfg=configuration();rows=read(Path(cfg['sources']['train']['manifest']));labels=cfg['training_classes'];idx={n:i for i,n in enumerate(labels)}
    v=torch.tensor(feature_cache(subject,'train',rows),device='cuda')
    y=torch.tensor([idx[r['label']] for r in rows],device='cuda')
    w=torch.tensor([[idx[x] for x in r['words'][1:]] for r in rows],device='cuda')
    prompts=[s.format(n) for n in labels for s in data.TEMPLATES]
    tt=objective.tokens_with_prefix(subject.tokenizer,prompts)
    with torch.no_grad():
        z=subject.model.encode_text(subject.tokenizer(prompts).cuda())
        frozen=data.norm(data.norm(z).reshape(len(labels),3,-1).mean(1)).detach()
    return v,y,w,frozen,tt,rows

def preflight_calibrate(subject,loaded):
    dest=OUT/subject.subject['id']
    if (dest/'calibration.json').exists():return read(dest/'calibration.json')
    model=subject.model;v,y,w,frozen,tt,rows=loaded;n=len(frozen);p=prefix(model,42)
    with torch.no_grad():
        star=model.token_embedding.weight[int(subject.tokenizer(['*'])[0,1])]
        a=objective.encode_prefix(model,tt[:12],star,False);b=model.encode_text(tt[:12])
        identity_error=float(abs(a-b).max());assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)
    a=objective.encode_prefix(model,tt[:12],p,True);b=objective.encode_prefix(model,tt[:12],p,False)
    trim_error=float(abs(a-b).max());assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)
    ga=torch.autograd.grad(a.square().mean(),p)[0];gb=torch.autograd.grad(b.square().mean(),p)[0]
    grad_error=float(abs(ga-gb).max());assert torch.allclose(ga,gb,atol=2e-5,rtol=2e-4)
    ids=stream(y.cpu().numpy(),42);native_scale=float(model.logit_scale.exp());gg=[]
    for j in range(16):
        ix=ids[j*32:(j+1)*32];t=objective.prototype(model,tt,p,n)
        s=torch.einsum('bsd,cd->bsc',v[ix],t);ref=torch.einsum('bsd,cd->bsc',v[ix],frozen)
        loss,terms=objective.loss(s,ref,ARMS['ranking'],y[ix],w[ix],1.,native_scale)
        gg.append({k:float(torch.autograd.grad(terms[k],p,retain_graph=True)[0].norm()) for k in ('ce','nuisance')})
        if j==0:
            parity=float(abs(torch.autograd.grad(loss,p,retain_graph=True)[0]-torch.autograd.grad(terms['shared'],p,retain_graph=True)[0]).max())
            assert parity==0.
    k=float(np.median([r['ce'] for r in gg])/np.median([r['nuisance'] for r in gg]));assert np.isfinite(k) and k>0
    cal=dict(kappa=k,frozen_logit_scale=native_scale,seed42_training_gradients=gg,
        identity_error=identity_error,trim_error=trim_error,trim_gradient_error=grad_error,
        zero_weight_gradient_error=parity,trainable_parameters=p.numel(),calibration_training_only=True,
        sequence_sha256=ah(ids),script_sha256=sha(Path(__file__)))
    dump(dest/'calibration.json',cal);print('CALIBRATION',subject.subject['id'],k,identity_error,grad_error,flush=True)
    return cal

def train(subject,loaded,cal,seed,arm):
    root=OUT/subject.subject['id'];dest=root/'runs'/f'seed{seed}'/arm
    if (dest/'complete.json').exists():
        done=read(dest/'complete.json');verify_files(done['files']);return done
    dest.mkdir(parents=True,exist_ok=False)
    model=subject.model;v,y,w,frozen,tt,rows=loaded;n=len(frozen);p=prefix(model,seed)
    initial=ah(p.detach().cpu().numpy());ids=stream(y.cpu().numpy(),seed)
    opt=torch.optim.SGD([p],lr=.002);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2752,eta_min=.00005)
    records=[];chunk=[];start=time.monotonic()
    for step in range(2752):
        ix=ids[step*32:(step+1)*32];t=objective.prototype(model,tt,p,n)
        scores=torch.einsum('bsd,cd->bsc',v[ix],t);ref=torch.einsum('bsd,cd->bsc',v[ix],frozen)
        loss,parts=objective.loss(scores,ref,ARMS[arm],y[ix],w[ix],cal['kappa'],cal['frozen_logit_scale'])
        opt.zero_grad(set_to_none=True);loss.backward();gn=p.grad.detach().norm()
        assert torch.isfinite(loss) and torch.isfinite(gn);opt.step();sched.step()
        chunk.append(dict(loss=float(loss.detach()),gradient_norm=float(gn),**{k:float(z.detach()) for k,z in parts.items()}))
        if (step+1)%172==0:
            records.append(dict(updates=step+1,train={k:float(np.mean([r[k] for r in chunk])) for k in chunk[0]}));chunk=[]
            print('UPDATE',subject.subject['id'],seed,arm,step+1,'seconds',round(time.monotonic()-start,1),flush=True)
    torch.save(dict(prefix=p.detach().cpu(),seed=seed,arm=arm,adapter_location='prefix',updates=2752,
        initial_hash=initial,protocol_sha256=sha(root/'protocol.json'),calibration_sha256=sha(root/'calibration.json')),
        dest/'last.pt')
    dump(dest/'history.json',records)
    done=dict(name=arm,seed=seed,checkpoint=str(dest/'last.pt'),sha256=sha(dest/'last.pt'),
        initial_hash=initial,sequence_hash=ah(ids),updates=2752,selection='fixed_last',
        seconds=time.monotonic()-start,files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    dump(dest/'complete.json',done);return done

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);ap.add_argument('--model',choices=MODELS,required=True);a=ap.parse_args()
    if a.action=='freeze':freeze(a.model);return
    verify(a.model);dest=OUT/a.model;log(dest,'start');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    subject=load_subject(a.model,device='cuda');loaded=load_training(subject)
    # Frozen visual encoder is not involved in prompt optimization.
    subject.model.visual.cpu();cal=preflight_calibrate(subject,loaded);results=[]
    for seed in SEEDS:
        pp=prefix(subject.model,seed);initial=dest/f'initial_prefix_seed{seed}.pt'
        if not initial.exists():torch.save(dict(prefix=pp.detach().cpu(),adapter_location='prefix',updates=0,seed=seed),initial)
        group=[]
        for arm in ARMS:
            log(dest,'training_start',seed=seed,arm=arm);group.append(train(subject,loaded,cal,seed,arm));log(dest,'training_complete',seed=seed,arm=arm)
        assert len({r['initial_hash'] for r in group})==len({r['sequence_hash'] for r in group})==1
        results.extend(group)
    dump(dest/'models.json',results);dump(dest/'training_complete.json',dict(models_sha256=sha(dest/'models.json'),runs=6,
        matched_data_initialization_sequence_budget=True,no_checkpoint_selection=True,evaluation_pending=True))
    log(dest,'complete')

if __name__=='__main__':main()
