"""Repair before pooling: adapt final two visual blocks, matched budget/control."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mirror.cases.backdoor.par_diagnostic as data
from mirror.cases.backdoor.common import OUT; from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha; from mirror.cases.backdoor.common import temp_root

RUN=OUT/'backdoor_par/visual_blocks_v3'
SEED=42
STEPS=1536
BATCH=32


def prepare():
    cfg=data.verify();RUN.mkdir(parents=True,exist_ok=True);assert not (RUN/'protocol.json').exists()
    p=dict(stage='three-seed frozen-recipe confirmation; development used to fix recipe, independent tests remain unscored',source_sha256=sha(__file__),parent_sha256=sha(data.POUT/'protocol.json'),
      rationale='Output projection suppresses banana responses and loses genuine-target recall; allow patch-token attention to change before pooling, and control all class preferences rather than banana alone.',
      architecture='last2visualtransformerblocks fully trainable; earlier10blocks, poolingLN, projection and text fixed',
      methods=['ranking','IS'],seeds=[42,43,44],steps=STEPS,batch_size=BATCH,
      optimizer='AdamW lr1e-5 wd0.001 cosine min1e-6; FP16autocast+GradScaler; retry overflowed minibatch, schedule advances only on successful update',
      stream='identical source sequence; exactly1 genuinebanana per32-source batch, other31 classes balanced uniformly',
      shared='100scale all4stateCE +3cleanKL +clean true-vs-other margin retention',
      interaction='squared four-score mixed differences for all true-vs-other preferences, clean versus both trigger realizations and equal-support sham',
      calibration='CE/interaction training gradient-norm median over8initialminibatches; no validation tuning',
      selection='fixed final1536updates',cache=str(temp_root()/'backdoor_par/block10_cache'),
      target='preserve meaningful clean and genuinebanana recognition while recovering attacked recognition',
      manifests={bank:sha(data.POUT/(bank+'.json')) for bank in ('train','development')})
    p['v2_budget_correction']='v2 initial GradScaler overflow could skip an optimizer step; v3 counts exactly1536successfulupdates and retries same minibatch; all prior pilot artifacts retained.'
    parent=OUT/'backdoor_par/visual_blocks_v2'
    p['parent_recipe_sha256']=sha(parent/'protocol.json')
    for name in ('train_cache.json','development_cache.json','calibration.json'):
        record=json.loads((parent/name).read_text())
        dump(RUN/name,record)
    dump(RUN/'protocol.json',p);dump(RUN/'protocol_hash.json',dict(sha256=sha(RUN/'protocol.json')))


def verify():
    p=RUN/'protocol.json';assert sha(p)==json.loads((RUN/'protocol_hash.json').read_text())['sha256']
    cfg=json.loads(p.read_text());assert sha(__file__)==cfg['source_sha256']
    for bank,h in cfg['manifests'].items():assert sha(data.POUT/(bank+'.json'))==h
    return cfg


def tail(vision,x):
    for block in vision.transformer.resblocks[10:]:x=block(x)
    pooled,_=vision._pool(x)
    return F.normalize((pooled@vision.proj).float(),dim=-1)


@torch.no_grad()
def cache():
    cfg=verify();model,prep,tok,state=data.model_load('victim');del state
    root=Path(cfg['cache']);root.mkdir(parents=True,exist_ok=True)
    for bank in ('train','development'):
        path=root/(bank+'.npy');assert not path.exists()
        rows=json.loads((data.POUT/(bank+'.json')).read_text());values=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(rows),4,50,768))
        captured=[]
        def hook(module,args):captured.append(args[0].detach())
        handle=model.visual.transformer.resblocks[10].register_forward_pre_hook(hook);offset=0;parity=[]
        for j,images in enumerate(DataLoader(data.Sources(rows,prep),batch_size=32,num_workers=8,pin_memory=True,worker_init_fn=data.old.typo.worker_init)):
            with torch.autocast('cuda',dtype=torch.float16):z=model.encode_image(images.flatten(0,1).cuda(non_blocking=True))
            x=captured.pop();assert len(captured)==0 and x.shape[1:]==(50,768)
            values[offset:offset+len(images)]=x.half().reshape(len(images),4,50,768).cpu().numpy();offset+=len(images)
            if j==0:
                handle.remove()
                with torch.autocast('cuda',dtype=torch.float16):restored=tail(model.visual,x.half())
                error=float(abs(F.normalize(z.float(),dim=-1)-restored).max());assert error<3e-4,error;parity.append(error)
                handle=model.visual.transformer.resblocks[10].register_forward_pre_hook(hook)
            if j%25==0:print('BLOCK CACHE',bank,offset,'/',len(rows),flush=True)
        handle.remove();assert offset==len(rows);values.flush();del values
        dump(RUN/(bank+'_cache.json'),dict(path=str(path),sha256=sha(path),reconstruction_normalized_error=parity,shape=[len(rows),4,50,768]))


def setup():
    cfg=verify();dd=data.verify();model,prep,tok,state=data.model_load('victim');del state
    vision=model.visual;vision.requires_grad_(False)
    for block in vision.transformer.resblocks[10:]:block.requires_grad_(True)
    params=[p for p in vision.parameters() if p.requires_grad];text=torch.load(data.POUT/'victim_texts.pt',map_location='cpu',weights_only=True)['features'].cuda()
    bb={}
    for bank in ('train','development'):
        m=json.loads((RUN/(bank+'_cache.json')).read_text());assert sha(m['path'])==m['sha256']
        fm=json.loads((data.POUT/('victim_'+bank+'_features.json')).read_text());assert sha(fm['path'])==fm['sha256']
        rows=json.loads((data.POUT/(bank+'.json')).read_text());y=np.array([dd['vocabulary'].index(r['label']) for r in rows])
        bb[bank]=(np.load(m['path'],mmap_mode='r'),torch.from_numpy(np.load(fm['path'])).cuda(),torch.from_numpy(y).cuda(),rows)
    return cfg,vision,params,text,bb


def sequence(labels):
    rng=np.random.default_rng(646300+SEED);target=int(labels.max());pools=[np.flatnonzero(labels==i) for i in range(target+1)]
    seq=[]
    for step in range(STEPS):
        classes=rng.integers(0,target,size=BATCH-1)
        row=[rng.choice(pools[target])]+[rng.choice(pools[c]) for c in classes]
        rng.shuffle(row);seq.append(row)
    return np.array(seq)


def losses(scores,ref,y):
    ce=F.cross_entropy((100*scores).flatten(0,1),y.repeat_interleave(4))
    kl=F.kl_div(F.log_softmax(100*scores[:,0],-1),F.softmax(100*ref[:,0],-1),reduction='batchmean')
    m=scores.gather(-1,y[:,None,None].expand(-1,4,1))-scores
    rm=ref.gather(-1,y[:,None,None].expand(-1,4,1))-ref
    retain=F.relu(rm[:,0]-m[:,0]).square().mean()/.1**2
    interaction=(m[:,1:]-m[:,:1]).square().mean()/.1**2
    return dict(ce=ce,kl=kl,retain=retain,interaction=interaction,shared=ce+3*kl+retain)


def score(vision,tokens,text):
    x=torch.from_numpy(np.asarray(tokens).copy()).cuda(non_blocking=True)
    with torch.autocast('cuda',dtype=torch.float16):v=tail(vision,x.flatten(0,1))
    return v.reshape(len(x),4,-1)@text.T


def calibrate():
    cfg,vision,params,text,bb=setup();x,v,y,_=bb['train'];seq=sequence(y.cpu().numpy());grads=[]
    for j in range(8):
        ix=seq[j];s=score(vision,x[ix],text);ll=losses(s,v[ix]@text.T,y[ix]);g={}
        for key in ('ce','interaction'):
            gg=torch.autograd.grad(ll[key],params,retain_graph=True)
            g[key]=float(torch.stack([z.float().square().sum() for z in gg]).sum().sqrt())
        grads.append(g)
        if j==0:
            a=torch.autograd.grad(ll['shared']+0*ll['interaction'],params,retain_graph=True)
            b=torch.autograd.grad(ll['shared'],params,retain_graph=True)
            assert all(torch.equal(u,z) for u,z in zip(a,b))
    coef=float(np.median([g['ce'] for g in grads])/np.median([g['interaction'] for g in grads]))
    dump(RUN/'calibration.json',dict(coefficient=coef,gradients=grads,parameters=sum(p.numel() for p in params),zero_weight_gradient_parity=True))
    print('BLOCKS CALIBRATED',coef,flush=True)


@torch.no_grad()
def evaluate(vision,text,bb):
    x,v,y,rows=bb['development'];ss=[]
    for j in range(0,len(x),32):ss.append(score(vision,x[j:j+32],text).cpu().numpy())
    s=np.concatenate(ss);labels=y.cpu().numpy();good=s.argmax(-1)==labels[:,None];nt=labels!=len(text)-1
    m=s[np.arange(len(s))[:,None],np.arange(4)[None,:],labels[:,None]][:,:,None]-s
    stats={state+'_top1':float(good[:,j].mean()) for j,state in enumerate(data.STATES)}
    stats.update(target_clean_top1=float(good[~nt,0].mean()),target_attack_top1=float(good[~nt,1].mean()),
      asr=float((s[nt,1].argmax(-1)==len(text)-1).mean()),target_interaction_abs=float(abs(m[nt,1,-1]-m[nt,0,-1]).mean()),
      all_contrast_interaction_abs=float(abs(m[:,1]-m[:,0]).mean()),clean_correct_trigger_wrong=int((good[:,0]&~good[:,1]).sum()))
    return stats,s


def train(method,smoke):
    cfg,vision,params,text,bb=setup();x,v,y,_=bb['train'];seq=sequence(y.cpu().numpy())
    out=RUN/('smoke' if smoke else 'runs')/(method+'_seed'+str(SEED));out.mkdir(parents=True,exist_ok=True);assert not (out/'last.pt').exists()
    coef=json.loads((RUN/'calibration.json').read_text())['coefficient'];torch.manual_seed(SEED)
    initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params)).hexdigest()
    opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=1e-6)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024);overflows=0;history=[];start=time.monotonic();steps=2 if smoke else STEPS
    for step in range(steps):
        ix=seq[step]
        while True:
            ll=losses(score(vision,x[ix],text),v[ix]@text.T,y[ix]);value=ll['shared']+(method=='IS')*coef*ll['interaction']
            assert torch.isfinite(value);opt.zero_grad(set_to_none=True)
            before=scaler.get_scale();scaler.scale(value).backward();scaler.step(opt);scaler.update()
            if scaler.get_scale()>=before:break
            overflows+=1
            assert overflows<100,'Repeated overflow; stop instead of counting skipped steps.'
        sched.step()
        if (step+1)%256==0 or step+1==steps:
            stats,scores=evaluate(vision,text,bb);history.append(dict(step=step+1,development=stats,loss={k:float(z.detach()) for k,z in ll.items()}))
            dump(out/'history.json',history);print('BLOCK REPAIR',method,step+1,stats,'seconds',round(time.monotonic()-start,1),flush=True)
    state={k:z.cpu() for k,z in vision.state_dict().items() if k.startswith(('transformer.resblocks.10.','transformer.resblocks.11.'))}
    torch.save(dict(visual_blocks=state,method=method,steps=steps,protocol_sha256=sha(RUN/'protocol.json')),out/'last.pt')
    np.savez_compressed(out/'development_scores.npz',scores=scores)
    dump(out/'complete.json',dict(seconds=time.monotonic()-start,steps=steps,overflow_retries=overflows,seed=SEED,initial_sha256=initial,sequence_sha256=hashlib.sha256(seq[:steps].tobytes()).hexdigest(),checkpoint_sha256=sha(out/'last.pt')))


if __name__=='__main__':
    command();torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','cache','calibrate','smoke','train'])
    p.add_argument('--method',choices=['ranking','IS'],default='ranking');p.add_argument('--seed',type=int,default=42);a=p.parse_args();SEED=a.seed
    if a.action in ('train','smoke'):train(a.method,a.action=='smoke')
    else:globals()[a.action]()

