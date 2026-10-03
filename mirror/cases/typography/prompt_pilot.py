"""Shared soft prefix with matched ranking/IS losses and fixed data."""
from mirror.cases.typography.diagnose import *
import mirror.cases.typography.training_lib as old
from torch.nn import functional as F

RUN=ROOT/'mirror/cases/typography_prompt_20260926'
CONFIGS=old.CONFIGS

def command_log():
    RUN.mkdir(exist_ok=True)
    with (RUN/'commands.log').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+' '+shlex.join([sys.executable,*sys.argv])+'\n')

def network():
    model,_,tok=load_model();model.visual.cpu();model.eval().requires_grad_(False)
    return model,tok

def prefix_init(device='cuda',seed=42):
    g=torch.Generator(device='cpu').manual_seed(seed)
    return torch.nn.Parameter((.02*torch.randn(512,generator=g)).to(device))

def tokens_with_prefix(tok,prompts,device='cuda'):
    orig=tok(prompts).to(device);assert (orig.argmax(-1)<76).all(),'prefix would truncate a caption'
    star=int(tok(['*'])[0,1]);new=torch.cat([orig[:,:1],torch.full_like(orig[:,:1],star),orig[:,1:-1]],1)
    assert torch.equal(new.argmax(-1),orig.argmax(-1)+1)
    return new

def encode_prefix(model,tokens,prefix,trim=True):
    length=int(tokens.argmax(-1).max())+1 if trim else 77;tokens=tokens[:,:length]
    x=model.token_embedding(tokens).clone();x[:,1]=prefix
    x=model.transformer(x+model.positional_embedding[:length],attn_mask=model.attn_mask[:length,:length])
    x=model.ln_final(x)
    return x[torch.arange(len(tokens),device=tokens.device),tokens.argmax(-1)]@model.text_projection

def prototype(model,tokens,prefix,nclasses):
    return norm(norm(encode_prefix(model,tokens,prefix)).reshape(nclasses,3,-1).mean(1))

def loss(scores,reference,cand,y,w,coef,frozen_scale):
    ce=F.cross_entropy((cand['scale']*scores).flatten(0,1),y.repeat_interleave(5))
    mm=old.margins(scores,y,w);fm=old.margins(reference,y,w)
    identity=F.kl_div(F.log_softmax(frozen_scale*scores[:,:2].flatten(0,1),dim=-1),F.softmax(frozen_scale*reference[:,:2].flatten(0,1),dim=-1),reduction='batchmean')
    agreement=F.relu(.045-mm).square().mean()
    retention=F.relu(fm[:,:2].mean((0,2))-mm[:,:2].mean((0,2))).square().mean()/.1**2
    shared=ce+3*identity+1.2*agreement+retention
    nuisance=torch.stack([mm[:,i]-mm[:,j] for i in range(1,5) for j in range(i+1,5)],1).square().mean()/.1**2
    return shared+cand['weight']*coef*nuisance,dict(ce=ce,identity=identity,agreement=agreement,retention=retention,shared=shared,nuisance=nuisance)

def evaluate(model,tokens,prefix,v,y,w,n):
    with torch.no_grad():
        t=prototype(model,tokens,prefix,n);s=torch.einsum('bsd,cd->bsc',v,t);m=old.margins(s,y,w).cpu().numpy()
    conf=np.stack([m[:,3,0],m[:,4,1]],1);inter=m[:,2]-conf
    r=dict(clean=float((m[:,0]>0).mean()),blank=float((m[:,1]>0).mean()),conflict=float((conf>0).mean()),congruent=float((m[:,2]>0).mean()),clean_top1=float((s[:,0].argmax(-1)==y).float().mean()),interaction=float(abs(inter).mean()))
    r['utility']=(r['clean']+r['blank']+r['conflict'])/3
    return s.cpu().numpy(),r

def verify_prompt():
    cfg=verify();assert sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    pp=json.loads((RUN/'protocol.json').read_text())
    for p,h in pp['files'].items():assert sha(p)==h,p
    return cfg

def prepare():
    cfg=verify();torch.set_num_threads(2);assert not (RUN/'protocol.json').exists()
    paths=[Path(__file__),CODE/'PROMPT_PILOT_PROTOCOL.md',CODE/'train.py',OUT/'protocol.json',OUT/'texts.pt',OUT/'train_standard_features.json',OUT/'development_standard_features.json']
    dump(RUN/'protocol.json',dict(files={str(p):sha(p) for p in paths},configurations=CONFIGS,seed=42,updates=1024,epochs=16,adapter_location='prefix',optimizer='SGD',lr=.002,eta_min=.00005,shared_KL_weight=3.,external_status='developmental',model_weights_sha256=WEIGHT_SHA))
    (RUN/'protocol.sha256').write_text(sha(RUN/'protocol.json')+'\n')
    model,tok=network();prefix=prefix_init();v,y,w,base,_=old.data('train');base=base[:len(cfg['seen_classes'])]
    dv,dy,dw,full,_=old.data('development');vocab=torch.load(OUT/'texts.pt',map_location='cpu')['vocabulary']
    tt=tokens_with_prefix(tok,[s.format(n) for n in vocab for s in TEMPLATES]);n=len(cfg['seen_classes']);train_tt=tt[:n*3]
    # Identity substitution agrees with the original encoder on literal-star captions.
    with torch.no_grad():
        star=model.token_embedding.weight[int(tok(['*'])[0,1])]
        x=encode_prefix(model,train_tt[:12],star,False);z=model.encode_text(train_tt[:12]);identity_error=float(abs(x-z).max());assert identity_error<1e-5
    short=encode_prefix(model,train_tt[:12],prefix,True);long=encode_prefix(model,train_tt[:12],prefix,False)
    trim_error=float(abs(short-long).max());assert torch.allclose(short,long,atol=1e-5,rtol=1e-5)
    g1=torch.autograd.grad(short.square().mean(),prefix)[0];g2=torch.autograd.grad(long.square().mean(),prefix)[0]
    grad_error=float(abs(g1-g2).max());assert torch.allclose(g1,g2,atol=2e-5,rtol=2e-4)
    ids=np.random.default_rng(4200).permutation(len(v));gg=[];fscale=float(model.logit_scale.exp())
    for start in range(0,512,32):
        ix=ids[start:start+32];t=prototype(model,train_tt,prefix,n);s=torch.einsum('bsd,cd->bsc',v[ix],t);ref=torch.einsum('bsd,cd->bsc',v[ix],base)
        value,terms=loss(s,ref,dict(scale=100,weight=0),y[ix],w[ix],1.,fscale)
        norms={key:float(torch.autograd.grad(terms[key],prefix,retain_graph=True)[0].norm()) for key in ('ce','nuisance')};gg.append(norms)
        if start==0:
            parity_error=float(abs(torch.autograd.grad(value,prefix,retain_graph=True)[0]-torch.autograd.grad(terms['shared'],prefix,retain_graph=True)[0]).max());assert parity_error==0.
    coef=float(np.median([r['ce'] for r in gg])/np.median([r['nuisance'] for r in gg]))
    dump(RUN/'calibration.json',dict(nuisance_coefficient=coef,training_gradients=gg,frozen_logit_scale=fscale))
    dump(RUN/'implementation_tests.json',dict(identity_error=identity_error,trim_forward_error=trim_error,trim_gradient_error=grad_error,zero_nuisance_loss_gradient_error=parity_error,trainable_parameters=prefix.numel(),frozen_model_parameters=all(not p.requires_grad for p in model.parameters())))
    scores,stats=evaluate(model,tt,prefix,dv,dy,dw,len(vocab));np.savez_compressed(RUN/'initial_development.npz',scores=scores)
    dump(RUN/'initial_development.json',stats);print('PREPARED',coef,stats,flush=True)

def train(name,smoke=False):
    cfg=verify_prompt();torch.set_num_threads(2);model,tok=network();prefix=prefix_init();initial_hash=hashlib.sha256(prefix.detach().cpu().numpy().tobytes()).hexdigest()
    cand=CONFIGS[name];v,y,w,base,_=old.data('train');dv,dy,dw,full,_=old.data('development');n=len(cfg['seen_classes']);base=base[:n]
    vocab=torch.load(OUT/'texts.pt',map_location='cpu')['vocabulary'];tt=tokens_with_prefix(tok,[s.format(nm) for nm in vocab for s in TEMPLATES]);train_tt=tt[:n*3]
    cal=json.loads((RUN/'calibration.json').read_text());coef=cal['nuisance_coefficient'];fscale=cal['frozen_logit_scale']
    opt=torch.optim.SGD([prefix],lr=.002);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=1024,eta_min=.00005)
    dest=RUN/('smoke' if smoke else 'runs')/name;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'last.pt').exists()
    history=[];updates=0;start=time.monotonic();sequence=hashlib.sha256()
    for epoch in range(1,2 if smoke else 17):
        ids=np.random.default_rng(4200+epoch).permutation(len(v));ids=ids[:128] if smoke else ids;logs=[]
        for j in range(0,len(ids),32):
            ix=ids[j:j+32];sequence.update(ix.tobytes());t=prototype(model,train_tt,prefix,n)
            scores=torch.einsum('bsd,cd->bsc',v[ix],t);ref=torch.einsum('bsd,cd->bsc',v[ix],base)
            value,terms=loss(scores,ref,cand,y[ix],w[ix],coef,fscale);assert torch.isfinite(value)
            opt.zero_grad(set_to_none=True);value.backward();gn=torch.nn.utils.clip_grad_norm_([prefix],1.);assert torch.isfinite(gn);opt.step();scheduler.step();updates+=1
            logs.append({k:float(u.detach()) for k,u in terms.items()})
        scores,stats=evaluate(model,tt,prefix,dv,dy,dw,len(vocab));history.append(dict(epoch=epoch,updates=updates,development=stats,lr=opt.param_groups[0]['lr'],train={k:float(np.mean([r[k] for r in logs])) for k in logs[0]}));dump(dest/'history.json',history)
        print('EPOCH',name,epoch,json.dumps(stats),'seconds',time.monotonic()-start,flush=True)
    assert updates==(4 if smoke else 1024)
    torch.save(dict(prefix=prefix.detach().cpu(),candidate=cand,initial_hash=initial_hash,updates=updates,protocol_sha256=sha(RUN/'protocol.json'),adapter_location='prefix'),dest/'last.pt')
    np.savez_compressed(dest/'development.npz',scores=scores);dump(dest/'complete.json',dict(updates=updates,initial_hash=initial_hash,sequence_hash=sequence.hexdigest(),sha256=sha(dest/'last.pt'),seconds=time.monotonic()-start))

def select():
    verify_prompt();rr=[];initial=set();sequences=set()
    for name in CONFIGS:
        dest=RUN/'runs'/name;meta=json.loads((dest/'complete.json').read_text());assert meta['updates']==1024 and sha(dest/'last.pt')==meta['sha256'];initial.add(meta['initial_hash']);sequences.add(meta['sequence_hash'])
        r=json.loads((dest/'history.json').read_text())[-1]['development'];rr.append(dict(name=name,checkpoint=str(dest/'last.pt'),sha256=meta['sha256'],**r))
    assert len(initial)==len(sequences)==1
    selected={label:sorted([r for r in rr if r['name'].startswith(prefix)],key=lambda r:(-r['utility'],-r['clean_top1'],r['interaction'],r['name']))[0] for label,prefix in [('ranking','R_'),('IS','IS_')]}
    selected.update(same_scale_ablation='R_s'+str(CONFIGS[selected['IS']['name']]['scale']),all_candidates=rr,identical_init_source_sequence=True)
    dump(RUN/'selection.json',selected);(RUN/'selection.sha256').write_text(sha(RUN/'selection.json')+'\n');print('SELECTED',json.dumps(selected),flush=True)

def main():
    command_log();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','smoke','select']);ap.add_argument('name',nargs='?');args=ap.parse_args()
    if args.action in ('train','smoke'):train(args.name,args.action=='smoke')
    else:globals()[args.action]()

if __name__=='__main__':main()
