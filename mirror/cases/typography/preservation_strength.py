"""Matched preservation and 6x suppression: isolated, fixed seed-42 arms."""
from mirror.cases.typography.diagnose import *
import mirror.cases.typography.strength_pilot as prior
import mirror.cases.typography.train as original
import mirror.cases.typography.broad_support_filtered as bank
import mirror.cases.typography.objective as base

RUN=ROOT/'mirror/cases/typography_preservation_20260929'
CONFIGS={'R_preserve':dict(weight=0.,preserve=True),
         'IS4_preserve':dict(weight=4.,preserve=True),
         'IS6_original':dict(weight=6.,preserve=False),
         'IS6_preserve':dict(weight=6.,preserve=True)}


def command():
    RUN.mkdir(exist_ok=True)
    with (RUN/'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),argv=sys.argv))+'\n')


def geometry(features,reference):
    diff=features@features.T-reference@reference.T
    use=~torch.eye(len(features),dtype=torch.bool,device=features.device)
    return diff[use].square().mean()/.1**2


def objective(scores,reference,features,frozen_text,y,w,cfg,coef,fscale,gcoef):
    value,terms=base.loss(scores,reference,dict(scale=100,weight=cfg['weight']),y,w,coef,fscale)
    geom=geometry(features,frozen_text)
    extra=3*terms['identity']+gcoef*geom if cfg['preserve'] else geom.detach()*0
    total=value+extra if cfg['preserve'] else value
    return total,{**terms,'geometry':geom,'preservation_extra':extra,
                  'shared_total':terms['shared']+extra,'total':total}


def verify():
    original.verify_prompt()
    assert sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    pp=json.loads((RUN/'protocol.json').read_text())
    for p,h in pp['files'].items():assert sha(p)==h,p
    return pp


def prepare():
    assert not (RUN/'protocol.json').exists();torch.set_num_threads(2);original.verify_prompt()
    cfg,model,tok,prefix,v,y,w,t,tt,rows=original.setup()
    calibration=json.loads((original.RUN/'calibration.json').read_text())
    coef=calibration['nuisance_coefficient'];fscale=float(model.logit_scale.exp())
    ids=bank.stream(y,updates=2752);grads=[]
    for j in range(0,512,32):
        ix=ids[j:j+32];features=base.prototype(model,tt,prefix,len(t))
        s=torch.einsum('bsd,cd->bsc',v[ix],features);ref=torch.einsum('bsd,cd->bsc',v[ix],t)
        value,terms=base.loss(s,ref,dict(scale=100,weight=0),y[ix],w[ix],coef,fscale)
        geom=geometry(features,t)
        grads.append(dict(ce=float(torch.autograd.grad(terms['ce'],prefix,retain_graph=True)[0].norm()),
                          geometry=float(torch.autograd.grad(geom,prefix)[0].norm())))
    gcoef=.25*float(np.median([r['ce'] for r in grads])/np.median([r['geometry'] for r in grads]))
    assert np.isfinite(gcoef) and gcoef>0
    # Geometry must vanish at its reference and is unchanged by an orthogonal
    # permutation applied to BOTH sets of feature coordinates.
    perm=torch.randperm(t.shape[-1],generator=torch.Generator().manual_seed(42)).cuda()
    with torch.no_grad():
        features=base.prototype(model,tt,prefix,len(t));g=geometry(features,t)
        error=float((g-geometry(features[:,perm],t[:,perm])).abs());assert error<1e-5
        assert float(geometry(t,t))==0
    ix=ids[:32];features=base.prototype(model,tt,prefix,len(t))
    s=torch.einsum('bsd,cd->bsc',v[ix],features);ref=torch.einsum('bsd,cd->bsc',v[ix],t)
    r,terms=objective(s,ref,features,t,y[ix],w[ix],CONFIGS['R_preserve'],coef,fscale,gcoef)
    gr=torch.autograd.grad(r,prefix,retain_graph=True)[0]
    zero=dict(CONFIGS['IS4_preserve'],weight=0.)
    z,_=objective(s,ref,features,t,y[ix],w[ix],zero,coef,fscale,gcoef)
    gz=torch.autograd.grad(z,prefix,retain_graph=True)[0];assert torch.equal(gr,gz) and torch.equal(r,z)
    full,_=objective(s,ref,features,t,y[ix],w[ix],CONFIGS['IS4_preserve'],coef,fscale,gcoef)
    gf=torch.autograd.grad(full,prefix,retain_graph=True)[0]
    gn=torch.autograd.grad(terms['nuisance'],prefix,retain_graph=True)[0]
    gap=float((gf-gr-4*coef*gn).abs().max());assert gap<2e-4,gap
    old,_=base.loss(s,ref,dict(scale=100,weight=6),y[ix],w[ix],coef,fscale)
    new,_=objective(s,ref,features,t,y[ix],w[ix],CONFIGS['IS6_original'],coef,fscale,gcoef)
    oldg=torch.autograd.grad(old,prefix,retain_graph=True)[0];newg=torch.autograd.grad(new,prefix)[0]
    assert torch.equal(old,new) and torch.equal(oldg,newg)
    files=[Path(__file__),CODE/'PRESERVATION_STRENGTH_20260929_PROTOCOL.md',
           CODE/'strength_pilot.py',CODE/'prompt_unclipped.py',original.RUN/'protocol.json',
           original.RUN/'calibration.json',original.RUN/'selection.json']
    prior4=prior.RUN/'runs/IS_w4/last.pt';files.append(prior4)
    pp=dict(seed=42,configurations=CONFIGS,updates=2752,batch_size=32,
            geometry_coefficient=gcoef,nuisance_coefficient=coef,gradient_calibration=grads,
            files={str(p):sha(p) for p in files},selection='fixed final checkpoints; full-grid evaluation, no winner selection',
            initialization_hash=hashlib.sha256(prefix.detach().cpu().numpy().tobytes()).hexdigest(),
            sequence_hash=hashlib.sha256(ids.tobytes()).hexdigest())
    dump(RUN/'protocol.json',pp);(RUN/'protocol.sha256').write_text(sha(RUN/'protocol.json')+'\n')
    dump(RUN/'implementation_checks.json',dict(zero_nuisance_preservation_gradient_parity=True,
         interaction_gradient_identity_error=gap,original_objective_gradient_parity=True,
         geometry_at_frozen=0,common_permutation_error=error,frozen_encoders=all(not p.requires_grad for p in model.parameters())))
    diagnostic={}
    with torch.no_grad():
        for name,path in [('original_IS',original.RUN/'runs/IS_s100_w1/last.pt'),('IS4',prior4)]:
            token=torch.load(path,map_location='cuda')['prefix'];a=base.prototype(model,tt,token,len(t))
            distance=torch.cdist(a,a);frozen_distance=torch.cdist(t,t)
            use=~torch.eye(len(t),dtype=torch.bool,device='cuda')
            diagnostic[name]=dict(training_class_geometry_loss=float(geometry(a,t)),
                 average_cosine_to_frozen=float((a*t).sum(-1).mean()),
                 pair_distance_ratio=float(distance[use].mean()/frozen_distance[use].mean()))
    dump(RUN/'training_caption_diagnosis.json',diagnostic)
    print('PREPARED',gcoef,json.dumps(diagnostic),flush=True)


def train(name,smoke=False):
    pp=verify();torch.set_num_threads(2)
    cfg,model,tok,prefix,v,y,w,t,tt,rows=original.setup()
    arm=CONFIGS[name];coef=pp['nuisance_coefficient'];gcoef=pp['geometry_coefficient'];fscale=float(model.logit_scale.exp())
    ids=bank.stream(y,updates=2752);ih=hashlib.sha256(prefix.detach().cpu().numpy().tobytes()).hexdigest()
    sh=hashlib.sha256(ids.tobytes()).hexdigest();assert ih==pp['initialization_hash'] and sh==pp['sequence_hash']
    prior_meta=json.loads((original.RUN/'runs/IS_s100_w1/complete.json').read_text())
    assert ih==prior_meta['initial_hash'] and sh==prior_meta['sequence_hash']
    dest=RUN/('smoke' if smoke else 'runs')/name;dest.mkdir(parents=True,exist_ok=False)
    optim=torch.optim.SGD([prefix],lr=.002);sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=2752,eta_min=.00005)
    dev={}
    if not smoke:
        for bank_name in ('original','added'):
            dv,dy,dw,dt,dr,labels=prior.load_bank(bank_name)
            dev[bank_name]=(dv,dy,dw,torch.einsum('bsd,cd->bsc',dv,dt),
                            base.tokens_with_prefix(tok,[s.format(n) for n in labels for s in TEMPLATES]),len(labels))
    history=[];logs=[];start=time.monotonic()
    for step in range(4 if smoke else 2752):
        ix=ids[step*32:(step+1)*32];features=base.prototype(model,tt,prefix,len(t))
        scores=torch.einsum('bsd,cd->bsc',v[ix],features);reference=torch.einsum('bsd,cd->bsc',v[ix],t)
        value,terms=objective(scores,reference,features,t,y[ix],w[ix],arm,coef,fscale,gcoef)
        assert torch.isfinite(value);optim.zero_grad(set_to_none=True);value.backward()
        gn=prefix.grad.detach().norm();assert torch.isfinite(gn);optim.step();sched.step()
        logs.append({**{k:float(u.detach()) for k,u in terms.items()},'gradient_norm':float(gn)})
        if (step+1)%172==0 or smoke and step==3:
            rec=dict(updates=step+1,seconds=time.monotonic()-start,
                     train={k:float(np.mean([r[k] for r in logs])) for k in logs[0]})
            with torch.no_grad():
                for bank_name,(dv,dy,dw,df,dtt,n) in dev.items():
                    ds=torch.einsum('bsd,cd->bsc',dv,base.prototype(model,dtt,prefix,n))
                    rec[bank_name]=prior.statistics(ds,df,dy,dw)
                    if step==2751:np.savez_compressed(dest/f'development_{bank_name}.npz',scores=ds.cpu().numpy())
            history.append(rec);logs=[];dump(dest/'history.json',history)
            print('UPDATE',name,json.dumps(rec),flush=True)
    if not smoke:
        checkpoint=dest/'last.pt';torch.save(dict(prefix=prefix.detach().cpu(),candidate=arm,updates=2752,
             initial_hash=ih,adapter_location='prefix',protocol_sha256=sha(RUN/'protocol.json')),checkpoint)
        dump(dest/'complete.json',dict(updates=2752,initial_hash=ih,sequence_hash=sh,sha256=sha(checkpoint),seconds=time.monotonic()-start))


def register():
    verify();assert not (RUN/'registry.json').exists();reg={}
    for name,arm in CONFIGS.items():
        p=RUN/'runs'/name/'last.pt';meta=json.loads((p.parent/'complete.json').read_text());assert sha(p)==meta['sha256']
        reg[name]=dict(checkpoint=str(p),sha256=meta['sha256'],config=arm)
    dump(RUN/'registry.json',reg);(RUN/'registry.sha256').write_text(sha(RUN/'registry.json')+'\n')
    print('REGISTERED ALL FIXED ARMS',flush=True)


def main():
    command();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','smoke','train','register']);ap.add_argument('name',nargs='?',choices=list(CONFIGS));a=ap.parse_args()
    if a.action in ('smoke','train'):train(a.name,a.action=='smoke')
    else:globals()[a.action]()

if __name__=='__main__':main()
