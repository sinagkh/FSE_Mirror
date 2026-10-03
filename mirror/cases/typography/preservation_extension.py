"""Two fixed typography extensions; reuse the validated training loop unchanged."""
from mirror.cases.typography.diagnose import *
import mirror.cases.typography.preservation_strength as previous

PREVIOUS_RUN = previous.RUN
RUN = ROOT/'mirror/cases/typography_preservation_extension_20260929'
CONFIGS = {'IS8_preserve': dict(weight=8., preserve=True, mode='raw'),
           'IS6_directional_preserve': dict(weight=6., preserve=True, mode='directional')}
original, base, bank = previous.original, previous.base, previous.bank
raw_objective = previous.objective
DIRECTIONAL_COEFFICIENT = None


def command():
    RUN.mkdir(exist_ok=True)
    with (RUN/'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), argv=sys.argv))+'\n')


def verify_previous():
    original.verify_prompt()
    p=PREVIOUS_RUN/'protocol.json'
    assert sha(p)==(PREVIOUS_RUN/'protocol.sha256').read_text().strip()
    pp=json.loads(p.read_text())
    for path,h in pp['files'].items(): assert sha(path)==h,path
    reg=PREVIOUS_RUN/'registry.json'
    assert sha(reg)==(PREVIOUS_RUN/'registry.sha256').read_text().strip()
    control=json.loads(reg.read_text())['R_preserve']
    assert sha(control['checkpoint'])==control['sha256']
    return pp


def directional(scores,features,y,w):
    margins=base.old.margins(scores,y,w)
    contrasts=torch.stack([margins[:,i]-margins[:,j]
                           for i in range(1,5) for j in range(i+1,5)],1)
    distances=(features[y,None,:]-features[w]).norm(dim=-1)
    normalized=contrasts/distances[:,None,:].clamp_min(1e-6)
    return normalized.square().mean()/.1**2, distances, normalized


def objective(scores,reference,features,frozen_text,y,w,cfg,coef,fscale,gcoef):
    if cfg['mode']=='raw':
        return raw_objective(scores,reference,features,frozen_text,y,w,cfg,coef,fscale,gcoef)
    shared,terms=raw_objective(scores,reference,features,frozen_text,y,w,
                             dict(cfg,weight=0.),coef,fscale,gcoef)
    penalty,distances,_=directional(scores,features,y,w)
    value=shared if cfg['weight']==0 else shared+cfg['weight']*DIRECTIONAL_COEFFICIENT*penalty
    return value, {**terms,'raw_nuisance':terms['nuisance'],'nuisance':penalty,
                   'caption_distance_min':distances.min(),
                   'caption_distance_mean':distances.mean(),'total':value}


def verify():
    global DIRECTIONAL_COEFFICIENT
    verify_previous()
    p=RUN/'protocol.json'; assert sha(p)==(RUN/'protocol.sha256').read_text().strip()
    pp=json.loads(p.read_text())
    for path,h in pp['files'].items(): assert sha(path)==h,path
    DIRECTIONAL_COEFFICIENT=pp['directional_coefficient']
    return pp


def prepare():
    global DIRECTIONAL_COEFFICIENT
    assert not (RUN/'protocol.json').exists()
    old=verify_previous(); cfg,model,tok,prefix,v,y,w,t,tt,rows=original.setup()
    coef=old['nuisance_coefficient']; gcoef=old['geometry_coefficient']
    scale=float(model.logit_scale.exp()); ids=bank.stream(y,updates=2752); calibration=[]
    for j in range(0,512,32):
        ix=ids[j:j+32]; text=base.prototype(model,tt,prefix,len(t))
        s=torch.einsum('bsd,cd->bsc',v[ix],text); ref=torch.einsum('bsd,cd->bsc',v[ix],t)
        _,terms=base.loss(s,ref,dict(scale=100,weight=0),y[ix],w[ix],coef,scale)
        d,_,_=directional(s,text,y[ix],w[ix])
        calibration.append(dict(raw=float(torch.autograd.grad(terms['nuisance'],prefix,retain_graph=True)[0].norm()),
                                directional=float(torch.autograd.grad(d,prefix)[0].norm())))
    DIRECTIONAL_COEFFICIENT=coef*np.median([r['raw'] for r in calibration])/np.median([r['directional'] for r in calibration])
    assert np.isfinite(DIRECTIONAL_COEFFICIENT) and DIRECTIONAL_COEFFICIENT>0
    ix=ids[:32]; text=base.prototype(model,tt,prefix,len(t))
    s=torch.einsum('bsd,cd->bsc',v[ix],text); ref=torch.einsum('bsd,cd->bsc',v[ix],t)
    args=(s,ref,text,t,y[ix],w[ix]); checks={}
    raw,terms=raw_objective(*args,CONFIGS['IS8_preserve'],coef,scale,gcoef)
    new,_=objective(*args,CONFIGS['IS8_preserve'],coef,scale,gcoef)
    graw=torch.autograd.grad(raw,prefix,retain_graph=True)[0]
    gnew=torch.autograd.grad(new,prefix,retain_graph=True)[0]
    assert torch.equal(raw,new) and torch.equal(graw,gnew)
    checks['raw8_exact_loss_gradient_parity']=True
    reference,_=raw_objective(*args,dict(weight=0.,preserve=True),coef,scale,gcoef)
    gr=torch.autograd.grad(reference,prefix,retain_graph=True)[0]
    for name,arm in CONFIGS.items():
        zero,_=objective(*args,dict(arm,weight=0.),coef,scale,gcoef)
        gz=torch.autograd.grad(zero,prefix,retain_graph=True)[0]
        assert torch.equal(zero,reference) and torch.equal(gz,gr)
        checks[name+'_zero_weight_exact_ranking_parity']=True
    d,dist,z=directional(s,text,y[ix],w[ix])
    half,_,_=directional(s*.5,text*.5,y[ix],w[ix])
    assert torch.allclose(d,half,atol=1e-6,rtol=1e-6)
    unit=(text[y[ix],None,:]-text[w[ix]])/dist[:,:,None]
    vv=torch.stack([v[ix,i]-v[ix,j] for i in range(1,5) for j in range(i+1,5)],1)
    geometric=torch.einsum('bkd,bqd->bkq',vv,unit)
    error=float((z-geometric).abs().max()); assert error<2e-5,error
    full,_=objective(*args,CONFIGS['IS6_directional_preserve'],coef,scale,gcoef)
    gf=torch.autograd.grad(full,prefix,retain_graph=True)[0]
    gd=torch.autograd.grad(d,prefix)[0]
    gap=float((gf-gr-6*DIRECTIONAL_COEFFICIENT*gd).abs().max()); assert gap<2e-4,gap
    assert torch.isfinite(gf).all()
    checks.update(directional_scaling_invariance=True,projection_identity_error=error,
                  directional_gradient_identity_error=gap,initial_min_caption_distance=float(dist.min()),
                  frozen_encoders=all(not p.requires_grad for p in model.parameters()))
    files=[Path(__file__),CODE/'PRESERVATION_EXTENSION_20260929_PROTOCOL.md',
           CODE/'preservation_extension_retest.py',CODE/'preservation_extension_summary.py',
           CODE/'run_preservation_extension_20260929.sh',Path(previous.__file__),
           PREVIOUS_RUN/'protocol.json',PREVIOUS_RUN/'registry.json']
    ih=hashlib.sha256(prefix.detach().cpu().numpy().tobytes()).hexdigest()
    sh=hashlib.sha256(ids.tobytes()).hexdigest()
    assert ih==old['initialization_hash'] and sh==old['sequence_hash']
    pp=dict(seed=42,updates=2752,batch_size=32,configurations=CONFIGS,
            geometry_coefficient=gcoef,nuisance_coefficient=coef,
            directional_coefficient=float(DIRECTIONAL_COEFFICIENT),gradient_calibration=calibration,
            initialization_hash=ih,sequence_hash=sh,files={str(p):sha(p) for p in files},
            selection='fixed final checkpoints; both arms fully reported',matched_control='R_preserve')
    dump(RUN/'protocol.json',pp); (RUN/'protocol.sha256').write_text(sha(RUN/'protocol.json')+'\n')
    dump(RUN/'implementation_checks.json',checks)
    print('PREPARED',DIRECTIONAL_COEFFICIENT,json.dumps(checks),flush=True)


def attach():
    # Change module globals in this process only; the earlier source/artifacts stay intact.
    previous.RUN=RUN; previous.CONFIGS=CONFIGS; previous.verify=verify; previous.objective=objective


def main():
    command(); ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','smoke','train','register'])
    ap.add_argument('name',nargs='?',choices=list(CONFIGS)); a=ap.parse_args()
    if a.action=='prepare': prepare()
    else:
        attach()
        if a.action=='register': previous.register()
        else: previous.train(a.name,a.action=='smoke')


if __name__=='__main__': main()
