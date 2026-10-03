"""Matched soft-prompt pilot with broader lexical training support."""
from mirror.cases.typography.diagnose import *
import mirror.cases.typography.objective as base
import mirror.cases.typography.training_lib as old
import mirror.cases.typography.broad_support_filtered as bank
RUN=bank.DEST/'matched_duration'
CONFIGS={'R_s100':dict(scale=100,weight=0.),'IS_s100_w1':dict(scale=100,weight=1.)}
SEED=42
network=base.network
tokens_with_prefix=base.tokens_with_prefix
encode_prefix=base.encode_prefix
prototype=base.prototype

def command_log():
    RUN.mkdir(exist_ok=True)
    with (RUN/'commands.log').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+' '+shlex.join([sys.executable,*sys.argv])+'\n')

def verify_prompt():
    cfg=bank.verify_data();p=RUN/'protocol.json';assert sha(p)==(RUN/'protocol.sha256').read_text().strip()
    for fn,h in json.loads(p.read_text())['files'].items():assert sha(fn)==h,fn
    return cfg

def setup():
    cfg=bank.verify_data();torch.set_num_threads(2);model,tok=network();prefix=base.prefix_init()
    v,y,w,t,rows=bank.data('train');n=len(cfg['training_classes'])
    tt=tokens_with_prefix(tok,[s.format(label) for label in cfg['training_classes'] for s in TEMPLATES])
    return cfg,model,tok,prefix,v,y,w,t[:n],tt,rows

def prepare():
    assert not (RUN/'protocol.json').exists()
    cfg,model,tok,prefix,v,y,w,t,tt,rows=setup()
    files=[Path(__file__),CODE/'broad_duration_retest.py',CODE/'BROAD_SUPPORT_PROTOCOL.md',CODE/'BROAD_SUPPORT_AMENDMENT.md',CODE/'BROAD_DURATION_PROTOCOL.md',
           CODE/'broad_support_filtered.py',CODE/'objective.py',CODE/'train.py',
           bank.DEST/'filtered_data_protocol.json']
    dump(RUN/'protocol.json',dict(files={str(p):sha(p) for p in files},seed=42,updates=2752,batch_size=32,
       unique_sources=len(rows),training_classes=len(cfg['training_classes']),configurations=CONFIGS,
       optimizer='SGD',lr=.002,eta_min=.00005,clipping=False,shared_KL_weight=3.,shared_margin_retention=1.,
       selection='fixed final checkpoints; no search',public_status='developmental'))
    (RUN/'protocol.sha256').write_text(sha(RUN/'protocol.json')+'\n')
    ids=bank.stream(y,updates=2752);fscale=float(model.logit_scale.exp());gg=[];n=len(t)
    for j in range(0,512,32):
        ix=ids[j:j+32];features=prototype(model,tt,prefix,n)
        s=torch.einsum('bsd,cd->bsc',v[ix],features);ref=torch.einsum('bsd,cd->bsc',v[ix],t)
        loss,terms=base.loss(s,ref,CONFIGS['R_s100'],y[ix],w[ix],1.,fscale)
        gg.append({k:float(torch.autograd.grad(terms[k],prefix,retain_graph=True)[0].norm()) for k in ('ce','nuisance')})
        if j==0:
            parity=float(abs(torch.autograd.grad(loss,prefix,retain_graph=True)[0]-torch.autograd.grad(terms['shared'],prefix,retain_graph=True)[0]).max());assert parity==0
    coefficient=float(np.median([r['ce'] for r in gg])/np.median([r['nuisance'] for r in gg]))
    dump(RUN/'calibration.json',dict(nuisance_coefficient=coefficient,training_gradients=gg,frozen_logit_scale=fscale))
    dump(RUN/'implementation_tests.json',dict(nuisance_zero_gradient_parity_error=parity,trainable_parameters=prefix.numel(),
       frozen_encoders=all(not p.requires_grad for p in model.parameters()),sequence_hash=hashlib.sha256(ids.tobytes()).hexdigest(),
       no_excluded_label_in_training=not (set(cfg['training_classes'])&bank.EXCLUDE)))
    torch.save(dict(prefix=prefix.detach().cpu(),updates=0,adapter_location='prefix'),RUN/'initial_prefix.pt')
    print('PREPARED',len(rows),n,coefficient,flush=True)

def train(name,smoke=False):
    verify_prompt();cfg,model,tok,prefix,v,y,w,t,tt,rows=setup();n=len(t)
    dv,dy,dw,oldtext,_=old.data('development');vocab=torch.load(OUT/'texts.pt',map_location='cpu')['vocabulary']
    evaltt=tokens_with_prefix(tok,[s.format(label) for label in vocab for s in TEMPLATES])
    cand=CONFIGS[name];cal=json.loads((RUN/'calibration.json').read_text());coef=cal['nuisance_coefficient'];fscale=cal['frozen_logit_scale']
    opt=torch.optim.SGD([prefix],lr=.002);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2752,eta_min=.00005)
    count=4 if smoke else 2752;ids=bank.stream(y,updates=2752)[:count*32]
    initial=hashlib.sha256(prefix.detach().cpu().numpy().tobytes()).hexdigest()
    dest=RUN/('smoke' if smoke else 'runs')/name;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'last.pt').exists()
    history=[];logs=[];start=time.monotonic()
    for step in range(count):
        ix=ids[step*32:(step+1)*32];features=prototype(model,tt,prefix,n)
        scores=torch.einsum('bsd,cd->bsc',v[ix],features);ref=torch.einsum('bsd,cd->bsc',v[ix],t)
        loss,terms=base.loss(scores,ref,cand,y[ix],w[ix],coef,fscale);assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();gn=prefix.grad.detach().norm();assert torch.isfinite(gn)
        opt.step();scheduler.step();logs.append({**{k:float(z.detach()) for k,z in terms.items()},'gradient_norm':float(gn)})
        if (step+1)%172==0 or step+1==count:
            scores,stats=base.evaluate(model,evaltt,prefix,dv,dy,dw,len(vocab))
            history.append(dict(updates=step+1,development_original=stats,train={k:float(np.mean([r[k] for r in logs])) for k in logs[0]}))
            logs=[];dump(dest/'history.json',history);print('UPDATE',name,step+1,json.dumps(stats),'sec',time.monotonic()-start,flush=True)
    torch.save(dict(prefix=prefix.detach().cpu(),candidate=cand,initial_hash=initial,updates=count,
       protocol_sha256=sha(RUN/'protocol.json'),calibration_sha256=sha(RUN/'calibration.json'),adapter_location='prefix'),dest/'last.pt')
    np.savez_compressed(dest/'development.npz',scores=scores)
    dump(dest/'complete.json',dict(updates=count,initial_hash=initial,sequence_hash=hashlib.sha256(ids.tobytes()).hexdigest(),
       sha256=sha(dest/'last.pt'),seconds=time.monotonic()-start))

def select():
    verify_prompt();rr=[];ih=set();sh=set()
    for name in CONFIGS:
        dest=RUN/'runs'/name;m=json.loads((dest/'complete.json').read_text())
        assert m['updates']==2752 and sha(dest/'last.pt')==m['sha256'];ih.add(m['initial_hash']);sh.add(m['sequence_hash'])
        rr.append(dict(name=name,checkpoint=str(dest/'last.pt'),sha256=m['sha256'],
             **json.loads((dest/'history.json').read_text())[-1]['development_original']))
    assert len(ih)==len(sh)==1
    sel=dict(ranking=rr[0],IS=rr[1],same_scale_ablation='R_s100',all_candidates=rr,
         initial_prefix=dict(name='initial_prefix',checkpoint=str(RUN/'initial_prefix.pt'),sha256=sha(RUN/'initial_prefix.pt')),
         selection='fixed final checkpoints; no search',identical_init_source_sequence=True)
    dump(RUN/'selection.json',sel);(RUN/'selection.sha256').write_text(sha(RUN/'selection.json')+'\n')
    print('REGISTERED',json.dumps(sel),flush=True)

def retest():
    import mirror.cases.typography.broad_duration_retest as rt
    rt.main()

def main():
    command_log();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','smoke','select','retest']);ap.add_argument('name',nargs='?');a=ap.parse_args()
    if a.action in ('train','smoke'):train(a.name,a.action=='smoke')
    else:globals()[a.action]()
if __name__=='__main__':main()

