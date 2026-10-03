"""Replicate fixed typography preservation recipes without modifying old runs."""
import mirror.cases.typography.strength_retest as rt  # Preserve the validated standalone import context.
from mirror.cases.typography.diagnose import *
import mirror.cases.typography.replicate as confirmation
import mirror.cases.typography.preservation_strength as trainer
import mirror.cases.typography.preservation_extension as pilot
import subprocess

RUN = ROOT/'mirror/cases/typography_preservation_replication_20260929'
ARMS = {'IS8_preserve': dict(weight=8., preserve=True, mode='raw'),
        'R_preserve': dict(weight=0., preserve=True, mode='raw')}


def read(p):
    return json.loads(Path(p).read_text())


def command():
    RUN.mkdir(exist_ok=True)
    with (RUN/'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
            argv=[sys.executable,*sys.argv],cwd=str(Path.cwd())))+'\n')


def original(seed):
    return ROOT/'mirror/cases/typography_broad_confirmation_20260926'/f'seed{seed}'


def seed_root(seed):
    return RUN/f'seed{seed}'


def verify():
    p=RUN/'protocol.json'
    assert sha(p)==read(RUN/'protocol_hash.json')['sha256']
    pp=read(p)
    for path,h in pp['files'].items():
        assert sha(path)==h,path
    return pp


def prepare():
    assert not (RUN/'protocol.json').exists()
    pp=pilot.verify()
    files=[Path(__file__),CODE/'preservation_replication_analysis.py',
           CODE/'run_preservation_replication.sh',CODE/'PRESERVATION_REPLICATION_PROTOCOL.md',
           Path(trainer.__file__),Path(pilot.__file__),Path(confirmation.__file__),
           Path(rt.__file__),Path(rt.legacy.__file__),Path(rt.digital.__file__),
           Path(rt.public.__file__),Path(rt.sugar.__file__),Path(trainer.base.__file__),
           pilot.RUN/'protocol.json',pilot.PREVIOUS_RUN/'protocol.json',
           ROOT/'mirror/tools/aggregate.py',
           OUT/'external_retest/protocol.json',OUT/'external_retest/cache.json',
           OUT/'sugarcrepe/cache.json']
    reused={}
    for arm,parent in [('IS8_preserve',pilot.RUN),('R_preserve',pilot.PREVIOUS_RUN)]:
        r=parent/'runs'/arm; done=read(r/'complete.json')
        assert done['updates']==2752 and sha(r/'last.pt')==done['sha256']
        files += [r/'last.pt',r/'complete.json']
        reused[arm]=dict(checkpoint=str(r/'last.pt'),sha256=done['sha256'])
    for seed in [43,44]:
        confirmation.SEED,confirmation.RUN=seed,original(seed)
        confirmation.verify_prompt()
        old=read(original(seed)/'runs/R_s100/complete.json')
        other=read(original(seed)/'runs/IS_s100_w1/complete.json')
        for k in ['initial_hash','sequence_hash','updates']:
            assert old[k]==other[k]
        cfg=dict(seed=seed,updates=2752,batch_size=32,configurations=ARMS,
                 nuisance_coefficient=pp['nuisance_coefficient'],
                 geometry_coefficient=pp['geometry_coefficient'],
                 initialization_hash=old['initial_hash'],sequence_hash=old['sequence_hash'],
                 selection='fixed final update; no checkpoint or seed selection')
        dump(seed_root(seed)/'protocol.json',cfg)
        files += [seed_root(seed)/'protocol.json',original(seed)/'selection.json',
                  original(seed)/'text_caches.json',original(seed)/'protocol.json',
                  original(seed)/'runs/R_s100/complete.json',
                  original(seed)/'runs/IS_s100_w1/complete.json']
    dump(RUN/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        new_seeds=[43,44],all_seeds=[42,43,44],arms=ARMS,reused_seed42=reused,
        updates=2752,batch_size=32,trainable_parameters=512,
        shared_preservation='Clean/blank KL weight 6 plus training-caption Gram retention; no extra data',
        nuisance_coefficient=pp['nuisance_coefficient'],geometry_coefficient=pp['geometry_coefficient'],
        comparison='IS8+P, matched ranking+P, existing IS4, existing ranking, frozen',
        endpoints='All original direct/transfer, all twelve contrasts and normalized contrasts, official RTA100, SCAM, SynthSCAM, NoSCAM, clean, full SugarCrepe and all categories',
        statistics='Means and sample SD; 5000 paired source and crossed seed/source bootstrap draws',
        selection='All seeds, final checkpoints. Recipe choice awaits discussion after complete evaluation.',
        known_results='Seed42 and existing IS4 public outcomes known; developmental seed replication',
        backbone_ports='Pending a single recipe choice; no port training in this job',
        storage='Existing caches read-only; new output on root disk; final checkpoints only',
        files={str(p):sha(p) for p in files}))
    dump(RUN/'protocol_hash.json',dict(sha256=sha(RUN/'protocol.json')))
    print('PREPARED',RUN,flush=True)


def attach(seed):
    verify()
    confirmation.SEED,confirmation.RUN=seed,original(seed)
    confirmation.verify_prompt()
    trainer.RUN=seed_root(seed)
    trainer.CONFIGS=ARMS
    trainer.verify=lambda: read(seed_root(seed)/'protocol.json')
    trainer.original.RUN=original(seed)
    trainer.original.setup=confirmation.setup
    trainer.bank.stream=confirmation.stream


def smoke(seed):
    attach(seed)
    cfg,model,tok,prefix,v,y,w,text,tt,rows=confirmation.setup()
    pp=read(seed_root(seed)/'protocol.json')
    ids=confirmation.stream(y,updates=2752)
    assert hashlib.sha256(prefix.detach().cpu().numpy().tobytes()).hexdigest()==pp['initialization_hash']
    assert hashlib.sha256(ids.tobytes()).hexdigest()==pp['sequence_hash']
    features=trainer.base.prototype(model,tt,prefix,len(text))
    ix=ids[:32]
    scores=torch.einsum('bsd,cd->bsc',v[ix],features)
    ref=torch.einsum('bsd,cd->bsc',v[ix],text)
    args=(scores,ref,features,text,y[ix],w[ix])
    kwargs=(pp['nuisance_coefficient'],float(model.logit_scale.exp()),pp['geometry_coefficient'])
    r,terms=trainer.objective(*args,ARMS['R_preserve'],*kwargs)
    z,_=trainer.objective(*args,dict(ARMS['IS8_preserve'],weight=0.),*kwargs)
    gr=torch.autograd.grad(r,prefix,retain_graph=True)[0]
    gz=torch.autograd.grad(z,prefix,retain_graph=True)[0]
    assert torch.equal(r,z) and torch.equal(gr,gz)
    full,_=trainer.objective(*args,ARMS['IS8_preserve'],*kwargs)
    gf=torch.autograd.grad(full,prefix,retain_graph=True)[0]
    gn=torch.autograd.grad(terms['nuisance'],prefix)[0]
    error=float((gf-gr-8*pp['nuisance_coefficient']*gn).abs().max())
    assert error<3e-4 and torch.isfinite(gf).all()
    assert all(not p.requires_grad for p in model.parameters())
    dump(seed_root(seed)/'implementation_checks.json',dict(zero_interaction_exact_loss_gradient_parity=True,
        isolated_interaction_gradient_error=error,initialization_and_stream_match=True,
        training_sources=len(rows),classes=len(text),parameters=prefix.numel(),frozen_encoders=True))
    del model,prefix,features,scores,ref,v,text
    torch.cuda.empty_cache()
    for arm in ARMS:
        trainer.train(arm,smoke=True)
    print('SMOKE PASSED',seed,flush=True)


def train(seed,arm):
    attach(seed)
    out=seed_root(seed)/'runs'/arm
    if not (out/'complete.json').exists():
        trainer.train(arm,smoke=False)
    done=read(out/'complete.json');pp=read(seed_root(seed)/'protocol.json')
    assert sha(out/'last.pt')==done['sha256'] and done['updates']==2752
    assert done['initial_hash']==pp['initialization_hash'] and done['sequence_hash']==pp['sequence_hash']
    dump(out/'matching_check.json',dict(initialization_equal=True,source_stream_equal=True,
        updates_equal=True,shared_losses_equal=True,only_interaction_weight_differs=True,
        final_checkpoint_sha256=done['sha256']))
    print('TRAINING COMPLETE',seed,arm,flush=True)


def evaluate(seed,arm):
    verify();torch.set_num_threads(2)
    out=seed_root(seed)/'evaluation'/arm/'retest'
    if (out/'complete.json').exists():
        for p,h in read(out/'complete.json')['files'].items():
            assert sha(out/p)==h,p
        return
    if out.exists():
        # Preserve interrupted scoring output; final checkpoints are never changed.
        out.rename(out.with_name('interrupted_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')))
    out.mkdir(parents=True)
    old=read(original(seed)/'selection.json')
    ck=seed_root(seed)/'runs'/arm/'last.pt'
    assert sha(ck)==read(ck.parent/'complete.json')['sha256']
    reg=dict(ranking=old['ranking'],previous_IS=old['IS'],same_scale_ablation=old['ranking'],
             initial_prefix=old['initial_prefix'],IS=dict(checkpoint=str(ck),sha256=sha(ck)))
    dump(out/'protocol.json',dict(seed=seed,candidate=arm,registry=reg,
        parent_protocol_sha256=sha(RUN/'protocol.json'),
        internal_IS_key='Candidate slot only; scientific identity is candidate field'))
    rt.RUN,rt.DEST,rt.legacy.RUN,rt.pilot.ORIGINAL=out.parent,out,out,original(seed)
    mm=rt.text_banks(reg)
    rt.legacy.digital_score(mm);rt.fresh_score(mm);rt.complete_targets(mm)
    rt.legacy.sugar_score(mm);rt.legacy.external_score(mm);rt.public_contrasts()
    dump(out/'complete.json',dict(seed=seed,candidate=arm,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print('EVALUATION COMPLETE',seed,arm,flush=True)


def wait_gpu():
    while True:
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not active:
            return
        print('WAITING FOR OTHER GPU WORK',active,flush=True)
        time.sleep(20)


def status(state):
    dump(RUN/'status.json',dict(state=state,time_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    command();torch.set_num_threads(2)
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','verify','smoke','train','evaluate','wait_gpu','status'])
    ap.add_argument('--seed',type=int,choices=[43,44],default=43)
    ap.add_argument('--arm',choices=list(ARMS),default='IS8_preserve')
    ap.add_argument('--state',default='running')
    args=ap.parse_args()
    if args.action in ['prepare','verify','wait_gpu']:globals()[args.action]()
    elif args.action=='status':status(args.state)
    elif args.action=='smoke':smoke(args.seed)
    elif args.action=='train':train(args.seed,args.arm)
    else:evaluate(args.seed,args.arm)
