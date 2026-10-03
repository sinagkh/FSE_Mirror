"""Replicate the chosen typography objective, without selecting on port outcomes."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from mirror.cases.typography import backbone_setup as base
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import log; from mirror.core.io import verify_files

OLD = base.OUT
OUT = ROOT / 'mirror/cases/typography_backbone_replication_20260929'
SEEDS = (42, 43, 44)


def freeze(model):
    old = base.verify(model)
    dest = OUT / model
    if (dest / 'protocol.json').exists():
        verify_files(read(dest / 'protocol.json')['inputs'])
        return
    cal = read(OLD / model / 'calibration.json')
    regs = read(OLD / model / 'models.json')
    ranking = [r for r in regs if r['name'] == 'ranking']
    assert sorted(r['seed'] for r in ranking) == list(SEEDS)
    paths = [Path(__file__), Path(base.__file__), Path(base.objective.__file__),
             OLD/model/'protocol.json', OLD/model/'calibration.json',
             OLD/model/'models.json', OLD/model/'train_features.json']
    paths += [Path(r['checkpoint']) for r in ranking]
    inputs = {str(p): sha(p) for p in paths}
    dump(dest/'protocol.json', dict(
        model=model, seeds=SEEDS, selected_objective='plain shared-prefix IS',
        gradient_calibrated_strength=4., coefficient=4*cal['kappa'],
        ranking_reused=ranking, original_protocol=old,
        training_changes=['interaction coefficient only'],
        optimizer='SGD', lr=.002, eta_min=.00005, updates=2752,
        batch_size=32, checkpoint_selection='fixed last',
        shared_losses=dict(KL=3., margin_floor=1.2, margin_retention=1.),
        prefix_initialization='same seeded .02 Gaussian',
        evaluation='same source banks, public-test protocols and preprocessing; no port-specific tuning',
        inputs=inputs))
    dump(dest/'calibration.json', cal)
    log(dest, 'frozen')


def run(model):
    freeze(model)
    dest = OUT/model
    log(dest, 'start')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    subject = base.load_subject(model, device='cuda')
    # Read the unchanged original image cache, not a fresh rendering.
    base.OUT = OLD
    loaded = base.load_training(subject)
    subject.model.visual.cpu()
    torch.cuda.empty_cache()
    cal = read(dest/'calibration.json')
    ranking = read(dest/'protocol.json')['ranking_reused']
    verify_files({r['checkpoint']: r['sha256'] for r in ranking})
    for r in ranking:
        assert r['updates'] == 2752 and r['selection'] == 'fixed_last'
        assert base.ah(base.prefix(subject.model, r['seed']).detach().cpu().numpy()) == r['initial_hash']
        assert base.ah(base.stream(loaded[1].cpu().numpy(), r['seed'])) == r['sequence_hash']
    if not (dest/'smoke.json').exists():
        v,y,w,frozen,tt,_ = loaded
        p = base.prefix(subject.model,42)
        opt = torch.optim.SGD([p], lr=.002)
        sequence = base.stream(y.cpu().numpy(),42)
        records = []
        for step in range(3):
            ix=sequence[step*32:(step+1)*32]
            t=base.objective.prototype(subject.model,tt,p,len(frozen))
            s=torch.einsum('bsd,cd->bsc',v[ix],t)
            ref=torch.einsum('bsd,cd->bsc',v[ix],frozen)
            r,terms=base.objective.loss(s,ref,dict(scale=100,weight=0),y[ix],w[ix],cal['kappa'],cal['frozen_logit_scale'])
            loss,_=base.objective.loss(s,ref,dict(scale=100,weight=4),y[ix],w[ix],cal['kappa'],cal['frozen_logit_scale'])
            assert torch.allclose(loss-r,4*cal['kappa']*terms['nuisance'],atol=1e-5,rtol=1e-5)
            g=torch.autograd.grad(r-terms['shared'],p,retain_graph=True)[0]
            assert not torch.count_nonzero(g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
            opt.step()
            records.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(p.grad.norm())))
        dump(dest/'smoke.json',dict(records=records,ranking_init_stream_budget_verified=True,
             zero_weight_gradient_parity=True,only_extra_term_is_interaction=True,
             frozen_backbone=all(not p.requires_grad for p in subject.model.parameters())))
        print('SMOKE_OK',model,flush=True)
    base.OUT=OUT
    base.ARMS={'ranking':dict(scale=100,weight=0.),'IS':dict(scale=100,weight=4.)}
    regs=[]
    for seed in SEEDS:
        other=next(r for r in ranking if r['seed']==seed)
        log(dest,'training_start',seed=seed)
        result=base.train(subject,loaded,cal,seed,'IS')
        assert result['initial_hash']==other['initial_hash']
        assert result['sequence_hash']==other['sequence_hash']
        regs += [other,result]
        log(dest,'training_complete',seed=seed,seconds=result['seconds'])
    dump(dest/'models.json',regs)
    dump(dest/'training_complete.json',dict(runs=3,reused_ranking=3,
         matched_data_initialization_sequence_budget=True,models_sha256=sha(dest/'models.json')))
    log(dest,'complete')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('freeze','run'))
    ap.add_argument('--model',choices=base.MODELS,required=True)
    a=ap.parse_args()
    globals()[a.action](a.model)
