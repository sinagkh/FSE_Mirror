"""Shared-caption dropout, matched routing objectives, and cache-based training."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import _state_hash

OUT = ROOT / 'clip/interbind_routing_common_noise_20260923'
PLAN = ROOT / 'FSE_VLM/plan/23_common_noise_routing_retry.md'
ARM = 'CN'
CROSS_PRIORITY = 4.


class PairedCaptionDropout(nn.Module):
    """Common random numbers within both layouts and all compared captions.

    Different source/color blocks get independent feature masks. During eval
    this is identity, so saved A/B weights use the original adapter evaluator.
    """
    def __init__(self, p=.05):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if x.ndim != 3 or len(x) % 2:
            raise ValueError('Expected adjacent direct/swapped caption batches')
        keep = x.new_empty((len(x)//2, 1, x.shape[-1])).bernoulli_(1-self.p)/(1-self.p)
        return x * keep.repeat_interleave(2, 0)


def make_model(cfg, device, shared=True):
    model = TextLowRankAdapter(768, cfg.rank, cfg.alpha, cfg.dropout).to(device)
    if shared:
        model.drop = PairedCaptionDropout(cfg.dropout)
    return model


def objective(parts, weights):
    # Preserve every original guard/target; only raise the cross-loss priority.
    return prior.objective(parts, weights, 'IP') + (CROSS_PRIORITY-1)*weights['cross']*parts['cross']


def available_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    free, total = torch.cuda.mem_get_info()
    if free < 6*1024**3:
        raise RuntimeError('GPU busy: fewer than 6 GiB free; no jobs interrupted')
    print('CUDA_FREE_GIB', free/1024**3, 'TOTAL_GIB', total/1024**3, flush=True)


def diagnose():
    """Training data only, original final checkpoint, zero optimizer updates."""
    prior.verify(); prior.configure(); available_gpu()
    cache, spec, contexts, cfg = prior.load_training('cuda')
    model = make_model(cfg, 'cuda', shared=False)
    old = read(prior.OUT/'models.json')
    entry = next(m for m in old if m['arm']=='IP')
    assert sha(entry['checkpoint']) == entry['sha256']
    model.load_state_dict(torch.load(entry['checkpoint'], map_location='cuda', weights_only=False)['state_dict'])
    blocks = np.random.default_rng(20260923).permutation(1280)[:24]
    ids = prior.paired_rows(torch.tensor(blocks, device='cuda'))
    names = ('binding', 'cross_abs', 'response', 'absolute_preference')

    def values():
        scores = cache.images[ids] @ model(cache.texts[ids,:4]).transpose(1,2)
        d,c,e,b,_ = prior.measurements(scores,spec,contexts)
        return [d,c,e,b]

    with torch.inference_mode():
        model.eval(); reference = values()
        report = {}
        for mode in ('independent_caption_dropout', 'paired_common_dropout'):
            model.drop = nn.Dropout(cfg.dropout) if mode.startswith('independent') else PairedCaptionDropout(cfg.dropout)
            model.train(); errors=[]; losses=[]
            for repeat in range(32):
                torch.manual_seed(230000+repeat); torch.cuda.manual_seed_all(230000+repeat)
                current = values()
                errors.append([float((a-b).square().mean().sqrt()) for a,b in zip(current,reference)])
                torch.manual_seed(230000+repeat); torch.cuda.manual_seed_all(230000+repeat)
                parts=prior.components(model,cache,spec,contexts,cfg,ids)
                losses.append({k:float(v) for k,v in parts.items()})
            report[mode] = dict(mean_rmse_from_deterministic=dict(zip(names,np.mean(errors,axis=0).tolist())),
                mean_losses={k:float(np.mean([r[k] for r in losses])) for k in losses[0]})
    original=read(prior.OUT/'normalization.json')
    dump(OUT/'training_only_diagnosis.json',dict(original_checkpoint=entry,
        original_gradient_cosine_order=original['gradient_cosine_order'],
        original_gradient_cosines=original['mean_gradient_cosines'],blocks=blocks.tolist(),
        diagnostic_seed=20260923,dropout_repetitions=32,training_data_only=True,
        optimizer_updates=0,script_sha256=sha(Path(__file__)),dropout_diagnosis=report))
    print('TRAINING_ONLY_DIAGNOSIS', report, flush=True)


def freeze():
    prior.verify()
    paths=[Path(__file__),PLAN,OUT/'training_only_diagnosis.json',
        prior.OUT/'protocol.json',prior.OUT/'normalization.json',
        prior.OUT/'training_config.json',prior.OUT/'batch_schedule.json',prior.OUT/'models.json',
        prior.OUT/'runs/IP/initial.pt',
        ROOT/'clip/interbind_routing_same_class_20260923/evaluation_protocol.json',
        ROOT/'clip/interbind_routing_same_class_20260923/evaluation/complete.json']
    paths += [Path(__file__).with_name(n+'.py') for n in
        ('routing_relative_pilot','routing_same_class_evaluate','routing_common_noise_evaluate','repair')]
    paths += [Path(__file__).parent/'tests/test_routing_common_noise.py']
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in paths},
        arm=ARM,seed=42,epochs=12,updates=648,selection='fixed_last',
        initialization='Original frozen-identity seed42 A/B initialization, not a continuation',
        cross_priority=CROSS_PRIORITY,other_weights='Original training-only gradient normalizers and protection priority 4 unchanged',
        dropout='p=.05; one 64-feature mask per source/color block, shared across captions and both layouts',
        source_pairs=640,independent_evaluation_anchors=888,
        scope='Exploratory development after observing the 888-anchor evaluation; not untouched confirmation',
        training_data_unchanged=True,evaluation_ids_unchanged=True,thresholds_unchanged=True,
        other_seeds=False,hyperparameter_sweep=False,natural_benchmark_selection=False))
    print('FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    protocol=read(OUT/'protocol.json');verify_files(protocol['inputs']);prior.verify()
    return protocol


def train():
    p=verify();prior.configure();available_gpu()
    cache,spec,contexts,cfg=prior.load_training('cuda')
    weights=read(prior.OUT/'normalization.json')['weights']
    schedule=read(prior.OUT/'batch_schedule.json')
    assert len(schedule)==12 and all(sorted(r)==list(range(1280)) for r in schedule)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=make_model(cfg,'cuda');initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    original=torch.load(prior.OUT/'runs/IP/initial.pt',map_location='cpu',weights_only=False)
    assert _state_hash(initial)==original['state_hash']
    # Preflight: every term is finite and the new loss has a finite nonzero gradient.
    model.train();torch.manual_seed(420001);torch.cuda.manual_seed_all(420001)
    ids=prior.paired_rows(torch.arange(24,device='cuda'))
    parts=prior.components(model,cache,spec,contexts,cfg,ids);loss=objective(parts,weights)
    norm=prior.norm_grads(loss,model)
    assert np.isfinite(float(loss)) and np.isfinite(norm) and norm>0
    dump(OUT/'preflight.json',dict(identity_hash=_state_hash(initial),matches_original_initialization=True,
        loss=float(loss.detach()),gradient_norm=norm,components={k:float(v.detach()) for k,v in parts.items()},
        optimizer_updates=0,protocol_sha256=sha(OUT/'protocol.json')))
    dest=OUT/'runs'/ARM;dest.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state_dict=initial,state_hash=_state_hash(initial)),dest/'initial.pt')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    history=[];diagnostics=[dict(epoch=0,**prior.train_diagnostics(model,cache,spec,contexts))]
    started=time.monotonic()
    for epoch,order in enumerate(schedule,1):
        model.train();torch.manual_seed(420000+epoch);torch.cuda.manual_seed_all(420000+epoch)
        for start in range(0,1280,24):
            ids=prior.paired_rows(torch.tensor(order[start:start+24],device='cuda'))
            parts=prior.components(model,cache,spec,contexts,cfg,ids);total=objective(parts,weights)
            opt.zero_grad(set_to_none=True);total.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(total) and torch.isfinite(norm)
            opt.step();history.append(dict(epoch=epoch,rows=ids.cpu().tolist(),loss=float(total.detach()),
                gradient_norm=float(norm),components={k:float(v.detach()) for k,v in parts.items()}))
        diagnostics.append(dict(epoch=epoch,**prior.train_diagnostics(model,cache,spec,contexts)))
        print('TRAIN',ARM,epoch,'seconds',round(time.monotonic()-started,1),diagnostics[-1],flush=True)
    assert len(history)==p['updates']
    old_history=prior.lines(prior.OUT/'runs/IP/history.jsonl')
    assert [r['rows'] for r in history]==[r['rows'] for r in old_history]
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(dict(state_dict=state,arm=ARM,seed=42,selection='fixed_last',epochs=12,updates=len(history),
        configuration=asdict(cfg),objective_protocol_sha256=sha(OUT/'protocol.json'),weights=weights,
        cross_priority=CROSS_PRIORITY,dropout_scheme='paired_caption_common_random_numbers',
        initial_state_hash=_state_hash(initial),cache_sha256=cfg.expected_cache_sha256,optimizer=opt.state_dict()),dest/'last.pt')
    jsonl(dest/'history.jsonl',history);jsonl(dest/'training_diagnostics.jsonl',diagnostics)
    entry=dict(arm=ARM,seed=42,checkpoint=str(dest/'last.pt'),sha256=sha(dest/'last.pt'),
        initial_state_hash=_state_hash(initial),seconds=time.monotonic()-started,updates=len(history),selection='fixed_last')
    dump(dest/'complete.json',entry)
    models=read(prior.OUT/'models.json')+[entry];dump(OUT/'models.json',models)
    dump(OUT/'training_complete.json',dict(models_sha256=sha(OUT/'models.json'),protocol_sha256=sha(OUT/'protocol.json'),
        matched_initialization=True,matched_batches=True,matched_epochs=True,
        dropout_distribution_changed=True,checkpoint_sha256=entry['sha256'],
        old_results_unchanged=True,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    print('TRAINING_COMPLETE',entry,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['diagnose','freeze','train'])
    args=parser.parse_args();log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
