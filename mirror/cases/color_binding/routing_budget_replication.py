"""Fixed 36-epoch original ranking and common-noise IS, conditional 3 seeds."""
import argparse
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.cases.color_binding import routing_common_noise_pilot as cn
from mirror.core.repair import _state_hash

OUT=ROOT/'clip/interbind_routing_budget_replication_20260923'
PLAN=ROOT/'FSE_VLM/plan/24_routing_budget_and_replication.md'
EPOCHS=36
SEEDS=(42,43,44)


def schedule_for(seed,epochs=EPOCHS):
    rng=np.random.default_rng(seed)
    return [rng.permutation(1280).tolist() for _ in range(epochs)]


def objective(parts,weights,arm):
    if arm=='IS':return cn.objective(parts,weights)
    if arm=='R':return parts['ranking']
    raise ValueError(arm)


def freeze():
    cn.verify()
    paths=[PLAN,Path(__file__),Path(__file__).with_name('routing_budget_replication_evaluate.py'),
        Path(__file__).with_name('routing_budget_replication_stats.py'),
        Path(__file__).with_name('run_routing_budget_replication.sh'),
        Path(__file__).parent/'tests/test_routing_budget_replication.py',
        cn.OUT/'protocol.json',cn.OUT/'models.json',cn.OUT/'runs/CN/last.pt',
        cn.OUT/'evaluation/complete.json',cn.OUT/'evaluation/per_example.csv',
        prior.OUT/'protocol.json',prior.OUT/'training_config.json',prior.OUT/'normalization.json',
        prior.OUT/'batch_schedule.json',ROOT/'clip/interbind_routing_same_class_20260923/evaluation_protocol.json']
    paths += [Path(__file__).with_name(n+'.py') for n in
        ('routing_common_noise_pilot','routing_relative_pilot','routing_same_class_evaluate',
         'routing_same_class_data','routing_adapter_reaudit_v2','repair')]
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in paths},epochs=EPOCHS,
        seeds=SEEDS,arms=['F','R','IS'],updates=54*EPOCHS,selection='fixed_last',
        cross_priority=cn.CROSS_PRIORITY,ranking_objective='Original four-way CE + .2 embedding anchoring; no IS guards',
        shared_dropout=True,training_sources=640,development_anchors=888,
        seed42_gate=dict(cross_ratio_max=.88,binding_above_frozen=True,
            binding_fraction_better_min=.5,cross_fraction_better_min=.5,
            exchange_drop_max=.01,word_accuracy_min=.99,object_accuracy_drop_max=.01,
            views=['canvas','swapped_canvas'],color='red-blue'),
        seed_expansion='Only after seed42 gate, before any new natural-benchmark scoring',
        bootstrap=dict(replicates=2000,seed=20260923,anchor_strata='object pair',
            seed_item_draws='shared across matched arms; common item draw across sampled seeds'),
        development=True,endpoint_control=False,hyperparameter_search=False,unchanged_evaluation_ids=True))
    print('FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    p=read(OUT/'protocol.json');verify_files(p['inputs']);cn.verify();return p


def train(seed):
    p=verify();assert seed in SEEDS
    if seed!=42:
        gate=read(OUT/'seed42_gate.json');assert gate['passed']
        assert gate['protocol_sha256']==sha(OUT/'protocol.json')
    prior.configure();cn.available_gpu();cache,spec,contexts,cfg=prior.load_training('cuda')
    cfg=replace(cfg,seed=seed,epochs=EPOCHS)
    weights=read(prior.OUT/'normalization.json')['weights'];schedule=schedule_for(seed)
    if seed==42:assert schedule[:12]==read(prior.OUT/'batch_schedule.json')
    dest=OUT/f'seed{seed}';dest.mkdir(parents=True,exist_ok=False)
    jsonl(dest/'batch_schedule.jsonl',[dict(epoch=i+1,blocks=x) for i,x in enumerate(schedule)])
    dump(dest/'training_config.json',asdict(cfg))
    registrations=[dict(arm='F',seed=0,checkpoint=None)]
    starts=[];finishes=[];first_parts=[]
    for arm in ('IS','R'):
        target=dest/arm;target.mkdir()
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=cn.make_model(cfg,'cuda');state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ih=_state_hash(state);starts.append(ih)
        if seed==42:
            original=torch.load(prior.OUT/'runs/IP/initial.pt',map_location='cpu',weights_only=False)
            assert ih==original['state_hash']
        torch.save(dict(state_dict=state,state_hash=ih,seed=seed),target/'initial.pt')
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        history=[];diagnostics=[dict(epoch=0,**prior.train_diagnostics(model,cache,spec,contexts))]
        begun=time.monotonic()
        for epoch,order in enumerate(schedule,1):
            model.train();torch.manual_seed(seed*10000+epoch);torch.cuda.manual_seed_all(seed*10000+epoch)
            for start in range(0,1280,24):
                ids=prior.paired_rows(torch.tensor(order[start:start+24],device='cuda'))
                parts=prior.components(model,cache,spec,contexts,cfg,ids)
                loss=objective(parts,weights,arm);opt.zero_grad(set_to_none=True);loss.backward()
                gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                assert torch.isfinite(loss) and torch.isfinite(gn)
                opt.step();history.append(dict(epoch=epoch,rows=ids.cpu().tolist(),loss=float(loss.detach()),
                    gradient_norm=float(gn),components={k:float(v.detach()) for k,v in parts.items()}))
            diagnostics.append(dict(epoch=epoch,**prior.train_diagnostics(model,cache,spec,contexts)))
            if seed==42 and arm=='IS' and epoch==12:
                old=torch.load(cn.OUT/'runs/CN/last.pt',map_location='cuda',weights_only=False)['state_dict']
                error=max(float((model.state_dict()[k]-old[k]).abs().max()) for k in old)
                assert error==0,error
                dump(dest/'epoch12_replay.json',dict(max_parameter_error=error,reference_sha256=sha(cn.OUT/'runs/CN/last.pt'),
                    new_checkpoint_saved=False,no_evaluation_or_selection=True))
            print('TRAIN',seed,arm,epoch,'seconds',round(time.monotonic()-begun,1),diagnostics[-1],flush=True)
        assert len(history)==p['updates']
        final_rng=sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes());finishes.append(final_rng)
        first_parts.append(history[0]['components'])
        final={k:v.detach().cpu() for k,v in model.state_dict().items()}
        torch.save(dict(state_dict=final,arm=arm,seed=seed,epochs=EPOCHS,updates=len(history),selection='fixed_last',
            configuration=asdict(cfg),protocol_sha256=sha(OUT/'protocol.json'),weights=weights,
            cross_priority=cn.CROSS_PRIORITY if arm=='IS' else None,dropout_scheme='paired_caption_common_random_numbers',
            initial_state_hash=ih,optimizer=opt.state_dict()),target/'last.pt')
        jsonl(target/'history.jsonl',history);jsonl(target/'training_diagnostics.jsonl',diagnostics)
        entry=dict(arm=arm,seed=seed,checkpoint=str(target/'last.pt'),sha256=sha(target/'last.pt'),
            initial_state_hash=ih,final_rng_sha256=final_rng,seconds=time.monotonic()-begun,
            epochs=EPOCHS,updates=len(history),selection='fixed_last')
        dump(target/'complete.json',entry);registrations.append(entry);del model,opt
    assert len(set(starts))==len(set(finishes))==1
    assert first_parts[0]==first_parts[1]
    registrations.sort(key=lambda m:['F','R','IS'].index(m['arm']))
    dump(dest/'models.json',registrations)
    dump(dest/'training_complete.json',dict(models_sha256=sha(dest/'models.json'),
        protocol_sha256=sha(OUT/'protocol.json'),matched_initialization=True,matched_batches=True,
        matched_dropout_rng=True,matched_epochs=True,first_batch_components_equal=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    print('SEED_TRAINING_COMPLETE',seed,flush=True)


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','train'])
    parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
    log(OUT,'start',stage=args.action,seed=args.seed)
    try:
        if args.action=='freeze':freeze()
        else:train(args.seed)
    except BaseException as exc:log(OUT,'failed',stage=args.action,seed=args.seed,error=repr(exc));raise
    log(OUT,'complete',stage=args.action,seed=args.seed)


if __name__=='__main__':main()
