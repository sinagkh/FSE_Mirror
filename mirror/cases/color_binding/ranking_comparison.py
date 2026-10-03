"""Finite review control; original experiment files are read-only."""
import argparse
from dataclasses import asdict; from dataclasses import replace
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.cases.color_binding import routing_common_noise_pilot as cn
from mirror.cases.color_binding import routing_budget_replication as old
from mirror.core.repair import _state_hash
from mirror.core.features import verify_cache
from mirror.core.metrics import bank_arrays; from mirror.core.metrics import adapt; from mirror.core.metrics import routing
from mirror.cases.color_binding.behavioral_pilot import lines

OUT = ROOT/'clip/interbind_ranking_comparison_20260925'
PLAN = ROOT/'FSE_VLM/plan/36_primary_ranking_comparison.md'
DEV = ROOT/'clip/interbind_routing_same_class_20260923'
BASE = ROOT/'clip/interbind_phase_bc_completion_20260923'
MODEL = 'openclip_laion_l14'
SEEDS = (42, 43, 44)
CANDIDATES = [dict(name=f'R_s{s}', family='R', scale=s, retention=0.) for s in (1,10,100)] + [
    dict(name=f'RG_s{s}_g{g:g}', family='RG', scale=s, retention=g)
    for s in (1,10,100) for g in (.25,1.)]
IS = dict(name='IS', family='IS', scale=1, retention=1.)


def configure():
    prior.configure()
    cn.available_gpu()


def loss_for(model, cache, spec, contexts, cfg, ids, weights, candidate):
    captured = []
    hook = model.register_forward_hook(lambda _m, _args, result: captured.append(result))
    try:
        parts = prior.components(model, cache, spec, contexts, cfg, ids)
    finally:
        hook.remove()
    assert len(captured) == 1
    scores = (cache.images[ids] @ captured[0][:,:6].transpose(1,2))[:,:,:4]
    labels = torch.arange(4, device=scores.device).repeat(len(ids))
    ce = F.cross_entropy(candidate['scale']*scores.reshape(-1,4), labels)
    guards = prior.objective(parts, weights, 'G')
    if candidate['family'] == 'IS':
        loss = cn.objective(parts, weights)
    elif candidate['family'] == 'R':
        loss = ce + cfg.legacy_ranking.anchor_weight*parts['ranking_anchor']
    else:
        assert candidate['family'] == 'RG'
        loss = ce + candidate['retention']*guards
    if candidate['scale'] == 1:
        torch.testing.assert_close(ce, parts['ranking_endpoint'], rtol=0, atol=0)
        if candidate['family'] == 'R':
            torch.testing.assert_close(loss, parts['ranking'], rtol=0, atol=0)
    return loss, parts, ce, guards


def freeze():
    cache, spec, contexts, cfg = prior.load_training('cpu')
    files = [PLAN, Path(__file__), Path(__file__).with_name('ranking_comparison_evaluate.py'),
             Path(__file__).parent/'tests/test_ranking_comparison.py',
             prior.OUT/'normalization.json', prior.OUT/'training_config.json',
             DEV/'rows.jsonl', DEV/'features/complete.json', BASE/'same_rule_confirmation/rows.jsonl']
    files += [Path(__file__).with_name(n+'.py') for n in ('repair','routing_relative_pilot',
        'routing_common_noise_pilot','routing_budget_replication','completion_metrics','strengthening_statistics')]
    for seed in SEEDS:
        for arm in ('IS','R'):
            files += [old.OUT/f'seed{seed}/{arm}'/n for n in ('initial.pt','last.pt','complete.json','history.jsonl')]
    for path in (DEV/'features', BASE/'same_rule_confirmation/features'/MODEL):
        verify_cache(path)
        files += [path/n for n in ('images.npy','texts.npy','index.jsonl')]
    for benchmark in ('aro','coco'):
        files += [BASE/'preservation/features'/MODEL/benchmark/'features.npz',
                  BASE/'preservation/manifests'/benchmark/'rows.jsonl']
    files += [prior.SUGAR/n for n in ('features.npz','indices.json','frozen_seed0.csv')]
    dump(OUT/'protocol.json', dict(inputs={str(p):sha(p) for p in files},
        candidates=CANDIDATES, reference=IS, seeds=SEEDS, device='cuda', threads=4,
        config=asdict(replace(cfg, epochs=36)), cache_shape=list(cache.images.shape),
        expected_cache_sha256=cfg.expected_cache_sha256, updates=1944,
        selection='fixed_last; seed42 development selects one recipe per ranking family',
        primary='IS minus selected RG, red-blue direct-canvas exchange accuracy',
        development_only_grid=True, confirmation_previously_observed=True,
        new_checkpoint_test_scores_seen=False, bootstrap_draws=10000, bootstrap_seed=20260925))
    print('FROZEN', sha(OUT/'protocol.json'), flush=True)


def train(candidate, seed, loaded):
    cache, spec, contexts, cfg = loaded
    cfg = replace(cfg, seed=seed, epochs=36)
    weights = read(prior.OUT/'normalization.json')['weights']
    dest = OUT/'runs'/f'seed{seed}'/candidate['name']
    if (dest/'complete.json').exists():
        done = read(dest/'complete.json'); verify_files(done['files']); return done
    dest.mkdir(parents=True, exist_ok=False)
    initial = torch.load(old.OUT/f'seed{seed}/IS/initial.pt', map_location='cpu', weights_only=False)
    other = torch.load(old.OUT/f'seed{seed}/R/initial.pt', map_location='cpu', weights_only=False)
    assert initial['state_hash'] == other['state_hash']
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); model = cn.make_model(cfg, 'cuda')
    model.load_state_dict(initial['state_dict'])
    assert _state_hash(model.state_dict()) == initial['state_hash']
    torch.save(initial, dest/'initial.pt')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = old.schedule_for(seed); history = []; started = time.monotonic()
    for epoch, order in enumerate(schedule, 1):
        model.train(); torch.manual_seed(seed*10000+epoch); torch.cuda.manual_seed_all(seed*10000+epoch)
        for start in range(0,1280,24):
            ids = prior.paired_rows(torch.tensor(order[start:start+24],device='cuda'))
            loss, parts, ce, guard = loss_for(model,cache,spec,contexts,cfg,ids,weights,candidate)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(grad)
            optimizer.step()
            history.append(dict(epoch=epoch,rows=ids.tolist(),loss=float(loss.detach()),
                ce=float(ce.detach()),retention=float(guard.detach()),gradient_norm=float(grad),
                components={k:float(v.detach()) for k,v in parts.items()}))
        if epoch%6==0:
            print('TRAIN',seed,candidate['name'],epoch,'seconds',round(time.monotonic()-started,1),flush=True)
    assert len(history)==1944
    rng = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    replay = None
    if candidate['name'] in ('IS','R_s1'):
        arm = 'IS' if candidate['name']=='IS' else 'R'
        previous = torch.load(old.OUT/f'seed{seed}/{arm}/last.pt',map_location='cuda',weights_only=False)['state_dict']
        replay = max(float((previous[k]-model.state_dict()[k]).abs().max()) for k in previous)
        assert replay == 0, ('Reference replay failed',replay)
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        configuration=asdict(cfg),candidate=candidate,seed=seed,optimizer=optimizer.state_dict(),
        initial_state_hash=initial['state_hash'],protocol_sha256=sha(OUT/'protocol.json'),
        epochs=36,updates=len(history),selection='fixed_last',device='cuda'),dest/'last.pt')
    jsonl(dest/'history.jsonl',history)
    done = dict(**candidate,seed=seed,checkpoint=str(dest/'last.pt'),sha256=sha(dest/'last.pt'),
        initial_state_hash=initial['state_hash'],final_rng_sha256=rng,original_replay_max_parameter_error=replay,
        batch_schedule_sha256=hashlib.sha256(np.asarray(schedule,dtype=np.int64).tobytes()).hexdigest(),
        first_batch_components=history[0]['components'],updates=1944,seconds=time.monotonic()-started,
        files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    dump(dest/'complete.json',done)
    return done


def select_candidate(records, family):
    options = [r for r in records if r['family']==family]
    eligible = [r for r in options if r['eligible']]
    # Stable tie break, defined without test scores.
    choice = sorted(eligible or options,key=lambda r:(-r['mean_exchange'],r['scale'],-r['retention']))[0]
    return dict(name=choice['name'], any_eligible=bool(eligible),
        candidate=next(c for c in CANDIDATES if c['name']==choice['name']),
        selection_score=choice['mean_exchange'])


def development(registry):
    dest=OUT/'development';dest.mkdir(exist_ok=False)
    texts=np.load(DEV/'features/texts.npy');images=np.load(DEV/'features/images.npy',mmap_mode='r')
    all_index=lines(DEV/'features/index.jsonl');meta={r['anchor_id']:r for r in all_index}
    per=[];summary=[]
    regs=[dict(name='F',family='F',seed=0,checkpoint=None),*registry]
    for color in ('red-blue','green-yellow'):
        for view in ('canvas','swapped_canvas'):
            v,t,idx=bank_arrays(DEV/'features','routing',color,view)
            assert len(idx)==888
            for reg in regs:
                x=np.einsum('nid,njd->nij',v,adapt(t,reg['checkpoint']))
                m=routing(x);at=adapt(texts,reg['checkpoint']);om=[]
                for r in idx:
                    obj=v[len(om)]@at[r['text_indices'][8:10]].T
                    om.append((obj[:,0]>obj[:,1]).mean())
                m['object_guard']=np.array(om)
                summary.append(dict(name=reg['name'],color=color,view=view,**{k:float(a.mean()) for k,a in m.items() if not k.startswith('contrast/')}))
                for j,r in enumerate(idx):
                    per.append(dict(name=reg['name'],seed=reg['seed'],color=color,view=view,anchor_id=r['anchor_id'],
                        scores=x[j].tolist(),**{k:float(a[j]) for k,a in m.items()}))
    jsonl(dest/'per_example.jsonl',per);jsonl(dest/'summary.jsonl',summary)
    choices=[]
    frozen={r['view']:r for r in summary if r['name']=='F' and r['color']=='red-blue'}
    for c in CANDIDATES:
        rows=[r for r in summary if r['name']==c['name'] and r['color']=='red-blue']
        assert len(rows)==2
        eligible=all(min(r['word1_accuracy'],r['word2_accuracy'])>=.99 and
            r['object_guard']>=frozen[r['view']]['object_guard']-.01 for r in rows)
        choices.append(dict(**c,eligible=eligible,mean_exchange=float(np.mean([r['exchange_accuracy'] for r in rows]))))
    result=dict(candidates=choices,selected={f:select_candidate(choices,f) for f in ('R','RG')},
        protocol_sha256=sha(OUT/'protocol.json'),development_sha256=sha(dest/'per_example.jsonl'),
        new_checkpoint_confirmation_scored=False,new_checkpoint_natural_scored=False)
    dump(dest/'selection.json',result);print('SELECTED',result['selected'],flush=True)


def run():
    verify_files(read(OUT/'protocol.json')['inputs'])
    loaded=prior.load_training('cuda');configure()
    # Fail on any reference mismatch before training the new recipes.
    seed42=[train(c,42,loaded) for c in [CANDIDATES[0],IS,*CANDIDATES[1:]]]
    if not (OUT/'development/selection.json').exists():development(seed42)
    selected=read(OUT/'development/selection.json')['selected']
    candidates={c['name']:c for c in [CANDIDATES[0],IS,*[s['candidate'] for s in selected.values()]]}
    records=[]
    for seed in SEEDS:
        for c in candidates.values():
            if seed==42 or c['name'] not in ('IS','R_s1'):
                records.append(train(c,seed,loaded));continue
            arm='IS' if c['name']=='IS' else 'R';path=old.OUT/f'seed{seed}/{arm}'
            done=read(path/'complete.json');assert sha(done['checkpoint'])==done['sha256']
            history0=next(iter(lines(path/'history.jsonl')))
            records.append(dict(**c,seed=seed,checkpoint=done['checkpoint'],sha256=done['sha256'],
                initial_state_hash=done['initial_state_hash'],final_rng_sha256=done['final_rng_sha256'],
                batch_schedule_sha256=hashlib.sha256(np.asarray(old.schedule_for(seed),dtype=np.int64).tobytes()).hexdigest(),
                first_batch_components=history0['components'],updates=done['updates'],reused_original=True))
    for seed in SEEDS:
        rows=[r for r in records if r['seed']==seed]
        for field in ('initial_state_hash','final_rng_sha256','batch_schedule_sha256'):
            assert len({r[field] for r in rows})==1
        assert all(r['first_batch_components']==rows[0]['first_batch_components'] for r in rows)
    dump(OUT/'checkpoints_frozen.json',dict(models=records,selection_sha256=sha(OUT/'development/selection.json'),
        matched_initialization=True,matched_batches=True,matched_dropout=True,matched_updates=True,
        first_batch_components_equal=True,new_checkpoint_test_scores_seen=False))
    print('TRAINING_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();configure()
    log(OUT,'start',stage=a.action)
    try:globals()[a.action]()
    except BaseException as e:log(OUT,'failed',stage=a.action,error=repr(e));raise
    log(OUT,'complete',stage=a.action)


if __name__=='__main__':main()
