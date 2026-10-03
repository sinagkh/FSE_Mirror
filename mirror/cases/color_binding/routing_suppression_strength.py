"""Bounded CPU-only routing cross-priority pilot; original recipes untouched."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
from dataclasses import asdict; from dataclasses import replace
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from mirror.core.io import ROOT; from mirror.core.io import dump; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.cases.color_binding import routing_common_noise_pilot as cn
from mirror.cases.color_binding import routing_budget_replication as old
from mirror.cases.color_binding import ranking_followup_pilot as prev
from mirror.core import metrics
from mirror.core.repair import _state_hash

OUT = ROOT/'clip/interbind_routing_suppression_20260929'
PLAN = ROOT/'FSE_VLM/plan/76_routing_suppression_strength_20260929.md'
ARMS = ('IS4', 'IS8', 'IS16', 'Ranking')
INITIAL = old.OUT/'seed42/IS/initial.pt'


def hash_array(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def objective(parts, weights, arm, ce):
    if arm == 'Ranking':
        return ce + prior.objective(parts, weights, 'G')
    priority = int(arm[2:])
    return cn.objective(parts, weights) + (priority-4)*weights['cross']*parts['cross']


def configure():
    prior.configure()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()


def freeze():
    paths = [PLAN, Path(__file__), INITIAL, prior.OUT/'normalization.json',
             prior.OUT/'training_config.json', prev.OUT/'training_encoding.json',
             prev.DEV/'features/complete.json', prev.DEV/'rows.jsonl']
    paths += [Path(m.__file__) for m in (prior, cn, old, prev, metrics)]
    record = dict(inputs={str(p): sha(p) for p in paths}, arms=ARMS, seed=42,
                  epochs=36, updates=1944, device='cpu', checkpoint_selection='fixed_last',
                  tune_only='cross target priority 4/8/16', gradients_and_guards='unchanged',
                  historical_test_results_seen=True, candidate_test_results_seen=False,
                  development_anchors=888, automatically_run_other_seeds=False)
    dump(OUT/'protocol.json', record)
    print('FROZEN', sha(OUT/'protocol.json'), flush=True)


def verify():
    protocol = read(OUT/'protocol.json')
    verify_files(protocol['inputs'])
    return protocol


def new_model(cfg):
    torch.manual_seed(42)
    model = cn.make_model(cfg, 'cpu')
    model.load_state_dict(torch.load(INITIAL, map_location='cpu', weights_only=False)['state_dict'])
    return model


def loss_parts(model, cache, spec, contexts, cfg, ids):
    captures = []
    hook = model.register_forward_hook(lambda _m, _a, z: captures.append(z))
    try:
        parts = prior.components(model, cache, spec, contexts, cfg, ids)
    finally:
        hook.remove()
    assert len(captures) == 1
    scores = cache.images[ids] @ captures[0][:, :4].transpose(1, 2)
    ce = F.cross_entropy(100*scores.reshape(-1, 4), torch.arange(4).repeat(len(ids)))
    return parts, ce


def preflight():
    verify(); configure()
    loaded = prior.load_training('cpu'); cache, spec, contexts, cfg = loaded
    forms = prev.template_cache(cache)
    current = replace(cache, texts=forms[:, 1])
    model = new_model(cfg).train()
    ids = prior.paired_rows(torch.arange(24))
    torch.manual_seed(420001)
    parts, ce = loss_parts(model, current, spec, contexts, cfg, ids)
    weights = read(prior.OUT/'normalization.json')['weights']
    params = tuple(model.parameters())
    def grads(loss):
        return torch.cat([g.flatten() for g in torch.autograd.grad(loss, params, retain_graph=True)])
    original = cn.objective(parts, weights)
    replay = objective(parts, weights, 'IS4', ce)
    assert torch.equal(original, replay)
    assert torch.equal(grads(original), grads(replay))
    for priority in (8, 16):
        a = grads(objective(parts, weights, 'IS'+str(priority), ce)) - grads(replay)
        b = grads((priority-4)*weights['cross']*parts['cross'])
        torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-7)
    # Diagnose weighted gradient competition at the current final checkpoint.
    entry = next(r for r in read(prev.OUT/'checkpoints_frozen.json')['models']
                 if r['name'] == 'IS_templates' and r['seed'] == 42)
    assert sha(entry['checkpoint']) == entry['sha256']
    model.load_state_dict(torch.load(entry['checkpoint'], map_location='cpu', weights_only=False)['state_dict'])
    model.eval()
    order = np.random.default_rng(20260929).permutation(1280)[:192]
    records = []
    for j in range(0, len(order), 24):
        ids = prior.paired_rows(torch.tensor(order[j:j+24]))
        parts, ce = loss_parts(model, current, spec, contexts, cfg, ids)
        terms = dict(cross=4*weights['cross']*parts['cross'],
                     binding=weights['binding']*parts['binding'],
                     response=weights['response']*parts['response'],
                     preference=4*weights['preference']*parts['preference'],
                     guards=prior.objective(parts, weights, 'G'))
        gg = {k: grads(v) for k, v in terms.items()}
        records.append(dict(weighted_loss={k: float(v.detach()) for k,v in terms.items()},
                            gradient_norm={k: float(v.norm()) for k,v in gg.items()},
                            cross_gradient_cosine={k: float(F.cosine_similarity(gg['cross'], v, dim=0)) for k,v in gg.items()}))
    dump(OUT/'preflight.json', dict(current_objective_gradient_exact=True,
          increased_priority_changes_only_cross_gradient=True, device='cpu',
          original_checkpoint=entry, gradient_diagnosis=records,
          cache_shapes=[list(x.shape) for x in (cache.images, cache.texts, cache.natural_texts, forms)],
          torch_version=torch.__version__))
    print('PREFLIGHT_PASS', flush=True)


def train(arm):
    verify(); configure()
    assert (OUT/'preflight.json').exists()
    dest = OUT/'runs'/arm
    dest.mkdir(parents=True, exist_ok=False)
    cache, spec, contexts, cfg = prior.load_training('cpu')
    cfg = replace(cfg, epochs=36, seed=42)
    forms = prev.template_cache(cache)
    weights = read(prior.OUT/'normalization.json')['weights']
    schedule = old.schedule_for(42)
    reps = prev.representation_schedule(42)
    model = new_model(cfg)
    initial_hash = _state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []; start = time.monotonic()
    for ei, order in enumerate(schedule):
        choice = torch.as_tensor(np.repeat(reps[ei], 2))
        current = replace(cache, texts=forms[torch.arange(2560), choice])
        model.train(); torch.manual_seed(420000+ei+1)
        for j in range(0, 1280, 24):
            ids = prior.paired_rows(torch.tensor(order[j:j+24]))
            parts, ce = loss_parts(model, current, spec, contexts, cfg, ids)
            loss = objective(parts, weights, arm, ce)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(grad)
            optimizer.step()
            history.append(dict(epoch=ei+1, rows=ids.tolist(), loss=float(loss.detach()),
                gradient_norm=float(grad), ce=float(ce.detach()),
                components={k: float(v.detach()) for k,v in parts.items()}))
        if (ei+1) % 6 == 0:
            print('TRAIN', arm, ei+1, 'seconds', round(time.monotonic()-start, 1), flush=True)
    assert len(history) == 1944
    checkpoint = dest/'last.pt'
    torch.save(dict(state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},
        configuration=asdict(cfg), optimizer=optimizer.state_dict(), seed=42, arm=arm,
        selection='fixed_last', epochs=36, updates=1944, device='cpu', weights=weights,
        initial_state_hash=initial_hash, protocol_sha256=sha(OUT/'protocol.json')), checkpoint)
    jsonl(dest/'history.jsonl', history)
    receipt = dict(name=arm, seed=42, checkpoint=str(checkpoint), sha256=sha(checkpoint),
        initial_state_hash=initial_hash, updates=1944, device='cpu',
        schedule_sha256=hash_array(np.asarray(schedule)), representation_sha256=hash_array(reps),
        final_rng_sha256=hash_array(torch.get_rng_state().numpy()),
        first_components=history[0]['components'], seconds=time.monotonic()-start)
    dump(dest/'complete.json', receipt)
    print('TRAIN_COMPLETE', arm, receipt['seconds'], flush=True)


def evaluate():
    verify(); configure()
    regs = [read(OUT/'runs'/arm/'complete.json') for arm in ARMS]
    for key in ('initial_state_hash', 'updates', 'schedule_sha256',
                'representation_sha256', 'final_rng_sha256'):
        assert len({r[key] for r in regs}) == 1, key
    assert all(r['first_components'] == regs[0]['first_components'] for r in regs)
    regs = [dict(name='F', seed=0, checkpoint=None), *regs]
    result = prev.development(regs, OUT/'development.jsonl')
    aggregate = prev.means(result)
    # Persist all per-anchor score tables, including individual templates, before selection.
    feature = np.load(prev.OUT/'training_template_features.npy')
    lookup = {s:i for i,s in enumerate(read(prev.OUT/'training_template_strings.json'))}
    groups = prev.lines(prev.DEV/'features/text_groups.jsonl')
    arrays = {}; context_means = {}
    for view in ('canvas', 'swapped_canvas'):
        v, t, idx = metrics.bank_arrays(prev.DEV/'features', 'routing', 'red-blue', view)
        arrays[view+'/ids'] = np.asarray([r['anchor_id'] for r in idx])
        forms = np.stack([feature[[lookup[s] for ti in r['text_indices'][:4]
                        for s in groups[ti]['templates']]].reshape(4,3,768).transpose(1,0,2) for r in idx])
        for reg in regs:
            scores = metrics.score_arrays(v, t, reg['checkpoint'])
            arrays[view+'/'+reg['name']] = scores
            arrays[view+'/'+reg['name']+'/templates'] = np.einsum('nid,nkjd->nkij', v, metrics.adapt(forms, reg['checkpoint']))
            values = metrics.routing(scores)
            context_means[view+'/'+reg['name']] = {k:float(a.mean()) for k,a in values.items() if k.startswith('contrast/')}
    with (OUT/'development_scores.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    frozen = aggregate['F']; current = aggregate['IS4']
    binding_names = [r['name'] for r in metrics.all_contexts() if r['kind']=='binding']
    candidates = []
    for arm in ('IS8', 'IS16'):
        a = aggregate[arm]
        binding_ok = all(np.mean([context_means[v+'/'+arm]['contrast/'+n] for v in ('canvas','swapped_canvas')])
                         >= np.mean([context_means[v+'/F']['contrast/'+n] for v in ('canvas','swapped_canvas')])
                         for n in binding_names)
        checks = dict(cross_20percent=a['cross'] <= .8*frozen['cross'],
            binding_contexts=binding_ok, pooled_accuracy=a['exchange_accuracy'] >= current['exchange_accuracy']-.005,
            individual_accuracy=a['individual_exchange'] >= current['individual_exchange']-.01)
        for view in ('canvas','swapped_canvas'):
            r = next(x for x in result if x['name']==arm and x['view']==view)
            f = next(x for x in result if x['name']=='F' and x['view']==view)
            checks[view+'/words'] = min(r['word1_accuracy'],r['word2_accuracy']) >= .99
            checks[view+'/object'] = r['object_guard'] >= f['object_guard']-.01
        candidates.append(dict(name=arm, eligible=all(checks.values()), checks=checks))
    eligible = [r['name'] for r in candidates if r['eligible']]
    selected = min(eligible, key=lambda n:(aggregate[n]['cross'], -aggregate[n]['exchange_accuracy'], int(n[2:]))) if eligible else None
    dump(OUT/'development_means.json', aggregate)
    dump(OUT/'development_contexts.json', context_means)
    dump(OUT/'selection.json', dict(selected=selected, candidates=candidates, source='888 development anchors only',
          test_scoring_started=False, seed=42, matched_training_receipts=True,
          protocol_sha256=sha(OUT/'protocol.json')))
    print('DEVELOPMENT', {k:{m:round(v[m],6) for m in ('binding','cross','exchange_accuracy','individual_exchange','preference')} for k,v in aggregate.items()}, flush=True)
    print('SELECTION', selected, candidates, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','preflight','train','evaluate'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    log(OUT, 'start', action=args.action, arm=args.arm)
    try:
        if args.action == 'train': train(args.arm)
        else: globals()[args.action]()
    except BaseException as exc:
        log(OUT, 'failed', action=args.action, error=repr(exc)); raise
    log(OUT, 'complete', action=args.action, arm=args.arm)


if __name__ == '__main__':
    main()
