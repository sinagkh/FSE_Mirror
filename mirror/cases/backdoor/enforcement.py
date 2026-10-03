"""Bounded backdoor strength study; reuse immutable original trainer/evaluator."""
import argparse
import copy
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime; from datetime import timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mirror.cases.backdoor.test_data as data
import mirror.cases.backdoor.evaluate as ev
import mirror.cases.backdoor.train as tr
import mirror.cases.backdoor.par_visual_blocks_v3 as base
from mirror.cases.backdoor.common import ROOT; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha

DEST = ROOT / 'clip/fse_backdoor_strength_20260929'
BANKS = ('development', 'banana87', 'banana1000', 'imagenetv2')


def read(path):
    return json.loads(Path(path).read_text())


def log_command():
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / 'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
            argv=[sys.executable, *sys.argv], cwd=str(Path.cwd()))) + '\n')


def original_root(case):
    return tr.ORIGINAL_RUN if case == 'stripes' else tr.directory(case)


def original_run(case, method='IS'):
    return original_root(case) / 'runs' / (method + '_seed42')


def arm_root(case, strength):
    return DEST / case / ('IS' + str(strength))


def verify():
    cfg = read(DEST / 'protocol.json')
    assert sha(DEST / 'protocol.json') == read(DEST / 'protocol_hash.json')['sha256']
    for name, digest in cfg['files'].items():
        assert sha(name) == digest, name
    return cfg


def prepare():
    assert not (DEST / 'protocol.json').exists(), 'Do not overwrite a frozen study.'
    ev.verify()
    assert shutil.disk_usage(DEST).free > 2 * 1024**3
    cases = {}
    files = [Path(__file__), Path(base.__file__), Path(tr.__file__), Path(ev.__file__),
             Path(data.__file__), DEST / 'PROTOCOL.md']
    for case in data.CASES:
        tr.CASE = case
        tr.verify()
        root = tr.directory(case)
        coefficient = read(root / 'calibration.json')['coefficient']
        cases[case] = dict(original_coefficient=coefficient, training=read(root / 'protocol.json'),
                          original_results=str(root / 'evaluation'))
        files += [root / 'protocol.json', root / 'calibration.json']
        for bank in ('train', 'development'):
            meta = read(root / (bank + '_cache.json'))
            assert sha(meta['path']) == meta['sha256']
            files.append(root / (bank + '_cache.json'))
        for method in ('IS', 'ranking'):
            path = original_run(case, method)
            done = read(path / 'complete.json')
            assert done['steps'] == 1536 and sha(path / 'last.pt') == done['checkpoint_sha256']
            files += [path / 'complete.json']
        for bank in BANKS:
            summary = read(root / 'evaluation' / bank / 'summary.json')
            for method in ('victim', 'ranking_seed42', 'IS_seed42'):
                path = root / 'evaluation' / bank / (method + '_records.npz')
                assert sha(path) == summary['records_sha256'][method]
        for strength in (1, 2, 4):
            new = arm_root(case, strength)
            dump(new / 'calibration.json', dict(coefficient=coefficient * strength,
                 original_coefficient=coefficient, multiplier=strength,
                 inherited_from=str(root / 'calibration.json')))
            dump(new / 'protocol.json', dict(case=case, seed=42, multiplier=strength,
                 original_protocol_sha256=sha(root / 'protocol.json'), steps=1536,
                 sole_training_change='multiply original interaction coefficient; everything else reused'))
    dump(DEST / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
         stage='developmental strength study after prior benchmark outcomes were known',
         cases=cases, strengths=[2, 4], seeds=[42], final_checkpoint=True,
         no_automatic_selection_or_replication=True, banks=list(BANKS), processing=list(data.GRID),
         evaluation='same 21 states, source IDs, captions, precision, batch size4; six CPU loaders',
         validation='original 1x stripes full replay; all-case evaluation parity; paired source bootstrap conditional on seed42',
         environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda),
         files={str(path): sha(path) for path in files}))
    dump(DEST / 'protocol_hash.json', dict(sha256=sha(DEST / 'protocol.json')))
    # Verify that coefficient scaling changes only the explicit interaction gradient.
    torch.manual_seed(29)
    scores = (torch.randn(3, 4, 87, dtype=torch.float64) * .04).requires_grad_()
    reference = torch.randn_like(scores) * .04
    labels = torch.tensor([0, 12, 86])
    terms = base.losses(scores, reference, labels)
    g1 = torch.autograd.grad(terms['shared'] + terms['interaction'], scores, retain_graph=True)[0]
    g4 = torch.autograd.grad(terms['shared'] + 4 * terms['interaction'], scores, retain_graph=True)[0]
    gi = torch.autograd.grad(terms['interaction'], scores)[0]
    error = float(abs(g4 - g1 - 3 * gi).max())
    assert error < 1e-12
    dump(DEST / 'loss_identity_test.json', dict(error=error, shared_losses_unchanged=True))
    print('PREPARED', DEST, flush=True)


def train(case, strength, smoke=False):
    verify()
    tr.CASE, tr.METHOD = case, 'IS'
    tr.verify()
    base.RUN, base.SEED = arm_root(case, strength), 42
    base.setup = tr.setup  # Original victim, tokens, labels, text, parameters, mode.
    out = base.RUN / ('smoke' if smoke else 'runs') / 'IS_seed42'
    if (out / 'complete.json').exists():
        done = read(out / 'complete.json')
        assert done['checkpoint_sha256'] == sha(out / 'last.pt')
        print('VERIFIED EXISTING', out, flush=True)
        return
    assert shutil.disk_usage(DEST).free > 1024**3
    base.train('IS', smoke)  # Original optimizer, schedule, loss, RNG, forwards.
    done = read(out / 'complete.json')
    previous = read(original_run(case) / 'complete.json')
    assert done['initial_sha256'] == previous['initial_sha256']
    if not smoke:
        assert done['steps'] == previous['steps'] == 1536
        assert done['sequence_sha256'] == previous['sequence_sha256']
    parity = dict(initialization_equal=True, stream_equal=not smoke,
                  original_receipt_sha256=sha(original_run(case) / 'complete.json'))
    if strength == 1 and not smoke:
        old = torch.load(original_run(case) / 'last.pt', map_location='cpu', weights_only=True)['visual_blocks']
        new = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)['visual_blocks']
        error = max(float(abs(old[k] - new[k]).max()) for k in old)
        a = np.load(original_run(case) / 'development_scores.npz')['scores']
        b = np.load(out / 'development_scores.npz')['scores']
        score_error = float(abs(a - b).max())
        assert error < 1e-6 and score_error < 1e-5, (error, score_error)
        parity.update(weight_max_error=error, score_max_error=score_error,
                      prediction_agreement=float((a.argmax(-1) == b.argmax(-1)).mean()))
    dump(out / 'matching_check.json', parity)


def load_tail(model, path):
    receipt = read(path.parent / 'complete.json')
    assert receipt['steps'] == 1536 and receipt['checkpoint_sha256'] == sha(path)
    state = torch.load(path, map_location='cpu', weights_only=True)['visual_blocks']
    blocks = torch.nn.ModuleList([copy.deepcopy(b) for b in model.visual.transformer.resblocks[10:]])
    for j, block in enumerate(blocks):
        prefix = f'transformer.resblocks.{10+j}.'
        block.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    return blocks.eval().requires_grad_(False)


@torch.no_grad()
def evaluate(case, bank, smoke=False):
    verify()
    tr.CASE = case
    tr.verify()
    dest = DEST / case / ('evaluation_smoke' if smoke else 'evaluation') / bank
    if (dest / 'complete.json').exists():
        receipt = read(dest / 'complete.json')
        for name, digest in receipt['records_sha256'].items():
            assert sha(dest / (name + '_records.npz')) == digest
        print('VERIFIED EXISTING', dest, flush=True)
        return
    model, prep, tok = tr.model_load(case)
    tv = ev.texts(model, tok, case, 'victim', bank in ('banana1000', 'imagenetv2'))
    target = 954 if bank in ('banana1000', 'imagenetv2') else 86
    if smoke:
        checkpoints = {m + '_seed42': original_run(case, m) / 'last.pt' for m in ('ranking', 'IS')}
    else:
        checkpoints = {'IS' + str(w) + '_seed42': arm_root(case, w) / 'runs/IS_seed42/last.pt' for w in (2, 4)}
    tails = {name: load_tail(model, path) for name, path in checkpoints.items()}
    source = ev.Sources(case, bank, prep, with_controls=False, limit=8 if smoke else None)
    rows, n = source.rows, len(source.rows)
    vocab = read(data.original.POUT / 'protocol.json')['vocabulary']
    labels = np.array([target if bank.startswith('banana') else
        r['label'] if bank == 'imagenetv2' else vocab.index(r['label']) for r in rows])
    values = {name: ev.empty_records(n) for name in tails}
    captured = []
    hook = model.visual.transformer.resblocks[10].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    start, seen = time.monotonic(), []
    loader = DataLoader(source, batch_size=4, num_workers=2 if smoke else 6,
        pin_memory=True, prefetch_factor=1, worker_init_fn=data.original.old.typo.worker_init)
    for j, (indices, images, _) in enumerate(loader):
        ix = indices.numpy()
        with torch.autocast('cuda', dtype=torch.float16):
            model.encode_image(images[:, 0].flatten(0, 1).cuda(non_blocking=True))
            tokens = captured.pop()
            assert not captured
            features = {}
            for name, blocks in tails.items():
                z = tokens
                for block in blocks:
                    z = block(z)
                pooled, _ = model.visual._pool(z)
                features[name] = F.normalize((pooled @ model.visual.proj).float(), dim=-1)
        for name, z in features.items():
            scores = (z @ tv.T).reshape(len(ix), len(ev.STATES), -1)
            ev.record(values[name], ix, scores, labels, target)
        seen.extend(ix.tolist())
        if j % 100 == 0:
            print('STRENGTH EVALUATE', case, bank, len(seen), '/', n, round(time.monotonic() - start, 1), flush=True)
    hook.remove()
    assert sorted(seen) == list(range(n))
    parity = {}
    if smoke:
        for name, value in values.items():
            old = np.load(tr.directory(case) / 'evaluation' / bank / (name + '_records.npz'))
            error = max(float(abs(value[k] - old[k][:n]).max()) for k in
                        ('true_score', 'target_margin', 'allclass_interaction_abs'))
            agreement = float((value['pred'] == old['pred'][:n]).mean())
            assert error < 3e-4 and agreement == 1, (case, bank, name, error, agreement)
            parity[name] = dict(max_score_error=error, prediction_agreement=agreement)
    dest.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, value in values.items():
        path = dest / (name + '_records.npz')
        assert not path.exists()
        np.savez_compressed(path, **value, labels=labels, ids=np.array([r['id'] for r in rows]), states=np.array(ev.STATES))
        digests[name] = sha(path)
    dump(dest / 'complete.json', dict(case=case, bank=bank, n=n, seconds=time.monotonic() - start,
         records_sha256=digests, source_sha256=sha(__file__), protocol_sha256=sha(DEST / 'protocol.json'),
         checkpoints={str(p): sha(p) for p in checkpoints.values()}, parity=parity))
    print('EVALUATION COMPLETE', case, bank, flush=True)


if __name__ == '__main__':
    log_command()
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'smoke', 'evaluate', 'eval_smoke'])
    parser.add_argument('--case', choices=data.CASES, default='stripes')
    parser.add_argument('--strength', type=int, choices=[1, 2, 4], default=2)
    parser.add_argument('--bank', choices=BANKS, default='development')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action in ('train', 'smoke'):
        train(args.case, args.strength, args.action == 'smoke')
    else:
        evaluate(args.case, args.bank, args.action == 'eval_smoke')
