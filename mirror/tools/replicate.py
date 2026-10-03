"""Fixed backdoor 2x / plain typography 4x seed replication, isolated outputs."""
from mirror.paths import ARTIFACT_ROOT
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime; from datetime import timezone

ROOT = ARTIFACT_ROOT
DEST = ROOT / 'clip/fse_selected_strength_replication_20260929'
sys.path.insert(0, str(ROOT / 'mirror/cases/typography'))
import numpy as np
import torch
from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha

SEEDS = (43, 44)
CASES = ('stripes', 'triangles', 'text')
BANKS = ('development', 'banana87', 'banana1000', 'imagenetv2')


def read(p):
    return json.loads(Path(p).read_text())


def command():
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / 'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
                               argv=[sys.executable, *sys.argv], cwd=str(Path.cwd()))) + '\n')


def typo_original(seed):
    return ROOT / 'mirror/cases/typography_broad_confirmation_20260926' / f'seed{seed}'


def verify():
    cfg = read(DEST / 'protocol.json')
    assert sha(DEST / 'protocol.json') == read(DEST / 'protocol_hash.json')['sha256']
    for p, h in cfg['files'].items():
        assert sha(p) == h, p
    return cfg


def prepare():
    import mirror.cases.backdoor.enforcement as p
    import mirror.cases.typography.replicate as t
    assert not (DEST / 'protocol.json').exists()
    p.verify()
    files = [Path(__file__), Path(p.__file__), Path(p.base.__file__), Path(p.tr.__file__),
             Path(p.ev.__file__), Path(t.__file__), Path(t.base.__file__), Path(t.bank.__file__),
             ROOT / 'mirror/cases/typography/strength_retest.py',
             ROOT / 'mirror/cases/typography/broad_duration_retest.py',
             ROOT / 'mirror/cases/typography/retest.py']
    recipes = {}
    for case in CASES:
        original = p.original_root(case)
        root = DEST / 'backdoor' / case
        coefficient = read(p.tr.directory(case) / 'calibration.json')['coefficient']
        dump(root / 'calibration.json', dict(coefficient=2 * coefficient,
             original_coefficient=coefficient, multiplier=2))
        dump(root / 'protocol.json', dict(case=case, multiplier=2, seeds=SEEDS,
             steps=1536, selection='fixed final successful update', original=str(original)))
        files += [root / 'protocol.json', root / 'calibration.json']
        for seed in SEEDS:
            receipts = []
            for method in ('ranking', 'IS'):
                r = original / 'runs' / f'{method}_seed{seed}'
                done = read(r / 'complete.json')
                assert done['steps'] == 1536 and sha(r / 'last.pt') == done['checkpoint_sha256']
                receipts.append(done)
                files += [r / 'complete.json', r / 'last.pt']
            assert receipts[0]['initial_sha256'] == receipts[1]['initial_sha256']
            assert receipts[0]['sequence_sha256'] == receipts[1]['sequence_sha256']
        r = p.arm_root(case, 2) / 'runs/IS_seed42'
        assert sha(r / 'last.pt') == read(r / 'complete.json')['checkpoint_sha256']
        files += [r / 'last.pt', r / 'complete.json']
        recipes[case] = dict(multiplier=2, coefficient=2 * coefficient, original=str(original))
    for seed in SEEDS:
        original = typo_original(seed)
        t.RUN = original
        t.verify_prompt()
        root = DEST / 'typography' / f'seed{seed}'
        dump(root / 'calibration.json', read(original / 'calibration.json'))
        dump(root / 'protocol.json', dict(seed=seed, multiplier=4, updates=2752,
             batch_size=32, selection='fixed final update', original=str(original),
             preservation='plain 4x: original shared guards, no P package'))
        files += [root / 'protocol.json', root / 'calibration.json', original / 'selection.json',
                  original / 'text_caches.json']
        receipts = []
        for method in ('R_s100', 'IS_s100_w1'):
            r = original / 'runs' / method
            done = read(r / 'complete.json')
            assert done['updates'] == 2752 and sha(r / 'last.pt') == done['sha256']
            receipts.append(done)
            files += [r / 'complete.json', r / 'last.pt']
        assert receipts[0]['initial_hash'] == receipts[1]['initial_hash']
        assert receipts[0]['sequence_hash'] == receipts[1]['sequence_hash']
    r = ROOT / 'mirror/cases/typography_strength_20260929/runs/IS_w4'
    assert sha(r / 'last.pt') == read(r / 'complete.json')['sha256']
    files += [r / 'last.pt', r / 'complete.json']
    dump(DEST / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
         new_seeds=SEEDS, all_seeds=[42,43,44], backdoor=recipes,
         typography=dict(multiplier=4, preservation='unchanged original guards; no P'),
         prior_outcomes='Seed42 strength and public results known before this fixed replication',
         selection='all requested attacks/seeds; final checkpoints; no new recipe or seed selection',
         matching='reuse original trainers; verify matched initialization, stream and update count',
         evaluation='same complete seed42 test banks and processing; matched original ranking/IS and frozen',
         statistics='mean and sample SD; paired source and crossed seed/source percentile bootstrap',
         gpu_policy='wait for other GPU jobs before each GPU action; do not stop unrelated jobs',
         storage='reuse all existing caches; only final checkpoints; no downloads',
         manuscript='no automatic manuscript replacement', files={str(f): sha(f) for f in files}))
    dump(DEST / 'protocol_hash.json', dict(sha256=sha(DEST / 'protocol.json')))
    print('PREPARED', DEST, flush=True)


def train_backdoor(case, seed, smoke=False):
    import mirror.cases.backdoor.enforcement as p
    verify()
    p.tr.CASE, p.tr.METHOD = case, 'IS'
    p.tr.verify()
    p.base.RUN, p.base.SEED = DEST / 'backdoor' / case, seed
    p.base.setup = p.tr.setup
    out = p.base.RUN / ('smoke' if smoke else 'runs') / f'IS_seed{seed}'
    if not (out / 'complete.json').exists():
        assert shutil.disk_usage(DEST).free > 700 * 1024**2
        p.base.train('IS', smoke)
    done = read(out / 'complete.json')
    assert sha(out / 'last.pt') == done['checkpoint_sha256']
    old = read(p.original_root(case) / 'runs' / f'ranking_seed{seed}' / 'complete.json')
    assert done['initial_sha256'] == old['initial_sha256']
    if not smoke:
        assert done['steps'] == old['steps'] == 1536
        assert done['sequence_sha256'] == old['sequence_sha256']
    dump(out / 'matching_check.json', dict(initial_equal=True, stream_equal=not smoke,
         update_count_equal=not smoke, shared_losses_unchanged=True, original=old))


def train_typography(seed, smoke=False):
    import mirror.cases.typography.replicate as t
    verify()
    t.SEED, t.RUN = seed, typo_original(seed)
    t.verify_prompt()
    old = read(t.RUN / 'runs/R_s100/complete.json')
    t.RUN = DEST / 'typography' / f'seed{seed}'
    t.CONFIGS = {'IS_s100_w4': dict(scale=100, weight=4.)}
    t.verify_prompt = verify
    # All forwards, RNG, optimizer and guards stay in the original trainer.
    out = t.RUN / ('smoke' if smoke else 'runs') / 'IS_s100_w4'
    if not (out / 'complete.json').exists():
        t.train('IS_s100_w4', smoke)
    done = read(out / 'complete.json')
    assert sha(out / 'last.pt') == done['sha256']
    assert done['initial_hash'] == old['initial_hash']
    if not smoke:
        assert done['updates'] == old['updates'] == 2752
        assert done['sequence_hash'] == old['sequence_hash']
    dump(out / 'matching_check.json', dict(initial_equal=True, stream_equal=not smoke,
         update_count_equal=not smoke, shared_losses_unchanged=True, no_P_losses=True, original=old))


@torch.no_grad()
def evaluate_backdoor(case, bank, smoke=False):
    import mirror.cases.backdoor.enforcement as p
    from torch.utils.data import DataLoader
    from torch.nn import functional as F
    verify()
    p.tr.CASE = case
    p.tr.verify()
    root = DEST / 'backdoor' / case / ('evaluation_smoke' if smoke else 'evaluation') / bank
    if (root / 'complete.json').exists():
        for name, h in read(root / 'complete.json')['records_sha256'].items():
            assert sha(root / f'{name}_records.npz') == h
        return
    model, prep, tok = p.tr.model_load(case)
    tv = p.ev.texts(model, tok, case, 'victim', bank in ('banana1000','imagenetv2'))
    target = 954 if bank in ('banana1000','imagenetv2') else 86
    checkpoints = {f'IS2_seed{s}': DEST / 'backdoor' / case / 'runs' / f'IS_seed{s}/last.pt' for s in SEEDS}
    if smoke:
        checkpoints = {f'{m}_seed43': p.original_root(case) / 'runs' / f'{m}_seed43/last.pt' for m in ('ranking','IS')}
    tails = {name: p.load_tail(model, path) for name, path in checkpoints.items()}
    source = p.ev.Sources(case, bank, prep, with_controls=False, limit=8 if smoke else None)
    rows, n = source.rows, len(source.rows)
    vocab = read(p.data.original.POUT / 'protocol.json')['vocabulary']
    labels = np.array([target if bank.startswith('banana') else r['label'] if bank=='imagenetv2'
                       else vocab.index(r['label']) for r in rows])
    values = {name: p.ev.empty_records(n) for name in tails}
    captured = []
    hook = model.visual.transformer.resblocks[10].register_forward_pre_hook(lambda module,args: captured.append(args[0]))
    loader = DataLoader(source, batch_size=4, num_workers=2 if smoke else 6, pin_memory=True,
         prefetch_factor=1, worker_init_fn=p.data.original.old.typo.worker_init)
    seen, start = [], time.monotonic()
    for j, (indices, images, _) in enumerate(loader):
        ix = indices.numpy()
        with torch.autocast('cuda', dtype=torch.float16):
            model.encode_image(images[:,0].flatten(0,1).cuda(non_blocking=True))
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
            p.ev.record(values[name], ix, (z @ tv.T).reshape(len(ix),len(p.ev.STATES),-1), labels,target)
        seen.extend(ix.tolist())
        if j % 100 == 0:
            print('REPLICATE EVAL', case, bank, len(seen), n, round(time.monotonic()-start,1), flush=True)
    hook.remove()
    assert sorted(seen) == list(range(n))
    parity = {}
    if smoke:
        oldroot = ROOT / 'clip/fse_four_case_20260927/backdoor_completion_v1' / case / 'evaluation' / bank
        for name, value in values.items():
            with np.load(oldroot / f'{name}_records.npz') as old:
                error = max(float(abs(value[k]-old[k][:n]).max()) for k in ('true_score','target_margin','allclass_interaction_abs'))
                agree = float((value['pred']==old['pred'][:n]).mean())
                assert error < 3e-4 and agree == 1, (case,bank,name,error,agree)
                parity[name] = dict(max_score_error=error,prediction_agreement=agree)
    root.mkdir(parents=True,exist_ok=True)
    hashes = {}
    for name, value in values.items():
        path = root / f'{name}_records.npz'
        assert not path.exists()
        np.savez_compressed(path, **value, labels=labels, ids=np.array([r['id'] for r in rows]), states=np.array(p.ev.STATES))
        hashes[name] = sha(path)
    dump(root / 'complete.json', dict(case=case,bank=bank,seeds=SEEDS,n=n,seconds=time.monotonic()-start,
         records_sha256=hashes,parity=parity,checkpoints={str(p):sha(p) for p in checkpoints.values()}))


def evaluate_typography(seed):
    import mirror.cases.typography.strength_retest as rt
    verify()
    torch.set_num_threads(2)
    run = DEST / 'typography' / f'seed{seed}'
    out = run / 'retest'
    if (out / 'complete.json').exists():
        for p,h in read(out / 'complete.json')['files'].items():
            assert sha(out / p) == h, p
        return
    original = typo_original(seed)
    old = read(original / 'selection.json')
    path = run / 'runs/IS_s100_w4/last.pt'
    assert sha(path) == read(path.parent / 'complete.json')['sha256']
    reg = dict(ranking=old['ranking'],previous_IS=old['IS'],same_scale_ablation=old['ranking'],
         initial_prefix=old['initial_prefix'],IS=dict(checkpoint=str(path),sha256=sha(path)))
    dump(run / 'evaluation_registry.json',reg)
    out.mkdir(parents=True,exist_ok=True)
    rt.RUN, rt.DEST, rt.legacy.RUN, rt.pilot.ORIGINAL = run, out, out, original
    # Reuse precisely the pilot evaluator, but matched to this seed's baselines.
    mm = rt.text_banks(reg)
    rt.legacy.digital_score(mm)
    rt.fresh_score(mm)
    rt.complete_targets(mm)
    rt.legacy.sugar_score(mm)
    rt.legacy.external_score(mm)
    rt.public_contrasts()
    files = [p for p in out.rglob('*') if p.is_file()]
    dump(out / 'complete.json',dict(seed=seed,multiplier=4,no_P=True,
         files={str(p.relative_to(out)):sha(p) for p in files}))


def wait_gpu():
    stable = 0
    while stable < 3:
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        util = int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
        stable = stable + 1 if not apps and util < 10 else 0
        dump(DEST / 'gpu_wait.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
             gpu_pids=apps.splitlines(),utilization=util,consecutive_free_checks=stable,
             status='ready' if stable==3 else 'waiting_for_other_jobs'))
        print('GPU GATE', util, apps.replace('\n',','), stable, flush=True)
        if stable < 3:
            time.sleep(20)


if __name__ == '__main__':
    command()
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','verify','wait_gpu','train_backdoor','smoke_backdoor',
        'train_typography','smoke_typography','evaluate_backdoor','smoke_evaluation','evaluate_typography'])
    ap.add_argument('--seed',type=int,choices=SEEDS,default=43)
    ap.add_argument('--case',choices=CASES,default='stripes')
    ap.add_argument('--bank',choices=BANKS,default='development')
    a = ap.parse_args()
    if a.action in ('prepare','verify','wait_gpu'):
        globals()[a.action]()
    elif a.action in ('train_backdoor','smoke_backdoor'):
        train_backdoor(a.case,a.seed,a.action.startswith('smoke'))
    elif a.action in ('train_typography','smoke_typography'):
        train_typography(a.seed,a.action.startswith('smoke'))
    elif a.action in ('evaluate_backdoor','smoke_evaluation'):
        evaluate_backdoor(a.case,a.bank,a.action.startswith('smoke'))
    else:
        evaluate_typography(a.seed)
