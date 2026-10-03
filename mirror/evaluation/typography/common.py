"""Create-only campaign artifacts and paired seed/source statistics."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files

OUT = ROOT / 'clip/fse_two_case_completion_20260927'
SEEDS = (42, 43, 44)
PLAN = ROOT / 'FSE_VLM/plan/54_two_case_completion_and_native_colorswap.md'


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def freeze(dest, inputs, **details):
    inputs = [Path(__file__), PLAN, *map(Path, inputs)]
    value = dict(inputs={str(p): sha(p) for p in inputs}, **details)
    p = dest / 'protocol.json'
    if p.exists():
        assert read(p) == json.loads(json.dumps(value)), 'Protocol changed'
    else:
        dump(p, value)
        with (dest / 'protocol.sha256').open('x') as f:
            f.write(sha(p) + '\n')
    verify_files(value['inputs'])
    log(dest, 'start')


def save_array(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **values)


def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        pd.DataFrame(rows).to_csv(f, index=False)


def finish(dest, **details):
    files = {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*')
             if p.is_file() and p.name not in ('commands.jsonl', 'complete.json')}
    dump(dest / 'complete.json', dict(files=files, **details))
    log(dest, 'complete', **details)


def summaries(values, metrics, clusters, meta, denominators=None, draws=5000):
    """values: method -> [seeds, items, metrics]; ratios average seed ratios.

    Source draws shared by all methods and seeds; seed draws shared by paired
    trained methods. A deterministic external method is broadcast, not treated
    as three fits. Source-only and crossed intervals are both saved.
    """
    _, cluster = np.unique(clusters, return_inverse=True)
    nc = int(cluster.max()) + 1
    rng = np.random.default_rng(54260927)
    weights = rng.multinomial(nc, np.full(nc, 1 / nc), size=draws).astype(float)
    seed_weights = rng.multinomial(3, np.full(3, 1 / 3), size=draws) / 3
    cache = {}
    for name, raw in values.items():
        a = np.asarray(raw, float)
        assert a.ndim == 3 and len(a) in (1, 3) and a.shape[2] == len(metrics)
        den = np.ones_like(a) if denominators is None else np.asarray(denominators[name], float)
        assert den.shape == a.shape
        sums = np.stack([[np.bincount(cluster, weights=s[:, j], minlength=nc)
                          for j in range(a.shape[2])] for s in a])
        dsums = np.stack([[np.bincount(cluster, weights=s[:, j], minlength=nc)
                           for j in range(a.shape[2])] for s in den])
        point = np.divide(a.sum(1), den.sum(1), out=np.full((len(a), a.shape[2]), np.nan), where=den.sum(1)>0)
        source, crossed = [], []
        for start in range(0, draws, 100):
            w = weights[start:start+100]
            n = np.einsum('bc,smc->bsm', w, sums, optimize=True)
            d = np.einsum('bc,smc->bsm', w, dsums, optimize=True)
            per = np.divide(n, d, out=np.full_like(n, np.nan), where=d>0)
            source.append(np.nanmean(per, 1))
            crossed.append(np.nanmean(per, 1) if len(a)==1 else
                           np.sum(per * seed_weights[start:start+100, :, None], 1))
        cache[name] = (point, np.concatenate(source), np.concatenate(crossed))
    comparisons = [(n, None) for n in values]
    if 'IS' in values:
        comparisons += [('IS', n) for n in values if n != 'IS']
    result = []
    for a, b in comparisons:
        p, item, crossed = cache[a]
        if b is not None:
            q, qi, qc = cache[b]
            p, item, crossed = p-q, item-qi, crossed-qc
        for j, metric in enumerate(metrics):
            if not np.isfinite(p[:, j]).any():
                continue
            lo, hi = np.nanquantile(crossed[:, j], [.025, .975])
            il, ih = np.nanquantile(item[:, j], [.025, .975])
            result.append(dict(**meta, comparison=a if b is None else a+' - '+b,
                metric=metric, mean=float(np.nanmean(p[:, j])),
                sample_sd=float(np.nanstd(p[:, j], ddof=1)) if len(p)>1 else None,
                per_seed=p[:, j].tolist(), n_items=len(cluster), n_source_clusters=nc,
                independent_fits=len(values[a]) if b is None else None,
                ci95_low=float(lo), ci95_high=float(hi),
                item_ci95_low=float(il), item_ci95_high=float(ih)))
    return result


def run_logged(command):
    OUT.mkdir(parents=True, exist_ok=True)
    log(OUT, 'command_start', argv=command)
    r = subprocess.run(command, cwd=ROOT)
    log(OUT, 'command_end', argv=command, returncode=r.returncode)
    if r.returncode:
        raise SystemExit(r.returncode)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    run_logged(args.command)
