"""Inference-only timing of the routing (color-binding) test generator and frozen LAION L/14 audit.

Reuses FSE_VLM/Mirror: routing_light_tint.render (75% tint), scorers.load_subject /
encode_images / encode_texts, rendering.captions (canonical order),
indirect_generalization.prompts (reversed order, as in routing_tint_balance_data),
completion_metrics.routing. Mirrors feature_cache.cache_bank's pipelining
(8 render threads, chunks of 16 anchors, image batch 64, text batch 128) without
its file writes and checkpoint hashing. Writes only results/timing_routing.json.
"""
from mirror.paths import ARTIFACT_ROOT
import argparse; import json; import platform; import statistics; import sys; import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import cv2

ROOT = ARTIFACT_ROOT
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'FSE_VLM'))
from mirror.cases.color_binding import routing_light_tint as p                      # noqa: E402
from mirror.cases.color_binding import indirect_generalization as ind               # noqa: E402
from mirror.core import metrics                                                   # noqa: E402
from mirror.core.encoders import load_subject; from mirror.core.encoders import legacy_unit            # noqa: E402
from mirror.cases.color_binding.rendering import captions                           # noqa: E402
from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for      # noqa: E402

COLORS = p.COLORS[:2]          # red/blue, green/yellow (paper primary)
ALPHA = .75
WORKERS, IMG_BATCH, TXT_BATCH = 8, 64, 128
SAVED = ROOT / 'clip/interbind_routing_tint_balance_20260930/midpoint75/directional_retest/analysis/primary'


def sync(): torch.cuda.synchronize()


def render(row): return p.render(row, COLORS, ALPHA)


def generate(rows):
    out = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(rows), 2 * WORKERS):
            out.extend(pool.map(render, rows[start:start + 2 * WORKERS]))
    return out


def caption_sets(rows):
    canon = []; seen = set()
    for r in rows:
        for c in COLORS:
            for g in captions(r['family'], r['objects'], c):
                k = json.dumps(g)
                if k not in seen: seen.add(k); canon.append(g)
    pairs = sorted({tuple(r['objects']) for r in rows}); rev = {}
    for nouns in pairs:
        for c in COLORS:
            rev[(nouns, '-'.join(c))] = ind.prompts(nouns, c, 'canvas', 'reverse_order')
    rev_strings = list(dict.fromkeys(s for v in rev.values() for g in v for s in g))
    return canon, rev, rev_strings


def encode_texts(scorer, canon, rev, rev_strings):
    tc = scorer.encode_texts(canon, batch_size=TXT_BATCH).numpy(); ci = {json.dumps(g): i for i, g in enumerate(canon)}
    ts = scorer.encode_texts(rev_strings, batch_size=TXT_BATCH).numpy(); si = {s: i for i, s in enumerate(rev_strings)}
    return tc, ci, ts, si


def text_block(rows, color, order, tc, ci, rev, ts, si):
    colors = tuple(color.split('-'))
    if order == 'canonical':
        return np.stack([tc[[ci[json.dumps(g)] for g in captions(r['family'], r['objects'], colors)]] for r in rows])
    x = np.stack([ts[[[si[s] for s in g] for g in rev[(tuple(r['objects']), color)]]] for r in rows])  # n,4,3,d
    return legacy_unit(torch.from_numpy(x.mean(2))).numpy()          # text_for pooling


def image_block(feats, names_per_row, color, view):
    a, b = color.split('-')
    names = [f'{color}/{view}/{s}_{t}' for s in (a, b) for t in (a, b)]
    return np.stack([feats[i][[nm.index(n) for n in names]] for i, nm in enumerate(names_per_row)])


def score_all(rows, feats, names_per_row, texts):
    tc, ci, rev, ts, si = texts; out = {}; tot = dict(exch=0, wrong=0, tests=0, scores=0)
    for color in map('-'.join, COLORS):
        for order in ('canonical', 'reversed'):
            t = text_block(rows, color, order, tc, ci, rev, ts, si)
            for view in p.VIEWS:
                v = image_block(feats, names_per_row, color, view)
                x = np.einsum('nid,njd->nij', v, t, optimize=True)
                m = metrics.routing(x)
                q1 = x[:, 1, 1] - x[:, 1, 2]; q2 = x[:, 2, 2] - x[:, 2, 1]
                tot['exch'] += 2 * len(x); tot['wrong'] += int((q1 <= 0).sum() + (q2 <= 0).sum())
                tot['tests'] += 17 * len(x); tot['scores'] += x.size
                out[(color, order, view)] = (x, m)
    return out, tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repeats', type=int, default=3); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rec = dict(scenario='routing', bank='400 held-out anchors, 75% tint, red/blue + green/yellow, both noun orders, both layouts',
               render_threads=WORKERS, image_batch=IMG_BATCH, text_batch=TXT_BATCH, render_chunk_anchors=2 * WORKERS,
               precision='image and text FP32, TF32 off; scores CPU einsum', gpu=torch.cuda.get_device_name(0),
               cpu=platform.processor() or 'AMD EPYC 7R32', torch=torch.__version__, setup={})
    rows = lines(p.CONFIRM / 'rows.jsonl')
    if a.limit: rows = rows[:a.limit]
    t = time.perf_counter(); annotations_for('train2017'); annotations_for('val2017'); rec['setup']['coco_annotation_index_s'] = time.perf_counter() - t
    t = time.perf_counter(); scorer = load_subject(p.prior.MODEL, device='cuda'); sync(); rec['setup']['model_load_and_checkpoint_hash_s'] = time.perf_counter() - t
    canon, rev, rev_strings = caption_sets(rows)
    # warm-up (untimed)
    w = generate(rows[:64]); scorer.encode_images([im for ims, _, _ in w for im in ims], batch_size=IMG_BATCH)
    encode_texts(scorer, canon, rev, rev_strings); sync(); del w
    G, EI, ET, ES, P = [], [], [], [], []
    for rep in range(a.repeats):
        t = time.perf_counter(); gen = generate(rows); G.append(time.perf_counter() - t)
        flat = [im for ims, _, _ in gen for im in ims]; names = [nm for _, nm, _ in gen]
        t = time.perf_counter(); f = scorer.encode_images(flat, batch_size=IMG_BATCH).numpy(); sync(); EI.append(time.perf_counter() - t)
        feats = f.reshape(len(rows), 16, -1)
        t = time.perf_counter(); tx = encode_texts(scorer, canon, rev, rev_strings); sync(); ET.append(time.perf_counter() - t)
        texts = (tx[0], tx[1], rev, tx[2], tx[3])
        t = time.perf_counter(); res, tot = score_all(rows, feats, names, texts); ES.append(time.perf_counter() - t)
        # pipelined, cache_bank-style: render threads + GPU encoding of pending images
        t = time.perf_counter(); pending, chunks, pnames = [], [], []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for start in range(0, len(rows), 2 * WORKERS):
                for ims, nm, _ in pool.map(render, rows[start:start + 2 * WORKERS]):
                    pending.extend(ims); pnames.append(nm)
                    if len(pending) >= IMG_BATCH:
                        chunks.append(scorer.encode_images(pending, batch_size=IMG_BATCH)); pending = []
        if pending: chunks.append(scorer.encode_images(pending, batch_size=IMG_BATCH))
        pf = torch.cat(chunks).numpy().reshape(len(rows), 16, -1)
        ptx = encode_texts(scorer, canon, rev, rev_strings); pres, _ = score_all(rows, pf, pnames, (ptx[0], ptx[1], rev, ptx[2], ptx[3])); sync()
        P.append(time.perf_counter() - t)
        print(f'REP {rep}: G={G[-1]:.2f} E_img={EI[-1]:.2f} E_txt={ET[-1]:.2f} E_score={ES[-1]:.3f} pipelined={P[-1]:.2f}', flush=True)
    def summ(x): return dict(median=statistics.median(x), min=min(x), max=max(x), runs=x)
    rec['timing_s'] = dict(generation=summ(G), exec_image_prep_encode=summ(EI), exec_text_encode=summ(ET), exec_score_and_checks=summ(ES),
                           execution_total=summ([x + y + z for x, y, z in zip(EI, ET, ES)]), end_to_end_pipelined=summ(P))
    rec['work'] = dict(sources=len(rows), images_rendered=len(flat), image_forwards=-(-len(flat) // IMG_BATCH),
                       canonical_caption_groups=len(canon), caption_strings_encoded=3 * len(canon) + len(rev_strings),
                       scores=tot['scores'], interaction_tests=tot['tests'], decision_checks=tot['exch'], failing_decisions=tot['wrong'],
                       exchange_accuracy_pct=100 * (1 - tot['wrong'] / tot['exch']))
    # fidelity versus saved per-block score arrays
    diffs, pdiffs, flips = [], [], 0
    for (color, order, view), (x, m) in res.items():
        saved = np.load(SAVED / f'{color}_{order}_scores.npz')[f'{view}/Frozen'][:len(rows)]
        diffs.append(float(abs(x - saved).max())); pdiffs.append(float(abs(pres[(color, order, view)][0] - saved).max()))
        q = lambda z: np.stack([z[:, 1, 1] - z[:, 1, 2], z[:, 2, 2] - z[:, 2, 1]], 1) > 0
        flips += int((q(x) != q(saved)).sum())
    rec['fidelity'] = dict(max_abs_score_diff=max(diffs), max_abs_score_diff_pipelined=max(pdiffs), exchange_decision_flips=flips,
                           failing_recomputed=tot['wrong'], failing_saved_expected=3054,
                           binding_mean=float(np.mean([m['binding'].mean() for _, m in res.values()])),
                           cross_mean=float(np.mean([m['cross'].mean() for _, m in res.values()])),
                           assignment_mean=float(np.mean([2 * m['response'].mean() for _, m in res.values()])))
    saved_meta = json.loads((ROOT / 'clip/interbind_routing_tint_balance_20260930/midpoint75/directional_retest/features/primary/complete.json').read_text())
    rec['historical_reference'] = dict(cache_bank_elapsed_s=saved_meta['elapsed_seconds'], cache_bank_images=saved_meta['n_images'],
                                       note='Original cache_bank run: 3 color pairs (24 images/anchor) incl. pixel-hash checks and file writes')
    (HERE / 'results' / ('timing_routing.json' if not a.limit else 'smoke_routing.json')).write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps({k: rec[k] for k in ('timing_s', 'work', 'fidelity', 'setup')}, indent=1))


if __name__ == '__main__':
    main()
