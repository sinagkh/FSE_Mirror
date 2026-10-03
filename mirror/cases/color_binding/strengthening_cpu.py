"""CPU-only, versioned P0/P1 follow-up; never selects checkpoints or examples."""
import argparse
import itertools
import json
import os
from pathlib import Path

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError("Launch with CUDA_VISIBLE_DEVICES='' (GPU work is prohibited)")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.behavioral_pilot import CAL; from mirror.cases.color_binding.behavioral_pilot import CACHE; from mirror.cases.color_binding.behavioral_pilot import QUAL; from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.core.metrics import bank_arrays; from mirror.core.metrics import adapt
from mirror.cases.color_binding.routing_feature_geometry import geometry

OUT = ROOT / 'clip/interbind_strengthening_cpu_20260925'
PRIOR = ROOT / 'clip/interbind_phase_bc_completion_20260923'
CONFIRM = PRIOR / 'same_rule_confirmation'
DIAGNOSTIC = ROOT / 'clip/interbind_routing_diagnostic_20260923/measurements.jsonl'
MODELS = ('openclip_laion_l14', 'openclip_laion_b32', 'openai_clip_l14', 'aro_negclip_b32')
SEEDS = (42, 43, 44)
NBOOT = 10000
RNGSEED = 20260925
METRICS = ('response', 'absolute_preference', 'exchange_alignment', 'text_exchange_norm',
           'image_exchange_norm', 'surplus', 'exchange_accuracy')


def clean(obj):
    if isinstance(obj, dict): return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)): return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (float, np.floating)): return float(obj) if np.isfinite(obj) else None
    return obj


def ci(values):
    values = np.asarray(values)
    good = values[np.isfinite(values)]
    return dict(ci95=np.quantile(good, [.025, .975]).tolist() if len(good) else None,
                defined_bootstrap_draws=len(good), undefined_bootstrap_draws=len(values)-len(good))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x-x.mean(-1, keepdims=True), y-y.mean(-1, keepdims=True)
    den = np.sqrt((x*x).sum(-1)*(y*y).sum(-1))
    return np.divide((x*y).sum(-1), den, out=np.full(den.shape, np.nan), where=den>1e-15)


def agreement(x, y):
    observed = (x == y).mean(-1)
    categories = np.unique(np.concatenate((x.ravel(), y.ravel())))
    expected = sum((x == k).mean(-1)*(y == k).mean(-1) for k in categories)
    kappa = np.divide(observed-expected, 1-expected, out=np.full(np.shape(observed), np.nan), where=expected<1-1e-15)
    return observed, kappa


def registry_path(model):
    return PRIOR/'routing_controls/models.json' if model==MODELS[0] else PRIOR/'breadth'/model/'routing/models.json'


def freeze():
    paths = [Path(__file__), Path(__file__).with_name('routing_feature_geometry.py'),
             Path(__file__).with_name('completion_metrics.py'), CAL, DIAGNOSTIC,
             ROOT/'FSE_VLM/plan/32_strong_paper_research_plan.md']
    for model in MODELS:
        path = registry_path(model); paths.append(path)
        verify_files({m['checkpoint']: m['sha256'] for m in read(path) if m['checkpoint']})
    inputs = {str(p): sha(p) for p in paths}
    dump(OUT/'protocol.json', dict(version='strong-paper-cpu-followup-v1', inputs=inputs,
        device='cpu', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], torch_threads=4,
        bootstrap_draws=NBOOT, bootstrap_seed=RNGSEED, seeds=SEEDS,
        inference='post-specified descriptive analysis of already-observed data; no new confirmation claim',
        stability='within-model and pooled cross-view Pearson/Spearman, shifts, absolute differences, regime transitions; bootstrap shared source IDs, never individual model-view rows',
        geometry='all available final F/R/IS and term-removal controls; all three trained seeds; fixed calibration; exact score identities checked',
        geometry_uncertainty='paired source-connected-component bootstrap and crossed seed plus source-component bootstrap; same source draws across seeds and arms; sample-SD of seed means',
        angle_policy='undefined zero-norm angles retained as missing and denominators reported',
        negative_results='Retain all outcomes; investigate concrete development hypotheses without selecting on confirmation or benchmark scores.',
        no_gpu=True, no_training_in_P1=True, no_manuscript_edits=True))


def stability():
    dest = OUT/'diagnostic_stability'; dest.mkdir(parents=True, exist_ok=False)
    verify_files(read(OUT/'protocol.json')['inputs'])
    frame = pd.DataFrame(lines(DIAGNOSTIC))
    assert not frame.duplicated(['model', 'source_id', 'view']).any()
    views = ('audit', 'swapped_canvas', 'in_situ')
    records, transitions = [], []
    for va, vb in itertools.combinations(views, 2):
        a = frame[frame.view == va].set_index(['source_id', 'model']).sort_index()
        b = frame[frame.view == vb].set_index(['source_id', 'model']).loc[a.index]
        for model in ['pooled_shared_source_bootstrap', *sorted(frame.model.unique())]:
            aa = a if model.startswith('pooled') else a[a.index.get_level_values('model') == model]
            bb = b.loc[aa.index]; sources = sorted(set(aa.index.get_level_values('source_id')))
            n = len(sources); width = len(aa)//n
            assert len(aa) == n*width
            draws = np.random.default_rng(RNGSEED).integers(0, n, (NBOOT, n))
            base = dict(model=model, view_a=va, view_b=vb, n_sources=n, n_rows=len(aa))
            for field in ('response', 'preference', 'response_raw', 'preference_raw'):
                x, y = aa[field].to_numpy(float), bb[field].to_numpy(float)
                xx = x.reshape(n, width)[draws].reshape(NBOOT, -1)
                yy = y.reshape(n, width)[draws].reshape(NBOOT, -1)
                for name, point, boot in (
                    ('pearson', corr(x, y), corr(xx, yy)),
                    ('spearman', corr(rankdata(x), rankdata(y)), corr(rankdata(xx, axis=1), rankdata(yy, axis=1))),
                    ('mean_shift_b_minus_a', (y-x).mean(), (yy-xx).mean(1)),
                    ('median_shift_b_minus_a', np.median(y-x), np.median(yy-xx, axis=1)),
                    ('median_absolute_difference', np.median(abs(y-x)), np.median(abs(yy-xx), axis=1)),
                    ('q90_absolute_difference', np.quantile(abs(y-x), .9), np.quantile(abs(yy-xx), .9, axis=1))):
                    records.append(dict(**base, quantity=field, statistic=name, estimate=float(point), **ci(boot)))
            labels = sorted(frame.regime.unique()); label_map = {v: i for i, v in enumerate(labels)}
            for kind, x, y in [('three_regimes', aa.regime.map(label_map).to_numpy(), bb.regime.map(label_map).to_numpy()),
                               ('response_sign', (aa.response>0).to_numpy(int), (bb.response>0).to_numpy(int))]:
                est = agreement(x, y)
                boot = agreement(x.reshape(n, width)[draws].reshape(NBOOT, -1), y.reshape(n, width)[draws].reshape(NBOOT, -1))
                for j, name in enumerate(('agreement', 'kappa')):
                    records.append(dict(**base, quantity=kind, statistic=name, estimate=float(est[j]), **ci(boot[j])))
                for i, j in itertools.product(sorted(set(x)|set(y)), repeat=2):
                    transitions.append(dict(**base, quantity=kind, from_label=labels[i] if kind=='three_regimes' else str(i),
                        to_label=labels[j] if kind=='three_regimes' else str(j), count=int(((x==i)&(y==j)).sum())))
    jsonl(dest/'statistics.jsonl', clean(records)); jsonl(dest/'transitions.jsonl', clean(transitions))
    pd.DataFrame(records).to_csv(dest/'statistics.csv', index=False, mode='x')
    text = ['# Cross-view diagnostic stability', '',
        'Post-specified analysis of all 49 valid pilot sources and seven frozen models. 10,000 source-cluster bootstrap draws; models on a shared image are not independent samples. Pointwise 95% percentile intervals, not simultaneous tests.', '',
        'Continuous response and preference are primary. Regime labels are secondary because a sign split near zero can change despite similar continuous values. In-situ is a changed visual context, not a repeated measurement of identical inputs.', '',
        '| Views | Model | Response r [95% CI] | Preference r [95% CI] | Three-regime κ |', '|---|---|---|---|---|']
    rr = pd.DataFrame(records)
    for (va, vb, m), g in rr.groupby(['view_a', 'view_b', 'model'], sort=False):
        cells = []
        for quantity, stat in [('response','pearson'), ('preference','pearson'), ('three_regimes','kappa')]:
            r = g[(g.quantity==quantity)&(g.statistic==stat)].iloc[0]
            cells.append(f"{r.estimate:.3f} [{r.ci95[0]:.3f}, {r.ci95[1]:.3f}]")
        text.append(f'| {va} → {vb} | {m} | '+ ' | '.join(cells) + ' |')
    text += ['', 'Pooled raw-score correlations are also saved but must not substitute for within-model correlations: models differ in score scales and cross-model offsets can inflate a pooled correlation. No calibration or classification threshold was fitted here. Full shift distributions, rank correlations, source counts and transition tables accompany this report.']
    with (dest/'REPORT.md').open('x') as f: f.write('\n'.join(text)+'\n')
    dump(dest/'complete.json', dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}, no_gpu=True, training=False))


def source_clusters(source_lists):
    """Join anchors sharing any image; resample independent connected components."""
    parent = list(range(len(source_lists))); owner = {}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, sources in enumerate(source_lists):
        for s in sources:
            if s in owner: parent[root(i)] = root(owner[s])
            else: owner[s] = i
    _, inverse = np.unique([root(i) for i in range(len(parent))], return_inverse=True)
    return inverse


def paired_intervals(values, clusters):
    """S x N x M arrays. Preserve pairing, resample source components and seeds."""
    s, n, m = values.shape; ncl = int(max(clusters))+1
    sums = np.zeros((s, ncl, m)); counts = np.zeros_like(sums)
    for k in range(ncl):
        x = values[:, clusters==k]
        sums[:,k] = np.nansum(x, axis=1); counts[:,k] = np.isfinite(x).sum(1)
    rng = np.random.default_rng(RNGSEED)
    weights = rng.multinomial(ncl, np.full(ncl, 1/ncl), NBOOT).astype(float)
    seed_weights = rng.multinomial(s, np.full(s, 1/s), NBOOT).astype(float)/s
    numerator = np.einsum('bk,skm->bsm', weights, sums, optimize=True)
    denom = np.einsum('bk,skm->bsm', weights, counts, optimize=True)
    means = np.divide(numerator, denom, out=np.full_like(numerator, np.nan), where=denom>0)
    item = np.mean(means, axis=1)
    both = np.sum(means*seed_weights[:,:,None], axis=1)
    per_seed = np.nanmean(values, axis=1)
    return [dict(mean=float(np.mean(per_seed[:,j])), sample_sd=float(np.std(per_seed[:,j], ddof=1)) if s>1 else None,
                 per_seed=per_seed[:,j].tolist(), ci95_source=ci(item[:,j]), ci95_seed_source=ci(both[:,j]),
                 n_defined=int(np.isfinite(values[:,:,j]).sum()), n_source_components=ncl) for j in range(m)]


def cases(model):
    yield ('confirmation', CONFIRM/'features'/model, lines(CONFIRM/'rows.jsonl'), '', ('canvas','swapped_canvas'))
    for part in ('pilot', 'reserve'):
        yield (part, CACHE/'features'/model/part/'routing',
               [r for r in lines(QUAL/part/'accepted_rows.jsonl') if r['family']=='routing'],
               'blend90_luminance/', ('canvas','swapped_canvas','in_situ'))


def final_geometry():
    dest = OUT/'feature_geometry'; dest.mkdir(parents=True, exist_ok=False)
    verify_files(read(OUT/'protocol.json')['inputs']); cal = read(CAL)
    summaries, changes, files, checks = [], [], {}, []
    for model in MODELS:
        unit = cal['units'][model]['unit']; models = read(registry_path(model))
        verify_files({m['checkpoint']:m['sha256'] for m in models if m['checkpoint']})
        prior_path = (PRIOR/'routing_final_evaluation/routing_per_example.jsonl' if model==MODELS[0]
                      else PRIOR/'breadth_evaluation'/model/'routing/per_example.jsonl')
        prior = pd.DataFrame(lines(prior_path))
        if 'evaluated_family' in prior: prior=prior[prior.evaluated_family=='routing']
        key = ['bank','color','view','arm','seed','anchor_id']
        prior = prior.set_index(key); assert prior.index.is_unique
        files[str(prior_path)] = sha(prior_path)
        records=[]
        for bank, path, rows, prefix, views in cases(model):
            if not path.exists():
                raise FileNotFoundError(path)
            for name in ('images.npy','texts.npy','index.jsonl','complete.json'):
                files[str(path/name)] = sha(path/name)
            meta = {r['anchor_id']:r for r in rows}
            for color, view in itertools.product(('red-blue','green-yellow','purple-orange'), views):
                v,t,idx=bank_arrays(path,'routing',color,view,prefix,meta)
                groups=source_clusters([meta[r['anchor_id']]['source_ids'] for r in idx])
                byarm={}
                for m in models:
                    tt=adapt(t,m['checkpoint'])
                    g=geometry(v,tt,unit,cal['kappa']-cal['tau_fraction'])
                    g['surplus']=g['response']-g['absolute_preference']
                    g['exchange_accuracy']=((g['response']+g['preference']>0).astype(float)+(g['response']-g['preference']>0))/2
                    # Independent score computation verifies the identity and replays saved scores.
                    scores=np.einsum('nid,njd->nij',np.asarray(v,float),np.asarray(tt,float))
                    marg=np.stack((scores[:,1,1]-scores[:,1,2], scores[:,2,2]-scores[:,2,1]),1)
                    err=max(abs(g['response']*unit-marg.mean(1)).max(), abs(g['preference']*unit-(marg[:,0]-marg[:,1])/2).max())
                    assert err<1e-12, err
                    saved=prior.loc[[(bank,color,view,m['arm'],m['seed'],r['anchor_id']) for r in idx]]
                    replay=max(abs(g['response']-saved.response.to_numpy()).max(), abs(g['absolute_preference']-saved.preference.to_numpy()).max())*unit
                    assert replay<2e-6, (model,bank,color,view,m,replay)
                    checks.append(dict(model=model,bank=bank,color=color,view=view,arm=m['arm'],seed=m['seed'],identity_error_raw=float(err),saved_replay_error_raw=float(replay)))
                    byarm[(m['arm'],m['seed'])]=np.stack([g[k] for k in METRICS],axis=1)
                    for j,r in enumerate(idx):
                        records.append(dict(model=model,bank=bank,color=color,view=view,arm=m['arm'],seed=m['seed'],anchor_id=r['anchor_id'],
                            source_ids=json.dumps(meta[r['anchor_id']]['source_ids']), group='+'.join(meta[r['anchor_id']]['objects']),
                            source_component=int(groups[j]), **{k:float(a[j]) for k,a in g.items()}))
                for arm in sorted({m['arm'] for m in models}):
                    values=np.stack([byarm[(arm,0 if arm=='F' else s)] for s in SEEDS])
                    for metric, stat in zip(METRICS,paired_intervals(values,groups)):
                        summaries.append(dict(model=model,bank=bank,color=color,view=view,arm=arm,n_anchors=len(idx),metric=metric,**stat))
                    if arm=='IS': continue
                    diff=np.stack([byarm[('IS',s)]-byarm[(arm,0 if arm=='F' else s)] for s in SEEDS])
                    for metric, stat in zip(METRICS,paired_intervals(diff,groups)):
                        changes.append(dict(model=model,bank=bank,color=color,view=view,contrast='IS-'+arm,n_anchors=len(idx),metric=metric,paired=True,**stat))
            print('GEOMETRY',model,bank,'complete',flush=True)
        pd.DataFrame(records).to_csv(dest/f'{model}_per_example.csv',index=False,mode='x')
        del prior, records
    jsonl(dest/'summary.jsonl',clean(summaries)); jsonl(dest/'paired_changes.jsonl',clean(changes))
    jsonl(dest/'replay_checks.jsonl',checks); dump(dest/'input_hashes.json',files)
    pd.DataFrame(summaries).to_csv(dest/'summary.csv',index=False,mode='x')
    pd.DataFrame(changes).to_csv(dest/'paired_changes.csv',index=False,mode='x')
    report=['# Final-checkpoint embedding geometry', '',
        'All saved final frozen, ranking, IS and available term-removal controls; all three trained seeds. Every original pilot/reserve and fresh same-rule confirmation view and all three color pairs. Post-specified descriptive follow-up; no checkpoint or subset selection.', '',
        'e = ½⟨v_rb−v_br, t_rb−t_br⟩ and b = ⟨(v_rb+v_br)/2, t_rb−t_br⟩. Response is the product of image-difference norm, text-difference norm, and their alignment. Scores and saved response/preference measurements are independently replayed. Image embeddings are unchanged by this text-side patch. Angles at zero difference norm are undefined, not zero.', '',
        '| Model | Bank/view (red–blue) | Arm | e | |b| | Alignment | Text difference norm | Exchange accuracy |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    ss=pd.DataFrame(summaries)
    for (model,bank,view,arm),g in ss[(ss.color=='red-blue')&ss.arm.isin(['F','R','IS'])].groupby(['model','bank','view','arm']):
        v=g.set_index('metric')['mean']
        report.append(f'| {model} | {bank}/{view} | {arm} | {v.response:.3f} | {v.absolute_preference:.3f} | {v.exchange_alignment:.3f} | {v.text_exchange_norm:.3f} | {100*v.exchange_accuracy:.2f}% |')
    report += ['', 'Per-seed means, sample standard deviations, paired changes, and 10,000-draw source-only and crossed seed/source-component percentile intervals are in the accompanying tables. Anchors sharing any source image are joined before resampling; the same source draw is used across arms and seeds. With only three seeds, these intervals do not establish robustness to an arbitrary training-seed population.', '',
        'This describes score geometry, not a causal location inside the network. The bound e_raw ≤ ||v_rb−v_br|| is necessary for an arbitrary unit-text repair; passing it does not prove a shared low-rank patch can achieve the bound. No conclusion about repair feasibility follows from a small nonzero image difference alone.']
    with (dest/'REPORT.md').open('x') as f:f.write('\n'.join(report)+'\n')
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},gpu=False,training=False,
        max_replay_error_raw=max(r['saved_replay_error_raw'] for r in checks),identity_checks=len(checks)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','stability','geometry']);args=ap.parse_args()
    torch.set_num_threads(4);log(OUT,'start',action=args.action,device='cpu')
    try: {'freeze':freeze,'stability':stability,'geometry':final_geometry}[args.action]()
    except BaseException as exc:log(OUT,'failed',action=args.action,error=repr(exc));raise
    log(OUT,'complete',action=args.action)


if __name__=='__main__':main()
