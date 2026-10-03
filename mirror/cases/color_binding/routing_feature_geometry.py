"""Descriptive cached-feature diagnosis; no encoder, optimizer, or fitted oracle."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.routing_adapter_reaudit_v2 import OLD; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import CACHE; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import OUT as REAUDIT; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import CAL; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import MODEL; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import adapted_text
from mirror.cases.color_binding.behavioral_pilot import lines

OUT = ROOT/'clip/interbind_routing_geometry_20260923'
EDGES = ((0, 1), (0, 2), (1, 3), (2, 3))


def ratio(a, b):
    return np.divide(a, b, out=np.full_like(np.asarray(a, float), np.nan), where=b > 1e-12)


def geometry(v, t, unit, required_response):
    """RR,RB,BR,BB unit embeddings. Undefined angles remain NaN, never zero."""
    v, t = np.asarray(v, float), np.asarray(t, float)
    dv, dt, vm = v[:, 1]-v[:, 2], t[:, 1]-t[:, 2], (v[:, 1]+v[:, 2])/2
    nv, nt, nm = [np.linalg.norm(x, axis=1) for x in (dv, dt, vm)]
    iv, it = [np.mean([np.linalg.norm(x[:, i]-x[:, j], axis=1) for i, j in EDGES], axis=0) for x in (v, t)]
    dot = lambda a, b: np.einsum('nd,nd->n', a, b)
    e, b = .5*dot(dv, dt), dot(vm, dt)
    return dict(image_exchange_norm=nv, image_single_flip_norm=iv, image_exchange_to_single=ratio(nv, iv),
        text_exchange_norm=nt, text_single_flip_norm=it, text_exchange_to_single=ratio(nt, it),
        exchange_alignment=ratio(2*e, nv*nt), shared_image_preference_alignment=ratio(b, nm*nt),
        response=e/unit, preference=b/unit, absolute_preference=abs(b)/unit,
        current_difference_response_ceiling=.5*nv*nt/unit,
        unrestricted_unit_text_response_ceiling=nv/unit,
        necessary_capacity_test_fails=(nv < required_response*unit).astype(float),
        image_midpoint_exchange_inner_product=dot(vm, dv))


def freeze():
    # This is post-specified after the complete score-lattice re-audit, not a blind test.
    protocol = read(REAUDIT/'protocol.json'); verify_files(protocol['inputs'])
    complete = read(REAUDIT/'score_complete.json')
    checks = {str(REAUDIT/k): v for k, v in complete['files'].items()}
    for p in [Path(__file__), Path(__file__).with_name('routing_adapter_reaudit_v2.py'), CAL,
              REAUDIT/'protocol.json', REAUDIT/'models.json', REAUDIT/'score_complete.json']:
        checks[str(p)] = sha(p)
    for bank in ('routing_seen', 'routing_heldout'):
        for view in ('direct', 'swapped'):
            folder = OLD/f'audit_cache/{bank}'; meta = read(folder/f'{view}.json')
            checks[str(folder/f'{view}.npy')] = meta['sha256']
            checks[str(folder/'text.pt')] = meta['text_sha256']
            checks[str(OLD/f'manifests/{bank}.jsonl')] = meta['manifest_sha256']
            checks[str(folder/f'{view}.json')] = sha(folder/f'{view}.json')
    checks.update({str(CACHE/k): v for k, v in read(CACHE/'complete.json')['files'].items()})
    for m in read(REAUDIT/'models.json'):
        if m['checkpoint']: checks[m['checkpoint']] = m['sha256']
    verify_files(checks)
    dump(OUT/'protocol.json', dict(inputs=checks, scope='post-specified descriptive diagnosis of all previously re-audited models and rows',
        metrics='exchange and one-color-change embedding norms; exchange alignment; shared-image preference; exact e/b replay; necessary unit-text capacity bound',
        exclusions='none beyond frozen prior source list', fitting=False, training=False, gpu=False, reserve=False,
        inference='descriptive per-seed means and source quantiles, not neural-component causal attribution; no weak/strong cutoff search'))


def main():
    log(OUT, 'start')
    try:
        freeze(); torch.set_num_threads(2)
        cal = read(CAL); unit = cal['units'][MODEL]['unit']
        # Same sufficient response floor as the complete-context specification.
        required = cal['kappa']-cal['tau_fraction']
        models = read(REAUDIT/'models.json'); records = []

        def retain(bank, view, m, ids, v, t):
            measured = geometry(v, t, unit, required)
            for j, a in enumerate(ids):
                records.append(dict(bank=bank, view=view, arm=m['arm'], seed=m['seed'], anchor_id=a,
                    **{k: float(x[j]) for k, x in measured.items()}))

        for bank in ('routing_seen', 'routing_heldout'):
            manifest = lines(OLD/f'manifests/{bank}.jsonl')
            tx = torch.load(OLD/f'audit_cache/{bank}/text.pt', map_location='cpu', weights_only=False)
            ids = [tx['groups'].index(tuple(r['objects'])) for r in manifest]
            for m in models:
                with torch.inference_mode(): t = adapted_text(tx['base'], m).numpy()[ids, :4]
                for view in ('direct', 'swapped'):
                    v = np.load(OLD/f'audit_cache/{bank}/{view}.npy')
                    retain(bank, view, m, [r['anchor_id'] for r in manifest], v, t)
        wanted = {r['anchor_id'] for r in lines(REAUDIT/'frozen_direct_regimes.jsonl')}
        index = [r for r in lines(CACHE/'index.jsonl') if r['anchor_id'] in wanted]
        images, text = np.load(CACHE/'images.npy', mmap_mode='r'), np.load(CACHE/'texts.npy')
        for m in models:
            with torch.inference_mode(): t = adapted_text(text, m).numpy()
            for view in ('canvas', 'swapped_canvas', 'in_situ'):
                vv, tt = [], []
                for r in index:
                    ii = [i for i, n in enumerate(r['state_names']) if n.startswith('blend90_luminance/red-blue/'+view+'/')]
                    assert len(ii) == 4
                    vv.append(images[r['image_offset']+np.asarray(ii)]); tt.append(t[r['text_indices'][:4]])
                retain('new_pilot', view, m, [r['anchor_id'] for r in index], np.stack(vv), np.stack(tt))
        frame = pd.DataFrame(records); keys = ['bank', 'view', 'arm', 'seed', 'anchor_id']
        prior = pd.read_csv(REAUDIT/'per_example.csv').set_index(keys)
        current = frame.set_index(keys); assert current.index.is_unique and set(current.index) == set(prior.index)
        error = np.max(abs(current[['response', 'preference']]-prior.loc[current.index, ['response', 'preference']]).to_numpy())
        assert error*unit < 2e-6, error
        summaries = []
        for key, g in frame.groupby(['bank', 'view', 'arm']):
            for metric in frame.columns.difference(keys):
                per = g.groupby('seed')[metric].mean()
                vals = g[metric].dropna().to_numpy()
                summaries.append(dict(zip(['bank', 'view', 'arm'], key), metric=metric, n_anchors=int(g.anchor_id.nunique()),
                    per_seed={str(k): (float(v) if np.isfinite(v) else None) for k, v in per.items()},
                    mean=float(per.mean()) if len(vals) else None,
                    sample_sd=float(per.std(ddof=1)) if len(per)>1 and len(vals) else None,
                    source_model_quantiles=np.quantile(vals, [.05, .5, .95]).tolist() if len(vals) else None,
                    n_defined=len(vals)))
        with (OUT/'per_example.csv').open('x') as f: frame.to_csv(f, index=False)
        jsonl(OUT/'summary.jsonl', summaries)
        with (OUT/'summary.csv').open('x') as f: pd.DataFrame(summaries).to_csv(f, index=False)
        report = ['# Routing feature geometry: cached-feature diagnosis', '',
            'All nine existing L/14 adapters and frozen; same historical banks and 49 new pilot anchors. No fitting, encoder inference, GPU or reserve outcomes.', '',
            'For unit features, e = (v_rb-v_br)·(t_rb-t_br)/2 and b = ((v_rb+v_br)/2)·(t_rb-t_br). '
            'Thus e factors into image-difference size, text-difference size and their alignment. A small norm is not itself a behavioral failure; the sign and size of e relative to |b| determine the two exchange decisions. '
            'The image midpoint is orthogonal to the image difference for exactly unit vectors. This is score geometry, not proof of a neural root cause.', '',
            '| Bank/view | Arm | Image exchange/single-flip norm | Text exchange/single-flip norm | Exchange alignment | e (units) | |b| (units) |',
            '|---|---|---:|---:|---:|---:|---:|']
        for (bank, view, arm), g in frame.groupby(['bank', 'view', 'arm']):
            avg = g.mean(numeric_only=True)
            report.append(f"| {bank}/{view} | {arm} | {avg.image_exchange_to_single:.3f} | {avg.text_exchange_to_single:.3f} | {avg.exchange_alignment:.4f} | {avg.response:.4f} | {avg.absolute_preference:.4f} |")
        report += ['', 'All distributions and seed means are retained. Ratios use the mean norm of the four single-word/color changes, not endpoint correctness. '
            'For fixed unit image features, e cannot exceed ||v_rb-v_br|| with arbitrary unit text features. Failing that necessary capacity test rules out the current sufficient response floor for a text-only repair on that row; passing it does not prove a shared low-rank adapter can attain the floor or preserve other decisions. '
            'No diagnostic cutoffs were optimized and no model or example was selected based on these values.']
        with (OUT/'REPORT.md').open('x') as f: f.write('\n'.join(report)+'\n')
        dump(OUT/'complete.json', dict(protocol_sha256=sha(OUT/'protocol.json'), rows=len(frame), max_e_b_replay_error_raw=float(error*unit),
            files={n: sha(OUT/n) for n in ('per_example.csv', 'summary.jsonl', 'summary.csv', 'REPORT.md')}, training=False, gpu=False, reserve=False))
    except BaseException as exc:
        log(OUT, 'failed', error=repr(exc)); raise
    log(OUT, 'complete')


if __name__ == '__main__': main()
