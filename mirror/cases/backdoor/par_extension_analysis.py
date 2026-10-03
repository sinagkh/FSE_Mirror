"""Paired source-level comparisons; immutable original-victim bug cohorts."""
import argparse
import json
import numpy as np
import mirror.cases.backdoor.test_data as data
import mirror.cases.backdoor.train as training
from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha


def paired_ci(differences, mask, rng):
    """Cross seed and source resampling, keeping arms and states paired.

    With one trained seed this is strictly an item interval, not seed uncertainty.
    The mask is fixed by the poisoned model, never by either repaired arm.
    """
    differences = differences[:, mask]
    s, n = differences.shape
    if not n:
        return None
    means = []
    for begin in range(0, 4000, 64):
        k = min(64, 4000-begin)
        indices = rng.integers(0, n, (k, n))
        seeds = rng.integers(0, s, (k, s))
        values = np.zeros((k, n), np.float32)
        for j in range(s):
            values += differences[seeds[:, j, None], indices]/s
        means.extend(values.mean(1).tolist())
    seed_means = differences.mean(1)
    return dict(mean=float(seed_means.mean()), ci95=np.quantile(means,[.025,.975]).tolist(),
        seed_values=seed_means.tolist(), sample_sd=float(seed_means.std(ddof=1)) if s>1 else None,
        n=n, seeds=s, interval='paired source bootstrap' if s==1 else 'crossed seed/source bootstrap', replicates=4000)


def analyze(case, bank):
    root = training.directory(case)/'evaluation'/bank
    summary = json.loads((root/'summary.json').read_text()); seeds = summary['seeds']
    output = root/'paired_analysis.json'
    if output.exists():
        assert json.loads(output.read_text())['summary_sha256'] == sha(root/'summary.json')
        return
    rows = {}
    for name, digest in summary['records_sha256'].items():
        path = root/(name+'_records.npz'); assert sha(path)==digest
        with np.load(path) as z:
            rows[name] = {k:z[k] for k in z.files}
    ref = rows['victim']; y = ref['labels']; target = 954 if bank in ('imagenetv2','banana1000') else 86
    for r in rows.values():
        assert np.array_equal(r['ids'],ref['ids']) and np.array_equal(r['labels'],y)
    competitors = ['ranking','clean_only','PAR'] + (['exact_filter','tolerant_filter'] if case=='stripes' else ['blend_inversion'])
    rng = np.random.default_rng(646711); results = {}
    for comp in competitors:
        pairs = [(rows['IS_seed'+str(seed)], rows[comp+'_seed'+str(seed)] if comp in ('ranking','clean_only') else rows[comp]) for seed in seeds]
        result = {}
        for gi, grid in enumerate(data.GRID):
            c, a = 1+gi*4, 2+gi*4
            fg = ref['pred']==y[:,None]; bug = fg[:,c]&~fg[:,a]
            arrays = {k:[] for k in ('clean_accuracy','attacked_accuracy','asr','retained_repair','clean_regression','target_interaction_abs','allclass_interaction_abs')}
            for isr, cr in pairs:
                ig, cg = isr['pred']==y[:,None], cr['pred']==y[:,None]
                arrays['clean_accuracy'].append(ig[:,c].astype(float)-cg[:,c])
                arrays['attacked_accuracy'].append(ig[:,a].astype(float)-cg[:,a])
                arrays['asr'].append((isr['pred'][:,a]==target).astype(float)-(cr['pred'][:,a]==target))
                arrays['retained_repair'].append((ig[:,c]&ig[:,a]).astype(float)-(cg[:,c]&cg[:,a]))
                arrays['clean_regression'].append((~ig[:,c]).astype(float)-(~cg[:,c]))
                arrays['target_interaction_abs'].append(abs(isr['target_margin'][:,a]-isr['target_margin'][:,c])-abs(cr['target_margin'][:,a]-cr['target_margin'][:,c]))
                arrays['allclass_interaction_abs'].append(isr['allclass_interaction_abs'][:,gi,0]-cr['allclass_interaction_abs'][:,gi,0])
            masks = dict(asr=y!=target,target_interaction_abs=y!=target,retained_repair=bug,clean_regression=fg[:,c])
            result[grid] = {metric:paired_ci(np.stack(vals),masks.get(metric,np.ones(len(y),bool)),rng) for metric,vals in arrays.items()}
        results['IS_minus_'+comp] = result
    dump(output,dict(case=case,bank=bank,summary_sha256=sha(root/'summary.json'),
        source_sha256=sha(__file__),units='proportions for behavior, raw cosine for interactions',comparisons=results,
        caution='Multiple registered processing conditions and comparisons are reported descriptively; no family-wise significance claim.'))
    print('PAIRED ANALYSIS',case,bank,results['IS_minus_ranking']['identity'],flush=True)


def report(case):
    root = training.directory(case)
    lines = ['# '+case+' fixed-recipe extension', '', 'All results use the fixed final checkpoint. No new recipe selection used ImageNetV2.', '',
             'The released PAR defense is an operational comparison with different cleanup data and information.', '']
    for bank in ('development','banana87','banana1000','imagenetv2'):
        path = root/'evaluation'/bank/'summary.json'
        if not path.exists():
            lines += ['## '+bank, '', 'Not completed.', '']; continue
        d = json.loads(path.read_text())
        lines += ['## '+bank, '', f"Sources: {d['n']}; repair seeds: {d['seeds']}.", '',
                  '| Method | Clean | Attacked | ASR | Target absolute interaction | All-class absolute interaction |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, entry in d['methods'].items():
            s = entry['grid']['identity']
            asr = '—' if s['asr'] is None else f"{100*s['asr']:.2f}"
            interaction = '—' if s['target_interaction_abs'] is None else f"{s['target_interaction_abs']:.5f}"
            lines.append(f"| {name} | {100*s['clean_top1']:.2f} | {100*s['attacked_top1']:.2f} | {asr} | {interaction} | {s['allclass_interaction_abs']:.5f} |")
        lines += ['', 'The full processing grid, per-source records, and paired intervals are saved in this bank’s evaluation directory.', '']
    (root/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n')
    # Explicit review gate. The queue never silently launches more seeds.
    dump(root/'replication_pending.json',dict(status='development review required before seeds43/44',
        no_automatic_replication=True,development_summary_sha256=sha(root/'evaluation/development/summary.json')))


if __name__ == '__main__':
    command(); parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['analyze','report'])
    parser.add_argument('--case',choices=data.CASES,required=True)
    parser.add_argument('--bank',choices=['development','banana87','banana1000','imagenetv2'],default='development')
    args=parser.parse_args()
    analyze(args.case,args.bank) if args.action=='analyze' else report(args.case)
