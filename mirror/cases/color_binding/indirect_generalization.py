"""Frozen-checkpoint, source-paired syntax and familiar-noun recombination audit."""
import argparse
import ast
from collections import Counter; from collections import defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.core.metrics import adapt; from mirror.core.metrics import routing
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.core.encoders import DEFAULT_REGISTRY; from mirror.core.encoders import load_subject; from mirror.core.encoders import legacy_unit
from mirror.cases.color_binding.strengthening_statistics import intervals
from mirror.cases.color_binding.behavioral_pilot import lines

OUT = ROOT / 'clip/interbind_indirect_generalization_20260925'
PLAN = ROOT / 'FSE_VLM/plan/37_indirect_generalization_comparison.md'
OLD_SCRIPT = ROOT / 'mirror/utils/strong_paper/eval_routing_behavior_stress.py'
COMPARISON = ROOT / 'clip/interbind_ranking_comparison_20260925'
CONFIRM = ROOT / 'clip/interbind_phase_bc_completion_20260923/same_rule_confirmation'
TRAIN = ROOT / 'clip/interbind_routing_repair_pilot_20260923'
MODEL = 'openclip_laion_l14'
SEEDS = (42, 43, 44)
COLORS = (('red', 'blue'), ('green', 'yellow'))
VIEWS = ('canvas', 'swapped_canvas')
FAMILIES = ('train_templates', 'reverse_order', 'attribute_clause', 'left_right')
GROUPS = {**{name: list(range(3*i, 3*i+3)) for i, name in enumerate(FAMILIES)},
          'nonspatial_unseen': list(range(3, 9)), 'all_unseen': list(range(3, 12))}
ARMS = ('F', 'R_s1', 'R_s100', 'RG_s100_g0.25', 'IS')
METRICS = ('exchange_accuracy', 'caption_accuracy', 'both_correct',
           'reverse_image_choice', 'single_word_accuracy', 'strict_all',
           'consensus_caption_accuracy', 'consensus_exchange_accuracy',
           'binding', 'cross', 'response', 'preference')


@lru_cache(None)
def original_templates():
    """Execute only the archived, standalone template function, not its CLI."""
    tree = ast.parse(OLD_SCRIPT.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'template_prompts')
    namespace = {'colors_for_bits': lambda bits, colors: [colors[b] for b in bits]}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(OLD_SCRIPT), 'exec'), namespace)
    return namespace['template_prompts']


def prompts(objects, colors, view, family):
    result = []
    for bits in itertools.product((0, 1), repeat=2):
        nouns, state = tuple(objects), bits
        if family == 'left_right' and view == 'swapped_canvas':
            nouns, state = nouns[::-1], state[::-1]
        result.append(original_templates()(family, nouns, state, colors))
    return result  # four caption states, three strings each


def recombine(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row['objects'])].append(row)
    pairs = sorted(groups)
    assert len(pairs) == 4
    groups = {p: sorted(groups[p], key=lambda r: r['anchor_id']) for p in pairs}
    assert {len(v) for v in groups.values()} == {100}
    result = []
    for g, pair in enumerate(pairs):
        other = pairs[(g+1) % len(pairs)]
        for first, second in zip(groups[pair], groups[other]):
            sources = [first['sources'][0], second['sources'][1]]
            key = [(s['image_id'], s['ann_id']) for s in sources]
            aid = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
            hashes = first['source_image_sha256'] | second['source_image_sha256']
            wanted = {s['image_sha256'] for s in sources}
            result.append(dict(anchor_id='recombine_'+aid, family='routing',
                objects=[pair[0], other[1]], sources=sources,
                source_ids=[s['image_id'] for s in sources],
                original_anchor_ids=[first['anchor_id'], second['anchor_id']],
                source_image_sha256={p: h for p, h in hashes.items() if h in wanted}))
    return sorted(result, key=lambda r: (r['objects'], r['anchor_id']))


def configure():
    import cv2
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def freeze():
    rows = lines(CONFIRM/'rows.jsonl')
    train = lines(TRAIN/'training_rows.jsonl')
    assert len(rows) == 400 and len(train) == 640
    combined = recombine(rows)
    held_ids = {i for r in rows for i in r['source_ids']}
    training_ids = {i for r in train for i in r['source_ids']}
    training_ids.update(r['natural_guard']['image_id'] for r in train)
    assert len(held_ids) == 800 and not (held_ids & training_ids)
    assert Counter(i for r in combined for i in r['source_ids']) == Counter(held_ids)
    training_pairs = {frozenset(r['objects']) for r in train}
    vocab = set().union(*training_pairs)
    assert all(set(r['objects']) <= vocab and frozenset(r['objects']) not in training_pairs
               for r in combined)
    registry = read(COMPARISON/'checkpoints_frozen.json')['models']
    models = [dict(name='F', seed=0, checkpoint=None)] + registry
    assert Counter(r['name'] for r in registry) == Counter({a: 3 for a in ARMS if a != 'F'})
    verify_files({r['checkpoint']: r['sha256'] for r in registry})
    paths = [PLAN, Path(__file__), Path(__file__).parent/'tests/test_indirect_generalization.py',
             OLD_SCRIPT, COMPARISON/'protocol.json', COMPARISON/'checkpoints_frozen.json',
             CONFIRM/'rows.jsonl', TRAIN/'training_rows.jsonl',
             TRAIN/'features/text_groups.jsonl', TRAIN/'features/natural_templates.json',
             DEFAULT_REGISTRY, DEFAULT_REGISTRY.with_name('activation_addendum.json')]
    paths += [Path(__file__).with_name(n+'.py') for n in
              ('completion_metrics', 'strengthening_statistics', 'rendering', 'rendering_v3',
               'routing_relative_data', 'feature_cache', 'scorers', 'io')]
    original = CONFIRM/'features'/MODEL
    verify_cache(original)
    paths += [original/n for n in ('images.npy', 'texts.npy', 'index.jsonl', 'complete.json')]
    task_strings = {t for r in lines(TRAIN/'features/text_groups.jsonl') for t in r['templates']}
    natural_strings = {t for row in read(TRAIN/'features/natural_templates.json') for t in row}
    pairs = sorted({tuple(r['objects']) for r in rows + combined})
    templates = []
    for pair, color, view, family in itertools.product(pairs, COLORS, VIEWS, FAMILIES):
        for state, variants in enumerate(prompts(pair, color, view, family)):
            for j, text in enumerate(variants):
                templates.append(dict(objects=pair, color='-'.join(color), view=view,
                    family=family, state=state, variant=j, text=text))
    unseen = {r['text'] for r in templates if r['family'] != 'train_templates'}
    assert not unseen.intersection(task_strings | natural_strings)
    jsonl(OUT/'unseen_pair_rows.jsonl', combined)
    jsonl(OUT/'templates.jsonl', templates)
    dump(OUT/'protocol.json', dict(inputs={str(p): sha(p) for p in paths}, models=models,
        generated={str(OUT/n): sha(OUT/n) for n in ('unseen_pair_rows.jsonl', 'templates.jsonl')},
        seeds=SEEDS, colors=COLORS, views=VIEWS, groups=GROUPS,
        primary='IS-R_s100: seen_pairs, red-blue, both views, nonspatial_unseen exchange_accuracy',
        secondary='unseen pairings with pooled training-template captions; all registered families',
        cohorts={'seen_pairs': len(rows), 'unseen_pairs': len(combined)},
        source_overlap_with_training=0, exact_unseen_prompt_overlap=0,
        n_sources_per_bank=800, shared_sources_across_banks=True,
        unseen_pair_counts=dict(Counter('+'.join(r['objects']) for r in combined)),
        all_nouns_trained=True, all_recombined_pairs_untrained=True,
        templates_from_archived_paper=True, checkpoint_selection=False,
        draws=10000, bootstrap_seed=20260925, post_specified=True,
        original_confirmation_previously_scored=True, new_outcomes_seen=False))
    print('PROTOCOL_FROZEN', sha(OUT/'protocol.json'), flush=True)


def verify():
    p = read(OUT/'protocol.json')
    verify_files(p['inputs']); verify_files(p['generated'])
    verify_files({r['checkpoint']: r['sha256'] for r in p['models'] if r['checkpoint']})
    return p


def render(row):
    from mirror.cases.color_binding.repair_trainbank import source
    from mirror.cases.color_binding.routing_relative_data import paste_states
    ss = [source(s) for s in row['sources']]
    h, w = max(s[0].shape[0] for s in ss), sum(s[0].shape[1] for s in ss)
    packed = np.zeros((h, w, 3), np.uint8); masks = []; left = 0
    for rgb, mask, _ in ss:
        hh, ww = mask.shape
        packed[:hh, left:left+ww] = rgb
        mm = np.zeros((h, w), bool); mm[:hh, left:left+ww] = mask
        masks.append(mm); left += ww
    images, names, hashes, checks = [], [], [], []
    for color, view in itertools.product(COLORS, VIEWS):
        ii, nn, hh, cc = paste_states(packed, masks, color, view == 'swapped_canvas')
        images += ii; names += ['-'.join(color)+'/'+n for n in nn]; hashes += hh; checks += cc
    assert all(c['edit']['outside_unchanged'] for c in checks)
    return images, names, hashes, checks


def gallery():
    from PIL import Image; from PIL import ImageDraw
    from mirror.cases.color_binding.repair_trainbank import annotations_for
    verify(); configure(); annotations_for('train2017')
    rows = lines(OUT/'unseen_pair_rows.jsonl')
    chosen = [next(r for r in rows if tuple(r['objects']) == pair)
              for pair in sorted({tuple(r['objects']) for r in rows})]
    page = Image.new('RGB', (8*160, 4*200), 'white'); draw = ImageDraw.Draw(page)
    entries = []
    for j, row in enumerate(chosen):
        images, names, hashes, checks = render(row)
        draw.text((4, j*200+3), ' / '.join(row['objects'])+'  '+row['anchor_id'], fill='black')
        for k in range(8):
            page.paste(Image.fromarray(images[k]).resize((160,160)), (k*160, j*200+22))
            draw.text((k*160+2, j*200+184), names[k].replace('red-blue/', ''), fill='black')
        entries.append(dict(anchor_id=row['anchor_id'], objects=row['objects'], hashes=hashes,
                            outside_unchanged=all(c['edit']['outside_unchanged'] for c in checks)))
    dest=OUT/'gallery.png'
    with dest.open('xb') as stream: page.save(stream, format='PNG')
    jsonl(OUT/'gallery_selection.jsonl', entries)


def encode():
    from mirror.cases.color_binding.repair_trainbank import annotations_for
    from mirror.cases.color_binding.rendering import captions
    p = verify(); configure()
    review = read(OUT/'visual_review.json')
    assert review['accepted'] and review['gallery_sha256'] == sha(OUT/'gallery.png')
    if torch.cuda.mem_get_info()[0] < 12*1024**3:
        raise RuntimeError('GPU busy: other jobs will not be interrupted')
    scorer = load_subject(MODEL, device='cuda')
    templates = lines(OUT/'templates.jsonl')
    unique = list(dict.fromkeys(r['text'] for r in templates))
    text = scorer.encode_texts(unique, batch_size=128).numpy()
    with (OUT/'template_features.npy').open('xb') as stream: np.save(stream, text)
    dump(OUT/'template_index.json', unique)
    # Reconstruct the exact original pooled inputs from individual encodings.
    lookup = {s: i for i, s in enumerate(unique)}
    cache = CONFIRM/'features'/MODEL
    groups = lines(cache/'text_groups.jsonl'); old = np.load(cache/'texts.npy')
    errors=[]
    for group in groups:
        if not all(t in lookup for t in group['templates']): continue
        pooled = legacy_unit(torch.from_numpy(text[[lookup[t] for t in group['templates']]]).mean(0)).numpy()
        errors.append(float(np.max(np.abs(pooled-old[group['index']]))))
    assert errors and max(errors) < 2e-6
    print('TEXT_ENCODING', len(unique), 'pooled_replay_error', max(errors), flush=True)
    annotations_for('train2017'); checks=[]
    def renderer(row):
        ii, nn, hh, cc = render(row)
        checks.append(dict(anchor_id=row['anchor_id'], n_checks=len(cc),
            outside_unchanged=all(c['edit']['outside_unchanged'] for c in cc),
            qualified=sum(c['edit']['passes'] for c in cc)))
        return ii, nn, hh
    def captions_for(family, nouns):
        return [g for color in COLORS for g in captions(family, nouns, color)]
    cache_bank(OUT/'unseen_pair_features', lines(OUT/'unseen_pair_rows.jsonl'), scorer,
        renderer, captions_for, registry_path=DEFAULT_REGISTRY,
        input_hashes={str(OUT/n): sha(OUT/n) for n in ('protocol.json','unseen_pair_rows.jsonl','visual_review.json')},
        image_batch_size=64, text_batch_size=128, render_workers=8,
        details=dict(colors=COLORS, views=VIEWS, new_noun_combinations=True,
                     score_dependent_exclusions=False))
    jsonl(OUT/'pixel_checks.jsonl', sorted(checks, key=lambda r:r['anchor_id']))
    dump(OUT/'encoding_complete.json', dict(protocol_sha256=sha(OUT/'protocol.json'),
        max_original_text_replay_error=max(errors), n_replayed_groups=len(errors),
        files={str(OUT/n): sha(OUT/n) for n in ('template_features.npy','template_index.json',
            'unseen_pair_features/complete.json','pixel_checks.jsonl')},
        score_forward_count=0, n_images=6400))
    print('ENCODING_COMPLETE', flush=True)


def metrics(scores):
    """Per anchor, average template outcomes; no all-corners primary oracle.

    scores has axes [anchor, template, image_state, caption_state].
    """
    x=np.asarray(scores, float); n, k, _, _=x.shape
    flat=x.reshape(-1,4,4); r=routing(flat)
    correct=np.stack((x[:,:,1,1]>x[:,:,1,2],x[:,:,2,2]>x[:,:,2,1]),-1)
    full=x.diagonal(axis1=2,axis2=3)>np.where(np.eye(4,dtype=bool),-np.inf,x).max(3)
    consensus=routing(x.mean(1))
    result={m:r[m].reshape(n,k).mean(1) for m in
            ('exchange_accuracy','caption_accuracy','binding','cross','response','preference')}
    result.update(both_correct=correct.all(2).mean(1),
        reverse_image_choice=np.stack((x[:,:,1,1]>x[:,:,2,1],x[:,:,2,2]>x[:,:,1,2]),-1).mean((1,2)),
        single_word_accuracy=(r['word1_accuracy']+r['word2_accuracy']).reshape(n,k).mean(1)/2,
        strict_all=full.all((1,2)).astype(float),
        consensus_caption_accuracy=consensus['caption_accuracy'],
        consensus_exchange_accuracy=consensus['exchange_accuracy'])
    return np.stack([result[m] for m in METRICS],-1)


def feature_bank(bank, color, view):
    cache=CONFIRM/'features'/MODEL if bank=='seen_pairs' else OUT/'unseen_pair_features'
    meta=lines(CONFIRM/'rows.jsonl' if bank=='seen_pairs' else OUT/'unseen_pair_rows.jsonl')
    idx=lines(cache/'index.jsonl'); assert [r['anchor_id'] for r in idx]==[r['anchor_id'] for r in meta]
    images=np.load(cache/'images.npy',mmap_mode='r'); texts=np.load(cache/'texts.npy')
    vi=[];ti=[]; cindex=COLORS.index(tuple(color))
    for r in idx:
        names=['-'.join(color)+'/'+view+'/'+a+'_'+b for a,b in itertools.product(color,repeat=2)]
        vi.append(images[r['image_offset']+np.array([r['state_names'].index(s) for s in names])])
        ti.append(texts[r['text_indices'][cindex*4:(cindex+1)*4]])
    return np.stack(vi),np.stack(ti),meta


def summarize(values, meta, strata):
    packed={name:np.stack([values[(name, 0 if name=='F' else s)] for s in SEEDS]) for name in ARMS}
    contrasts=[('IS','F'),('IS','R_s1'),('IS','R_s100'),('IS','RG_s100_g0.25'),('R_s100','F')]
    labels=list(packed.items())+[(a+' - '+b,packed[a]-packed[b]) for a,b in contrasts]
    matrix=np.concatenate([v for _,v in labels],-1)
    statistics=intervals(matrix,np.arange(matrix.shape[1]),strata)
    return [dict(**meta,comparison=label,metric=metric,per_seed_ids=list(SEEDS),
                 **statistics[i*len(METRICS)+j])
            for i,(label,_) in enumerate(labels) for j,metric in enumerate(METRICS)]


def score():
    p=verify(); configure(); enc=read(OUT/'encoding_complete.json'); verify_files(enc['files'])
    assert read(OUT/'visual_review.json')['accepted']
    text=np.load(OUT/'template_features.npy'); strings=read(OUT/'template_index.json')
    lookup={s:i for i,s in enumerate(strings)}
    dest=OUT/'evaluation'; dest.mkdir(exist_ok=False)
    summary=[]; per_example=[]; raw=[]
    for bank in ('seen_pairs','unseen_pairs'):
        for color in COLORS:
            both=defaultdict(dict)
            for view in VIEWS:
                v, pooled, rows=feature_bank(bank,color,view)
                indices=np.array([[lookup[s] for family in FAMILIES
                    for variants in zip(*prompts(tuple(r['objects']),color,view,family))
                    for s in variants] for r in rows]).reshape(len(rows),12,4)
                t=text[indices]
                strata=np.array(['+'.join(r['objects']) for r in rows])
                grouped=defaultdict(dict); all_scores=[]; all_pooled=[]
                for reg in p['models']:
                    adapted=adapt(t,reg['checkpoint'])
                    x=np.einsum('nid,nkjd->nkij',v,adapted,optimize=True)
                    xp=np.einsum('nid,njd->nij',v,adapt(pooled,reg['checkpoint']),optimize=True)
                    all_scores.append(x.astype(np.float32)); all_pooled.append(xp.astype(np.float32))
                    for family, ix in list(GROUPS.items())+[('pooled_training_reference',None)]:
                        a=metrics(x[:,ix] if ix is not None else xp[:,None])
                        grouped[family][(reg['name'],reg['seed'])]=a
                        both[family].setdefault((reg['name'],reg['seed']),[]).append(a)
                        for j,row in enumerate(rows):
                            per_example.append(dict(bank=bank,color='-'.join(color),view=view,family=family,
                                name=reg['name'],seed=reg['seed'],anchor_id=row['anchor_id'],
                                source_ids=row['source_ids'],pair=strata[j],
                                **dict(zip(METRICS,a[j].tolist()))))
                name=bank+'_'+'-'.join(color)+'_'+view+'.npz'
                with (dest/name).open('xb') as stream:
                    np.savez_compressed(stream,scores=np.stack(all_scores),pooled_scores=np.stack(all_pooled))
                raw.append(dict(file=name,models=[dict(name=r['name'],seed=r['seed']) for r in p['models']],
                    bank=bank,color='-'.join(color),view=view,anchor_ids=[r['anchor_id'] for r in rows],
                    axes=['model','anchor','template','image_state','caption_state'],
                    template_families=FAMILIES,templates_per_family=3))
                for family, values in grouped.items():
                    summary += summarize(values,dict(bank=bank,color='-'.join(color),view=view,family=family),strata)
                print('SCORED',bank,'-'.join(color),view,flush=True)
            for family,values in both.items():
                summary += summarize({key:np.mean(a,axis=0) for key,a in values.items()},
                    dict(bank=bank,color='-'.join(color),view='both_layouts',family=family),strata)
    jsonl(dest/'summary.jsonl',summary);jsonl(dest/'per_example.jsonl',per_example);dump(dest/'score_index.json',raw)
    # Reproduce prior canonical metrics before interpreting any new comparison.
    old=lines(COMPARISON/'confirmation/summary.jsonl'); errors=[]
    for r in summary:
        if r['bank']!='seen_pairs' or r['view']=='both_layouts' or r['family']!='pooled_training_reference':continue
        if r['metric'] not in ('exchange_accuracy','caption_accuracy','binding','cross','response','preference','reverse_image_choice'):continue
        ref=next(a for a in old if all(a[k]==r[k] for k in ('color','view','comparison','metric')))
        errors.append(abs(ref['mean']-r['mean']))
    assert max(errors)<2e-5
    dump(dest/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        files={str(f):sha(f) for f in dest.iterdir() if f.is_file()},n_rows=len(per_example),
        original_metric_replay_max_error=max(errors),no_training=True))
    print('EVALUATION_COMPLETE',len(per_example),'original_replay_error',max(errors),flush=True)


def report():
    import csv
    from io import StringIO
    p=verify(); done=read(OUT/'evaluation/complete.json');verify_files(done['files'])
    rows=lines(OUT/'evaluation/summary.jsonl')
    def select(bank,color,view,family,label,metric):
        return next(r for r in rows if (r['bank'],r['color'],r['view'],r['family'],r['comparison'],r['metric'])
                    ==(bank,color,view,family,label,metric))
    table=['# Indirect-generalization comparison','','All settings and source IDs were frozen before new scores. '
        'Existing fixed checkpoints, seeds 42/43/44, 400 anchors per bank. Both banks share the same 800 '
        'source images; they are reported separately. No new training or manuscript edits.','',
        '## Primary and key secondary tests','',
        '| Test (red/blue) | Frozen | Original ranking | Scaled ranking | Ranking + retention | IS | IS − scaled [95% CI] |',
        '|---|---:|---:|---:|---:|---:|---:|']
    tests=[('seen_pairs','both_layouts','nonspatial_unseen','exchange_accuracy','Primary: unseen nonspatial wording, both layouts'),
           ('seen_pairs','canvas','all_unseen','exchange_accuracy','Nine unseen templates, direct layout'),
           ('seen_pairs','canvas','all_unseen','strict_all','Historical AU strict (36 decisions)'),
           ('seen_pairs','swapped_canvas','nonspatial_unseen','strict_all','Historical swap strict (24 decisions)'),
           ('unseen_pairs','both_layouts','pooled_training_reference','exchange_accuracy','New noun pairings, trained wording'),
           ('unseen_pairs','both_layouts','nonspatial_unseen','exchange_accuracy','New noun pairings + unseen wording')]
    for bank,view,family,metric,label in tests:
        values=[select(bank,'red-blue',view,family,a,metric) for a in ARMS]
        cells=[f"{100*r['mean']:.2f}"+(f" ± {100*r['sample_sd']:.2f}" if a!='F' else '') for a,r in zip(ARMS,values)]
        delta=select(bank,'red-blue',view,family,'IS - R_s100',metric);lo,hi=delta['ci95_seed_source']
        table.append('|'+label+'|'+'|'.join(cells)+f"|{100*delta['mean']:+.2f} [{100*lo:.2f}, {100*hi:.2f}]|")
    table += ['','Individual decisions are primary. The strict metrics require all four color-state decisions '
        'under all 9 (AU) or 6 (swap) templates. No all-corners rule selects examples. '
        'Both layouts and both evaluated colors were in training; the held-out axes are wording, sources '
        'and noun combinations. Intervals are paired, noun-pair-stratified, crossed seed/anchor bootstraps '
        'with 10,000 draws. Secondary intervals are pointwise.','',
        '## Every family, both layouts','',
        '| Bank | Colors | Wording | F | R | Scaled R | R + retention | IS | IS − scaled [95% CI] |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for bank,color,family in itertools.product(('seen_pairs','unseen_pairs'),map('-'.join,COLORS),list(GROUPS)+['pooled_training_reference']):
        vals=[select(bank,color,'both_layouts',family,a,'exchange_accuracy') for a in ARMS]
        d=select(bank,color,'both_layouts',family,'IS - R_s100','exchange_accuracy');lo,hi=d['ci95_seed_source']
        table.append('|'+bank+'|'+color+'|'+family+'|'+'|'.join(f"{r['mean']*100:.2f}" for r in vals)+
                     f"|{100*d['mean']:+.2f} [{100*lo:.2f}, {100*hi:.2f}]|")
    table += ['','## Provenance','',f"Protocol SHA256: `{sha(OUT/'protocol.json')}`.",
        f"Exact pooled-reference metric replay maximum error: {done['original_metric_replay_max_error']:.3g}.",
        'Full per-seed means, sample SD, source-only and seed/source intervals, all behavior and interaction '
        'metrics are in `evaluation/summary.jsonl`; anchor-level metrics in `evaluation/per_example.jsonl`; '
        'complete score matrices and their indices are alongside them. No new outcome was used to change '
        'any checkpoint, prompt, rendering, or sample rule.']
    with (OUT/'REPORT.md').open('x') as stream:stream.write('\n'.join(table)+'\n')
    with (OUT/'tables.csv').open('x') as stream:
        fields=['bank','color','view','family','comparison','metric','mean','sample_sd','per_seed',
                'ci95_source','ci95_seed_source','n_items','n_source_clusters']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    dump(OUT/'complete.json',dict(files={str(OUT/n):sha(OUT/n) for n in ('REPORT.md','tables.csv','evaluation/complete.json')},
         protocol_sha256=sha(OUT/'protocol.json'),paper_modified=False,training=False))
    print('\n'.join(table[:17]),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','gallery','encode','score','report'))
    args=parser.parse_args();log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
