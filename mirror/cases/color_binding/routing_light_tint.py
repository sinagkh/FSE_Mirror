"""One-seed, opacity-only routing sensitivity experiment; previous files read-only."""
import argparse
from collections import Counter
from dataclasses import asdict; from dataclasses import replace
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image; from PIL import ImageDraw
import torch
from torch.nn import functional as F

from mirror.core.io import ROOT; from mirror.core.io import dump; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import source; from mirror.cases.color_binding.repair_trainbank import annotations_for; from mirror.cases.color_binding.repair_trainbank import lines
from mirror.cases.color_binding.rendering import canvas_pair; from mirror.cases.color_binding.rendering import captions
from mirror.cases.color_binding.transparency_development import recolor
from mirror.cases.color_binding.generators import pixel_hash
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.cases.color_binding import routing_common_noise_pilot as cn
from mirror.cases.color_binding import routing_budget_replication as old
from mirror.cases.color_binding import ranking_followup_pilot as templates
from mirror.core import metrics
from mirror.core.repair import _state_hash; from mirror.core.repair import cache_sha256; from mirror.core.repair import derive_scales

OUT = ROOT / 'clip/interbind_routing_light40_seed42_20260930'
TRAIN = templates.TRAIN
CONFIRM = ROOT / 'clip/interbind_phase_bc_completion_20260923/same_rule_confirmation'
BASE = CONFIRM.parent
COLORS = (('red', 'blue'), ('green', 'yellow'), ('purple', 'orange'))
VIEWS = ('canvas', 'swapped_canvas')
ALPHA = .4
SEED = 42
INITIAL = old.OUT / 'seed42/IS/initial.pt'


def array_hash(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def bases(row):
    sources = [source(s) for s in row['sources']]
    h = max(s[0].shape[0] for s in sources)
    w = sum(s[0].shape[1] for s in sources)
    packed = np.zeros((h, w, 3), np.uint8)
    masks = []
    left = 0
    for rgb, mask, _ in sources:
        hh, ww = mask.shape
        packed[:hh, left:left+ww] = rgb
        m = np.zeros((h, w), bool)
        m[:hh, left:left+ww] = mask
        masks.append(m)
        left += ww
    return [canvas_pair(packed, masks, swapped=j == 1) for j in range(2)]


def render(row, colors=COLORS, alpha=ALPHA):
    images, names = [], []
    for view, (base, masks) in zip(VIEWS, bases(row)):
        assert not np.any(masks[0] & masks[1])
        for pair in colors:
            for a in pair:
                for b in pair:
                    image = recolor(recolor(base, masks[0], a, alpha), masks[1], b, alpha)
                    assert np.array_equal(image[~(masks[0] | masks[1])], base[~(masks[0] | masks[1])])
                    images.append(image)
                    names.append('-'.join(pair)+'/'+view+'/'+a+'_'+b)
    return images, names, [pixel_hash(im) for im in images]


def preview():
    cv2.setNumThreads(1)
    rows = lines(TRAIN/'training_rows.jsonl')
    chosen = sorted((r for r in rows if r['objects'] == ['car', 'boat']), key=lambda r:r['anchor_id'])[:3]
    chosen += [next(r for r in rows if r['objects'] == list(pair)) for pair in
               [('bus', 'airplane'), ('cat', 'couch'), ('dog', 'sports ball'), ('person', 'bicycle')]]
    reference = {r['anchor_id']:r for r in lines(TRAIN/'features/index.jsonl')}
    dest = OUT/'preview'
    dest.mkdir(parents=True, exist_ok=False)
    selected = []
    for j, row in enumerate(chosen):
        high, names, hashes = render(row, COLORS[:2], .9)
        assert hashes[:8] == reference[row['anchor_id']]['pixel_sha256']
        panel = Image.new('RGB', (4*256, 3*288), 'white')
        draw = ImageDraw.Draw(panel)
        for iy, alpha in enumerate((0., .4, .9)):
            ims, _, _ = render(row, COLORS[:1], alpha)
            for ix, im in enumerate(ims[:4]):
                panel.paste(Image.fromarray(im), (ix*256, iy*288+24))
                draw.text((ix*256+5, iy*288+5), f'{alpha:.0%} tint: '+['RR','RB','BR','BB'][ix], fill='black')
        file = dest/f'{j:02d}_{"_".join(row["objects"]).replace(" ", "-")}.png'
        panel.save(file)
        selected.append(dict(anchor_id=row['anchor_id'], objects=row['objects'], file=str(file), sha256=sha(file),
                             original90_pixels_exact=True))
    jsonl(dest/'selection.jsonl', selected)
    dump(dest/'complete.json', dict(selection='First three lexical car/boat training IDs, then first training ID per other pair',
         no_model_scores=True, opacity=.4, all_original90_replays_exact=True))
    print('PREVIEW_COMPLETE', flush=True)


def freeze():
    assert (OUT/'preview/review.json').is_file()
    train = lines(TRAIN/'training_rows.jsonl')
    test = lines(CONFIRM/'rows.jsonl')
    assert len(train)==640 and len(test)==400
    assert not {i for r in train for i in r['source_ids']} & {i for r in test for i in r['source_ids']}
    paths = [Path(__file__), TRAIN/'training_rows.jsonl', CONFIRM/'rows.jsonl',
             INITIAL, prior.OUT/'normalization.json', prior.OUT/'training_config.json',
             templates.OUT/'training_encoding.json', OUT/'preview/review.json',
             ROOT/'clip/interbind_targeted_suppression_20260925/checkpoints_frozen.json']
    paths += [Path(__file__).with_name(n+'.py') for n in ('rendering', 'rendering_v2', 'transparency_development',
        'repair_trainbank', 'routing_relative_pilot', 'routing_common_noise_pilot', 'ranking_followup_pilot',
        'routing_budget_replication', 'repair', 'completion_metrics', 'scorers', 'feature_cache')]
    inputs = {str(p):sha(p) for p in paths}
    inputs.update({p:h for r in train+test for p,h in r['source_image_sha256'].items()})
    verify_files(inputs)
    dump(OUT/'protocol.json', dict(inputs=inputs, seed=42, tint_strength=.4, original_tint=.9,
        selection_basis='User requested much less intense colors; visual-only opacity choice, before model scores',
        analysis='Post-specified one-seed renderer sensitivity experiment; previously observed test sources',
        repair_site='Main-paper text-only rank64 output map; image and text encoders frozen',
        arms=['Frozen','Ranking40','IS40'], training_source_pairs=640, test_source_pairs=400,
        training_colors=COLORS[:2], evaluation_colors=COLORS, layouts=VIEWS,
        epochs=36, updates=1944, blocks_per_update=24, optimizer='Same AdamW LR2e-4 WD.01 clip1',
        initialization_sha256=sha(INITIAL), targets_and_weights='Original main-paper targets, unit and gradient weights unchanged; frozen-relative references recomputed from lighter training images',
        ranking='Main-paper matched-retention ranking: CE(100*s) plus identical complete IS retention',
        preservation='Original natural training captions; no new data or guards',
        checkpoint_selection='Fixed last; no loss search, seed expansion, or score-based exclusions',
        evaluations=['lighter held-out 400 pairs: all three colors and both layouts',
                     'original90 held-out same 400 pairs: all colors and both layouts',
                     'full SugarCrepe, ARO Attribution, COCO retrieval from original frozen-feature caches'],
        statistics='5000 paired anchor-bootstrap draws, fixed object-pair strata; both layouts retained; conditional on seed42',
        storage='Only final trained checkpoints, source manifests, cached features, scores, logs and preview images'))
    dump(OUT/'protocol_hash.json', dict(sha256=sha(OUT/'protocol.json')))
    print('FROZEN', sha(OUT/'protocol.json'), flush=True)


def verify():
    p = read(OUT/'protocol.json')
    assert sha(OUT/'protocol.json')==read(OUT/'protocol_hash.json')['sha256']
    verify_files(p['inputs'])


def encode():
    verify()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.mem_get_info()[0] > 12*1024**3, 'GPU not free; stop without interrupting others'
    annotations_for('train2017'); annotations_for('val2017')
    scorer = load_subject(prior.MODEL, device='cuda')
    train = lines(TRAIN/'training_rows.jsonl')
    vocab = sorted({tuple(r['objects']) for r in train})
    for bank, path, colors in [('train',TRAIN/'training_rows.jsonl',COLORS[:2]), ('test',CONFIRM/'rows.jsonl',COLORS)]:
        dest = OUT/'features'/bank
        if (dest/'complete.json').exists():
            verify_cache(dest)
            continue
        def prompts(family, pair):
            pair = tuple(pair)
            wrong = vocab[(vocab.index(pair)+1)%len(vocab)]
            return [g for c in colors for g in captions(family,pair,c)] + [prior.prior.objects(pair),prior.prior.objects(wrong)]
        begin = time.monotonic()
        print('ENCODING_START', bank, flush=True)
        cache_bank(dest, lines(path), scorer, lambda row:render(row, colors), prompts,
                   registry_path=DEFAULT_REGISTRY, input_hashes={str(OUT/'protocol.json'):sha(OUT/'protocol.json'),str(path):sha(path)},
                   image_batch_size=64, text_batch_size=64, render_workers=8,
                   details=dict(tint_strength=.4, colors=colors, layouts=VIEWS, geometry='unchanged',outside_masks_unchanged=True))
        print('ENCODING_COMPLETE', bank, round(time.monotonic()-begin,1), flush=True)
    dump(OUT/'encoding_complete.json', dict(files={str(OUT/'features'/b/'complete.json'):sha(OUT/'features'/b/'complete.json') for b in ('train','test')}))


def light_training():
    cache, spec, contexts, cfg = prior.load_training('cpu')
    path = OUT/'features/train'
    verify_cache(path)
    index = lines(path/'index.jsonl')
    original = lines(TRAIN/'features/index.jsonl')
    assert [r['anchor_id'] for r in index] == [r['anchor_id'] for r in original]
    images = np.load(path/'images.npy')
    texts = np.load(path/'texts.npy')
    vv, tt = [], []
    for row in index:
        for ci, pair in enumerate(COLORS[:2]):
            for view in VIEWS:
                names = ['-'.join(pair)+'/'+view+'/'+a+'_'+b for a in pair for b in pair]
                vv.append(images[row['image_offset']+np.array([row['state_names'].index(n) for n in names])])
                tt.append(texts[np.array(row['text_indices'][ci*4:ci*4+4]+row['text_indices'][8:10])])
    assert np.max(np.abs(np.stack(tt)-cache.texts.numpy()))<2e-6
    cache = replace(cache, images=torch.tensor(np.stack(vv)), bank_id='routing_light40_seed42')
    cfg = replace(cfg, bank_id=cache.bank_id, expected_cache_sha256=cache_sha256(cache),
                  scales=derive_scales(cache,spec,(0,1,2,3)), seed=42, epochs=36)
    cache = replace(cache, images=cache.images.cuda(), texts=cache.texts.cuda(), natural_texts=cache.natural_texts.cuda())
    return cache,spec,contexts,cfg


def make(cfg):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = cn.make_model(cfg,'cuda')
    model.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=False)['state_dict'])
    return model


def parts_ce(model,current,spec,contexts,cfg,ids):
    captured=[]
    hook=model.register_forward_hook(lambda m,a,z:captured.append(z))
    try:
        parts=prior.components(model,current,spec,contexts,cfg,ids)
    finally:
        hook.remove()
    assert len(captured)==1
    scores=current.images[ids]@captured[0][:,:4].transpose(1,2)
    ce=F.cross_entropy(100*scores.reshape(-1,4),torch.arange(4,device='cuda').repeat(len(ids)))
    return parts,ce


def train():
    verify(); prior.configure()
    cache,spec,contexts,cfg=light_training()
    forms=templates.template_cache(cache)
    weights=read(prior.OUT/'normalization.json')['weights']
    schedule=old.schedule_for(SEED)
    reps=templates.representation_schedule(SEED)
    model=make(cfg).train()
    ids=prior.paired_rows(torch.tensor(schedule[0][:24],device='cuda'))
    current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),torch.tensor(np.repeat(reps[0],2),device='cuda')])
    check=[]
    for arm in ('Ranking40','IS40'):
        torch.manual_seed(420001);torch.cuda.manual_seed_all(420001)
        parts,ce=parts_ce(model,current,spec,contexts,cfg,ids)
        loss=cn.objective(parts,weights) if arm=='IS40' else ce+prior.objective(parts,weights,'G')
        grads=torch.autograd.grad(loss,tuple(model.parameters()))
        assert all(torch.isfinite(g).all() for g in grads)
        check.append(dict(arm=arm,loss=float(loss.detach()),components={k:float(v.detach()) for k,v in parts.items()},
                          rng=array_hash(torch.cuda.get_rng_state().cpu().numpy())))
    assert check[0]['components']==check[1]['components'] and check[0]['rng']==check[1]['rng']
    dump(OUT/'preflight.json',dict(shared_forward_components_and_rng=True,gradient_checks=check,
         cache_shape=list(cache.images.shape),configuration=asdict(cfg),cached_text_replay_within_2e6=True))
    receipts=[]
    for arm in ('Ranking40','IS40'):
        dest=OUT/'runs'/arm;dest.mkdir(parents=True,exist_ok=False)
        model=make(cfg);initial_hash=_state_hash(model.state_dict())
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        history=[];start=time.monotonic()
        for ei,order in enumerate(schedule):
            current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),torch.tensor(np.repeat(reps[ei],2),device='cuda')])
            model.train();torch.manual_seed(420001+ei);torch.cuda.manual_seed_all(420001+ei)
            for j in range(0,1280,24):
                ids=prior.paired_rows(torch.tensor(order[j:j+24],device='cuda'))
                parts,ce=parts_ce(model,current,spec,contexts,cfg,ids)
                loss=cn.objective(parts,weights) if arm=='IS40' else ce+prior.objective(parts,weights,'G')
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                assert torch.isfinite(loss) and torch.isfinite(norm)
                opt.step()
                history.append(dict(epoch=ei+1,rows=ids.cpu().tolist(),loss=float(loss.detach()),ce=float(ce.detach()),
                                    gradient_norm=float(norm),components={k:float(v.detach()) for k,v in parts.items()}))
            if (ei+1)%6==0:print('TRAIN',arm,ei+1,round(time.monotonic()-start,1),flush=True)
        assert len(history)==1944
        ckpt=dest/'last.pt'
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuration=asdict(cfg),
                   weights=weights,seed=42,arm=arm,updates=1944,selection='fixed_last',
                   protocol_sha256=sha(OUT/'protocol.json')),ckpt)
        jsonl(dest/'history.jsonl',history)
        r=dict(name=arm,seed=42,checkpoint=str(ckpt),sha256=sha(ckpt),initial_state_hash=initial_hash,
               schedule_sha256=array_hash(schedule),representation_sha256=array_hash(reps),
               final_rng_sha256=array_hash(torch.cuda.get_rng_state().cpu().numpy()),updates=1944,
               seconds=time.monotonic()-start,first_components=history[0]['components'])
        dump(dest/'complete.json',r);receipts.append(r)
    for k in ('initial_state_hash','schedule_sha256','representation_sha256','final_rng_sha256','updates'):
        assert receipts[0][k]==receipts[1][k],k
    assert receipts[0]['first_components']==receipts[1]['first_components']
    dump(OUT/'training_complete.json',dict(models=receipts,matched_streams_initialization_budget_guards=True))


def registry():
    oldregs=read(ROOT/'clip/interbind_targeted_suppression_20260925/checkpoints_frozen.json')['models']
    refs=[]
    for name in ('IS_templates','RG_selected'):
        r=next(r for r in oldregs if r['name']==name and r['seed']==42)
        refs.append(dict(name='IS90' if name=='IS_templates' else 'Ranking90',seed=42,checkpoint=r['checkpoint'],sha256=r['sha256']))
    regs=[dict(name='Frozen',seed=0,checkpoint=None),*refs,*read(OUT/'training_complete.json')['models']]
    verify_files({r['checkpoint']:r['sha256'] for r in regs if r['checkpoint']})
    return regs


def evaluate():
    verify();torch.set_num_threads(4)
    regs=registry();dump(OUT/'models.json',regs)
    rows=lines(CONFIRM/'rows.jsonl');meta={r['anchor_id']:r for r in rows}
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False)
    summary=[];records=[];scores={};bootstrap=[]
    for opacity,path in [(40,OUT/'features/test'),(90,CONFIRM/'features'/prior.MODEL)]:
        verify_cache(path)
        for pair in COLORS:
            color='-'.join(pair);values={}
            for view in VIEWS:
                v,t,idx=metrics.bank_arrays(path,'routing',color,view)
                assert [r['anchor_id'] for r in idx]==[r['anchor_id'] for r in rows]
                assert len(idx)==400
                for reg in regs:
                    x=metrics.score_arrays(v,t,reg['checkpoint']);m=metrics.routing(x)
                    m['both_exchange_correct']=((x[:,1,1]>x[:,1,2]) & (x[:,2,2]>x[:,2,1])).astype(float)
                    scores[f'{opacity}/{color}/{view}/{reg["name"]}']=x
                    values.setdefault(reg['name'],[]).append(m)
                    records.extend(dict(opacity=opacity,color=color,view=view,name=reg['name'],anchor_id=r['anchor_id'],
                        pair='+'.join(meta[r['anchor_id']]['objects']),**{k:float(z[j]) for k,z in m.items()}) for j,r in enumerate(idx))
            means={name:{k:np.mean([a[k] for a in vv],axis=0) for k in vv[0]} for name,vv in values.items()}
            groups=np.array(['+'.join(r['objects']) for r in rows]);rng=np.random.default_rng(20260930)
            draw=np.zeros((5000,len(rows)))
            for g in sorted(set(groups)):
                ix=np.flatnonzero(groups==g)
                draw[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=5000)/len(rows)
            for name,m in means.items():
                summary.append(dict(opacity=opacity,color=color,name=name,n=400,**{k:float(a.mean()) for k,a in m.items()}))
            for a,b in [('IS40','Ranking40'),('IS40','Frozen'),('IS40','IS90')]:
                for key in means[a]:
                    delta=means[a][key]-means[b][key]
                    bootstrap.append(dict(opacity=opacity,color=color,comparison=a+' - '+b,metric=key,
                         mean=float(delta.mean()),ci95=np.quantile(draw@delta,[.025,.975]).tolist(),conditional_seed=42))
    np.savez_compressed(dest/'routing_scores.npz',**scores)
    jsonl(dest/'routing_per_example.jsonl',records)
    jsonl(dest/'routing_summary.jsonl',summary)
    jsonl(dest/'routing_contrasts.jsonl',bootstrap)
    dump(dest/'routing_complete.json',dict(model_registry_sha256=sha(OUT/'models.json'),matched_test_rows=400,
         source_row_sha256=sha(CONFIRM/'rows.jsonl'),all_colors_and_layouts_reported=True))
    print('ROUTING_EVALUATION_COMPLETE',flush=True)


def natural():
    import pandas as pd
    from mirror.cases.color_binding.completion_preservation import retrieval_ranks
    verify();torch.set_num_threads(4);regs=registry();summary=[]
    dest=OUT/'evaluation/natural';dest.mkdir(exist_ok=False)
    for bank in ('sugarcrepe','aro','coco'):
        path=prior.SUGAR if bank=='sugarcrepe' else BASE/'preservation/features'/prior.MODEL/bank
        ff=np.load(path/'features.npz');vv=ff['images'];rows=[];scores={}
        if bank=='sugarcrepe':
            index=read(path/'indices.json');ref=pd.read_csv(path/'frozen_seed0.csv')
            vi={s:i for i,s in enumerate(index['names'])};ti={s:i for i,s in enumerate(index['prompts'])}
            v=vv[[vi[s] for s in ref.filename]];pos=[ti[s] for s in ref.caption];neg=[ti[s] for s in ref.negative_caption]
            rows=ref.to_dict('records')
        else:
            rows=lines(BASE/'preservation/manifests'/bank/'rows.jsonl');v=vv
            if bank=='aro':pos=[r['positive'] for r in rows];neg=[r['negative'] for r in rows]
            else:owner=np.array([r['image_index'] for r in rows])
        for reg in regs:
            t=metrics.adapt(ff['texts'],reg['checkpoint'])
            if bank=='coco':
                tr,ir,_,_=retrieval_ranks(v,t,owner,128,device='cuda')
                scores[reg['name']+'/t2i']=tr;scores[reg['name']+'/i2t']=ir
                for direction, ranks in [('t2i',tr),('i2t',ir)]:
                    summary.append(dict(benchmark=bank,direction=direction,name=reg['name'],n=len(ranks),accuracy=float(np.mean(ranks<=1))))
            else:
                x=np.stack([np.einsum('nd,nd->n',v,t[pos]),np.einsum('nd,nd->n',v,t[neg])],axis=1)
                scores[reg['name']]=x
                summary.append(dict(benchmark=bank,name=reg['name'],n=len(x),accuracy=float(np.mean(x[:,0]>x[:,1]))))
        np.savez_compressed(dest/(bank+'_scores.npz'),**scores)
        dump(dest/(bank+'_rows.json'),rows)
        print('NATURAL_COMPLETE',bank,flush=True)
    jsonl(dest/'summary.jsonl',summary)


def report():
    summary=lines(OUT/'evaluation/routing_summary.jsonl')
    nat=lines(OUT/'evaluation/natural/summary.jsonl')
    regs=registry();names=[r['name'] for r in regs]
    text=['# Routing with lighter tint: seed 42','',
          'Tint is reduced from 90% to 40%, retaining 60% original RGB inside the same object masks. '
          'Main-paper text-only IS and matched-retention Ranking receive the same lighter counterfactuals. '
          'The original data, captions, initialization, weights, optimizer and 1,944-update budget are unchanged. '
          'IS90/Ranking90 are archived seed-42 models, re-scored without retraining. No manuscript edits.','',
          '640 training source pairs; 400 held-out source pairs. Both layouts stay together in the 5,000-draw, '
          'object-pair-stratified paired bootstrap. This is a post-specified single-seed sensitivity study; '
          'intervals do not measure variation across training seeds.','']
    for opacity in (40,90):
        text += [f'## Test tint {opacity}%','', '| Color / measure | '+' | '.join(names)+' |', '|---|'+'---:|'*len(names)]
        for color in ['red-blue','green-yellow','purple-orange']:
            selected={r['name']:r for r in summary if r['opacity']==opacity and r['color']==color}
            for key in ('exchange_accuracy','caption_accuracy','word1_accuracy','word2_accuracy','binding','cross','response','preference'):
                factor=100 if 'accuracy' in key else 1
                text.append('| '+color+' / '+key+' | '+' | '.join(f'{selected[n][key]*factor:.3f}' for n in names)+' |')
        text.append('')
    text += ['## Natural-image preservation','', '| Metric | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
    for bank,direction in [('sugarcrepe',None),('aro',None),('coco','t2i'),('coco','i2t')]:
        selected={r['name']:r for r in nat if r['benchmark']==bank and r.get('direction')==direction}
        text.append('| '+bank+(' '+direction+' R@1' if direction else '')+' | '+' | '.join(f'{selected[n]["accuracy"]*100:.2f}' for n in names)+' |')
    text += ['', '## Paired contrasts on the lighter red/blue test','', '| Contrast | Metric | Difference [95% CI] |','|---|---|---:|']
    for r in lines(OUT/'evaluation/routing_contrasts.jsonl'):
        if r['opacity']!=40 or r['color']!='red-blue' or r['metric'] not in ('exchange_accuracy','binding','cross'):continue
        factor=100 if 'accuracy' in r['metric'] else 1
        text.append(f'| {r["comparison"]} | {r["metric"]} | {r["mean"]*factor:+.3f} [{r["ci95"][0]*factor:+.3f}, {r["ci95"][1]*factor:+.3f}] |')
    text += ['', 'All per-example scores and all 16 conditional interaction measurements are retained under `evaluation/`. '
             'Behavior values are percentages; interactions use the unchanged frozen calibration unit. '
             'Reducing opacity restores source chroma as well as texture: the visual check is recorded separately, '
             'and no examples are removed based on model scores.']
    with (OUT/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('commands.jsonl','run.log')}
    dump(OUT/'complete.json',dict(files=files,seed=42,all_requested_stages_complete=True))
    print('COMPLETE',OUT/'REPORT.md',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preview','freeze','encode','train','evaluate','natural','report'])
    args=parser.parse_args();log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
