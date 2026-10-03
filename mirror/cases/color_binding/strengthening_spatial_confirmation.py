"""Frozen-recipe three-seed spatial confirmation and official natural retest."""
import argparse
import os
from pathlib import Path
import time
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA')
import numpy as np
import pandas as pd
from PIL import Image
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.strengthening_cpu import OUT as ROOTOUT; from mirror.cases.color_binding.strengthening_cpu import clean
from mirror.cases.color_binding.strengthening_spatial import OUT as DATA; from mirror.cases.color_binding.strengthening_spatial import MODEL; from mirror.cases.color_binding.strengthening_spatial import measurements; from mirror.cases.color_binding.strengthening_spatial import encode
from mirror.cases.color_binding import strengthening_spatial_repair as repair
from mirror.cases.color_binding.strengthening_labclip import intervals
from mirror.cases.color_binding.strengthening_whatsup import OUT as WHATSUP
from mirror.core.metrics import adapt
from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding.routing_relative_pilot import SUGAR

OUT=DATA/'confirmation_results'
ARMS=('F','R','IS','no_response','no_preference')


def selected():
    selection=DATA/'final_recipe_selection.json' if (DATA/'final_recipe_selection.json').exists() else DATA/'recipe_selection.json'
    decision=read(selection)
    assert decision['selected_version'] is not None, 'No recipe passed the development adoption rule'
    repair.OUT=DATA/decision['selected_version']
    assert not decision['confirmation_seen'] and decision['further_revisions_allowed']==0
    return repair.OUT


def seeds():
    selected()
    for seed in (43,44):repair.train(seed)
    dump(DATA/'three_seed_checkpoints_frozen.json',dict(selected_version=read(DATA/'recipe_selection.json')['selected_version'],
        checkpoints=[m for seed in (42,43,44) for m in read(repair.OUT/f'seed{seed}/models.json')],
        confirmation_not_scored=True,benchmarks_not_scored=True,selection_sha256=sha(DATA/'recipe_selection.json')))


def features():
    selected();assert (DATA/'three_seed_checkpoints_frozen.json').is_file()
    encode(['confirmation','heldout_pairs'])
    torch.set_num_threads(8);scorer=load_subject(MODEL,device='cpu');dest=WHATSUP/'features';dest.mkdir(exist_ok=False)
    verify_files(read(WHATSUP/'metadata_complete.json')['files'])
    images=lines(WHATSUP/'image_hashes.jsonl');verify_files({r['path']:r['sha256'] for r in images})
    rows=lines(WHATSUP/'rows.jsonl');names=list(dict.fromkeys(r['image'] for r in rows));prompts=list(dict.fromkeys(s for r in rows for s in r['captions']))
    vi=[];start=time.monotonic()
    for i in range(0,len(names),16):
        vi.append(scorer.encode_images([Image.open(p).convert('RGB') for p in names[i:i+16]],batch_size=16).numpy())
        if i%160==0:print('WHATSUP_CPU_ENCODE',i+len(vi[-1]),len(names),'seconds',round(time.monotonic()-start,1),flush=True)
    tt=scorer.encode_texts(prompts,batch_size=32).numpy()
    with (dest/'features.npz').open('xb') as f:np.savez_compressed(f,images=np.concatenate(vi),texts=tt)
    dump(dest/'index.json',dict(images=names,texts=prompts));dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},model=MODEL,
        registry_sha256=sha(DEFAULT_REGISTRY),device='cpu',precision='fp32',scores_formed=False,metadata_sha256=sha(WHATSUP/'metadata_complete.json')))


def summary(values,metrics,clusters,meta):
    result=[]
    for label in [*ARMS,'IS-F','IS-R','IS-no_response','IS-no_preference']:
        x=values['IS']-values[label[3:]] if label.startswith('IS-') else values[label]
        for metric,s in zip(metrics,intervals(x,clusters,draws=10000)):
            result.append(dict(**meta,comparison=label,metric=metric,paired=label.startswith('IS-'),**s))
    return result


def evaluate():
    if (DATA/'SUPERSEDED_BY_FIREWALL_V3.md').exists():
        raise RuntimeError('Superseded source bank: cached features may be reused, but benchmark only clean v3 checkpoints.')
    folder=selected();p=read(DATA/'three_seed_checkpoints_frozen.json');verify_files({m['checkpoint']:m['sha256'] for m in p['checkpoints']})
    OUT.mkdir(parents=True,exist_ok=False);records=[];stats=[]
    regs=[dict(arm='F',seed=0,checkpoint=None),*p['checkpoints']]
    selection=DATA/'final_recipe_selection.json' if (DATA/'final_recipe_selection.json').exists() else DATA/'recipe_selection.json'
    dump(OUT/'protocol.json',dict(script_sha256=sha(Path(__file__)),checkpoint_registry_sha256=sha(DATA/'three_seed_checkpoints_frozen.json'),
        selection_sha256=sha(selection),bootstrap='10000 crossed seed/source draws; same item draws across seeds and arms; fixed frozen regimes; Controlled-A all four photos of the same object pair share one source cluster',
        multiplicity='One primary IS-minus-frozen same-pair confirmation accuracy contrast; other outcomes and ablation/regime comparisons are secondary, descriptive pointwise95 intervals, not simultaneous discovery claims',
        score_selection=False,recipe_changes=False,device='cpu'))
    for bank in ('confirmation','heldout_pairs'):
        path=DATA/'features'/bank;verify_files(read(path/'complete.json')['files']);v=np.load(path/'images.npy');t=np.load(path/'texts.npy');rows=lines(path/'rows.jsonl')
        values={};frozen=measurements(v,t);metrics=[k for k in frozen if k!='regime']
        for reg in regs:
            met=measurements(v,adapt(t,reg['checkpoint']));values[(reg['arm'],reg['seed'])]=np.stack([met[k] for k in metrics],1)
            for j,r in enumerate(rows):records.append(dict(bank=bank,arm=reg['arm'],seed=reg['seed'],anchor_id=r['anchor_id'],source_ids=r['source_ids'],frozen_regime=frozen['regime'][j],
                **{k:str(a[j]) if k=='regime' else float(a[j]) for k,a in met.items()}))
        for cohort in ['all',*sorted(set(frozen['regime']))]:
            mask=np.ones(len(rows),bool) if cohort=='all' else frozen['regime']==cohort
            arrays={a:np.stack([values[(a,0 if a=='F' else s)][mask] for s in (42,43,44)]) for a in ARMS}
            stats+=summary(arrays,metrics,np.arange(mask.sum()),dict(benchmark='spatial_canvas',category=bank,cohort=cohort))
    jsonl(OUT/'canvas_per_example.jsonl',clean(records));jsonl(OUT/'canvas_summary.jsonl',clean(stats))
    natural=[];natural_stats=[]
    # Independent photographic spatial test: official labels and subsets frozen earlier.
    verify_files(read(WHATSUP/'features/complete.json')['files']);ff=np.load(WHATSUP/'features/features.npz');idx=read(WHATSUP/'features/index.json')
    vi={s:i for i,s in enumerate(idx['images'])};ti={s:i for i,s in enumerate(idx['texts'])};rows=lines(WHATSUP/'rows.jsonl')
    for reg in regs:
        t=adapt(ff['texts'],reg['checkpoint'])
        for row in rows:
            scores=ff['images'][vi[row['image']]]@t[[ti[s] for s in row['captions']]].T
            official=float(scores[0]>max(scores[1:]));lr=None
            if row['left_right']:
                indices=row.get('left_right_caption_indices',[0,1]);assert 0 in indices
                other=next(j for j in indices if j!=0);lr=float(scores[0]>scores[other])
            cluster=row['source_id']
            if row['benchmark']=='controlled_a':
                cluster=Path(row['image']).name
                for relation in ('_left_of_','_right_of_','_on_','_under_'):
                    cluster=cluster.replace(relation,'_REL_')
            natural.append(dict(**row,source_cluster=cluster,arm=reg['arm'],seed=reg['seed'],scores=scores.tolist(),official_accuracy=official,left_right_accuracy=lr))
    frame=pd.DataFrame(natural)
    for benchmark in ('coco_two_object','controlled_a'):
        for cat in ('full','left_right','left_right_without_known_annotation_issue'):
            g=frame[frame.benchmark==benchmark]
            if cat!='full':g=g[g.left_right]
            if cat.endswith('issue'):
                if benchmark!='controlled_a':continue
                g=g[~g.annotation_issue_known.astype(bool)]
            metrics=['official_accuracy']+([] if cat=='full' else ['left_right_accuracy'])
            ref=g[g.arm=='F'].set_index('example_id').sort_index();_,clusters=np.unique(ref.source_cluster,return_inverse=True)
            arrays={a:np.stack([g[(g.arm==a)&(g.seed==(0 if a=='F' else s))].set_index('example_id').loc[ref.index,metrics].to_numpy(float) for s in (42,43,44)]) for a in ARMS}
            natural_stats+=summary(arrays,metrics,clusters,dict(benchmark='whatsup_'+benchmark,category=cat,cohort='all'))
    jsonl(OUT/'whatsup_per_example.jsonl',clean(natural))
    # Full official SugarCrepe preservation; cached features only, no new fitting.
    ff=np.load(SUGAR/'features.npz');idx=read(SUGAR/'indices.json');ref=pd.read_csv(SUGAR/'frozen_seed0.csv');vi={s:i for i,s in enumerate(idx['names'])};ti={s:i for i,s in enumerate(idx['prompts'])}
    v=ff['images'][[vi[s] for s in ref.filename]];pos=[ti[s] for s in ref.caption];neg=[ti[s] for s in ref.negative_caption];ss=[];vv={}
    for reg in regs:
        t=adapt(ff['texts'],reg['checkpoint']);m=np.einsum('nd,nd->n',v,t[pos]-t[neg]);vv[(reg['arm'],reg['seed'])]=(m>0).astype(float)
        ss.extend(dict(arm=reg['arm'],seed=reg['seed'],example_id=str(r.subset)+'/'+str(r.example_id),source_id=r.filename,category=r.subset,margin=float(m[j]),accuracy=float(m[j]>0)) for j,r in ref.iterrows())
    for cat in ['full',*sorted(ref.subset.unique())]:
        mask=np.ones(len(ref),bool) if cat=='full' else (ref.subset==cat).to_numpy();_,cluster=np.unique(ref.filename[mask],return_inverse=True)
        arrays={a:np.stack([vv[(a,0 if a=='F' else s)][mask,None] for s in (42,43,44)]) for a in ARMS}
        natural_stats+=summary(arrays,['accuracy'],cluster,dict(benchmark='SugarCrepe',category=cat,cohort='all'))
    jsonl(OUT/'sugarcrepe_per_example.jsonl',ss);jsonl(OUT/'natural_summary.jsonl',clean(natural_stats))
    allstats=stats+natural_stats;pd.DataFrame(allstats).to_csv(OUT/'tables.csv',index=False,mode='x')
    text=['# Three-seed prospective spatial debugging', '',
        f"Frozen recipe selected only on{read(DATA/'data_complete.json')['counts']['development']} development anchors before opening confirmation or natural test scores. Seeds42,43,44; same initialization/budget per arm. Primary ranking is two-way CE plus historical0.2 embedding anchoring, not ranking plus IS guards.", '',
        '| Test | N | Frozen | Ranking | IS | IS−frozen [95% seed/source CI] |','|---|---:|---:|---:|---:|---:|']
    frame=pd.DataFrame(allstats)
    wanted=frame[((frame.benchmark=='spatial_canvas')&(frame.metric=='exchange_accuracy')&(frame.cohort=='all'))|((frame.benchmark.str.startswith('whatsup'))&(frame.category=='left_right')&(frame.metric=='left_right_accuracy'))|((frame.benchmark=='SugarCrepe')&(frame.category=='full'))]
    for (b,c),g in wanted.groupby(['benchmark','category']):
        v=g.set_index('comparison');diff=v.loc['IS-F'];lo,hi=diff.ci95_seed_source
        text.append(f"| {b}/{c} | {int(v.loc['F','n_items'])} | {100*v.loc['F','mean']:.2f} | {100*v.loc['R','mean']:.2f} | {100*v.loc['IS','mean']:.2f} | {100*diff['mean']:+.2f} [{100*lo:+.2f},{100*hi:+.2f}] |")
    text+=['','All results, seeds, controls, exact continuous response/preference, official broader categories, and known-label sensitivity are retained in tables. Synthetic/cutout confirmation and photographic natural tests are different evidence, not pooled. Bounded assistant visual inspection is not human validation. Pointwise bootstrap intervals are not a correction for multiple comparisons.']
    with (OUT/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    dump(OUT/'complete.json',dict(files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()},gpu=False,training_selection_on_test=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seeds','features','evaluate']);a=p.parse_args()
    torch.set_num_threads(4);log(ROOTOUT,'start',stage='spatial_confirmation_'+a.action)
    try:globals()[a.action]()
    except BaseException as e:log(ROOTOUT,'failed',stage='spatial_confirmation_'+a.action,error=repr(e));raise
    log(ROOTOUT,'complete',stage='spatial_confirmation_'+a.action)


if __name__=='__main__':main()
