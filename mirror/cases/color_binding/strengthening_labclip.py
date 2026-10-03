"""Released LABCLIP, matched OpenAI L/14 base; score-only CPU evaluation."""
import json
import os
from pathlib import Path
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA explicitly')
import numpy as np
import pandas as pd
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.strengthening_cpu import OUT as ROOTOUT; from mirror.cases.color_binding.strengthening_cpu import PRIOR; from mirror.cases.color_binding.strengthening_cpu import cases; from mirror.cases.color_binding.strengthening_cpu import registry_path; from mirror.cases.color_binding.strengthening_cpu import source_clusters; from mirror.cases.color_binding.strengthening_cpu import clean
from mirror.cases.color_binding.strengthening_feasibility import fetch
from mirror.core.metrics import bank_arrays; from mirror.core.metrics import adapt; from mirror.core.metrics import routing
from mirror.cases.color_binding.behavioral_pilot import lines; from mirror.cases.color_binding.behavioral_pilot import CAL
from mirror.cases.color_binding.routing_relative_pilot import SUGAR
from mirror.core.encoders import DEFAULT_REGISTRY

OUT=ROOTOUT/'published_patches/labclip_evaluation'
MODEL='openai_clip_l14'


def aligned(t,state):
    x=torch.as_tensor(np.asarray(t),dtype=torch.float32)
    with torch.inference_mode():
        x=x@state['linear.weight'].float().T
        return (x/x.norm(dim=-1,keepdim=True)).numpy()


def intervals(x,cluster,draws=2000):
    """Paired crossed source/seed bootstrap in bounded chunks (pilot default2000)."""
    ns,n,m=x.shape;nc=int(cluster.max())+1;counts=np.bincount(cluster)
    sums=np.stack([[np.bincount(cluster,weights=ss[:,j],minlength=nc) for j in range(m)] for ss in x])
    rng=np.random.default_rng(20260925);a=[];b=[]
    for start in range(0,draws,100):
        size=min(100,draws-start)
        w=rng.multinomial(nc,np.full(nc,1/nc),size=size).astype(float)
        sw=rng.multinomial(ns,np.full(ns,1/ns),size=size)/ns
        per=np.einsum('bc,smc->bsm',w,sums,optimize=True)/(w@counts)[:,None,None]
        a.extend(per.mean(1));b.extend((per*sw[:,:,None]).sum(1))
    means=x.mean(1)
    return [dict(mean=float(means[:,j].mean()),per_seed=means[:,j].tolist(),sample_sd=float(means[:,j].std(ddof=1)) if ns>1 else None,
        ci95_source=np.quantile(np.asarray(a)[:,j],[.025,.975]).tolist(),ci95_seed_source=np.quantile(np.asarray(b)[:,j],[.025,.975]).tolist(),
        n_source_clusters=nc,n_items=n) for j in range(m)]


def summarize(arrays,cluster,metrics,meta):
    rows=[]
    for label in ['F','LABCLIP','R','IS','LABCLIP-F','IS-F','IS-R','IS-LABCLIP']:
        if '-' in label:
            a,b=label.split('-');x=arrays[a]-arrays[b]
        else:x=arrays[label]
        for metric,s in zip(metrics,intervals(x,cluster)):
            rows.append(dict(**meta,comparison=label,metric=metric,paired='-' in label,**s))
    return rows


def main():
    torch.set_num_threads(4);log(ROOTOUT,'start',stage='released_labclip_audit')
    OUT.mkdir(parents=True,exist_ok=False)
    provenance=next(r for r in read(ROOTOUT/'published_patches/availability.json') if r['name']=='labclip')
    weight='alignment/weights/labclip_coco_neg_L14.pt'
    if weight not in provenance['weight_files']:raise ValueError('Official L14 weight absent')
    url=f"https://raw.githubusercontent.com/kdariina/CLIP-not-BoW-unimodally/{provenance['commit']}/{weight}"
    fetch(url,OUT/'labclip_coco_neg_L14.pt')
    models=[r for r in read(registry_path(MODEL)) if r['arm'] in ('F','R','IS')]
    verify_files({m['checkpoint']:m['sha256'] for m in models if m['checkpoint']})
    dump(OUT/'protocol.json',dict(script_sha256=sha(Path(__file__)),official_commit=provenance['commit'],official_weight_url=url,
        official_weight_sha256=sha(OUT/'labclip_coco_neg_L14.pt'),model=MODEL,subject_registry_sha256=sha(DEFAULT_REGISTRY),
        selection='Preassigned COCO-hard-negative LABCLIP L/14 because matching OpenAI L/14 caches and IS/R checkpoints exist; no other published map screened',
        base='OpenAI CLIP ViT-L/14 QuickGELU, never LAION L/14',same_base_models=models,
        score='normalize(text @ linear.weight.T); cosine; learned temperature excluded from raw interaction units and irrelevant to caption ordering',
        templates='Routing uses same normalized pooled-template feature interface as our adapters; natural benchmark captions are single official captions, matching published forward exactly',
        categories='All routing confirmation/pilot/reserve colors and views; full SugarCrepe and all seven categories; full ARO plus unchanged either-color and exact83 subsets',
        bootstrap='2000 paired source-cluster draws; crossed matched seed/source draws for IS/R; public map is one released checkpoint, not three trained seeds',
        benchmark_tuning=False,training=False,gpu=False))
    state=torch.load(OUT/'labclip_coco_neg_L14.pt',map_location='cpu',weights_only=True)
    assert set(state)=={'t','linear.weight'} and state['linear.weight'].shape==(768,768)
    rng=np.random.default_rng(9);smoke=torch.tensor(rng.normal(size=(8,768)),dtype=torch.float32)
    layer=torch.nn.Linear(768,768,bias=False);layer.weight.data.copy_(state['linear.weight'])
    with torch.inference_mode():ref=layer(smoke);ref=ref/ref.norm(dim=-1,keepdim=True)
    assert np.allclose(ref.numpy(),aligned(smoke.numpy(),state),atol=1e-7)
    dump(OUT/'loader_check.json',dict(keys=list(state),temperature=float(state['t']),max_forward_error=float(abs(ref.numpy()-aligned(smoke.numpy(),state)).max())))
    records=[];stats=[];hashes={}
    entries=[*models,dict(arm='LABCLIP',seed=0,checkpoint=None)]
    for bank,path,meta,prefix,views in cases(MODEL):
        lookup={r['anchor_id']:r for r in meta}
        for name in ('images.npy','texts.npy','index.jsonl','complete.json'):hashes[str(path/name)]=sha(path/name)
        for color in ('red-blue','green-yellow','purple-orange'):
            for view in views:
                v,t,idx=bank_arrays(path,'routing',color,view,prefix,lookup);values={}
                cluster=source_clusters([lookup[r['anchor_id']]['source_ids'] for r in idx])
                for entry in entries:
                    tt=aligned(t,state) if entry['arm']=='LABCLIP' else adapt(t,entry['checkpoint'])
                    x=np.einsum('nid,njd->nij',v,tt);met=routing(x,MODEL)
                    metrics=[k for k in met if not k.startswith('contrast/')]
                    values[(entry['arm'],entry['seed'])]=np.stack([met[k] for k in metrics],1)
                    for j,r in enumerate(idx):records.append(dict(model=MODEL,bank=bank,color=color,view=view,arm=entry['arm'],seed=entry['seed'],
                        anchor_id=r['anchor_id'],source_ids=lookup[r['anchor_id']]['source_ids'],scores=x[j].tolist(),**{k:float(a[j]) for k,a in met.items()}))
                arrays={a:np.stack([values[(a,0 if a in ('F','LABCLIP') else s)] for s in (42,43,44)]) for a in ('F','LABCLIP','R','IS')}
                stats+=summarize(arrays,cluster,metrics,dict(bank=bank,color=color,view=view))
        print('LABCLIP_ROUTING_SCORED',bank,flush=True)
    jsonl(OUT/'routing_per_example.jsonl',records);jsonl(OUT/'routing_summary.jsonl',stats)
    natural=[];natural_stats=[]
    for benchmark in ('sugarcrepe','aro'):
        folder=(PRIOR/'benchmark_features'/MODEL/benchmark if benchmark=='sugarcrepe' else PRIOR/'preservation/features'/MODEL/benchmark)
        hashes[str(folder/'features.npz')]=sha(folder/'features.npz');ff=np.load(folder/'features.npz')
        if benchmark=='sugarcrepe':
            idx=read(folder/'indices.json');ref=pd.read_csv(SUGAR/'frozen_seed0.csv')
            vi={s:i for i,s in enumerate(idx['names'])};ti={s:i for i,s in enumerate(idx['prompts'])}
            v=ff['images'][[vi[s] for s in ref.filename]];pos=[ti[s] for s in ref.caption];neg=[ti[s] for s in ref.negative_caption]
            rows=[dict(example_id=str(r.subset)+'/'+str(r.example_id),source_id=r.filename,category=r.subset) for _,r in ref.iterrows()]
        else:
            rows=lines(PRIOR/'preservation/manifests/aro/rows.jsonl');v=ff['images'];pos=[r['positive'] for r in rows];neg=[r['negative'] for r in rows]
        values={}
        for entry in entries:
            tt=aligned(ff['texts'],state) if entry['arm']=='LABCLIP' else adapt(ff['texts'],entry['checkpoint'])
            margin=np.einsum('nd,nd->n',v,tt[pos]-tt[neg]);acc=(margin>0).astype(float)
            values[(entry['arm'],entry['seed'])]=acc
            natural.extend(dict(benchmark=benchmark,arm=entry['arm'],seed=entry['seed'],**r,margin=float(margin[j]),accuracy=float(acc[j])) for j,r in enumerate(rows))
        categories={'full':np.ones(len(rows),bool)}
        if benchmark=='sugarcrepe':categories.update({c:np.array([r['category']==c for r in rows]) for c in sorted({r['category'] for r in rows})})
        else:categories.update({c:np.array([r[c] for r in rows]) for c in ('either_red_blue','exact_red_blue')})
        for c,mask in categories.items():
            sources=np.array([r['source_id'] for r in rows])[mask];_,cluster=np.unique(sources,return_inverse=True)
            arrays={a:np.stack([values[(a,0 if a in ('F','LABCLIP') else s)][mask,None] for s in (42,43,44)]) for a in ('F','LABCLIP','R','IS')}
            natural_stats+=summarize(arrays,cluster,['accuracy'],dict(benchmark=benchmark,category=c))
        print('LABCLIP_NATURAL_SCORED',benchmark,len(rows),flush=True)
    jsonl(OUT/'natural_per_example.jsonl',natural);jsonl(OUT/'natural_summary.jsonl',natural_stats);dump(OUT/'cache_hashes.json',hashes)
    table=['# Released LABCLIP patch audit', '',
        'Preassigned official COCO-hard-negative LABCLIP L/14, evaluated against its own OpenAI L/14 base and our existing same-base ranking and IS adapters. No training, checkpoint search or benchmark tuning. LABCLIP is a single released checkpoint; R/IS retain seeds42,43,44. Frozen and LABCLIP are repeated only for matched paired contrasts, not counted as three trained replicates.', '',
        '| Dataset/category | N | Frozen | LABCLIP | Ranking | IS |', '|---|---:|---:|---:|---:|---:|']
    for key,group in pd.DataFrame(natural_stats).groupby(['benchmark','category']):
        vals=group.set_index('comparison');table.append('| '+' / '.join(key)+f" | {int(vals.loc['F','n_items'])} | "+' | '.join(f"{100*vals.loc[a,'mean']:.2f}" for a in ('F','LABCLIP','R','IS'))+' |')
    table+=['','| Routing bank/view (red–blue) | Frozen | LABCLIP | Ranking | IS |','|---|---:|---:|---:|---:|']
    frame=pd.DataFrame(stats)
    for (bank,view),g in frame[(frame.color=='red-blue')&(frame.metric=='exchange_accuracy')].groupby(['bank','view']):
        vals=g.set_index('comparison')['mean'];table.append(f'| {bank}/{view} | '+' | '.join(f'{100*vals[a]:.2f}' for a in ('F','LABCLIP','R','IS'))+' |')
    table += ['', 'All colors, views, continuous mechanism metrics, per-example scores and paired source/seed confidence intervals are saved. Official caption orientation and original subset IDs are unchanged. Routing uses the existing pooled-template interface; natural benchmark scores use the official single-caption forward. These are different adaptation datasets/objectives, not a matched-training-budget causal comparison. LABCLIP uses COCO training data, so overlap with COCO-derived audit sources must not be described as a novel-data generalization claim for LABCLIP.']
    with (OUT/'REPORT.md').open('x') as f:f.write('\n'.join(table)+'\n')
    dump(OUT/'complete.json',dict(files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()},gpu=False,training=False))
    log(ROOTOUT,'complete',stage='released_labclip_audit')


if __name__=='__main__':main()
