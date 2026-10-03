"""Bounded Phase-B natural failure accounting, with official labels unchanged."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
import requests
import torch
from mirror.cases.color_binding.completion_data import OUT as ROOTOUT; from mirror.cases.color_binding.completion_data import PLAN; from mirror.cases.color_binding.completion_data import configure
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.core.encoders import load_subject; from mirror.core.encoders import read_registry; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding.behavioral_pilot import OUT as PILOT; from mirror.cases.color_binding.behavioral_pilot import lines

OUT=ROOTOUT/'natural_grounding'
MANIFEST=ROOT/'clip/rebuttal_experiments_vlm/benchmarks/cola_routing/subset_freeze/full_official_examples.json'
COLORS={'red','blue','green','yellow','purple','orange','brown','white','black','gray','grey','pink'}


def category(row):
    a=re.findall(r'[a-z]+',row['caption_1'].lower());b=re.findall(r'[a-z]+',row['caption_2'].lower())
    ac=[x for x in a if x in COLORS];bc=[x for x in b if x in COLORS]
    if len(set(ac))==2 and len(ac)==2 and Counter(ac)==Counter(bc) and ac==bc[::-1] and [x for x in a if x not in COLORS]==[x for x in b if x not in COLORS]:
        return 'exact_color_word_exchange'
    return 'other_or_unresolved'


def connected_sources(rows):
    """Cluster COLA items linked by reuse of either natural image."""
    parents={}
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    for r in rows:
        a,b=(Path(r[k]).stem for k in ('image_1_path','image_2_path'))
        parents[find(a)]=find(b)
    return [find(Path(r['image_1_path']).stem) for r in rows]


def interval(values,clusters,n=2000):
    x=np.asarray(values,float);_,inv=np.unique(clusters,return_inverse=True)
    counts=np.bincount(inv);sums=np.bincount(inv,weights=x);m=len(counts)
    w=np.random.default_rng(20260923).multinomial(m,np.full(m,1/m),size=n)
    draws=(w@sums)/(w@counts)
    return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),n_items=len(x),n_clusters=m)


def prepare():
    rows=read(MANIFEST);assert len(rows)==210
    p=dict(inputs={str(x):sha(x) for x in [MANIFEST,PLAN,Path(__file__),DEFAULT_REGISTRY,
        DEFAULT_REGISTRY.with_name('activation_addendum.json'),PILOT/'natural_queries.jsonl']},
        models=[r['id'] for r in read_registry(DEFAULT_REGISTRY)['subjects']],
        metrics=['individual_text_decisions','individual_image_decisions','official_image_match','text_match','group_match'],
        subset_rule='Full official 210 COLA multi-object items. Text-only color-exchange tag is descriptive; no scores used.',
        taxonomy='Exact word exchange is a compatibility tag, not proof of causal leakage; all other failures unresolved.',
        uncertainty='2000 bootstrap draws over connected components of reused natural images',
        no_human_labels=True,no_training=True,no_selection=True)
    dump(OUT/'protocol.json',p)
    # Reuse existing files; download missing official URLs to a new directory.
    tasks={}
    for r in rows:
        for i in (1,2):tasks[r[f'image_{i}_path']]=r[f'image_{i}_url']
    def resolve(item):
        original,url=item;path=Path(original)
        if not path.is_file():
            path=OUT/'images'/Path(original).name;path.parent.mkdir(parents=True,exist_ok=True)
            response=requests.get(url,timeout=60);response.raise_for_status()
            with path.open('xb') as f:f.write(response.content)
        with Image.open(path) as im:im.verify()
        return original,dict(path=str(path),sha256=sha(path),url=url,reused=str(path)==original)
    with ThreadPoolExecutor(max_workers=4) as pool:mapping=dict(pool.map(resolve,tasks.items()))
    clusters=connected_sources(rows)
    selected=[]
    for r,cluster in zip(rows,clusters):
        x={**r,'compatibility':category(r),'source_cluster':cluster}
        for i in (1,2):x[f'resolved_image_{i}']=mapping[r[f'image_{i}_path']]
        selected.append(x)
    jsonl(OUT/'items.jsonl',selected);dump(OUT/'download_manifest.json',mapping)
    dump(OUT/'prepared.json',dict(protocol_sha256=sha(OUT/'protocol.json'),items_sha256=sha(OUT/'items.jsonl'),
        n_items=len(rows),n_images=len(mapping),n_clusters=len(set(clusters)),
        taxonomy_counts=dict(Counter(r['compatibility'] for r in selected)),no_model_scores=True))


def score(model):
    configure();p=read(OUT/'protocol.json');verify_files(p['inputs'])
    assert model in p['models'];assert sha(OUT/'items.jsonl')==read(OUT/'prepared.json')['items_sha256']
    rows=lines(OUT/'items.jsonl');dest=OUT/model
    if (dest/'complete.json').exists():verify_files(read(dest/'complete.json')['files']);return
    if dest.exists():raise FileExistsError(dest)
    free,_=torch.cuda.mem_get_info()
    if free<8*1024**3:raise RuntimeError('GPU busy')
    paths=sorted({r[f'resolved_image_{i}']['path'] for r in rows for i in (1,2)})
    for r in rows:
        for i in (1,2):
            x=r[f'resolved_image_{i}'];assert sha(x['path'])==x['sha256']
    prompts=list(dict.fromkeys(r[f'caption_{i}'] for r in rows for i in (1,2)))
    scorer=load_subject(model,device='cuda')
    v=scorer.encode_images([Image.open(x).convert('RGB') for x in paths],batch_size=32)
    t=scorer.encode_texts(prompts,batch_size=64)
    x=scorer.scores(v,t).numpy();vi={s:i for i,s in enumerate(paths)};ti={s:i for i,s in enumerate(prompts)}
    records=[]
    for r in rows:
        z=x[np.ix_([vi[r[f'resolved_image_{i}']['path']] for i in (1,2)],[ti[r[f'caption_{i}']] for i in (1,2)])]
        text=np.diag(z)>z[np.arange(2),1-np.arange(2)]
        image=np.diag(z)>z[1-np.arange(2),np.arange(2)]
        records.append({k:r[k] for k in ('example_id','caption_1','caption_2','compatibility','source_cluster')}|dict(model=model,
            scores=z.tolist(),individual_text_decisions=float(text.mean()),individual_image_decisions=float(image.mean()),
            official_image_match=bool(image.all()),text_match=bool(text.all()),group_match=bool(text.all() and image.all()),
            causal_category='unresolved'))
    jsonl(dest/'per_example.jsonl',records)
    with (dest/'features.npz').open('xb') as f:np.savez_compressed(f,images=v.numpy(),texts=t.numpy())
    dump(dest/'index.json',dict(images=paths,texts=prompts))
    dump(dest/'complete.json',dict(files={str(f):sha(f) for f in dest.iterdir() if f.is_file()},
         official_labels_unchanged=True,model=model,protocol_sha256=sha(OUT/'protocol.json')))
    print('COLA',model, {k:np.mean([r[k] for r in records]) for k in p['metrics']},flush=True)
    del scorer;gc.collect();torch.cuda.empty_cache()


def summarize():
    p=read(OUT/'protocol.json');verify_files(p['inputs']);summaries=[];allrows=[]
    for model in p['models']:
        d=OUT/model;verify_files(read(d/'complete.json')['files']);rows=lines(d/'per_example.jsonl');allrows+=rows
        for metric in p['metrics']:
            summaries.append(dict(model=model,metric=metric,**interval([r[metric] for r in rows],[r['source_cluster'] for r in rows])))
    jsonl(OUT/'summary.jsonl',summaries)
    jsonl(OUT/'all_failures.jsonl',[r for r in allrows if r['individual_text_decisions']<1 or r['individual_image_decisions']<1])
    natural=lines(PILOT/'natural_queries.jsonl');ns=[];failures=[]
    labels={r['id']:r for r in lines(ROOT/'clip/interbind_natural_localizer_quality_20260922/pilot/accepted_rows.jsonl')}
    for row in natural:
        if row['hits']['1']:continue
        top=labels[row['top_id']]
        # An annotated other object matching the query is co-occurrence, not causation.
        other=top.get('other_labeled_color_objects',[])
        failures.append(dict(model=row['model'],view=row['view'],query_id=row['query_id'],noun=row['noun'],query_color=row['color'],
            selected_id=row['top_id'],selected_object_color=top['color'],
            other_labeled_objects=other,causal_category='unresolved',
            inference='Selected named-object color disagrees with explicit dataset label; mechanism not identified by retrieval alone'))
    for (model,view),g in pd.DataFrame(natural).groupby(['model','view']):
        ns.append(dict(model=model,view=view,n_queries=len(g),n_hit1=sum(r['1'] for r in g.hits),n_hit5=sum(r['5'] for r in g.hits),
            hit1=float(np.mean([r['1'] for r in g.hits])),hit5=float(np.mean([r['5'] for r in g.hits])),
            interval='Not independent query trials; shared galleries. Descriptive only.'))
    jsonl(OUT/'vg_summary.jsonl',ns);jsonl(OUT/'vg_failures.jsonl',failures)
    report=['# Phase B natural grounding','',
        'Full official COLA multi-object labels; all seven frozen subjects. Individual decisions are primary; official image matching is also retained. Connected-image bootstrap accounts for reused images. These outcomes establish natural behavioral failures, not their causal attribution to an interaction.','',
        '| Model | Text decision % | Image decision % | Official image match % |',
        '|---|---:|---:|---:|']
    for model in p['models']:
        lookup={r['metric']:r for r in summaries if r['model']==model}
        report.append('| '+model+' | '+' | '.join(f"{100*lookup[k]['mean']:.2f}" for k in p['metrics'][:3])+' |')
    report+=['','VG per-noun retrieval failure records retain every observed error and explicit labels. Unexplained errors remain unresolved; no claim that every wrong answer is leakage or under-binding.',
        '',f'VG top-1 failure records across model/view/query: {len(failures)}. These are repeated queries, not independent images.',
        '', 'The earlier exact-rule natural two-object Visual Genome study produced zero validated items. Its no-go is retained; this program does not redefine it to manufacture a natural routing test.']
    with (OUT/'REPORT.md').open('x') as f:f.write('\n'.join(report)+'\n')
    dump(OUT/'complete.json',dict(files={str(OUT/n):sha(OUT/n) for n in ('summary.jsonl','all_failures.jsonl','vg_summary.jsonl','vg_failures.jsonl','REPORT.md')},
        natural_failure_grounding=True,causal_taxonomy_not_established=True,natural_two_object_vg='infeasible under frozen construction'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','score','summarize']);ap.add_argument('--model');a=ap.parse_args()
    log(OUT,'start',stage=a.action,model=a.model)
    try:
        if a.action=='score':score(a.model)
        else:globals()[a.action]()
    except BaseException as e:log(OUT,'failed',error=repr(e));raise
    log(OUT,'complete',stage=a.action,model=a.model)


if __name__=='__main__':main()
