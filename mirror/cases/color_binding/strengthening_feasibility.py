"""Score-blind CPU-only external provenance and COCO support inventory."""
import argparse
from collections import Counter; from collections import defaultdict
import itertools
import json
import os
from pathlib import Path
import urllib.request

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError("Use CUDA_VISIBLE_DEVICES=''")
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log
from mirror.cases.color_binding.strengthening_cpu import OUT
from mirror.cases.color_binding.repair_trainbank import COCO; from mirror.cases.color_binding.repair_trainbank import lines
from mirror.cases.color_binding.routing_same_class_data import eligible; from mirror.cases.color_binding.routing_same_class_data import PAIRS


def fetch(url, path):
    request=urllib.request.Request(url,headers={'User-Agent':'InterBind-research-provenance/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw=response.read(8*1024*1024+1)
        if len(raw)>8*1024*1024: raise ValueError('Metadata download exceeds 8MiB bound')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:f.write(raw)
    log(OUT,'metadata_download',url=url,path=str(path),sha256=sha(path),bytes=len(raw))
    return raw


def external():
    dest=OUT/'published_patches'; dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'protocol.json',dict(script_sha256=sha(Path(__file__)),weights_search='Official repository tree, releases, linked issue and public Hugging Face model search; bounded metadata only',
        no_model_scores=True,no_training=True,no_substitute_backbone=True))
    results=[]
    for name,repo in [('labclip','kdariina/CLIP-not-BoW-unimodally'),('whatsup','amitakamath/whatsup_vlms')]:
        folder=dest/name
        commit=json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/main',folder/'commit.json'))['sha']
        tree=json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1',folder/'tree.json'))
        releases=json.loads(fetch(f'https://api.github.com/repos/{repo}/releases',folder/'releases.json'))
        paths=[r['path'] for r in tree['tree'] if r['type']=='blob']
        wanted=(['README.md','LICENSE','alignment/learning_alignment.py','alignment/coco_alignment.py','alignment/alignment_datasets.py','alignment/ARO.ipynb','alignment/sugarcrepe_eval.ipynb'] if name=='labclip'
                else ['README.md','LICENSE','dataset_zoo/aro_datasets.py','dataset_zoo/constants.py'])
        for p in wanted:
            if p in paths: fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{p}',folder/p)
        weights=[p for p in paths if p.endswith(('.pt','.pth','.ckpt','.safetensors','.bin','.npz','.npy'))]
        issues=json.loads(fetch(f'https://api.github.com/repos/{repo}/issues?state=all&per_page=100',folder/'issues.json'))
        results.append(dict(name=name,repository=f'https://github.com/{repo}',commit=commit,
            files_in_tree=len(paths),weight_files=weights,release_assets=[a['browser_download_url'] for r in releases for a in r.get('assets',[])],
            issues=[dict(number=r['number'],title=r['title'],body=r.get('body'),url=r['html_url']) for r in issues]))
    for query in ('LABCLIP','CLIP-not-BoW'):
        fetch('https://huggingface.co/api/models?search='+query+'&limit=30',dest/f'hf_search_{query}.json')
    dump(dest/'availability.json',results)
    dump(dest/'metadata_complete.json',dict(files={str(p):sha(p) for p in dest.rglob('*') if p.is_file()},gpu=False,weights_executed=False))


def exclusions():
    paths=[ROOT/'clip/interbind_routing_same_class_20260923/exclusions.json',
           ROOT/'data/color_binding/object_pairs/rows.jsonl',
           ROOT/'data/color_binding/confirmation/same_rule_confirmation/rows.jsonl',
           ROOT/'data/color_binding/train/rows.jsonl',
           ROOT/'clip/interbind_source_quality_20260922/pilot/accepted_rows.jsonl',
           ROOT/'clip/interbind_source_quality_20260922/reserve/accepted_rows.jsonl',
           ROOT/'clip/fse_background_preservation_results_20260920/manifests/training.jsonl']
    # Guard images are also training data and must not leak into new evaluation.
    blocked=set();hashes=set()
    for p in paths:
        rows=lines(p) if p.suffix=='.jsonl' else [read(p)]
        for r in rows:
            blocked.update(int(i) for i in r.get('source_ids',[]))
            if r.get('image_id') is not None:blocked.add(int(r['image_id']))
            if r.get('natural_guard'):blocked.add(int(r['natural_guard']['image_id']))
            if r.get('donor'):blocked.add(int(r['donor']['image_id']))
            hashes.update(r.get('source_image_sha256',{}).values());hashes.update(r.get('image_hashes',[]))
    return blocked, hashes, paths


def box_separation(a,b):
    x,y,w,h=a['bbox'];u,v,p,q=b['bbox']
    overlap=max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))
    return overlap/min(w*h,p*q)<=.05


def inventory():
    dest=OUT/'data_feasibility';dest.mkdir(parents=True,exist_ok=False)
    blocked,hashes,paths=exclusions();files=paths+[COCO/f'annotations/instances_{s}.json' for s in ('train2017','val2017')]
    dump(dest/'protocol.json',dict(inputs={str(p):sha(p) for p in files},script_sha256=sha(Path(__file__)),
        eligibility='Existing 2%-60% annotation mask area, min box side24px, >=1px from image edge, not crowd; distinct nouns; box intersection <=5% of smaller box',
        no_model_scores=True,purpose='Pre-score feasibility only; decoded mask and content hashes must be checked during construction',
        narrow_pairs=PAIRS,spatial_candidate_vocabulary=['bicycle','bus','car','cat','chair','dog','person','truck']))
    dump(dest/'exclusions.json',dict(source_ids=sorted(blocked),image_hashes=sorted(hashes)))
    singles=Counter();cooccurrences=defaultdict(list);sources=[]
    for split in ('train2017','val2017'):
        data=read(COCO/f'annotations/instances_{split}.json');images={r['id']:r for r in data['images']};cats={r['id']:r['name'] for r in data['categories']};byimage=defaultdict(dict)
        for a in data['annotations']:
            iid=a['image_id'];noun=cats[a['category_id']]
            if iid in blocked or not eligible(a,images[iid]):continue
            old=byimage[iid].get(noun)
            if old is None or (a['area'],-a['id'])>(old['area'],-old['id']):byimage[iid][noun]=a
        for iid,anns in byimage.items():
            im=images[iid]
            if not (COCO/split/im['file_name']).is_file():continue
            for noun,a in anns.items():
                singles[noun]+=1;sources.append(dict(noun=noun,image_id=iid,ann_id=a['id'],file=im['file_name'],coco_split=split))
            for na,nb in itertools.combinations(sorted(anns),2):
                a,b=anns[na],anns[nb]
                if not box_separation(a,b):continue
                cooccurrences[(na,nb)].append(dict(image_id=iid,coco_split=split,file=im['file_name'],objects=[na,nb],ann_ids=[a['id'],b['id']]))
        print('COCO_METADATA_COUNTED',split,len(byimage),flush=True)
    jsonl(dest/'eligible_single_objects.jsonl',sources)
    jsonl(dest/'eligible_cooccurrences.jsonl',[r for pair in sorted(cooccurrences) for r in cooccurrences[pair]])
    counts={'+'.join(pair):len(rows) for pair,rows in sorted(cooccurrences.items())}
    narrow={'+'.join(p):len(cooccurrences[tuple(sorted(p))]) for p in PAIRS}
    dump(dest/'counts.json',dict(single_objects=dict(sorted(singles.items())),cooccurrences=counts,narrow_pairs=narrow,
        n_unique_pair_images=len({r['image_id'] for rr in cooccurrences.values() for r in rr}),n_excluded_sources=len(blocked)))
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},no_scores=True,no_gpu=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['external','inventory']);a=p.parse_args();log(OUT,'start',action=a.action)
    try:{'external':external,'inventory':inventory}[a.action]()
    except BaseException as e:log(OUT,'failed',action=a.action,error=repr(e));raise
    log(OUT,'complete',action=a.action)


if __name__=='__main__':main()
