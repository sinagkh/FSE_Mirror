"""Fixed ARO, COCO-val2017 and CIFAR-100 preservation: encode once, score all arms."""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset; from torch.utils.data import DataLoader
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.completion_data import OUT as ROOTOUT; from mirror.cases.color_binding.completion_data import configure
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY; from mirror.core.encoders import legacy_unit
from mirror.core.metrics import adapt
from mirror.cases.color_binding.completion_breadth import OUT as BREADTH
from mirror.cases.color_binding.completion_controls import OUT as CONTROLS
from mirror.cases.color_binding.completion_breadth_evaluate import benchmark_summary

OUT=ROOTOUT/'preservation'
PLAN=ROOT/'FSE_VLM/plan/27_completion_checks.md'
ARO=ROOT/'clip/aro_bows/aro_data'
COCO=ROOT/'clip/data/coco'
COCO_MANIFEST=ROOT/'clip/fse_repair_v5_results/retrieval/manifest.json'
BENCHMARKS=('aro','coco','cifar100')


class ImageRows(Dataset):
    def __init__(self, rows, preprocess, data=None):
        self.rows,self.preprocess,self.data=rows,preprocess,data
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        if self.data is not None:image=Image.fromarray(self.data[i])
        else:
            with Image.open(r['path']) as raw:image=raw.convert('RGB')
            if 'box' in r:image=image.crop(tuple(r['box']))
        return self.preprocess(image)


def prepare(benchmark):
    dest=OUT/'manifests'/benchmark
    if (dest/'complete.json').exists():return
    data=None
    if benchmark=='aro':
        raw=read(ARO/'visual_genome_attribution.json');rows=[];images=[];texts=[]
        for i,r in enumerate(raw):
            image=dict(path=str(ARO/'images'/r['image_path']),box=[r['bbox_x'],r['bbox_y'],r['bbox_x']+r['bbox_w'],r['bbox_y']+r['bbox_h']])
            images.append(image);texts += [r['false_caption'],r['true_caption']]
            words={a.strip().lower() for a in r['attributes']}
            rows.append(dict(example_id=str(i),source_id=r['image_path'],positive=2*i+1,negative=2*i,
                either_red_blue=bool(words&{'red','blue'}),exact_red_blue=words=={'red','blue'},attributes=r['attributes']))
        assert sum(r['exact_red_blue'] for r in rows)==83
        inputs={str(ARO/'visual_genome_attribution.json'):sha(ARO/'visual_genome_attribution.json')}
    elif benchmark=='coco':
        src=read(COCO_MANIFEST);ann=read(COCO/'annotations/captions_val2017.json')
        assert sha(COCO/'annotations/captions_val2017.json')==src['caption_annotation_sha256']
        # Reconstruct deterministic order from official annotations, then bind it.
        images=[dict(path=str(COCO/'val2017'/r['file_name']),image_id=r['id']) for r in sorted(ann['images'],key=lambda x:x['id'])]
        lookup={r['image_id']:i for i,r in enumerate(images)};captions=sorted(ann['annotations'],key=lambda x:x['id'])
        texts=[r['caption'] for r in captions]
        rows=[dict(example_id=str(r['id']),source_id=str(r['image_id']),image_index=lookup[r['image_id']]) for r in captions]
        assert len(images)==5000 and len(texts)==25014
        inputs={str(COCO_MANIFEST):sha(COCO_MANIFEST),str(COCO/'annotations/captions_val2017.json'):sha(COCO/'annotations/captions_val2017.json')}
    else:
        from torchvision.datasets import CIFAR100
        ds=CIFAR100(root=str(OUT/'datasets'),train=False,download=True)
        images=[dict(index=i) for i in range(len(ds))];data=ds.data
        texts=[f"a photo of a {name.replace('_',' ')}." for name in ds.classes]
        rows=[dict(example_id=str(i),source_id=str(i),target=int(y)) for i,y in enumerate(ds.targets)]
        inputs={str(OUT/'datasets/cifar-100-python/test'):sha(OUT/'datasets/cifar-100-python/test'),
                str(OUT/'datasets/cifar-100-python/meta'):sha(OUT/'datasets/cifar-100-python/meta')}
    assert data is not None or all(Path(r['path']).is_file() for r in images)
    dump(dest/'images.json',images);dump(dest/'texts.json',texts);jsonl(dest/'rows.jsonl',rows)
    if data is not None:
        with (dest/'pixels.npy').open('xb') as f:np.save(f,data)
    dump(dest/'complete.json',dict(benchmark=benchmark,inputs=inputs,plan_sha256=sha(PLAN),
        files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},n_images=len(images),n_texts=len(texts),selection=False))


def encode(model,benchmark):
    configure();prepare(benchmark);src=OUT/'manifests'/benchmark
    meta=read(src/'complete.json');verify_files(meta['inputs']);verify_files(meta['files'])
    dest=OUT/'features'/model/benchmark
    if (dest/'complete.json').exists():return
    assert torch.cuda.mem_get_info()[0]>8*1024**3,'GPU busy'
    scorer=load_subject(model,device='cuda');rows=read(src/'images.json')
    data=np.load(src/'pixels.npy',mmap_mode='r') if (src/'pixels.npy').exists() else None
    loader=DataLoader(ImageRows(rows,scorer.preprocess,data),batch_size=64,num_workers=8,pin_memory=True,shuffle=False)
    chunks=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for i,pixels in enumerate(loader):
            chunks.append(legacy_unit(scorer.model.encode_image(pixels.cuda(non_blocking=True))).cpu().numpy())
            if i%100==0:print('PRESERVATION_ENCODE',model,benchmark,i,len(loader),flush=True)
        texts=scorer.encode_texts(read(src/'texts.json'),batch_size=256).numpy()
    dest.mkdir(parents=True,exist_ok=False)
    with (dest/'features.npz').open('xb') as f:np.savez_compressed(f,images=np.concatenate(chunks),texts=texts)
    dump(dest/'complete.json',dict(model=model,benchmark=benchmark,manifest_sha256=sha(src/'complete.json'),
        files={str(dest/'features.npz'):sha(dest/'features.npz')},registry_sha256=sha(DEFAULT_REGISTRY),code_sha256=sha(Path(__file__)),
        fp16_encoder_fp32_normalized=True,no_model_scores=True))
    del scorer;gc.collect();torch.cuda.empty_cache()


def model_registry(model,family):
    path=CONTROLS/'models.json' if (model,family)==('openclip_laion_l14','routing') else BREADTH/model/family/'models.json'
    models=read(path);verify_files({r['checkpoint']:r['sha256'] for r in models if r['checkpoint']})
    return models,path


def retrieval_ranks(images,texts,caption_to_image,batch_size=256,device='cuda'):
    """All-gallery bidirectional ranks, strict-greater ties, best positive i2t."""
    v=torch.as_tensor(images,device=device);t=torch.as_tensor(texts,device=device)
    owner=torch.as_tensor(caption_to_image,device=device);tr=[];ir=[];tp=[];ip=[]
    with torch.inference_mode():
        for start in range(0,len(t),batch_size):
            s=t[start:start+batch_size]@v.T
            positive=s[torch.arange(len(s),device=device),owner[start:start+len(s)]]
            tr.extend((1+(s>positive[:,None]).sum(1)).cpu().tolist());tp.extend(positive.cpu().tolist())
        for start in range(0,len(v),batch_size):
            s=v[start:start+batch_size]@t.T
            is_positive=owner[None]==torch.arange(start,start+len(s),device=device)[:,None]
            positive=s.masked_fill(~is_positive,-torch.inf).max(1).values
            ir.extend((1+(s>positive[:,None]).sum(1)).cpu().tolist());ip.extend(positive.cpu().tolist())
    return np.asarray(tr),np.asarray(ir),np.asarray(tp),np.asarray(ip)


def evaluate(model,family,benchmark):
    configure();models,registry=model_registry(model,family);src=OUT/'features'/model/benchmark
    verify_files(read(src/'complete.json')['files']);ff=np.load(src/'features.npz');v=ff['images'];base=ff['texts']
    from mirror.cases.color_binding.behavioral_pilot import lines
    rows=lines(OUT/'manifests'/benchmark/'rows.jsonl');dest=OUT/'evaluation'/model/family/benchmark
    if (dest/'complete.json').exists():return
    dest.mkdir(parents=True,exist_ok=False);records=[];arms=list(dict.fromkeys(r['arm'] for r in models))
    for m in models:
        t=adapt(base,m['checkpoint']);common=dict(model=model,repair_family=family,arm=m['arm'],seed=m['seed'])
        if benchmark=='aro':
            pos=np.asarray([r['positive'] for r in rows]);neg=np.asarray([r['negative'] for r in rows])
            ps=np.einsum('nd,nd->n',v,t[pos]);ns=np.einsum('nd,nd->n',v,t[neg])
            records.extend(dict(**common,**r,positive_score=float(ps[i]),negative_score=float(ns[i]),accuracy=float(ps[i]>ns[i])) for i,r in enumerate(rows))
        elif benchmark=='cifar100':
            scores=v@t.T;order=np.argsort(-scores,axis=1,kind='stable');targets=np.asarray([r['target'] for r in rows])
            for i,r in enumerate(rows):records.append(dict(**common,**r,prediction=int(order[i,0]),target_score=float(scores[i,targets[i]]),
                top1=float(order[i,0]==targets[i]),top5=float(targets[i] in order[i,:5]),scores=scores[i].tolist()))
        else:
            owner=np.asarray([r['image_index'] for r in rows]);tr,ir,tp,ip=retrieval_ranks(v,t,owner)
            records.extend(dict(**common,**r,direction='t2i',rank=int(tr[i]),positive_score=float(tp[i]),recall1=float(tr[i]<=1),recall5=float(tr[i]<=5)) for i,r in enumerate(rows))
            image_ids=[r['image_id'] for r in read(OUT/'manifests/coco/images.json')]
            records.extend(dict(**common,example_id=str(i),source_id=str(image_ids[i]),direction='i2t',rank=int(ir[i]),positive_score=float(ip[i]),recall1=float(ir[i]<=1),recall5=float(ir[i]<=5)) for i in range(len(v)))
        print('PRESERVATION_SCORED',model,family,benchmark,m['arm'],m['seed'],flush=True)
    jsonl(dest/'per_example.jsonl',records);frame=pd.DataFrame(records);summary=[]
    if benchmark=='aro':groups=[('full',frame),('either_red_blue',frame[frame.either_red_blue]),('exact_red_blue',frame[frame.exact_red_blue])];metrics=['accuracy']
    elif benchmark=='cifar100':groups=[('full',frame)];metrics=['top1','top5']
    else:groups=[(d,frame[frame.direction==d]) for d in ('t2i','i2t')];metrics=['recall1','recall5']
    for name,sub in groups:summary+=benchmark_summary(sub,metrics,arms,dict(model=model,repair_family=family,benchmark=benchmark,category=name))
    jsonl(dest/'summary.jsonl',summary)
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},
        registry_sha256=sha(registry),feature_sha256=sha(src/'complete.json'),plan_sha256=sha(PLAN),code_sha256=sha(Path(__file__)),no_selection=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','encode','evaluate']);ap.add_argument('--model');ap.add_argument('--family');ap.add_argument('--benchmark',choices=BENCHMARKS,required=True)
    a=ap.parse_args();log(OUT,'start',**vars(a))
    try:
        if a.action=='prepare':prepare(a.benchmark)
        elif a.action=='encode':encode(a.model,a.benchmark)
        else:evaluate(a.model,a.family,a.benchmark)
    except BaseException as e:log(OUT,'failed',error=repr(e));raise
    log(OUT,'complete',**vars(a))


if __name__=='__main__':main()
