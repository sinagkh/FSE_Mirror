"""Uniform 75% routing retests, including joint maps and released LABCLIP.

Raw scores remain cosines. SigLIP's frozen affine scale is applied only for
the original calibrated interaction units; no score-temperature fitting.
"""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from mirror.cases.color_binding import test_data as data
from mirror.cases.color_binding import results as stats
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files

base=data.base;p=data.p;a=base.ev.analysis;ind=data.ind;tt=base.ev.tt
OUT=data.OUT
STUDIES=('openclip_laion_l14','openai_clip_l14','google_siglip_b16_224','joint')
LAB=p.ROOT/'clip/interbind_strengthening_cpu_20260925/published_patches/labclip_evaluation/labclip_coco_neg_L14.pt'

def backbone(study):return 'openclip_laion_l14' if study=='joint' else study

def registry(study,seed,allarms=True):
    if study in ('joint',*data.MODELS):
        path=OUT/study/f'seed{seed}/models.json' if study=='joint' else OUT/study/'models.json'
        rows=[r for r in read(path) if r['seed']==seed]
    else:
        if seed==42:
            rows=[next(r for r in read(base.mid.OUT/'models.json') if r['name']=='Ranking'),
                  dict(read(base.OLD/'choice.json')['candidate']['model'],name='IS')]
        else:rows=[r for r in read(base.OUT/f'seed{seed}/models.json') if r['name'] in ('IS','Ranking')]
        rows += [r for r in read(OUT/study/'models.json') if r['seed']==seed]
    rows=sorted(rows,key=lambda r:(r['name']!='Ranking',r['name']!='IS',r['name']))
    if not allarms:rows=[r for r in rows if r['name'] in ('Ranking','IS','OrdinaryRanking')]
    if study=='openai_clip_l14':rows += [dict(name='LABCLIP',checkpoint=str(LAB),sha256=sha(LAB),seed=0,external_fixed=True)]
    verify_files({r['checkpoint']:r['sha256'] for r in rows})
    return [dict(name='Frozen',checkpoint=None,seed=0,external_fixed=True),*rows]

@lru_cache(None)
def checkpoint(path):return torch.load(path,map_location='cpu',weights_only=False)

def adapt(x,reg,side):
    if reg['checkpoint'] is None:return np.asarray(x)
    obj=checkpoint(reg['checkpoint']);isjoint=obj.get('mode')=='joint'
    if side=='image' and not isjoint:return np.asarray(x)
    t=torch.as_tensor(np.asarray(x)).float()
    with torch.inference_mode():
        if reg['name']=='LABCLIP':y=t@obj['linear.weight'].float().T
        else:
            state=obj['state_dict'];prefix=side+'_map.' if isjoint else ''
            aa,bb=state[prefix+'A.weight'].float(),state[prefix+'B.weight'].float()
            y=t+(t@aa.T)@bb.T
        return F.normalize(y,dim=-1).numpy()

def measurements(x,model):
    if model in data.MODELS:
        aff=read(data.breadth.OUT/model/'scale_preflight.json')
        native=np.asarray(x,float)*aff['native_scale']+aff['native_bias']
    else:native=x
    m=p.metrics.routing(native,model)
    m['both_correct']=((x[:,1,1]>x[:,1,2])&(x[:,2,2]>x[:,2,1])).astype(float)
    m['reverse_image_choice']=np.stack((x[:,1,1]>x[:,2,1],x[:,2,2]>x[:,1,2]),axis=1).mean(1)
    for ctx in p.metrics.all_contexts():
        if ctx['kind']=='unwanted':m['absolute/'+ctx['name']]=abs(m['contrast/'+ctx['name']])
    return m

def conditions(family):
    return tt.conditions(family) if family.endswith('_shift') else [(c,.75) for c,_ in data.olddata.ORIGINAL_CONDITIONS(family)]

@lru_cache(24)
def arrays(model,family,condition,view):
    folder=data.bank(model,family);idx=data.lines(folder/'index.jsonl')
    v=np.load(folder/'images.npy',mmap_mode='r');t=np.load(folder/'texts.npy')
    if family=='primary':colors=tuple(condition.split('-'));prefix=condition;ci=list(map('-'.join,p.COLORS)).index(condition)
    elif family.endswith('_shift'):colors=('red','blue');prefix=condition;ci=0
    else:
        color,strength=condition;colors=tuple(color.split('-'));prefix=base.ev.rt.condition_name(color,strength)
        ci=[c for c,_ in conditions(family)].index(color)
    names=[prefix+'/'+view+'/'+aa+'_'+bb for aa in colors for bb in colors]
    images=np.stack([v[r['image_offset']+np.array([r['state_names'].index(n) for n in names])] for r in idx])
    texts=t[np.array([r['text_indices'][ci*4:ci*4+4] for r in idx])]
    rows=data.lines(data.rowpath(family));assert [r['anchor_id'] for r in rows]==[r['anchor_id'] for r in idx]
    return images,texts,rows

@lru_cache(None)
def textbank(model):
    if model in data.MODELS:
        folder=OUT/model/'extra_text';ff=np.load(folder/'features.npy');ss=read(folder/'strings.json')
        return dict(zip(ss,ff))
    result=dict(zip(read(ind.OUT/'template_index.json'),np.load(ind.OUT/'template_features.npy')))
    folder=base.mid.ORIGINAL/'text';result.update(zip(read(folder/'strings.json'),np.load(folder/'features.npy')))
    return result

def wording(model,rows,colors,view,family,pooled=False):
    lookup=textbank(model)
    if model=='openclip_laion_l14' and not pooled:
        # Preserve the historical transfer cache exactly. The training cache
        # encodes some duplicate strings in different batches; exchanging them
        # can change float32 ties even when their embeddings agree to <3e-7.
        lookup=dict(zip(read(ind.OUT/'template_index.json'),np.load(ind.OUT/'template_features.npy')))
    x=np.asarray([[lookup[s] for variants in zip(*ind.prompts(tuple(r['objects']),colors,view,family)) for s in variants] for r in rows])
    x=x.reshape(len(rows),3,4,-1)
    if not pooled:return x
    return F.normalize(torch.tensor(x.mean(1)),dim=-1).numpy()

def scored(regs,views,model,individual=False):
    raw={};parts=[]
    for view,(v,t) in views.items():
        values={}
        for reg in regs:
            vi=adapt(v,reg,'image');te=adapt(t,reg,'text')
            x=np.einsum('nid,nkjd->nkij' if individual else 'nid,njd->nij',vi,te,optimize=True)
            m=measurements(x.reshape(-1,4,4),model)
            if individual:m={k:z.reshape(len(v),t.shape[1]).mean(1) for k,z in m.items()}
            values[reg['name']]=m;raw[view+'/'+reg['name']]=x
        parts.append(values)
    return a.average(parts),raw

class Recorder:
    """Save per-seed observations now; source/seed intervals are computed once."""
    def __init__(self,study,seed,family):
        self.dest=OUT/study/'evaluation'/f'seed{seed}'/family;self.dest.mkdir(parents=True,exist_ok=False)
        self.index=[];self.seed=seed
    def emit(self,test,values,rows,raw=None,details=None):
        keys=list(next(iter(values.values())))
        assert all(np.isfinite(np.asarray(v)).all() for m in values.values() for v in m.values())
        jsonl(self.dest/(test+'_per_example.jsonl'),[
            dict(name=name,test=test,anchor_id=r.get('anchor_id',r.get('example_id')),
                 source_ids=r.get('source_ids',[r.get('source_id')]),**{k:float(z[i]) for k,z in metrics.items()})
            for name,metrics in values.items() for i,r in enumerate(rows)])
        if raw is not None:np.savez_compressed(self.dest/(test+'_scores.npz'),**raw)
        self.index.append(dict(test=test,details=details,n_items=len(rows),metrics=keys,names=list(values)))
        print('SCORED',self.dest.parent.parent.parent.name,self.seed,test,flush=True)
    def finish(self):
        dump(self.dest/'score_index.json',self.index)
        dump(self.dest/'complete.json',dict(files={str(f):sha(f) for f in self.dest.iterdir() if f.is_file()}))

def primary(study,seed):
    model=backbone(study);regs=registry(study,seed);rec=Recorder(study,seed,'primary')
    for color in map('-'.join,p.COLORS):
        for order in ('canonical','reversed') if color!='purple-orange' else ('canonical',):
            views={}
            for view in p.VIEWS:
                v,t,rows=arrays(model,'primary',color,view)
                if order=='reversed':t=wording(model,rows,tuple(color.split('-')),view,'reverse_order',True)
                views[view]=(v,t)
            values,raw=scored(regs,views,model);rec.emit(color+'_'+order,values,rows,raw)
    rec.finish()

def transfer(study,seed):
    model=backbone(study);regs=registry(study,seed,False);rec=Recorder(study,seed,'transfer')
    for family in data.olddata.FAMILIES:
        parts=[];rawall={}
        for condition in conditions(family):
            views={}
            for view in p.VIEWS:
                v,t,rows=arrays(model,family,condition,view);views[view]=(v,t)
            values,raw=scored(regs,views,model);parts.append(values)
            rawall.update({str(condition)+'/'+k:z for k,z in raw.items()})
        rec.emit(family,a.average(parts),rows,rawall,dict(conditions=conditions(family)))
    rows=data.lines(data.rowpath('primary'))
    for colors in p.COLORS[:2]:
        for family in ('reverse_order','attribute_clause'):
            views={}
            for view in p.VIEWS:
                v,_,_=arrays(model,'primary','-'.join(colors),view)
                views[view]=(v,wording(model,rows,colors,view,family))
            values,raw=scored(regs,views,model,True)
            rec.emit(family+'_'+'-'.join(colors),values,rows,raw,dict(wordings=3,exposed_in_training=family=='reverse_order'))
    entries=read(tt.DEST/'caption_gallery.json');pairs={tuple(e['objects']):e['pair_index'] for e in entries}
    if model=='openclip_laion_l14':gallery=np.load(tt.DEST/'gallery_features.npy')
    else:
        lookup=textbank(model);gallery=F.normalize(torch.tensor(np.stack([np.mean([lookup[s] for s in e['templates']],axis=0) for e in entries])),dim=-1).numpy()
    correct=np.array([[4*pairs[tuple(r['objects'])]+j for j in range(4)] for r in rows]);parts=[];raw={}
    for view in p.VIEWS:
        v,_,_=arrays(model,'primary','red-blue',view);values={}
        for reg in regs:
            x=np.einsum('nid,jd->nij',adapt(v,reg,'image'),adapt(gallery,reg,'text'),optimize=True)
            z,_=a.gallery_metrics(x,correct);values[reg['name']]=dict(gallery_top1=z[:,0],object_pair_top1=z[:,1]);raw[view+'/'+reg['name']]=x
        parts.append(values)
    rec.emit('caption_gallery',a.average(parts),rows,raw,dict(captions=112));rec.finish()

def natural_path(model,bench):
    if model=='google_siglip_b16_224':return p.ROOT/f'clip/fse_two_case_completion_20260927/routing/{model}/features/natural_{bench}.npz'
    if bench=='sugarcrepe':
        return (p.prior.SUGAR if model=='openclip_laion_l14' else p.BASE/'benchmark_features'/model/bench)/'features.npz'
    return p.BASE/'preservation/features'/model/bench/'features.npz'

def natural(study,seed):
    model=backbone(study);regs=registry(study,seed,False);rec=Recorder(study,seed,'natural')
    for bench in ('sugarcrepe','aro','coco'):
        ff=np.load(natural_path(model,bench));values={};raw={}
        if bench=='sugarcrepe':
            ref=pd.read_csv(p.prior.SUGAR/'frozen_seed0.csv');index=read(p.prior.SUGAR/'indices.json')
            vi={s:i for i,s in enumerate(index['names'])};ti={s:i for i,s in enumerate(index['prompts'])}
            v=ff['images'][[vi[s] for s in ref.filename]];pos=[ti[s] for s in ref.caption];neg=[ti[s] for s in ref.negative_caption]
            rows=[dict(example_id=str(r.subset)+'/'+str(r.example_id),source_id=r.filename,category=r.subset) for _,r in ref.iterrows()]
        else:
            rows=data.lines(p.BASE/'preservation/manifests'/bench/'rows.jsonl');v=ff['images']
            if bench=='aro':pos=[r['positive'] for r in rows];neg=[r['negative'] for r in rows]
            else:owner=np.array([r['image_index'] for r in rows])
        for reg in regs:
            name=reg['name'];t=adapt(ff['texts'],reg,'text');vv=adapt(v,reg,'image')
            if bench=='coco':
                tr,ir,_,_=a.retrieval_ranks(vv,t,owner,128,device='cuda')
                values[name]=(tr,ir);raw[name+'/t2i']=tr;raw[name+'/i2t']=ir
            else:
                x=np.stack([np.einsum('nd,nd->n',vv,t[pos]),np.einsum('nd,nd->n',vv,t[neg])],axis=1)
                values[name]=dict(accuracy=(x[:,0]>x[:,1]).astype(float));raw[name]=x
        if bench=='coco':
            for d,direction in enumerate(('t2i','i2t')):
                rr=[dict(example_id=str(j),source_id=str(owner[j] if d==0 else j)) for j in range(len(next(iter(values.values()))[d]))]
                rec.emit('coco_'+direction,{n:{'recall'+str(k):(z[d]<=k).astype(float) for k in (1,5)} for n,z in values.items()},rr)
        else:
            rec.emit(bench+'_full',values,rows)
            masks={c:np.array([r['category']==c for r in rows]) for c in sorted({r['category'] for r in rows})} if bench=='sugarcrepe' else {c:np.array([r[c] for r in rows]) for c in ('either_red_blue','exact_red_blue')}
            for category,mask in masks.items():rec.emit(bench+'_'+category,{n:{k:z[mask] for k,z in m.items()} for n,m in values.items()},[r for r,yes in zip(rows,mask) if yes])
        np.savez_compressed(rec.dest/(bench+'_raw.npz'),**raw);dump(rec.dest/(bench+'_rows.json'),rows)
    rec.finish()

def prepare(study):
    paths=[data.PLAN,Path(__file__),Path(stats.__file__),Path(p.metrics.__file__),Path(a.__file__),OUT/'data_protocol.json']
    model=backbone(study)
    paths += [data.bank(model,f)/'complete.json' for f in ('primary',*data.olddata.FAMILIES)]
    paths += [natural_path(model,b) for b in ('sugarcrepe','aro','coco')]
    paths += [Path(r['checkpoint']) for s in (42,43,44) for r in registry(study,s) if r['checkpoint']]
    dest=OUT/study/'evaluation_protocol.json'
    if dest.exists():verify_files(read(dest)['inputs']);return
    dump(dest,dict(inputs={str(f):sha(f) for f in paths},tint=.75,study=study,
        seeds=[42,43,44],source_bootstrap=5000,seed_source_bootstrap=5000,
        primary='Pooled two trained colors, both caption orders and layouts; purple-orange secondary',
        external_fixed='Frozen and LABCLIP broadcast for paired draws, not three independent fits',
        full_assignment_D='Twice stored response e',no_new_subset_selection=True))

def evaluate(study):
    prepare(study)
    for seed in (42,43,44):
        for family in ('primary','transfer','natural'):
            done=OUT/study/'evaluation'/f'seed{seed}'/family/'complete.json'
            if done.exists():verify_files(read(done)['files']);continue
            globals()[family](study,seed)
    dump(OUT/study/'evaluation_complete.json',dict(inputs={str(OUT/study/'evaluation'/f'seed{s}'/f/'complete.json'):sha(OUT/study/'evaluation'/f'seed{s}'/f/'complete.json') for s in (42,43,44) for f in ('primary','transfer','natural')}))

def aggregate(study):
    verify_files(read(OUT/study/'evaluation_complete.json')['inputs'])
    stats.ROOTS=[OUT/study/'evaluation'/f'seed{s}' for s in (42,43,44)]
    # stats.load expects /analysis between the seed and family: adapt reader only.
    def load(family,test):
        samples=[];reference=None
        for root in stats.ROOTS:
            rr=data.lines(root/family/(test+'_per_example.jsonl'));names=list(dict.fromkeys(r['name'] for r in rr))
            groups={n:[r for r in rr if r['name']==n] for n in names};first=groups[names[0]]
            rows=[dict(anchor_id=r['anchor_id'],source_ids=r['source_ids']) for r in first]
            keys=[k for k in first[0] if k not in stats.METADATA]
            metadata=(rows,names,keys)
            if reference is None:reference=metadata
            assert metadata==reference
            for n in names:assert [(r['anchor_id'],r['source_ids']) for r in groups[n]]==[(r['anchor_id'],r['source_ids']) for r in first]
            samples.append(np.stack([np.array([[r[k] for k in keys] for r in groups[n]]) for n in names],axis=1))
        values=np.stack(samples);fi=names.index('Frozen');assert np.array_equal(values[:,:,fi],values[:1,:,fi].repeat(3,axis=0))
        return values,rows,names,keys
    dest=OUT/study/'results';dest.mkdir(exist_ok=False);records=[]
    for family in ('primary','transfer','natural'):
        for entry in read(stats.ROOTS[0]/family/'score_index.json'):
            records += stats.summarize(family,entry['test'],load(family,entry['test']))
    loaded=[load('primary',test) for test in stats.PRIMARY]
    assert all(z[1:]==loaded[0][1:] for z in loaded)
    records += stats.summarize('primary','trained_colors_both_orders',(np.mean([z[0] for z in loaded],axis=0),*loaded[0][1:]))
    jsonl(dest/'all_metrics.jsonl',records);stats.csvwrite(dest/'all_metrics.csv',records)
    dump(dest/'complete.json',dict(files={str(f):sha(f) for f in dest.iterdir() if f.is_file()},external_methods_single_fixed_model=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['evaluate','aggregate']);ap.add_argument('--study',choices=STUDIES,required=True);args=ap.parse_args()
    log(OUT,'evaluation_start',**vars(args));torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    globals()[args.action](args.study);log(OUT,'evaluation_complete',**vars(args))

if __name__=='__main__':main()
