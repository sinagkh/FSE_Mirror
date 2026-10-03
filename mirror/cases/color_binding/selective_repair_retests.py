"""Complete fixed-candidate retests; development selection is external and frozen."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.core.metrics import adapt; from mirror.core.metrics import bank_arrays
from mirror.cases.color_binding.strengthening_statistics import intervals

OUT=ROOT/'clip/interbind_selective_repair_retests_20260926'
PLAN=ROOT/'FSE_VLM/plan/40c_selective_repair_retests.md'
ORIGINAL=ROOT/'clip/interbind_selective_repair_pilot_20260925'
DIRECTIONAL=ROOT/'clip/interbind_selective_repair_directional_20260926'
CONNECTED=ROOT/'clip/interbind_connected_coverage_20260926'
STRENGTH=ROOT/'clip/interbind_strengthening_cpu_20260925'
COHORTS={'routing':(ORIGINAL/'routing','routing'),
    'background':(ORIGINAL/'background','background'),
    'full_scene':(DIRECTIONAL/'full_scene','full_scene'),
    'spatial':(DIRECTIONAL/'spatial','spatial'),
    **{b:(CONNECTED/b/'training/routing','routing') for b in ('disconnected','connected')}}


def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]


def source_clusters(rows):
    parent=list(range(len(rows)));owner={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        sources=list(r.get('source_ids',[r.get('source_id',r.get('anchor_id'))]))
        if 'donor' in r:sources.append(r['donor']['image_id'])
        for s in sources:
            s=str(s)
            if s in owner:parent[root(i)]=root(owner[s])
            else:owner[s]=i
    return np.unique([root(i) for i in range(len(rows))],return_inverse=True)[1]


def freeze(cohort):
    folder,family=COHORTS[cohort];sel=read(folder/'selection.json')
    regs=[dict(name='F',checkpoint=None,seed=0),
        dict(sel['ranking'],name='R'),dict(sel['is_candidate'],name='IS')]
    previous=next(r for r in read(folder/'models.json') if r['name']=='previous_IS')
    regs.append(previous)
    verify_files({r['checkpoint']:r['sha256'] for r in regs if r['checkpoint']})
    info=read(folder/'protocol.json');blocked=set(read(STRENGTH/'source_firewall_v3/blocked_coco_ids.json'))
    sources=set(map(int,info['training_source_ids']))
    dest=OUT/cohort
    dump(dest/'registry.json',dict(cohort=cohort,family=family,models=regs,selection=sel,
        training_source_ids=sorted(sources),known_benchmark_union_overlap=sorted(sources&blocked),
        inputs={str(p):sha(p) for p in (PLAN,Path(__file__),folder/'selection.json',folder/'protocol.json')},
        new_checkpoint_test_outcomes_seen=False,seed_conditional=True))
    if family=='routing':
        from mirror.cases.color_binding import ranking_transfer_extension as rt
        cc=read(CONNECTED/'construction.json')
        old={frozenset(p) for p in cc['original_pairs']}
        added={frozenset(r['objects']) for r in cc['offsets']['connected']}-old
        rows=lines(rt.DEST/'noun_rows.jsonl')
        jsonl(dest/'noun_pair_membership.jsonl',[dict(anchor_id=r['anchor_id'],objects=r['objects'],
            stratum='newly_trained_connected' if frozenset(r['objects']) in added else 'untrained_both') for r in rows])
    print('FROZEN',cohort,sel['adopted'],flush=True)


def registry(cohort):
    p=read(OUT/cohort/'registry.json');verify_files(p['inputs'])
    verify_files({r['checkpoint']:r['sha256'] for r in p['models'] if r['checkpoint']})
    return p


class Recorder:
    def __init__(self,dest,regs,training_sources=()):
        self.dest=dest;dest.mkdir(parents=True,exist_ok=False)
        self.regs=regs;self.stats=[];self.records=[];self.index=[];self.distributions=[]
        self.training_sources=set(map(str,training_sources))

    def emit(self,test,values,rows,raw=None):
        sources={str(s) for r in rows for s in r.get('source_ids',[])}
        assert not sources.intersection(self.training_sources),(test,'source overlap')
        names=list(values);metrics=list(next(iter(values.values())))
        packed={n:np.column_stack([values[n][k] for k in metrics]) for n in names}
        pairs=[('IS','R'),('IS','F'),('R','F'),('IS','previous_IS')]
        labels=list(packed.items())+[(a+' - '+b,packed[a]-packed[b]) for a,b in pairs if a in packed and b in packed]
        clusters=source_clusters(rows)
        stats=intervals(np.concatenate([a for _,a in labels],1)[None],clusters,draws=2000,seed=20260926)
        self.stats += [dict(test=test,comparison=n,metric=k,seed_ids=[42],
            inference='seed42 conditional; paired source-component interval',**stats[i*len(metrics)+j])
            for i,(n,_) in enumerate(labels) for j,k in enumerate(metrics)]
        for n,a in packed.items():
            self.records += [dict(test=test,name=n,anchor_id=r.get('anchor_id',r.get('example_id')),
                source_ids=r.get('source_ids',[r.get('source_id')]),**dict(zip(metrics,a[j].tolist()))) for j,r in enumerate(rows)]
            self.distributions += [dict(test=test,name=n,metric=k,quantiles=np.quantile(a[:,j],[0,.1,.5,.9,1]).tolist()) for j,k in enumerate(metrics)]
        if raw is not None:
            filename=test+'.npz'
            with (self.dest/filename).open('xb') as f:np.savez_compressed(f,scores=np.stack(raw))
            self.index.append(dict(file=filename,models=[r['name'] for r in self.regs],
                anchor_ids=[r.get('anchor_id',r.get('example_id')) for r in rows]))
        print('SCORED',test,len(rows),flush=True)

    def finish(self):
        jsonl(self.dest/'summary.jsonl',self.stats);jsonl(self.dest/'per_example.jsonl',self.records)
        jsonl(self.dest/'distribution_quantiles.jsonl',self.distributions);dump(self.dest/'score_index.json',self.index)
        dump(self.dest/'complete.json',dict(files={str(p):sha(p) for p in self.dest.iterdir() if p.is_file()}))


def measured(regs,v,t,family):
    from mirror.cases.color_binding.selective_repair_pilot import numeric_metrics
    values={};raw=[]
    for r in regs:
        x=np.einsum('nid,njd->nij',v,adapt(t,r['checkpoint']),optimize=True).astype(float)
        values[r['name']]=numeric_metrics(x,family);raw.append(x)
        if family in ('routing','full_scene'):
            from mirror.cases.color_binding.selective_repair_pilot import quantities
            q=quantities(torch.from_numpy(x),family)
            for kind in ('binding','cross'):
                clauses=q[kind].numpy()
                if kind=='cross':clauses=abs(clauses)
                for j in range(clauses.shape[1]):values[r['name']][f'{kind}_clause{j}']=clauses[:,j]
    return values,raw


def average(pieces):
    return {n:{k:np.mean([p[n][k] for p in pieces],0) for k in pieces[0][n]} for n in pieces[0]}


def subset(vals,mask):return {n:{k:v[mask] for k,v in a.items()} for n,a in vals.items()}


def routing_tests(rec):
    from mirror.cases.color_binding import indirect_generalization as ind; from mirror.cases.color_binding import ranking_transfer_extension as rt; from mirror.cases.color_binding import targeted_transfer as tt
    meta={r['anchor_id']:r for r in lines(ind.CONFIRM/'rows.jsonl')}
    for color in ('red-blue','green-yellow','purple-orange'):
        pieces=[]
        for view in ind.VIEWS:
            v,t,idx=bank_arrays(ind.CONFIRM/'features'/ind.MODEL,'routing',color,view)
            rows=[meta[r['anchor_id']] for r in idx];vals,raw=measured(rec.regs,v,t,'routing')
            rec.emit('direct_'+color+'_'+view,vals,rows,raw);pieces.append(vals)
        rec.emit('direct_'+color+'_both',average(pieces),rows)
    for family in ('noun_recombination','color_recombination'):
        pieces=[]
        for color,strength in rt.conditions(family):
            for view in ind.VIEWS:
                v,t,rows=rt.arrays(family,color,strength,view);vals,raw=measured(rec.regs,v,t,'routing')
                rec.emit(f'{family}_{color}_{view}',vals,rows,raw);pieces.append(vals)
        vv=average(pieces);rec.emit(family+'_all',vv,rows)
        if family=='noun_recombination':
            membership={r['anchor_id']:r['stratum'] for r in lines(rec.dest.parent/'noun_pair_membership.jsonl')}
            for s in ('untrained_both','newly_trained_connected'):
                mask=np.array([membership[r['anchor_id']]==s for r in rows])
                if mask.any():rec.emit(family+'_'+s,subset(vv,mask),[r for r,m in zip(rows,mask) if m])
    text=np.load(ind.OUT/'template_features.npy');lookup={s:i for i,s in enumerate(read(ind.OUT/'template_index.json'))}
    for bank in ('seen_pairs','unseen_pairs'):
        pieces=[]
        for view in ind.VIEWS:
            v,t,rows=ind.feature_bank(bank,('red','blue'),view)
            ids=np.array([[lookup[s] for fam in ind.FAMILIES for variants in zip(*ind.prompts(tuple(r['objects']),('red','blue'),view,fam))
                for s in variants] for r in rows]).reshape(len(rows),12,4)
            templates=text[ids][:,ind.GROUPS['nonspatial_unseen']];subpieces=[];rr=[]
            for j in range(6):
                vals,raw=measured(rec.regs,v,templates[:,j],'routing');subpieces.append(vals);rr.append(raw)
            vv=average(subpieces);pieces.append(vv)
            rec.emit(bank+'_individual_'+view,vv,rows,np.stack(rr,1))
        rec.emit(bank+'_individual_both',average(pieces),rows)
    for family in ('background_shift','geometry_shift'):
        pieces=[]
        for cond in tt.conditions(family):
            for view in ind.VIEWS:
                v,t,rows=tt.arrays(family,cond,view);vals,raw=measured(rec.regs,v,t,'routing')
                rec.emit(f'{family}_{cond}_{view}',vals,rows,raw);pieces.append(vals)
        rec.emit(family+'_all',average(pieces),rows)
    # Same complete 112-caption gallery, never restrict distractors by score.
    gallery=np.load(tt.DEST/'gallery_features.npy');g=read(tt.DEST/'caption_gallery.json')
    # The saved mapping is authoritative; preserve the prior gallery metric code.
    from mirror.cases.color_binding.targeted_evaluate import gallery_metrics
    pieces=[]
    for view in ind.VIEWS:
        v,_,idx=bank_arrays(ind.CONFIRM/'features'/ind.MODEL,'routing','red-blue',view)
        rows=[meta[r['anchor_id']] for r in idx]
        pairindex={tuple(e['objects']):e['pair_index'] for e in g}
        correct=np.array([[4*pairindex[tuple(r['objects'])]+i for i in range(4)] for r in rows])
        vals={};raw=[]
        for reg in rec.regs:
            x=np.einsum('nid,jd->nij',v,adapt(gallery,reg['checkpoint']),optimize=True)
            a,_=gallery_metrics(x,correct);vals[reg['name']]={'gallery_top1':a[:,0],'object_pair_top1':a[:,1]};raw.append(x)
        rec.emit('caption_gallery_'+view,vals,rows,raw);pieces.append(vals)
    rec.emit('caption_gallery_both',average(pieces),rows)


def full_scene_tests(rec):
    path=STRENGTH/'routing_transfer_v3';meta={r['anchor_id']:r for r in lines(path/'rows.jsonl')}
    for color in ('red-blue','green-yellow','purple-orange'):
        for view in ('canvas','swapped_canvas','in_situ'):
            v,t,idx=bank_arrays(path/'features/confirmation','routing',color,view)
            rows=[meta[r['anchor_id']] for r in idx];vals,raw=measured(rec.regs,v,t,'full_scene')
            test=f'{color}_{view}';rec.emit(test,vals,rows,raw)
            for s in sorted({r['stratum'] for r in rows}):
                mask=np.array([r['stratum']==s for r in rows]);rec.emit(test+'_'+s,subset(vals,mask),[r for r,m in zip(rows,mask) if m])


def background_tests(rec):
    path=ROOT/'clip/interbind_cache_v3_20260922/features/openclip_laion_l14/reserve/background'
    meta={r['anchor_id']:r for r in lines(ROOT/'clip/interbind_source_quality_20260922/reserve/accepted_rows.jsonl')}
    for color in ('red-blue','green-yellow','purple-orange'):
        pieces=[]
        for view in ('audit','green','hue_cast','scene_swap','original'):
            v,t,idx=bank_arrays(path,'background',color,view,'blend90_luminance/')
            rows=[meta[r['anchor_id']] for r in idx];vals,raw=measured(rec.regs,v,t,'background')
            rec.emit(f'{color}_{view}',vals,rows,raw)
            if view!='audit':pieces.append(vals)
        rec.emit(color+'_untrained_contexts',average(pieces),rows)


def spatial_tests(rec):
    for bank in ('confirmation','heldout_pairs'):
        path=STRENGTH/'spatial_v3/features'/bank;rows=lines(path/'rows.jsonl')
        vals,raw=measured(rec.regs,np.load(path/'images.npy'),np.load(path/'texts.npy'),'spatial')
        rec.emit(bank,vals,rows,raw)
    path=STRENGTH/'whatsup';ff=np.load(path/'features/features.npz');idx=read(path/'features/index.json')
    vi={s:i for i,s in enumerate(idx['images'])};ti={s:i for i,s in enumerate(idx['texts'])}
    rows=lines(path/'rows.jsonl')
    for benchmark in ('coco_two_object','controlled_a'):
        for category in ('full','left_right'):
            rr=[r for r in rows if r['benchmark']==benchmark and (category=='full' or r['left_right'])]
            cols=[r['left_right_caption_indices'] if benchmark=='controlled_a' and category=='left_right' else list(range(len(r['captions']))) for r in rr]
            assert all(0 in c for c in cols)
            v=ff['images'][[vi[r['image']] for r in rr]]
            t=ff['texts'][[[ti[r['captions'][i]] for i in c] for r,c in zip(rr,cols)]]
            vals={};raw=[]
            for reg in rec.regs:
                x=np.einsum('nd,njd->nj',v,adapt(t,reg['checkpoint']),optimize=True)
                vals[reg['name']]={'accuracy':(x[:,0]>x[:,1:].max(1)).astype(float)};raw.append(x)
            rec.emit('whatsup_'+benchmark+'_'+category,vals,rr,raw)


def controlled(cohort):
    torch.set_num_threads(4);p=registry(cohort);rec=Recorder(OUT/cohort/'controlled',p['models'],p['training_source_ids'])
    globals()[p['family']+'_tests'](rec);rec.finish()


def natural(cohort):
    import pandas as pd
    from mirror.cases.color_binding.completion_preservation import retrieval_ranks
    from mirror.cases.color_binding.routing_relative_pilot import SUGAR
    torch.set_num_threads(4);p=registry(cohort);rec=Recorder(OUT/cohort/'natural',p['models'])
    base=ROOT/'clip/interbind_phase_bc_completion_20260923/preservation'
    # CPU image/text scoring is sufficient; GPU is used only for complete COCO
    # gallery matrix products after the encoding job has released the device.
    assert torch.cuda.mem_get_info()[0]>18*1024**3,'GPU busy; no competing workload'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for benchmark in ('sugarcrepe','aro','coco'):
        path=SUGAR if benchmark=='sugarcrepe' else base/'features/openclip_laion_l14'/benchmark
        ff=np.load(path/'features.npz');v=ff['images'];base_text=ff['texts'];raw=[];values={}
        if benchmark=='sugarcrepe':
            idx=read(path/'indices.json');ref=pd.read_csv(path/'frozen_seed0.csv')
            vi={s:i for i,s in enumerate(idx['names'])};ti={s:i for i,s in enumerate(idx['prompts'])}
            v=v[[vi[s] for s in ref.filename]];pos=[ti[s] for s in ref.caption];neg=[ti[s] for s in ref.negative_caption]
            rows=[dict(example_id=str(r.subset)+'/'+str(r.example_id),source_id=r.filename,category=r.subset) for _,r in ref.iterrows()]
        else:
            rows=lines(base/'manifests'/benchmark/'rows.jsonl')
            if benchmark=='aro':pos=[r['positive'] for r in rows];neg=[r['negative'] for r in rows]
            else:owner=np.array([r['image_index'] for r in rows])
        for reg in rec.regs:
            t=adapt(base_text,reg['checkpoint'])
            if benchmark=='coco':
                tr,ir,_,_=retrieval_ranks(v,t,owner,256,device='cuda')
                values[reg['name']]=(tr,ir)
            else:
                ps=np.einsum('nd,nd->n',v,t[pos]);ns=np.einsum('nd,nd->n',v,t[neg])
                raw.append(np.stack((ps,ns),1));values[reg['name']]={'accuracy':(ps>ns).astype(float)}
        if benchmark=='coco':
            for direction,di in [('t2i',0),('i2t',1)]:
                vv={n:{f'recall{k}':(r[di]<=k).astype(float) for k in (1,5)} for n,r in values.items()}
                rr=rows if di==0 else [dict(example_id=str(i),source_id=str(i)) for i in range(len(v))]
                if di==0:rr=[dict(example_id=r['example_id'],source_id=str(r['image_index'])) for r in rows]
                rec.emit('coco_'+direction,vv,rr,[values[r['name']][di] for r in rec.regs])
        else:
            rec.emit(benchmark+'_full',values,rows,raw)
            cats=({c:np.array([r['category']==c for r in rows]) for c in sorted({r['category'] for r in rows})}
                  if benchmark=='sugarcrepe' else {c:np.array([r[c] for r in rows]) for c in ('either_red_blue','exact_red_blue')})
            for cat,mask in cats.items():rec.emit(benchmark+'_'+cat,subset(values,mask),[r for r,m in zip(rows,mask) if m])
    rec.finish()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','controlled','natural'));p.add_argument('cohort',choices=COHORTS)
    a=p.parse_args();log(OUT,'start',**vars(a))
    try:globals()[a.action](a.cohort)
    except BaseException as e:log(OUT,'failed',error=repr(e),**vars(a));raise
    log(OUT,'complete',**vars(a))


if __name__=='__main__':main()
