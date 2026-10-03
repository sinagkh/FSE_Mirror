"""Complete scoring for plan 39, with paired seed/source uncertainty."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.core.metrics import adapt; from mirror.core.metrics import bank_arrays; from mirror.core.metrics import routing
from mirror.cases.color_binding.completion_preservation import retrieval_ranks
from mirror.cases.color_binding.strengthening_statistics import intervals
from mirror.cases.color_binding import targeted_suppression as study
from mirror.cases.color_binding import targeted_transfer as transfer
from mirror.cases.color_binding import ranking_transfer_extension as prevtransfer
from mirror.cases.color_binding import indirect_generalization as ind

OUT=study.OUT
REGISTRY=OUT/'checkpoints_frozen_crn.json'
METRICS=prevtransfer.METRICS


def registry():
    p=read(REGISTRY);regs=p['models']
    verify_files({r['checkpoint']:r['sha256'] for r in regs if r['checkpoint']})
    return regs


def freeze():
    study.verify();transfer.verify();registry()
    paths=[Path(__file__),Path(__file__).with_name('completion_metrics.py'),
        Path(__file__).with_name('strengthening_statistics.py'),
        Path(__file__).parent/'tests/test_targeted_evaluate.py',
        REGISTRY,OUT/'protocol.json',OUT/'common_noise_retry/protocol.json',transfer.DEST/'protocol.json',
        transfer.DEST/'visual_review.json']
    dump(OUT/'scoring_protocol.json',dict(inputs={str(p):sha(p) for p in paths},
        all_registered_models_scored=True,no_checkpoint_selection=True,
        outcomes_previously_observed_for_old_conditions=True,new_condition_scores_seen=False,
        primary_metric='individual exchange decision accuracy; full-gallery Top-1 for caption-gallery family',
        uncertainty='10,000 paired source and crossed seed/source draws; all conditions remain within anchor',
        natural_categories='full and the unchanged previously specified subcategories'))
    print('SCORING_FROZEN',sha(OUT/'scoring_protocol.json'),flush=True)


def comparisons(names):
    pairs=[('IS_templates',b) for b in ('F','R_selected','RG_selected')]
    pairs += [('RG_selected','R_selected')]
    pairs += [('suppression_selected',b) for b in ('R_selected','continue_R','continue_RG','retain_only')]
    pairs += [('continue_R','R_selected')]
    return [(a,b) for a,b in pairs if a in names and b in names]


def summarize(values,metrics,meta,clusters):
    names=list(dict.fromkeys(n for n,s in values));out=[]
    for cohort,seeds in [('seed42',[42]),('three_seed',[42,43,44])]:
        available=[n for n in names if n=='F' or all((n,s) in values for s in seeds)]
        packed={n:np.stack([values[(n,0 if n=='F' else s)] for s in seeds]) for n in available}
        labels=list(packed.items())+[(a+' - '+b,packed[a]-packed[b]) for a,b in comparisons(available)]
        stats=intervals(np.concatenate([a for _,a in labels],-1),clusters,draws=10000)
        out += [dict(**meta,cohort=cohort,comparison=name,metric=metric,seed_ids=seeds,**stats[i*len(metrics)+j])
            for i,(name,_) in enumerate(labels) for j,metric in enumerate(metrics)]
    return out


def score_lattice(regs,v,t):
    values={};raw=[]
    for reg in regs:
        x=np.einsum('nid,njd->nij',v,adapt(t,reg['checkpoint']),optimize=True)
        values[(reg['name'],reg['seed'])]=prevtransfer.measurements(x);raw.append(x)
    return values,np.stack(raw)


def store_raw(dest,filename,raw,regs,rows,**extra):
    with (dest/filename).open('xb') as f:np.savez_compressed(f,scores=raw)
    return dict(file=filename,models=[dict(name=r['name'],seed=r['seed']) for r in regs],
        anchor_ids=[r['anchor_id'] for r in rows],**extra)


def controlled():
    ind.configure();regs=registry();dest=OUT/'controlled';dest.mkdir(exist_ok=False)
    summaries=[];records=[];index=[]
    def emit(test,vals,rows,clusters=None,metrics=METRICS):
        summaries.extend(summarize(vals,metrics,dict(test=test),np.arange(len(rows)) if clusters is None else clusters))
        for (name,seed),a in vals.items():
            records.extend(dict(test=test,name=name,seed=seed,anchor_id=r['anchor_id'],source_ids=r['source_ids'],
                **dict(zip(metrics,a[j].tolist()))) for j,r in enumerate(rows))
    for color in ('red-blue','green-yellow','purple-orange'):
        accum=defaultdict(list)
        for view in ind.VIEWS:
            v,t,idx=bank_arrays(ind.CONFIRM/'features'/ind.MODEL,'routing',color,view)
            meta={r['anchor_id']:r for r in lines(ind.CONFIRM/'rows.jsonl')};rows=[meta[r['anchor_id']] for r in idx]
            vals,raw=score_lattice(regs,v,t)
            test='direct_'+color+'_'+view;emit(test,vals,rows)
            index.append(store_raw(dest,test+'.npz',raw,regs,rows,kind='lattice'))
            for k,a in vals.items():accum[k].append(a)
        emit('direct_'+color+'_both',{k:np.mean(a,0) for k,a in accum.items()},rows)
    # Previously defined indirect tests, unchanged and complete.
    text=np.load(ind.OUT/'template_features.npy');lookup={s:i for i,s in enumerate(read(ind.OUT/'template_index.json'))}
    for bank in ('seen_pairs','unseen_pairs'):
        pooled=defaultdict(list);individual=defaultdict(list)
        for view in ind.VIEWS:
            v,t,rows=ind.feature_bank(bank,('red','blue'),view)
            vals,raw=score_lattice(regs,v,t)
            for k,a in vals.items():pooled[k].append(a)
            test=bank+'_trained_'+view;emit(test,vals,rows)
            index.append(store_raw(dest,test+'.npz',raw,regs,rows,kind='lattice'))
            ids=np.array([[lookup[s] for fam in ind.FAMILIES for variants in zip(*ind.prompts(tuple(r['objects']),('red','blue'),view,fam))
                for s in variants] for r in rows]).reshape(len(rows),12,4)
            templates=text[ids][:,ind.GROUPS['nonspatial_unseen']];values={};allraw=[]
            for reg in regs:
                x=np.einsum('nid,nkjd->nkij',v,adapt(templates,reg['checkpoint']),optimize=True)
                a=prevtransfer.measurements(x.reshape(-1,4,4)).reshape(len(v),6,-1).mean(1)
                key=(reg['name'],reg['seed']);values[key]=a;individual[key].append(a);allraw.append(x)
            test=bank+'_individual_'+view;emit(test,values,rows)
            index.append(store_raw(dest,test+'.npz',np.stack(allraw),regs,rows,kind='individual'))
        emit(bank+'_trained_both',{k:np.mean(a,0) for k,a in pooled.items()},rows)
        emit(bank+'_individual_both',{k:np.mean(a,0) for k,a in individual.items()},rows)
    for family in ('noun_recombination','color_recombination'):
        accum=defaultdict(list)
        for color,strength in prevtransfer.conditions(family):
            for view in ind.VIEWS:
                v,t,rows=prevtransfer.arrays(family,color,strength,view)
                vals,raw=score_lattice(regs,v,t)
                test=family+'_'+color+'_'+view
                emit(test,vals,rows,np.array([r['source_cluster'] for r in rows]))
                index.append(store_raw(dest,test+'.npz',raw,regs,rows,kind='lattice'))
                for k,a in vals.items():accum[k].append(a)
        emit(family+'_all',{k:np.mean(a,0) for k,a in accum.items()},rows,np.array([r['source_cluster'] for r in rows]))
    # Predefined exhaustive clause diagnostic, not a score-selected context subset.
    from mirror.cases.color_binding.routing_context_coverage import all_contexts
    from mirror.cases.color_binding.behavioral_pilot import CAL
    unit=read(CAL)['units'][ind.MODEL]['unit'];contexts=all_contexts();clause_rows=[]
    for reg in regs:
        if reg['name']!='suppression_selected':continue
        model_ix=next(j for j,r in enumerate(regs) if (r['name'],r['seed'])==(reg['name'],reg['seed']))
        reference_ix=next(j for j,r in enumerate(regs) if (r['name'],r['seed'])==('R_selected',reg['seed']))
        for view in ind.VIEWS:
            raw=np.load(dest/('direct_red-blue_'+view+'.npz'))['scores'].astype(float)
            current=raw[model_ix];incoming=raw[reference_ix]
            for context in contexts:
                a=np.einsum('nij,ij->n',current,context['weights'])/unit
                b=np.einsum('nij,ij->n',incoming,context['weights'])/unit
                if context['kind']=='unwanted':a,b=abs(a),abs(b)
                clause_rows.append(dict(seed=reg['seed'],view=view,context=context['name'],kind=context['kind'],
                    incoming_mean=float(b.mean()),current_mean=float(a.mean()),change=float((a-b).mean()),
                    decreased_fraction=float((a<b-1e-5).mean()),increased_fraction=float((a>b+1e-5).mean()),
                    n_anchors=len(a)))
    jsonl(dest/'all_clause_changes.jsonl',clause_rows)
    jsonl(dest/'summary.jsonl',summaries);jsonl(dest/'per_example.jsonl',records);dump(dest/'score_index.json',index)
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},
        checkpoint_registry_sha256=sha(REGISTRY)))
    print('CONTROLLED_COMPLETE',flush=True)


def gallery_metrics(x,correct):
    # x: anchors x four image states x 112 candidate captions.
    n=len(x);i=np.arange(n)[:,None];state=np.arange(4)[None]
    truth=x[i,state,correct];mask=np.eye(x.shape[-1],dtype=bool)[correct]
    top1=(truth>np.where(mask,-np.inf,x).max(-1)).mean(1)
    pairs=(correct[:,0]//4)
    good=np.arange(x.shape[-1])[None]//4==pairs[:,None]
    objectpair=(np.where(good[:,None],x,-np.inf).max(-1)>np.where(~good[:,None],x,-np.inf).max(-1)).mean(1)
    local=np.take_along_axis(x,correct[:,None,:],axis=2)
    m=prevtransfer.measurements(local)
    return np.column_stack((top1,objectpair,m)),local


def transfer_score():
    transfer.verify();ind.configure();regs=registry();dest=transfer.DEST/'evaluation';dest.mkdir(exist_ok=False)
    enc=read(transfer.DEST/'encoding_complete.json');verify_files(enc['files'])
    assert sha(transfer.DEST/'gallery_features.npy')==enc['gallery']['features_sha256']
    rows=lines(transfer.DEST/'rows.jsonl');summaries=[];records=[];index=[];adjusted=[]
    baseline={view:bank_arrays(ind.CONFIRM/'features'/ind.MODEL,'routing','red-blue',view)[:2] for view in ind.VIEWS}
    for family in transfer.FAMILIES[:2]:
        collective=defaultdict(list);familyrecords=[]
        for cond in transfer.conditions(family):
            combined=defaultdict(list)
            for view in transfer.VIEWS:
                v,t,rows=transfer.arrays(family,cond,view);values,raw=score_lattice(regs,v,t)
                bv,bt=baseline[view];_,rawbase=score_lattice(regs,bv,bt)
                metrics=(*METRICS,'decision_retention','decision_regression','exchange_change')
                for j,reg in enumerate(regs):
                    key=(reg['name'],reg['seed']);x=raw[j];b=rawbase[j]
                    correct=np.stack((x[:,1,1]>x[:,1,2],x[:,2,2]>x[:,2,1]),1)
                    oldcorrect=np.stack((b[:,1,1]>b[:,1,2],b[:,2,2]>b[:,2,1]),1)
                    # Unconditional joint rates allow exact aggregation; do not average conditional ratios.
                    retention=(correct&oldcorrect).mean(1);regression=(~correct&oldcorrect).mean(1)
                    change=correct.mean(1)-oldcorrect.mean(1)
                    a=np.column_stack((values[key],retention,regression,change))
                    combined[key].append(a);collective[key].append(a)
                    familyrecords.extend(dict(family=family,condition=cond,view=view,name=reg['name'],seed=reg['seed'],
                        anchor_id=r['anchor_id'],source_ids=r['source_ids'],**dict(zip(metrics,a[k].tolist()))) for k,r in enumerate(rows))
                index.append(store_raw(dest,family+'_'+cond+'_'+view+'.npz',raw,regs,rows,kind='lattice',family=family,condition=cond,view=view))
            summaries.extend(summarize({k:np.mean(a,0) for k,a in combined.items()},metrics,
                dict(family=family,condition=cond),np.arange(len(rows))))
        vals={k:np.mean(a,0) for k,a in collective.items()}
        summaries.extend(summarize(vals,metrics,dict(family=family,condition='all_conditions'),np.arange(len(rows))))
        delta=np.stack([vals[('IS_templates',s)][:,0]-vals[('R_selected',s)][:,0] for s in study.SEEDS])
        adjusted.append(dict(family=family,comparison='IS_templates - R_selected',metric='exchange_accuracy',
            mean=float(delta.mean()),ci9833=prevtransfer.family_adjusted_ci(delta,np.arange(len(rows)))))
        jsonl(dest/(family+'_per_example.jsonl'),familyrecords)
        print('TRANSFER_SCORED',family,flush=True)
    features=np.load(transfer.DEST/'gallery_features.npy');entries=read(transfer.DEST/'caption_gallery.json')
    pairindex={tuple(e['objects']):e['pair_index'] for e in entries}
    correct=np.array([[4*pairindex[tuple(r['objects'])]+s for s in range(4)] for r in rows])
    values=defaultdict(list);records=[];metrics=('gallery_top1','object_pair_accuracy',*METRICS)
    for view in ind.VIEWS:
        v,t=baseline[view];raw=[]
        for reg in regs:
            x=np.einsum('nid,jd->nij',v,adapt(features,reg['checkpoint']),optimize=True)
            a,local=gallery_metrics(x,correct);values[(reg['name'],reg['seed'])].append(a);raw.append(x)
            old=np.einsum('nid,njd->nij',v,adapt(t,reg['checkpoint']),optimize=True)
            np.testing.assert_allclose(local,old,atol=2e-6,rtol=0)
            records.extend(dict(family='caption_gallery',condition='112_captions',view=view,
                name=reg['name'],seed=reg['seed'],anchor_id=r['anchor_id'],source_ids=r['source_ids'],
                **dict(zip(metrics,a[j].tolist()))) for j,r in enumerate(rows))
        index.append(store_raw(dest,'caption_gallery_'+view+'.npz',np.stack(raw),regs,rows,kind='gallery',correct_indices=correct.tolist(),view=view))
    vals={k:np.mean(a,0) for k,a in values.items()}
    summaries.extend(summarize(vals,metrics,dict(family='caption_gallery',condition='all_conditions'),np.arange(len(rows))))
    delta=np.stack([vals[('IS_templates',s)][:,0]-vals[('R_selected',s)][:,0] for s in study.SEEDS])
    adjusted.append(dict(family='caption_gallery',comparison='IS_templates - R_selected',metric='gallery_top1',
        mean=float(delta.mean()),ci9833=prevtransfer.family_adjusted_ci(delta,np.arange(len(rows)))))
    jsonl(dest/'caption_gallery_per_example.jsonl',records);jsonl(dest/'summary.jsonl',summaries)
    dump(dest/'score_index.json',index);dump(dest/'family_adjusted_intervals.json',adjusted)
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},
        checkpoint_registry_sha256=sha(REGISTRY),test_protocol_sha256=sha(transfer.DEST/'protocol.json')))
    print('TRANSFER_SCORING_COMPLETE',flush=True)


def natural():
    ind.configure();study.cn.available_gpu();regs=registry();dest=OUT/'natural';dest.mkdir(exist_ok=False);stats=[]
    rc=study.rc
    for benchmark in ('sugarcrepe','aro','coco'):
        path=study.prior.SUGAR if benchmark=='sugarcrepe' else rc.BASE/'preservation/features'/rc.MODEL/benchmark
        ff=np.load(path/'features.npz');values={};records=[]
        if benchmark=='sugarcrepe':
            idx=read(path/'indices.json');ref=pd.read_csv(path/'frozen_seed0.csv')
            vi={s:i for i,s in enumerate(idx['names'])};ti={s:i for i,s in enumerate(idx['prompts'])}
            v=ff['images'][[vi[s] for s in ref.filename]];pos=[ti[s] for s in ref.caption];neg=[ti[s] for s in ref.negative_caption]
            rows=[dict(example_id=str(r.subset)+'/'+str(r.example_id),source_id=r.filename,category=r.subset) for _,r in ref.iterrows()]
        else:
            rows=lines(rc.BASE/'preservation/manifests'/benchmark/'rows.jsonl');v=ff['images']
            if benchmark=='aro':pos=[r['positive'] for r in rows];neg=[r['negative'] for r in rows]
            else:owner=np.array([r['image_index'] for r in rows])
        for reg in regs:
            t=adapt(ff['texts'],reg['checkpoint']);key=(reg['name'],reg['seed'])
            if benchmark=='coco':
                tr,ir,_,_=retrieval_ranks(v,t,owner,128,device='cuda');values[key]=(tr,ir)
                for direction,ranks in [('t2i',tr),('i2t',ir)]:
                    records.extend(dict(name=reg['name'],seed=reg['seed'],direction=direction,
                        example_id=str(j),source_id=str(owner[j] if direction=='t2i' else j),rank=int(rank)) for j,rank in enumerate(ranks))
            else:
                positive=np.einsum('nd,nd->n',v,t[pos]);negative=np.einsum('nd,nd->n',v,t[neg]);margin=positive-negative
                values[key]=(margin>0).astype(float)[:,None]
                records.extend(dict(name=reg['name'],seed=reg['seed'],**r,positive_score=float(positive[j]),
                    negative_score=float(negative[j]),margin=float(margin[j]),accuracy=float(margin[j]>0)) for j,r in enumerate(rows))
            print('NATURAL',benchmark,reg['name'],reg['seed'],flush=True)
        if benchmark=='coco':
            for d,direction in enumerate(('t2i','i2t')):
                vv={k:np.stack([z[d]<=q for q in (1,5)],1).astype(float) for k,z in values.items()}
                clusters=owner if d==0 else np.arange(len(ff['images']))
                stats+=summarize(vv,['recall1','recall5'],dict(benchmark=benchmark,category=direction),clusters)
        else:
            cats={'full':np.ones(len(rows),bool)}
            if benchmark=='sugarcrepe':cats.update({c:np.array([r['category']==c for r in rows]) for c in sorted({r['category'] for r in rows})})
            else:cats.update({c:np.array([r[c] for r in rows]) for c in ('either_red_blue','exact_red_blue')})
            for category,mask in cats.items():
                _,cl=np.unique(np.array([r['source_id'] for r in rows])[mask],return_inverse=True)
                stats+=summarize({k:a[mask] for k,a in values.items()},['accuracy'],dict(benchmark=benchmark,category=category),cl)
        jsonl(dest/(benchmark+'_per_example.jsonl'),records)
    jsonl(dest/'summary.jsonl',stats)
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},
        checkpoint_registry_sha256=sha(REGISTRY)))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','controlled','transfer_score','natural'))
    args=p.parse_args();log(OUT,'start',stage='evaluate_'+args.action)
    try:
        if args.action!='freeze':verify_files(read(OUT/'scoring_protocol.json')['inputs'])
        globals()[args.action]()
    except BaseException as e:log(OUT,'failed',stage='evaluate_'+args.action,error=repr(e));raise
    log(OUT,'complete',stage='evaluate_'+args.action)


if __name__=='__main__':main()
