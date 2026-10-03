"""Matched primary/transfer/cross-requirement/preservation scoring for each repair."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.completion_data import OUT as ROOTOUT; from mirror.cases.color_binding.completion_data import configure
from mirror.cases.color_binding.completion_breadth import OUT as TRAIN
from mirror.cases.color_binding.completion_confirmation import OUT as CONFIRM
from mirror.cases.color_binding.completion_benchmarks import OUT as BENCH
from mirror.core.metrics import bank_arrays; from mirror.core.metrics import score_arrays; from mirror.core.metrics import routing; from mirror.core.metrics import background; from mirror.core.metrics import adapt
from mirror.cases.color_binding.behavioral_pilot import CACHE; from mirror.cases.color_binding.behavioral_pilot import QUAL; from mirror.cases.color_binding.behavioral_pilot import NAT; from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.cases.color_binding.completion_natural import OUT as COLA
from mirror.cases.color_binding.routing_relative_pilot import SUGAR
from mirror.cases.color_binding.routing_budget_replication_stats import interval; from mirror.cases.color_binding.routing_budget_replication_stats import seed_weights; from mirror.cases.color_binding.routing_budget_replication_stats import cohort_weights
from mirror.cases.color_binding.outcomes import retrieval_query

OUT=ROOTOUT/'breadth_evaluation'


def summarize_arrays(values,metrics,groups,meta,arms):
    results=[];sw=seed_weights(3)
    for cohort,weights in cohort_weights(groups).items():
        for label,array in [(a,values[a]) for a in arms]+[('IS-'+a,values['IS']-values[a]) for a in arms if a!='IS']:
            s=interval(array,weights,sw)
            for j,k in enumerate(metrics):
                results.append(dict(**meta,cohort=cohort,metric=k,comparison=label,mean=float(s['mean'][j]),
                    sample_sd=0. if label=='F' else float(s['sample_sd'][j]),
                    per_seed=({'frozen':float(s['per_seed'][0,j])} if label=='F' else {str(seed):float(s['per_seed'][i,j]) for i,seed in enumerate((42,43,44))}),
                    ci95_item=s['ci95_item'][j].tolist(),ci95_seed_item=s['ci95_seed_item'][j].tolist(),
                    n_anchors=int((weights[0]>0).sum()),paired=label.startswith('IS-')))
    return results


def main_evaluate(model,family):
    configure();src=TRAIN/model/family;done=read(src/'complete.json');assert sha(src/'models.json')==done['models_sha256']
    models=read(src/'models.json');verify_files({m['checkpoint']:m['sha256'] for m in models if m['checkpoint']})
    dest=OUT/model/family;dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'protocol.json',dict(model=model,repair_family=family,models_sha256=sha(src/'models.json'),
        evaluator_sha256=sha(Path(__file__)),frozen_recipe=True,selection=False,primary_color='red-blue',
        primary_routing='400-anchor fresh four-pair confirmation; dog/ball unsupported',
        primary_background='unopened reserve independent contexts',both_tasks_scored=True))
    arms=['F','R','IS','G','no_cross' if family=='routing' else 'no_gap'];allrecords=[];summary=[]
    for evaluated_family in ('routing','background'):
        cases=[]
        if (CONFIRM/'features'/model/'complete.json').exists() and evaluated_family=='routing':
            cases.append(('confirmation',CONFIRM/'features'/model,lines(CONFIRM/'rows.jsonl'),''))
        for part in ('pilot','reserve'):
            rows=[r for r in lines(QUAL/part/'accepted_rows.jsonl') if r['family']==evaluated_family]
            cases.append((part,CACHE/'features'/model/part/evaluated_family,rows,'blend90_luminance/'))
        for bank,path,rows,prefix in cases:
            meta={r['anchor_id']:r for r in rows}
            views=['canvas','swapped_canvas']+([] if bank=='confirmation' else ['in_situ']) if evaluated_family=='routing' else ['audit','green','hue_cast','scene_swap','original']
            for color in ('red-blue','green-yellow','purple-orange'):
                for view in views:
                    v,t,idx=bank_arrays(path,evaluated_family,color,view,prefix,meta);groups=np.asarray(['+'.join(meta[r['anchor_id']]['objects']) for r in idx]);by_model={}
                    for m in models:
                        x=score_arrays(v,t,m['checkpoint']);met=(routing if evaluated_family=='routing' else background)(x,model)
                        metrics=[k for k in met if not k.startswith('contrast/')]
                        by_model[(m['arm'],m['seed'])]=np.stack([met[k] for k in metrics],1)
                        for j,r in enumerate(idx):allrecords.append(dict(model=model,repair_family=family,evaluated_family=evaluated_family,bank=bank,color=color,view=view,
                            arm=m['arm'],seed=m['seed'],anchor_id=r['anchor_id'],group=groups[j],
                            seen_class_stratum=meta[r['anchor_id']].get('seen_class_stratum','both_seen' if evaluated_family=='routing' else 'not_applicable'),
                            **{k:float(value[j]) for k,value in met.items()}))
                    values={arm:np.stack([by_model[(arm,0 if arm=='F' else s)] for s in (42,43,44)]) for arm in arms}
                    summary+=summarize_arrays(values,metrics,groups,dict(model=model,repair_family=family,evaluated_family=evaluated_family,bank=bank,color=color,view=view),arms)
            print('BREADTH_SCORED',model,family,evaluated_family,bank,len(rows),flush=True)
    jsonl(dest/'per_example.jsonl',allrecords);jsonl(dest/'summary.jsonl',summary)
    natural(models,model,family,dest,arms)
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},models_sha256=sha(src/'models.json'),
        independent_views=True,all_three_seeds=True,no_selection=True))


def benchmark_summary(frame,metrics,arms,meta):
    ids=['example_id'];ref=frame[frame.arm=='F'].set_index(ids).sort_index()
    unique,inv=np.unique(ref.source_id,return_inverse=True);counts=np.bincount(inv);n=len(unique)
    draws=np.random.default_rng(20260923).multinomial(n,np.full(n,1/n),size=2000)
    weights=(counts/counts.sum(),draws*counts[None]/(draws@counts)[:,None]);sw=seed_weights(3);values={}
    for arm in arms:
        arr=[]
        for seed in (42,43,44):
            x=ref[metrics].to_numpy(float) if arm=='F' else frame[(frame.arm==arm)&(frame.seed==seed)].set_index(ids).loc[ref.index,metrics].to_numpy(float)
            arr.append(np.stack([np.bincount(inv,weights=x[:,j])/counts for j in range(len(metrics))],1))
        values[arm]=np.stack(arr)
    result=[]
    for label,x in [(a,values[a]) for a in arms]+[('IS-'+a,values['IS']-values[a]) for a in arms if a!='IS']:
        s=interval(x,weights,sw)
        for j,k in enumerate(metrics):result.append(dict(**meta,metric=k,comparison=label,mean=float(s['mean'][j]),
            sample_sd=0. if label=='F' else float(s['sample_sd'][j]),ci95_item=s['ci95_item'][j].tolist(),ci95_seed_item=s['ci95_seed_item'][j].tolist(),
            per_seed=({'frozen':float(s['per_seed'][0,j])} if label=='F' else {str(seed):float(s['per_seed'][i,j]) for i,seed in enumerate((42,43,44))}),
            n_items=len(ref),n_source_clusters=n))
    return result


def natural(models,model,family,dest,arms):
    sugar_path=SUGAR if model=='openclip_laion_l14' else BENCH/model/'sugarcrepe'
    ff=np.load(sugar_path/'features.npz');idx=read(sugar_path/'indices.json');ref=pd.read_csv(SUGAR/'frozen_seed0.csv')
    vi={s:i for i,s in enumerate(idx['names'])};ti={s:i for i,s in enumerate(idx['prompts'])}
    v=ff['images'][[vi[s] for s in ref.filename]];pos=[ti[s] for s in ref.caption];neg=[ti[s] for s in ref.negative_caption];sc=[]
    for m in models:
        t=adapt(ff['texts'],m['checkpoint']);margin=np.einsum('nd,nd->n',v,t[pos]-t[neg])
        for j,r in ref.iterrows():sc.append(dict(arm=m['arm'],seed=m['seed'],example_id=str(r.subset)+'/'+str(r.example_id),category=r.subset,source_id=r.filename,margin=float(margin[j]),accuracy=float(margin[j]>0)))
    jsonl(dest/'sugarcrepe_per_example.jsonl',sc);frame=pd.DataFrame(sc);ns=[]
    for cat in ['full',*sorted(frame.category.unique())]:
        f=frame if cat=='full' else frame[frame.category==cat]
        ns+=benchmark_summary(f,['accuracy'],arms,dict(model=model,repair_family=family,benchmark='SugarCrepe',category=cat))
    ff=np.load(COLA/model/'features.npz');idx=read(COLA/model/'index.json');items=lines(COLA/'items.jsonl')
    vi={s:i for i,s in enumerate(idx['images'])};ti={s:i for i,s in enumerate(idx['texts'])};cc=[]
    for m in models:
        t=adapt(ff['texts'],m['checkpoint'])
        for r in items:
            x=ff['images'][[vi[r[f'resolved_image_{i}']['path']] for i in (1,2)]]@t[[ti[r[f'caption_{i}']] for i in (1,2)]].T
            txt=x.diagonal()>x[np.arange(2),1-np.arange(2)];im=x.diagonal()>x[1-np.arange(2),np.arange(2)]
            cc.append(dict(arm=m['arm'],seed=m['seed'],example_id=r['example_id'],source_id=r['source_cluster'],text_accuracy=float(txt.mean()),image_accuracy=float(im.mean()),official_image_match=float(im.all()),scores=x.tolist()))
    jsonl(dest/'cola_per_example.jsonl',cc)
    ns+=benchmark_summary(pd.DataFrame(cc),['text_accuracy','image_accuracy','official_image_match'],arms,dict(model=model,repair_family=family,benchmark='COLA',category='full_multi_object'))
    jsonl(dest/'natural_summary.jsonl',ns)
    vg=[];vs=[]
    for part in ('pilot','reserve'):
        labels={r['id']:r for r in lines(NAT/part/'accepted_rows.jsonl')}
        for view in ('natural_full_image','natural_foreground_context'):
            path=CACHE/'features'/model/part/view;v=np.load(path/'images.npy');base=np.load(path/'texts.npy')
            gallery={r['row_id']:r for r in lines(path/'gallery_index.jsonl')};queries=lines(path/'index.jsonl')
            for m in models:
                t=adapt(base,m['checkpoint']);current=[]
                for q in queries:
                    ids=q['gallery_ids'];scores=v[[gallery[i]['image_offset'] for i in ids]]@t[q['text_indices'][0]]
                    pos=[i for i in ids if labels[i]['color']==q['color']];neg=[i for i in ids if labels[i]['color']!=q['color']]
                    res=retrieval_query(scores,ids,pos,neg);current.append(res)
                    assert q['partition']==part
                    vg.append({**q,**res,'partition':part,'view':view,'arm':m['arm'],'seed':m['seed'],'scores':dict(zip(ids,scores.tolist()))})
                vs.append(dict(partition=part,view=view,arm=m['arm'],seed=m['seed'],n_queries=len(queries),
                    hit1=float(np.mean([r['hits']['1'] for r in current])),hit5=float(np.mean([r['hits']['5'] for r in current])),
                    uncertainty='Descriptive per seed; galleries overlap across queries.'))
    jsonl(dest/'vg_queries.jsonl',vg);jsonl(dest/'vg_summary.jsonl',vs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--family',required=True);a=ap.parse_args();log(OUT,'start',model=a.model,family=a.family)
    try:main_evaluate(a.model,a.family)
    except BaseException as e:log(OUT,'failed',error=repr(e));raise
    log(OUT,'complete',model=a.model,family=a.family)


if __name__=='__main__':main()
