"""Fixed-last pilot retest: all arms, fixed sources/regimes, no new fitting."""
import numpy as np
import pandas as pd
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import verify_files
from mirror.cases.color_binding.routing_repair_pilot import OUT; from mirror.cases.color_binding.routing_repair_pilot import SUGAR; from mirror.cases.color_binding.routing_repair_pilot import AUDIT_CACHE; from mirror.cases.color_binding.routing_repair_pilot import CAL; from mirror.cases.color_binding.routing_repair_pilot import MODEL; from mirror.cases.color_binding.routing_repair_pilot import verify
from mirror.cases.color_binding.routing_adapter_reaudit_v2 import adapted_text; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import measurements; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import OUT as REAUDIT; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import summary_interval
from mirror.cases.color_binding.diagnosis_guided_protocol import regime_contrasts
from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.cases.color_binding.outcomes import transitions; from mirror.cases.color_binding.outcomes import retrieval_query


def clustered_difference(values, sources, n=2000):
    x=np.asarray(values,float)
    if x.ndim==1:x=x[:,None]
    _,inverse=np.unique(sources,return_inverse=True);count=inverse.max()+1
    sums=np.zeros((count,x.shape[1]));np.add.at(sums,inverse,x)
    ns=np.bincount(inverse,minlength=count)
    w=np.random.default_rng(20260923).multinomial(count,np.full(count,1/count),size=n)
    draws=(w@sums)/(w@ns)[:,None]
    return dict(mean=x.mean(0).tolist(),ci95=np.quantile(draws,[.025,.975],axis=0).T.tolist(),n_sources=int(count),n_rows=len(x))


def evaluate():
    verify();torch.set_num_threads(4)
    done=read(OUT/'training_complete.json');assert sha(OUT/'models.json')==done['models_sha256']
    models=read(OUT/'models.json');assert [m['arm'] for m in models]==['F','R','G','I','P','IP','E']
    verify_files({m['checkpoint']:m['sha256'] for m in models if m['checkpoint']})
    enc=read(OUT/'encoding_complete.json');verify_files({str(OUT/k):v for k,v in enc['files'].items()})
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False)
    cal=read(CAL);wanted={r['anchor_id'] for r in lines(REAUDIT/'frozen_direct_regimes.jsonl')}
    index=[r for r in lines(AUDIT_CACHE/'index.jsonl') if r['anchor_id'] in wanted];assert len(index)==49
    images=np.load(AUDIT_CACHE/'images.npy',mmap_mode='r');texts=np.load(AUDIT_CACHE/'texts.npy')
    object_mapping=lines(OUT/'evaluation_object_mapping.jsonl');object_ids={r['anchor_id']:i for i,r in enumerate(object_mapping)}
    object_text=np.load(OUT/'evaluation_object_texts.npy');records=[];decision_records=[]
    old_frozen=pd.read_csv(REAUDIT/'per_example.csv');old_frozen=old_frozen[(old_frozen.arm=='frozen')&(old_frozen.bank=='new_pilot')].set_index(['view','anchor_id'])
    replay=0.
    for model in models:
        with torch.inference_mode():
            adapted=adapted_text(texts,model).numpy();objects=adapted_text(object_text,model).numpy()
        for view in ('canvas','swapped_canvas','in_situ'):
            matrices=[];guards=[]
            for r in index:
                ii=[j for j,n in enumerate(r['state_names']) if n.startswith('blend90_luminance/red-blue/'+view+'/')];assert len(ii)==4
                image=images[r['image_offset']+np.asarray(ii)];x=image@adapted[r['text_indices'][:4]].T;matrices.append(x)
                om=image.mean(0)@objects[object_ids[r['anchor_id']]].T;guards.append(float(om[0]-om[1]))
                for state in range(4):
                    wrong=[j for j in range(4) if j!=state];margin=float(x[state,state]-max(x[state,wrong]))
                    decision_records.append(dict(arm=model['arm'],seed=42,view=view,anchor_id=r['anchor_id'],state=state,comparison='four_candidate',margin=margin,correct=margin>0))
                for state,other in ((1,2),(2,1)):
                    margin=float(x[state,state]-x[state,other])
                    decision_records.append(dict(arm=model['arm'],seed=42,view=view,anchor_id=r['anchor_id'],state=state,comparison='exchange',margin=margin,correct=margin>0))
                if model['arm']=='F':
                    prior=old_frozen.loc[(view,r['anchor_id']),[f's_{a}_{b}' for a in range(4) for b in range(4)]].to_numpy(float).reshape(4,4)
                    replay=max(replay,float(abs(x-prior).max()))
            matrices=np.stack(matrices);met=measurements(matrices,cal)
            # Unlike argmax tie-breaking, strict behavioral comparisons count ties as failures.
            correct=np.eye(4,dtype=bool)[None]
            met['caption_accuracy']=(np.diagonal(matrices,axis1=1,axis2=2)>np.where(correct,-np.inf,matrices).max(2)).mean(1)
            met['object_guard']=(np.asarray(guards)>0).astype(float);met['object_guard_margin']=np.asarray(guards)
            for j,r in enumerate(index):
                records.append(dict(arm=model['arm'],seed=42,view=view,anchor_id=r['anchor_id'],
                    **{k:float(v[j]) for k,v in met.items()},**{f's_{a}_{b}':float(matrices[j,a,b]) for a in range(4) for b in range(4)}))
    assert replay<2e-6,replay
    frame=pd.DataFrame(records);decisions=pd.DataFrame(decision_records)
    with (dest/'per_example.csv').open('x') as f:frame.to_csv(f,index=False)
    with (dest/'decisions.csv').open('x') as f:decisions.to_csv(f,index=False)
    metrics=[k for k in frame if k not in {'arm','seed','view','anchor_id'} and not k.startswith('s_')]
    summaries=[];contrasts=[];transition_rows=[]
    for view,g in frame.groupby('view'):
        arrays={}
        for arm,h in g.groupby('arm'):
            h=h.sort_values('anchor_id');x=h[metrics].to_numpy()[None];arrays[arm]=x
            ci,_=summary_interval(x,np.random.default_rng(20260923))
            for j,k in enumerate(metrics):summaries.append(dict(view=view,arm=arm,metric=k,n_anchors=49,mean=float(x[0,:,j].mean()),ci95=ci[j].tolist()))
        for a,b in [('I','G'),('P','G'),('IP','I'),('IP','E'),('IP','R'),('IP','F')]:
            delta=arrays[a]-arrays[b];ci,_=summary_interval(delta,np.random.default_rng(20260923))
            for j,k in enumerate(metrics):contrasts.append(dict(view=view,contrast=a+'-'+b,metric=k,mean=float(delta[0,:,j].mean()),ci95=ci[j].tolist(),paired=True))
    for (view,kind),g in decisions.groupby(['view','comparison']):
        keys=['anchor_id','state'];f=g[g.arm=='F'].set_index(keys).sort_index()
        for arm,h in g.groupby('arm'):
            h=h.set_index(keys).loc[f.index]
            transition_rows.append(dict(view=view,comparison=kind,arm=arm,**transitions(~f.correct.to_numpy(bool),~h.correct.to_numpy(bool))))
    jsonl(dest/'summary.jsonl',summaries);jsonl(dest/'contrasts.jsonl',contrasts);jsonl(dest/'decision_transitions.jsonl',transition_rows)
    jsonl(dest/'regime_contrasts.jsonl',regime_contrasts(frame,lines(REAUDIT/'frozen_direct_regimes.jsonl')))
    with (dest/'summary.csv').open('x') as f:pd.DataFrame(summaries).to_csv(f,index=False)
    # Full official SugarCrepe, frozen features and all seven categories. No slices.
    features=np.load(SUGAR/'features.npz');v=features['images'];t=features['texts'];idx=read(SUGAR/'indices.json')
    reference=pd.read_csv(SUGAR/'frozen_seed0.csv');ii={s:i for i,s in enumerate(idx['names'])};tt={s:i for i,s in enumerate(idx['prompts'])}
    image_ids=np.array([ii[s] for s in reference.filename]);pos_ids=np.array([tt[s] for s in reference.caption]);neg_ids=np.array([tt[s] for s in reference.negative_caption])
    sugar={};sugar_summaries=[];sugar_contrasts=[];sugar_dir=dest/'sugarcrepe';sugar_dir.mkdir()
    for model in models:
        with torch.inference_mode():adapted=adapted_text(t,model).numpy()
        pos=np.einsum('nd,nd->n',v[image_ids],adapted[pos_ids]);neg=np.einsum('nd,nd->n',v[image_ids],adapted[neg_ids])
        if model['arm']=='F':
            assert max(abs(pos-reference.pos_score).max(),abs(neg-reference.neg_score).max())<2e-6
        h=reference[['subset','example_id','filename','caption','negative_caption']].copy()
        h['pos_score']=pos;h['neg_score']=neg;h['margin']=pos-neg;h['correct']=pos>neg;h['arm']=model['arm'];h['seed']=42
        with (sugar_dir/f'{model["arm"]}.csv').open('x') as f:h.to_csv(f,index=False)
        sugar[model['arm']]=h
        for subset in ['full',*sorted(h.subset.unique())]:
            a=h if subset=='full' else h[h.subset==subset]
            sugar_summaries.append(dict(arm=model['arm'],category=subset,accuracy=float(a.correct.mean()),n=len(a),n_images=int(a.filename.nunique())))
    for arm in ['R','G','I','P','IP','E']:
        for subset in ['full',*sorted(reference.subset.unique())]:
            mask=np.ones(len(reference),bool) if subset=='full' else reference.subset==subset
            diff=sugar[arm].correct.to_numpy(float)[mask]-sugar['F'].correct.to_numpy(float)[mask]
            sugar_contrasts.append(dict(contrast=arm+'-F',category=subset,**clustered_difference(diff,reference.filename.to_numpy()[mask])))
    for other in ['R','I','E']:
        for subset in ['full',*sorted(reference.subset.unique())]:
            mask=np.ones(len(reference),bool) if subset=='full' else reference.subset==subset
            diff=sugar['IP'].correct.to_numpy(float)[mask]-sugar[other].correct.to_numpy(float)[mask]
            sugar_contrasts.append(dict(contrast='IP-'+other,category=subset,**clustered_difference(diff,reference.filename.to_numpy()[mask])))
    jsonl(dest/'sugarcrepe_summary.jsonl',sugar_summaries);jsonl(dest/'sugarcrepe_contrasts.jsonl',sugar_contrasts)
    # Fixed natural retrieval: exact prior gallery/query definitions, both views.
    source={r['id']:r for r in lines(ROOT/'clip/interbind_natural_localizer_quality_20260922/pilot/accepted_rows.jsonl')}
    natural=[];natural_summary=[]
    for view in ('natural_foreground_context','natural_full_image'):
        path=AUDIT_CACHE.parent/view;v=np.load(path/'images.npy');t=np.load(path/'texts.npy')
        gallery={r['row_id']:r for r in lines(path/'gallery_index.jsonl')};queries=lines(path/'index.jsonl')
        for model in models:
            with torch.inference_mode():adapted=adapted_text(t,model).numpy()
            results=[]
            for query in queries:
                ids=query['gallery_ids'];offsets=[gallery[i]['image_offset'] for i in ids]
                scores=v[offsets]@adapted[query['text_indices'][0]]
                pos=[i for i in ids if source[i]['color']==query['color']];neg=[i for i in ids if source[i]['color']!=query['color']]
                r=retrieval_query(scores,ids,pos,neg);results.append(r)
                natural.append(dict(arm=model['arm'],seed=42,view=view,**query,**r,scores=dict(zip(ids,scores.tolist()))))
            natural_summary.append(dict(arm=model['arm'],view=view,n_queries=len(queries),n_gallery=len(gallery),
                hit1=float(np.mean([r['hits']['1'] for r in results])),hit5=float(np.mean([r['hits']['5'] for r in results])),
                uncertainty='Descriptive fixed-query preservation; overlapping galleries are not treated as independent Bernoulli trials'))
    jsonl(dest/'natural_queries.jsonl',natural);jsonl(dest/'natural_summary.jsonl',natural_summary)
    report=['# One-seed diagnosis-guided routing repair pilot','',
        'New L/14 seed42 checkpoints, not the historical paper runs. All six training arms use the same 1,280 four-color training lattices, 324 updates, initialization and dropout stream. Six epochs; fixed last only. No reserve scores or checkpoint/benchmark selection. Uncertainty is exploratory paired source bootstrap, not three-seed variability.','',
        '## Routing: individual exchange decisions and continuous diagnostics','',
        '| View | Arm | Exchange accuracy % | Four-candidate accuracy % | e | |b| | All binding | All absolute cross |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for view,g in frame.groupby('view'):
        for arm,h in g.groupby('arm'):
            m=h.mean(numeric_only=True)
            report.append(f'| {view} | {arm} | {100*m.exchange_accuracy:.2f} | {100*m.caption_accuracy:.2f} | {m.response:.4f} | {m.absolute_preference:.4f} | {m.all_binding:.4f} | {m.all_cross_abs:.4f} |')
    report+=['','F frozen; R historical ranking loss; G shared guards; I G+complete interactions; P G+preference; IP G+both; E G+matched exchange endpoint loss. Ranking is not given the new guards. All guarded arms share them. Natural and object-caption inputs are the same across arms.','',
        '## Full SugarCrepe preservation','', '| Arm | Accuracy % | N |','|---|---:|---:|']
    for r in sugar_summaries:
        if r['category']=='full':report.append(f'| {r["arm"]} | {100*r["accuracy"]:.2f} | {r["n"]} |')
    report+=['','All seven categories and paired source-cluster intervals are retained. Fixed natural retrieval results are in `natural_summary.jsonl`; per-query scores retain the exact original labels and galleries.','',
        '## Diagnostic tests and scope','',
        'The frozen direct-view regimes (30 nonpositive response, 16 preference-dominated, 3 both correct) are reused across all arms and outcome views. `regime_contrasts.jsonl` reports all specified component contrasts and effect-modification contrasts, including null/adverse effects. The smallest regime is descriptive only. `decision_transitions.jsonl` counts repaired and broken individual decisions; interaction direction and conformance remain separate.','',
        'No automatic extension to other seeds or claim of a final successful recipe. Read per-component changes, independent-view behavior and preservation together. A one-point preservation tolerance is an operational rule, not a guarantee or a benchmark-tuning target.']
    with (dest/'REPORT.md').open('x') as f:f.write('\n'.join(report)+'\n')
    dump(dest/'complete.json',dict(models_sha256=sha(OUT/'models.json'),protocol_sha256=sha(OUT/'protocol.json'),frozen_replay_max_error=replay,
        files={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()},reserve=False,automatic_seed_escalation=False))
    print('PILOT COMPLETE',dest/'REPORT.md',flush=True)
