"""Fixed-checkpoint same-class evaluation, paired source-disjoint anchor statistics."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.routing_same_class_data import OUT; from mirror.cases.color_binding.routing_same_class_data import MODELS; from mirror.cases.color_binding.routing_same_class_data import TRAIN; from mirror.cases.color_binding.routing_same_class_data import PAIRS; from mirror.cases.color_binding.routing_same_class_data import COLORS; from mirror.cases.color_binding.routing_same_class_data import bank; from mirror.cases.color_binding.routing_same_class_data import lines
from mirror.core.features import verify_cache
from mirror.cases.color_binding.routing_adapter_reaudit_v2 import adapted_text; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import measurements; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import CAL; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import MODEL
from mirror.cases.color_binding.routing_adequacy import preference_spec
from mirror.cases.color_binding.outcomes import transitions

ARMS=('F','R','G','I','P','IP','E')
CONTRASTS=tuple((a,'F') for a in ARMS[1:])+(('I','G'),('P','G'),('IP','I'),('IP','E'),('IP','R'))
META={'arm','seed','color','view','anchor_id','pair','source_ids'}


def bootstrap_weights(groups,n=2000,seed=20260923):
    """Fixed pair strata; each row is an independent two-source anchor."""
    groups=np.asarray(groups);rng=np.random.default_rng(seed);unique=sorted(set(groups))
    counts=np.zeros((n,len(groups)));macro=np.zeros(len(groups))
    for key in unique:
        ix=np.flatnonzero(groups==key);counts[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=n)
        macro[ix]=1/(len(unique)*len(ix))
    return {'micro':(np.full(len(groups),1/len(groups)),counts/len(groups)),
            'macro':(macro,counts*macro)}


def estimates(x,weights):
    point,draws=weights
    take=point>0
    return point[take]@x[take],np.quantile(draws[:,take]@x[take],[.025,.975],axis=0).T


def freeze():
    p,rows=bank();enc=read(OUT/'encoding_complete.json')
    assert sha(OUT/'features/complete.json')==enc['feature_complete_sha256']
    models=read(MODELS/'models.json');assert [r['arm'] for r in models]==list(ARMS)
    paths=[Path(__file__),Path(__file__).parent/'tests/test_routing_same_class.py',CAL,
        OUT/'bank_complete.json',OUT/'encoding_complete.json',OUT/'features/complete.json',
        MODELS/'models.json',MODELS/'evaluation_v2/complete.json',MODELS/'evaluation_v2/sugarcrepe_summary.jsonl']
    previous=read(MODELS/'evaluation_v2/complete.json')
    assert previous['models_sha256']==sha(MODELS/'models.json')
    assert sha(MODELS/'evaluation_v2/sugarcrepe_summary.jsonl')==previous['files']['sugarcrepe_summary.jsonl']
    paths += [Path(__file__).with_name(n+'.py') for n in ('outcomes','routing_adapter_reaudit_v2','routing_adequacy','requirements')]
    dump(OUT/'evaluation_protocol.json',dict(inputs={str(p):sha(p) for p in paths},models=models,
        primary_color='red-blue',both_trained_layouts=True,comparisons=CONTRASTS,seed=42,
        bootstrap=dict(replicates=2000,seed=20260923,unit='Independent two-source anchor',
            stratification='Fixed object pair',paired=True,seed_variability=False),
        aggregates=['micro','equal_pair_macro'],object_guard='Mean of four per-image correct-object versus distractor decisions',
        untouched_evaluation_ids=True,training=False,no_checkpoint_selection=True))


def score():
    p,rows=bank();ep=read(OUT/'evaluation_protocol.json');verify_files(ep['inputs'])
    models=ep['models'];verify_files({m['checkpoint']:m['sha256'] for m in models if m['checkpoint']})
    torch.set_num_threads(4);meta,done=verify_cache(OUT/'features')
    index=lines(OUT/'features/index.jsonl');assert [r['anchor_id'] for r in index]==[r['anchor_id'] for r in rows]
    images=np.load(OUT/'features/images.npy');texts=np.load(OUT/'features/texts.npy');cal=read(CAL)
    # Exact caption/template feature replay against the known training cache.
    old_groups=lines(TRAIN/'features/text_groups.jsonl');old_text=np.load(TRAIN/'features/texts.npy')
    group_map={tuple(r['templates']):r['index'] for r in old_groups}
    replay=max(float(np.max(abs(texts[r['index']]-old_text[group_map[tuple(r['templates'])]])))
               for r in lines(OUT/'features/text_groups.jsonl'))
    assert replay<2e-6,replay
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False);records=[];decisions=[]
    for model in models:
        with torch.inference_mode():t=adapted_text(texts,model).numpy()
        for color_index,colors in enumerate(COLORS):
            color='-'.join(colors)
            for view in ('canvas','swapped_canvas'):
                matrices=[];object_margins=[]
                for r in index:
                    ii=[j for j,s in enumerate(r['state_names']) if s.startswith(color+'/'+view+'/')]
                    expected=[color+'/'+view+'/'+a+'_'+b for a in colors for b in colors]
                    assert [r['state_names'][j] for j in ii]==expected
                    v=images[r['image_offset']+np.asarray(ii)]
                    x=v@t[r['text_indices'][4*color_index:4*color_index+4]].T;matrices.append(x)
                    obj=v@t[r['text_indices'][8:10]].T;object_margins.append(obj[:,0]-obj[:,1])
                    common=dict(arm=model['arm'],seed=42,color=color,view=view,anchor_id=r['anchor_id'])
                    for state in range(4):
                        wrong=[j for j in range(4) if j!=state]
                        mm=float(x[state,state]-max(x[state,wrong]))
                        decisions.append(dict(**common,state=state,kind='four_candidate',margin=mm,correct=mm>0))
                    for state,other in ((1,2),(2,1)):
                        mm=float(x[state,state]-x[state,other])
                        decisions.append(dict(**common,state=state,kind='exchange',margin=mm,correct=mm>0))
                x=np.stack(matrices);om=np.stack(object_margins);met=measurements(x,cal)
                met['caption_accuracy']=(x.diagonal(axis1=1,axis2=2)>np.where(np.eye(4,dtype=bool)[None],-np.inf,x).max(2)).mean(1)
                met['object_guard']=(om>0).mean(1);met['object_guard_margin']=om.mean(1)
                for j,r in enumerate(rows):
                    records.append(dict(arm=model['arm'],seed=42,color=color,view=view,anchor_id=r['anchor_id'],
                        pair='+'.join(r['objects']),source_ids=json_string(r['source_ids']),
                        **{k:float(a[j]) for k,a in met.items()},**{f's_{i}_{k}':float(x[j,i,k]) for i in range(4) for k in range(4)}))
        print('SCORED',model['arm'],len(rows),'anchors',flush=True)
    frame=pd.DataFrame(records);decision=pd.DataFrame(decisions)
    with (dest/'per_example.csv').open('x') as f:frame.to_csv(f,index=False)
    with (dest/'decisions.csv').open('x') as f:decision.to_csv(f,index=False)
    metrics=[k for k in frame if k not in META and not k.startswith('s_')]
    summaries=[];contrasts=[];direction_summary=[];direction_rows=[]
    definition,contexts=preference_spec(cal)
    for (color,view),cell in frame.groupby(['color','view']):
        ids=sorted(cell.anchor_id.unique());group=cell[cell.arm=='F'].set_index('anchor_id').loc[ids,'pair'].to_numpy()
        weightsets=bootstrap_weights(group)
        for pair in sorted(set(group)):
            take=np.flatnonzero(group==pair);w=bootstrap_weights(group[take])['micro']
            mean=np.zeros(len(ids));mean[take]=w[0];draw=np.zeros((2000,len(ids)));draw[:,take]=w[1]
            weightsets[pair]=(mean,draw)
        arrays={a:cell[cell.arm==a].set_index('anchor_id').loc[ids,metrics].to_numpy() for a in ARMS}
        for cohort,weights in weightsets.items():
            count=len(ids) if cohort in ('micro','macro') else int((group==cohort).sum())
            for arm,x in arrays.items():
                means,ci=estimates(x,weights)
                for j,k in enumerate(metrics):summaries.append(dict(color=color,view=view,cohort=cohort,arm=arm,metric=k,
                    n_anchors=count,n_source_images=2*count,mean=float(means[j]),ci95=ci[j].tolist()))
            for a,b in CONTRASTS:
                means,ci=estimates(arrays[a]-arrays[b],weights)
                for j,k in enumerate(metrics):contrasts.append(dict(color=color,view=view,cohort=cohort,contrast=a+'-'+b,
                    metric=k,n_anchors=count,mean=float(means[j]),ci95=ci[j].tolist(),paired=True))
        frozen=cell[cell.arm=='F'].set_index('anchor_id').loc[ids]
        for arm in ARMS:
            current=cell[cell.arm==arm].set_index('anchor_id').loc[ids];improvements=[]
            for c in contexts:
                before=frozen['contrast/'+c['contrast']].to_numpy();after=current['contrast/'+c['contrast']].to_numpy()
                delta=after-before if c['kind']=='binding' else abs(before)-abs(after)
                improvements.append((c['contrast'],c['kind'],delta))
            improvements += [('response','response',(current.response-frozen.response).to_numpy()),
                ('absolute_preference','preference',(frozen.absolute_preference-current.absolute_preference).to_numpy())]
            for j,anchor in enumerate(ids):
                direction_rows.append(dict(color=color,view=view,arm=arm,anchor_id=anchor,pair=group[j],
                    **{name:float(v[j]) for name,_,v in improvements}))
            for kind in ('binding','unwanted','response','preference'):
                delta=np.stack([v for _,k,v in improvements if k==kind],1)
                x=np.stack((delta.mean(1),(delta>1e-5).mean(1),(delta<-1e-5).mean(1)),1)
                for cohort,weights in weightsets.items():
                    mean,ci=estimates(x,weights)
                    direction_summary.append(dict(color=color,view=view,arm=arm,kind=kind,cohort=cohort,
                        n_contrasts=delta.shape[1],metrics=['directional_improvement','fraction_better','fraction_worse'],
                        mean=mean.tolist(),ci95=ci.tolist(),tolerance=1e-5))
    jsonl(dest/'summary.jsonl',summaries);jsonl(dest/'contrasts.jsonl',contrasts)
    jsonl(dest/'interaction_direction_summary.jsonl',direction_summary)
    with (dest/'summary.csv').open('x') as f:pd.DataFrame(summaries).to_csv(f,index=False)
    with (dest/'interaction_directions.csv').open('x') as f:pd.DataFrame(direction_rows).to_csv(f,index=False)
    trans=[]
    pair_map={r['anchor_id']:'+'.join(r['objects']) for r in rows};decision['pair']=decision.anchor_id.map(pair_map)
    for (color,view,kind),g in decision.groupby(['color','view','kind']):
        for pair in ['all',*sorted(g.pair.unique())]:
            h=g if pair=='all' else g[g.pair==pair];keys=['anchor_id','state']
            f=h[h.arm=='F'].set_index(keys).sort_index()
            for arm in ARMS:
                a=h[h.arm==arm].set_index(keys).loc[f.index]
                trans.append(dict(color=color,view=view,kind=kind,pair=pair,arm=arm,
                    **transitions(~f.correct.to_numpy(bool),~a.correct.to_numpy(bool))))
    jsonl(dest/'decision_transitions.jsonl',trans)
    report(dest,rows,frame,summaries,contrasts)
    dump(dest/'complete.json',dict(protocol_sha256=sha(OUT/'evaluation_protocol.json'),models_sha256=sha(MODELS/'models.json'),
        n_anchors=len(rows),n_source_images=2*len(rows),n_per_example_rows=len(frame),text_feature_replay_error=replay,
        files={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()},training=False,reserve=False,seed=42))
    print('EVALUATION COMPLETE',dest/'REPORT.md',flush=True)


def json_string(value):
    import json
    return json.dumps(value)


def report(dest,rows,frame,summaries,contrasts):
    counts=read(OUT/'bank_complete.json')['pair_counts']
    text=['# Same-class, new-source routing test (seed42)','',
        f'{len(rows)} independent two-source anchors / {2*len(rows)} source images; no image reused across anchors or repair/guard training. Existing fixed-final checkpoints only; no retraining. Post-specified matched-class diagnostic.','',
        'These are recolored cutouts from natural images pasted onto canvases, matching the training construction. They are not unedited natural-image benchmarks. Both layouts and both color pairs were trained.','',
        '## Counts','', '| Trained object pair | New anchors |','|---|---:|']
    for pair,n in sorted(counts.items()):text.append(f'| {pair} | {n} |')
    text += ['', '## Pooled individual decisions and mechanism','',
        '| Colors | Layout | Arm | Exchange % | Four-candidate % | Binding | Absolute cross | Response e | Absolute preference b |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for (color,view,arm),g in frame.groupby(['color','view','arm']):
        m=g.mean(numeric_only=True)
        text.append(f'| {color} | {view} | {arm} | {100*m.exchange_accuracy:.2f} | {100*m.caption_accuracy:.2f} | {m.all_binding:.4f} | {m.all_cross_abs:.4f} | {m.response:.4f} | {m.absolute_preference:.4f} |')
    text += ['', 'F frozen; R historical ranking objective; G guards; I guards+interactions/response; P guards+preference; IP combined; E guards+endpoint hinge. All trained arms use the same seed42 training bank and budget.','',
        '## Red/blue primary comparison: full IS minus frozen','',
        '| Layout | Aggregate | Exchange difference pp [95% CI] | Binding change [95% CI] | Absolute cross change [95% CI] |',
        '|---|---|---:|---:|---:|']
    for view in ('canvas','swapped_canvas'):
        for cohort in ('micro','macro'):
            rr={r['metric']:r for r in contrasts if r['color']=='red-blue' and r['view']==view and r['cohort']==cohort and r['contrast']=='IP-F'}
            values=[]
            for metric,scale in [('exchange_accuracy',100),('all_binding',1),('all_cross_abs',1)]:
                r=rr[metric];values.append(f"{scale*r['mean']:+.4f} [{scale*r['ci95'][0]:+.4f}, {scale*r['ci95'][1]:+.4f}]")
            text.append('| '+view+' | '+cohort+' | '+' | '.join(values)+' |')
    text += ['', 'All per-pair and equal-pair macro summaries, all clause/direction values, individual decision repair/break counts and matched controls are retained. Intervals use 2,000 paired anchor bootstrap draws within each fixed object pair; they describe source uncertainty conditional on seed42, not seed variability. Red/blue is primary; green/yellow is the predeclared secondary trained-color condition.','',
        '## Existing natural preservation (unchanged checkpoints)','', '| Arm | Full SugarCrepe % | N |','|---|---:|---:|']
    for r in lines(MODELS/'evaluation_v2/sugarcrepe_summary.jsonl'):
        if r['category']=='full':text.append(f"| {r['arm']} | {100*r['accuracy']:.2f} | {r['n']} |")
    text += ['', 'SugarCrepe scores are reused unchanged from the verified earlier evaluation, not used to select the new test or checkpoints. The previous 49-anchor unseen-class/in-situ evaluation remains valid and separate; this same-class result cannot establish unseen-class transfer. No automatic additional seeds or manuscript edits.']
    with (dest/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','score']);args=parser.parse_args()
    log(OUT,'start',stage='evaluation_'+args.action)
    try:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage='evaluation_'+args.action,error=repr(exc));raise
    log(OUT,'complete',stage='evaluation_'+args.action)


if __name__=='__main__':main()
