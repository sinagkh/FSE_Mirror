"""Create-only three-seed summaries of the fixed 75% routing replication."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from mirror.cases.color_binding import replicate as run
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import lines

OUT=run.OUT/'results'
SEEDS=(42,43,44)
ROOTS=[run.OLD,*[run.OUT/f'seed{s}' for s in SEEDS[1:]]]
DRAWS=5000
BOOTSEED=20260930
METADATA={'name','test','anchor_id','source_ids'}
PRIMARY=[c+'_'+o for c in ('red-blue','green-yellow') for o in ('canonical','reversed')]


def bootstrap(values,rows):
    """Crossed seed/source bootstrap; shared seed and source draws across arms.

    Sources are the same across seeds, so one cluster draw is shared across all
    resampled seeds rather than pretending each seed has independent test data.
    """
    assert values.ndim==3 and values.shape[:2]==(3,len(rows))
    cl=run.ev.analysis.source_clusters(rows);nc=int(cl.max())+1
    counts=np.bincount(cl,minlength=nc)
    sums=np.stack([np.stack([np.bincount(cl,weights=arr[:,j],minlength=nc)
                  for j in range(arr.shape[1])],axis=1) for arr in values])
    flat=sums.transpose(1,0,2).reshape(nc,-1)
    rng=np.random.default_rng(BOOTSEED);seedrng=np.random.default_rng(BOOTSEED+1)
    conditional=[];hierarchical=[]
    for start in range(0,DRAWS,100):
        n=min(100,DRAWS-start)
        w=rng.multinomial(nc,np.full(nc,1/nc),size=n)
        sw=seedrng.multinomial(3,np.full(3,1/3),size=n)/3
        draws=(w@flat).reshape(n,3,-1)/(w@counts)[:,None,None]
        conditional.append(draws.mean(1))
        hierarchical.append(np.einsum('bs,bsp->bp',sw,draws))
    return np.concatenate(conditional),np.concatenate(hierarchical),nc


def interval_record(seedmeans,items,hierarchical):
    seedmeans=np.asarray(seedmeans)
    assert np.isfinite(seedmeans).all() and np.isfinite(items).all() and np.isfinite(hierarchical).all()
    return dict(mean=float(seedmeans.mean()),sample_sd=float(seedmeans.std(ddof=1)),
        per_seed={str(s):float(a) for s,a in zip(SEEDS,seedmeans)},
        ci95_items=np.quantile(items,[.025,.975]).tolist(),
        ci95_seed_items=np.quantile(hierarchical,[.025,.975]).tolist(),
        ci_familywise95_seed_items=np.quantile(hierarchical,[.05/18,1-.05/18]).tolist())


def load(family,test):
    all_values=[];reference=None;names=None;keys=None
    for root in ROOTS:
        rr=lines(root/'analysis'/family/(test+'_per_example.jsonl'))
        ns=list(dict.fromkeys(r['name'] for r in rr));groups={n:[r for r in rr if r['name']==n] for n in ns}
        first=groups[ns[0]];kk=[k for k,v in first[0].items() if k not in METADATA]
        rows=[dict(anchor_id=r['anchor_id'],source_ids=r['source_ids']) for r in first]
        if reference is None:reference=rows;names=ns;keys=kk
        assert rows==reference and names==ns and keys==kk
        for n in names:assert [(r['anchor_id'],r['source_ids']) for r in groups[n]]==[(r['anchor_id'],r['source_ids']) for r in first]
        all_values.append(np.stack([np.array([[r[k] for k in keys] for r in groups[n]]) for n in names],axis=1))
    # [seed, source, method, metric]
    values=np.stack(all_values)
    fi=names.index('Frozen')
    assert np.allclose(values[:, :, fi],values[0:1,:,fi],atol=1e-7,rtol=1e-7)
    return values,reference,names,keys


def summarize(family,test,loaded):
    values,rows,names,keys=loaded
    shape=values.shape
    bi,bh,nc=bootstrap(values.reshape(3,len(rows),-1),rows)
    bi=bi.reshape(DRAWS,len(names),len(keys));bh=bh.reshape(DRAWS,len(names),len(keys))
    point=values.mean(1);records=[]
    base=dict(family=family,test=test,n_items=len(rows),n_source_clusters=nc,n_seeds=3,draws=DRAWS)
    for n,name in enumerate(names):
        for k,key in enumerate(keys):
            records.append(dict(base,comparison=name,metric=key,**interval_record(point[:,n,k],bi[:,n,k],bh[:,n,k])))
    i=names.index('IS')
    for n,name in enumerate(names):
        if n==i:continue
        for k,key in enumerate(keys):
            records.append(dict(base,comparison='IS - '+name,metric=key,
                **interval_record(point[:,i,k]-point[:,n,k],bi[:,i,k]-bi[:,n,k],bh[:,i,k]-bh[:,n,k])))
    if 'cross' in keys:
        k=keys.index('cross');f=names.index('Frozen')
        for n,name in enumerate(names):
            records.append(dict(base,comparison=name,metric='cross_reduction_percent',
                **interval_record(100*(1-point[:,n,k]/point[:,f,k]),100*(1-bi[:,n,k]/bi[:,f,k]),100*(1-bh[:,n,k]/bh[:,f,k]))))
    metric='exchange_accuracy' if 'exchange_accuracy' in keys else keys[0]
    print('AGGREGATED',family,test,{n:round(float(point[:,j,keys.index(metric)].mean()),5) for j,n in enumerate(names)},flush=True)
    return records


def diagnostic_ratios(color,order):
    rows=lines(run.p.CONFIRM/'rows.jsonl');names=['Frozen',*run.ARMS];n=len(rows)
    decisions={};words={};measures={}
    raw=[np.load(root/'analysis/primary'/f'{color}_{order}_scores.npz') for root in ROOTS]
    for name in names:
        xx=np.stack([np.stack([r[v+'/'+name] for v in run.p.VIEWS],axis=1) for r in raw])
        decisions[name]=np.stack((xx[:,:,:,1,1]>xx[:,:,:,1,2],xx[:,:,:,2,2]>xx[:,:,:,2,1]),axis=-1)
        words[name]=np.stack([((xx[:,:,:,i,i]>xx[:,:,:,i,i^1])&(xx[:,:,:,i,i]>xx[:,:,:,i,i^2])) for i in (1,2)],axis=-1)
        mm=run.ev.analysis.measurements(xx.reshape(-1,4,4))
        measures[name]={k:a.reshape(3,n,2) for k,a in mm.items()}
    ratios=[];cols=[]
    def add(label,num,den):
        ratios.append(label);cols.extend((num,den))
    for name in names:
        for metric,mask,state in [('repair',~decisions['Frozen'],decisions[name]),
                                  ('break',decisions['Frozen'],~decisions[name]),
                                  ('word_correct_assignment_error',words['Frozen'],~decisions[name])]:
            add(dict(type=metric,name=name),(state&mask).sum((2,3)),mask.sum((2,3)))
    f=measures['Frozen'];strata=np.where(f['response']<=0,0,np.where(f['response']<=f['preference'],1,2))
    for view in (0,1):
        for rid,regime in enumerate(('nonpositive_response','preference_dominated','both_correct')):
            mask=strata[:,:,view]==rid
            for name in names:
                add(dict(type='regime_accuracy',name=name,diagnosis_view=run.p.VIEWS[view],regime=regime),
                    measures[name]['exchange_accuracy'][:,:,1-view]*mask,mask.astype(float))
    arr=np.stack(cols,axis=-1);bi,bh,nc=bootstrap(arr,rows);pp=arr.mean(1)
    result=[];lookup={};bsvalues={}
    for j,label in enumerate(ratios):
        assert (pp[:,2*j+1]>0).all()
        if not ((bi[:,2*j+1]>0).all() and (bh[:,2*j+1]>0).all()):
            raise ValueError('A diagnostic stratum has empty bootstrap draws; handle explicitly')
        a=pp[:,2*j]/pp[:,2*j+1];b=bi[:,2*j]/bi[:,2*j+1];c=bh[:,2*j]/bh[:,2*j+1]
        rec=dict(color=color,order=order,n_items=n,n_source_clusters=nc,**label,**interval_record(a,b,c),
            denominator_per_seed={str(s):int(arr[k,:,2*j+1].sum()) for k,s in enumerate(SEEDS)})
        result.append(rec)
        if label['type']=='regime_accuracy':
            key=(label['diagnosis_view'],label['regime'],label['name']);bsvalues[key]=(a,b,c)
    for view in run.p.VIEWS:
        for arm in ('no_response','no_preference'):
            first,second=('nonpositive_response','preference_dominated') if arm=='no_response' else ('preference_dominated','nonpositive_response')
            arrays=[bsvalues[view,reg,name] for reg,name in ((first,'IS'),(first,arm),(second,'IS'),(second,arm))]
            a,b,c=[arrays[0][i]-arrays[1][i]-arrays[2][i]+arrays[3][i] for i in range(3)]
            result.append(dict(color=color,order=order,type='diagnosis_guided',name=arm,diagnosis_view=view,
                contrast=first+' minus '+second,n_items=n,n_source_clusters=nc,**interval_record(a,b,c)))
    return result


def csvwrite(path,records):
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in records)


def aggregate():
    run.verify()
    for root in ROOTS:
        assert (root/'complete.json').exists()
        for family in ('primary','transfer','natural','diagnostics'):
            verify_files(read(root/'analysis'/family/'complete.json')['files'])
    OUT.mkdir(parents=True,exist_ok=False)
    paper=run.p.ROOT/'FSE_VLM/manuscript_v6_3'
    paths=[Path(__file__),run.PLAN,run.OUT/'protocol.json',*[r/'complete.json' for r in ROOTS],
           *[paper/'generated'/f for f in ('routing_primary.tex','routing_transfer.tex','routing_components.tex','routing_diagnosis.tex','routing_preservation.tex')]]
    dump(OUT/'analysis_protocol.json',dict(inputs={str(f):sha(f) for f in paths},seeds=SEEDS,draws=DRAWS,
        source_rng_seed=BOOTSEED,seed_rng_seed=BOOTSEED+1,ddof=1,
        bootstrap='Shared connected-source draw across resampled seeds; paired methods; ratios recomputed in every draw',
        familywise='Bonferroni nine transfer tests; other CIs pointwise',
        pooled_primary='Equal average of two trained color pairs and two caption orders within source, before source resampling',
        seed42_development=True,manuscript_not_edited=True))
    records=[]
    for family in ('primary','transfer','natural'):
        index=read(ROOTS[0]/'analysis'/family/'score_index.json')
        for entry in index:
            test=entry['test'];records.extend(summarize(family,test,load(family,test)))
    loaded=[load('primary',test) for test in PRIMARY]
    assert all(z[1:]==loaded[0][1:] for z in loaded)
    pooled=(np.mean([z[0] for z in loaded],axis=0),*loaded[0][1:])
    records.extend(summarize('primary','trained_colors_both_orders',pooled))
    jsonl(OUT/'all_metrics.jsonl',records);csvwrite(OUT/'all_metrics.csv',records)
    diagnostics=[]
    for color in ('red-blue','green-yellow'):
        for order in ('canonical','reversed'):diagnostics.extend(diagnostic_ratios(color,order))
    jsonl(OUT/'diagnostics.jsonl',diagnostics);csvwrite(OUT/'diagnostics.csv',diagnostics)
    directions=[]
    for test in PRIMARY:
        for model in ('Ranking','IS'):
            for ctx in run.p.metrics.all_contexts():
                binding=ctx['kind']=='binding';key=('contrast/' if binding else 'absolute/')+ctx['name']
                r=next(r for r in records if r['family']=='primary' and r['test']==test and r['comparison']==model and r['metric']==key)
                f=next(r for r in records if r['family']=='primary' and r['test']==test and r['comparison']=='Frozen' and r['metric']==key)
                change={s:(r['per_seed'][s]-f['per_seed'][s])*(1 if binding else -1) for s in map(str,SEEDS)}
                d=dict(test=test,model=model,context=ctx['name'],kind=ctx['kind'],per_seed_directional_change=change,
                       all_seeds_expected_direction=all(v>0 for v in change.values()))
                if model=='IS':
                    diff=next(r for r in records if r['family']=='primary' and r['test']==test and r['comparison']=='IS - Frozen' and r['metric']==key)
                    d['ci95_seed_items']=diff['ci95_seed_items'] if binding else [-a for a in diff['ci95_seed_items'][::-1]]
                directions.append(d)
    jsonl(OUT/'directions.jsonl',directions)
    report(records,diagnostics,directions)
    dump(OUT/'complete.json',dict(all_three_seeds_completed=True,manuscript_unchanged=True,
         files={str(f):sha(f) for f in OUT.iterdir() if f.is_file()}))
    print('ALL_RESULTS_COMPLETE',OUT,flush=True)


def report(records,diagnostics,directions):
    def get(f,t,m,c):return next(r for r in records if (r['family'],r['test'],r['metric'],r['comparison'])==(f,t,m,c))
    def value(r,scale=100,sd=True):return f'{r["mean"]*scale:.2f}'+(f' ± {r["sample_sd"]*scale:.2f}' if sd else '')
    def difference(r,scale=100):
        lo,hi=r['ci95_seed_items'];return f'{r["mean"]*scale:+.2f} [{lo*scale:+.2f}, {hi*scale:+.2f}]'
    text=['# Routing at 75% tint: complete three-seed replication','',
        'Fixed standard-strength IS, seeds42/43/44. Seeds43/44 replicate the seed42 recipe without tuning. '
        'Seed42 contributed to development; the same previously observed evaluation banks were reused. '
        'Original tint-search decisions and results remain unchanged. No manuscript files were edited.','',
        'All learned arms receive the same sources, caption representations, guards, 1,944 updates and final-checkpoint rule. '
        'Within each seed, initialization, sample/representation schedules and dropout RNG consumption match exactly. '
        'A complete seed42 replay produced parameter error0. Only the prescribed objective terms differ.','',
        'Values are means ± sample SD across three seeds. Accuracy is percent; differences are percentage points. '
        'Intervals below are 5,000-draw paired seed-and-connected-source intervals. '
        'Item-only intervals, individual seeds, every metric and every category are in all_metrics.csv/jsonl. '
        'Frozen is one deterministic reference, repeated only for paired comparisons. Mechanisms use the unchanged calibrated units.','',
        '## Direct behavior and mechanism','',
        'Pooled primary:400 source pairs, both physical layouts, both trained color pairs and both noun orders; '
        'conditions are averaged within source, not counted as independent examples.','',
        '| Measure | Frozen | Ranking | IS | IS − ranking [95% CI] |',
        '|---|---:|---:|---:|---:|']
    selected=[]
    def addrow(label,f,t,m,scale=100):
        rr=[get(f,t,m,n) for n in ('Frozen','Ranking','IS')];delta=get(f,t,m,'IS - Ranking')
        text.append('| '+label+' | '+' | '.join(value(r,scale,n>0) for n,r in enumerate(rr))+' | '+difference(delta,scale)+' |')
        selected.extend(rr+[delta])
    for m in ('exchange_accuracy','caption_accuracy','word1_accuracy','word2_accuracy','both_correct','binding','cross','response','preference'):
        addrow(m,'primary','trained_colors_both_orders',m,1 if m in ('binding','cross','response','preference') else 100)
    text+=['','### Each color/order and seed','',
           '| Color/order | Method | Seed42 | Seed43 | Seed44 | Mean ± SD |','|---|---|---:|---:|---:|---:|']
    for t in (*PRIMARY,'purple-orange_canonical','trained_colors_both_orders'):
        for name in ('Frozen','Ranking','IS'):
            r=get('primary',t,'exchange_accuracy',name)
            text.append('| '+t+' | '+name+' | '+' | '.join(f'{r["per_seed"][str(s)]*100:.2f}' for s in SEEDS)+' | '+value(r,100,name!='Frozen')+' |')
    text+=['','## Indirect behavior','',
           'Reverse-order templates were exposed in training; they are not unseen wording. '
           'The attribute-clause tests use untrained wording and remain separate.','',
           '| Test | Frozen | Ranking | IS | IS − ranking [95% CI] |','|---|---:|---:|---:|---:|']
    for t in [*run.ev.olddata.FAMILIES,'caption_gallery','attribute_clause_red-blue','attribute_clause_green-yellow',
              'reverse_order_red-blue','reverse_order_green-yellow']:
        addrow(t,'transfer',t,'gallery_top1' if t=='caption_gallery' else 'exchange_accuracy')
    text+=['','## Natural-image preservation','',
           '| Test | Frozen | Ranking | IS | IS − ranking [95% CI] |','|---|---:|---:|---:|---:|']
    for t,m in [('sugarcrepe_full','accuracy'),('aro_full','accuracy'),('aro_either_red_blue','accuracy'),('aro_exact_red_blue','accuracy'),('coco_t2i','recall1'),('coco_i2t','recall1'),('coco_t2i','recall5'),('coco_i2t','recall5')]:
        addrow(t+'/'+m,'natural',t,m)
    text+=['','## Component controls','',
           '| Arm | Pooled assignment | Cross-effect | IS − arm assignment [95% CI] |',
           '|---|---:|---:|---:|']
    for name in ('IS',*run.ev.olddata.DELETIONS):
        r=get('primary','trained_colors_both_orders','exchange_accuracy',name);c=get('primary','trained_colors_both_orders','cross',name)
        delta='—' if name=='IS' else difference(get('primary','trained_colors_both_orders','exchange_accuracy','IS - '+name))
        text.append(f'| {name} | {value(r)} | {value(c,1)} | {delta} |')
    text+=['','no_interaction removes the binding, cross and response target losses; preference and all guards remain. '
           'It does not remove every interaction-dependent calculation.','',
           '## Full-context direction check','',
           '| Method | Seed | Binding means increased /32 | Cross-effect means reduced /32 |',
           '|---|---:|---:|---:|']
    for name in ('Ranking','IS'):
        for s in SEEDS:
            counts=[sum(r['per_seed_directional_change'][str(s)]>0 for r in directions if r['model']==name and r['kind']==k) for k in ('binding','unwanted')]
            text.append(f'| {name} | {s} | {counts[0]}/32 | {counts[1]}/32 |')
    text+=['','These are clause means across anchors, not guarantees on every example.','',
           '## Diagnosis-guided repair','',
           'Regimes are assigned from frozen scores in one physical layout; component benefit is measured in the other. '
           'Rows below are differences of paired component benefits between regimes, in percentage points.','',
           '| Color/order | Diagnosis layout | Component | Predicted regime advantage [95% CI] |',
           '|---|---|---|---:|']
    for r in diagnostics:
        if r['type']=='diagnosis_guided':text.append(f'| {r["color"]}/{r["order"]} | {r["diagnosis_view"]} | {r["name"]} | {difference(r)} |')
    text+=['','## Comparison with the current paper setting','',
           'The current manuscript uses90% tint and canonical-order training; this study uses75% tint, balanced noun-order exposure, '
           'and an additional single-word color-margin floor shared by Ranking and IS. '
           'Its main table reports red/blue canonical captions, whereas the pooled table above covers both trained colors and orders. '
           'The following comparison keeps the reported red/blue canonical metric, but changes the rendered images and training setting. '
           'It is descriptive, not a paired improvement on an identical test.','',
           '| Setting | Frozen assignment | Ranking assignment | IS assignment | IS − ranking |',
           '|---|---:|---:|---:|---:|',
           '| Current90% manuscript | 56.44 | 75.13 ±0.27 | 79.50 ±1.36 | +4.38 |']
    r=[get('primary','red-blue_canonical','exchange_accuracy',n) for n in ('Frozen','Ranking','IS')]
    text.append('| New75%, red/blue canonical | '+' | '.join(value(a,100,j>0) for j,a in enumerate(r))+' | '+difference(get('primary','red-blue_canonical','exchange_accuracy','IS - Ranking'))+' |')
    text+=['','A manuscript replacement would also need its text, running-example figures and table denominators updated. '
           'Existing90% OpenAI/SigLIP and published-patch comparisons remain valid at90%; they have not been rerun or relabeled as75%. '
           'The author will decide whether to replace the main setting or retain this as a sensitivity/extension experiment.','',
           '## Artifact map','',
           '- all_metrics.csv/jsonl: every seed, method, metric, test, SD, paired item and seed/item intervals.',
           '- diagnostics.csv/jsonl: repair/break, frozen-single-word-correct cohort, cross-layout regimes and component contrasts.',
           '- directions.jsonl: all32 binding and32 cross-effect directions, with seed-level changes.',
           '- paper_candidate_tables.csv: compact selected tables, including preservation and untrained-wording tests.',
           '- analysis_protocol.json: exact source hashes, sampling procedure and reference manuscript hashes.',
           '- ../seed43 and ../seed44: fixed checkpoints, configurations, training histories, per-example scores and complete evaluations.',
           '- Seed42 sources are reused read-only from the earlier directional_retest directory.']
    with (OUT/'REPORT.md').open('x') as stream:stream.write('\n'.join(text)+'\n')
    csvwrite(OUT/'paper_candidate_tables.csv',selected)
    dump(OUT/'followup_results.json',dict(seeds=SEEDS,all_metrics=records,diagnostics=diagnostics,directions=directions))


def selftest():
    # Constant per-seed values: item intervals are exact, hierarchical draws
    # retain seed variation; identical paired arms must have exactly zero CI.
    rows=[dict(anchor_id=str(i),source_ids=[i]) for i in range(8)]
    x=np.broadcast_to(np.array([1.,2.,3.])[:,None,None],(3,8,2)).copy()
    bi,bh,nc=bootstrap(x,rows)
    assert nc==8 and np.all(bi==2) and np.any(bh[:,0]!=2)
    assert np.array_equal(bh[:,0],bh[:,1])
    # Shared sources are one component and must be resampled together.
    assert len(set(run.ev.analysis.source_clusters([dict(source_ids=[1,2]),dict(source_ids=[2,3]),dict(source_ids=[4])])) )==2
    print('BOOTSTRAP_SELFTEST_PASSED',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['selftest','aggregate']);a=ap.parse_args()
    log(run.OUT,'analysis_start',action=a.action)
    try:globals()[a.action]()
    except BaseException as exc:log(run.OUT,'analysis_failed',action=a.action,error=repr(exc));raise
    log(run.OUT,'analysis_complete',action=a.action)
