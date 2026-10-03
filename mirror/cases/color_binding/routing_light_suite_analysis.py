"""Paper-facing seed42 retests after development-only cross-weight selection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mirror.cases.color_binding import routing_light_tint as p
from mirror.cases.color_binding import routing_light_suite_data as data
from mirror.cases.color_binding import routing_light_search as search
from mirror.cases.color_binding import indirect_generalization as ind
from mirror.cases.color_binding import ranking_transfer_extension as rt
from mirror.cases.color_binding import targeted_transfer as tt
from mirror.core.metrics import source_clusters
from mirror.cases.color_binding.targeted_evaluate import gallery_metrics
from mirror.cases.color_binding.completion_preservation import retrieval_ranks
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import lines
from mirror.core.features import verify_cache

OUT=data.OUT/'analysis'
DRAWS=5000
VIEWS=p.VIEWS
CORE=('exchange_accuracy','caption_accuracy','word1_accuracy','word2_accuracy','both_correct',
      'binding','cross','response','preference','reverse_image_choice')


def registry(components=False):
    selected=read(search.OUT/'selection.json')
    aliases={'Frozen':'Frozen','IS40':'IS_current','Ranking40':'Ranking','IS90':'IS90','Ranking90':'Ranking90'}
    regs=[dict(r,name=aliases[r['name']]) for r in p.registry()]
    regs.append(dict(selected['model'],name='IS'))
    if components:regs+=read(search.OUT/'ablations.json')['models']
    verify_files({r['checkpoint']:r['sha256'] for r in regs if r['checkpoint']})
    return regs


def freeze():
    data.verify();search.verify()
    files=[Path(__file__),Path(source_clusters.__code__.co_filename),Path(gallery_metrics.__code__.co_filename),
        Path(p.metrics.__file__),search.OUT/'selection.json',search.OUT/'ablations.json',
        data.OUT/'protocol.json',search.OUT/'protocol.json',p.OUT/'complete.json']
    dump(OUT/'protocol.json',dict(inputs={str(f):sha(f) for f in files},models=registry(True),
        metrics=list(CORE),seed=42,draws=DRAWS,selected_before_retests=True,
        reported_candidates='Development grid complete; full paper retests of current and development-selected IS, matched ranking, frozen and archived90 references',
        transfer='All original conditions in each of the six paper families; no slices selected by outcomes',
        ablation='Selected multiplier; same seed/init/stream/guards/1944 updates, named target losses zeroed',
        statistics='Paired connected-source cluster resampling; both layouts and all conditions within source; no seed SD from one seed',
        diagnostic_test='Frozen regime in one layout; full-minus-deletion accuracy in opposite layout',
        decision_cohort='Each mixed-state decision passes its two one-word alternatives; test its swap decision. Never filter the complete primary sample.',
        no_automatic_seed_expansion=True,no_manuscript_changes=True))
    print('ANALYSIS_FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():verify_files(read(OUT/'protocol.json')['inputs'])


def measurements(x):
    m=p.metrics.routing(x)
    m['both_correct']=((x[:,1,1]>x[:,1,2])&(x[:,2,2]>x[:,2,1])).astype(float)
    m['reverse_image_choice']=np.stack((x[:,1,1]>x[:,2,1],x[:,2,2]>x[:,1,2]),axis=1).mean(1)
    for ctx in p.metrics.all_contexts():
        if ctx['kind']=='unwanted':m['absolute/'+ctx['name']]=abs(m['contrast/'+ctx['name']])
    return m


def average(parts):
    return {name:{k:np.mean([z[name][k] for z in parts],axis=0) for k in parts[0][name]} for name in parts[0]}


def scored(regs,views,individual=False):
    raw={};parts=[]
    for view,(v,t) in views.items():
        vals={};memo={}
        for r in regs:
            key=r['checkpoint']
            if key not in memo:
                adapted=p.metrics.adapt(t,key)
                if individual:
                    x=np.einsum('nid,nkjd->nkij',v,adapted,optimize=True)
                    m=measurements(x.reshape(-1,4,4));m={k:a.reshape(len(v),t.shape[1]).mean(1) for k,a in m.items()}
                else:
                    x=np.einsum('nid,njd->nij',v,adapted,optimize=True);m=measurements(x)
                memo[key]=(x,m)
            x,m=memo[key];raw[view+'/'+r['name']]=x;vals[r['name']]=m
        parts.append(vals)
    return average(parts),raw


def bootstrap_matrix(matrix,rows):
    cl=source_clusters(rows);nc=int(cl.max())+1;counts=np.bincount(cl)
    sums=np.stack([np.bincount(cl,weights=matrix[:,j],minlength=nc) for j in range(matrix.shape[1])])
    rng=np.random.default_rng(20260930);out=[]
    for start in range(0,DRAWS,100):
        n=min(100,DRAWS-start);w=rng.multinomial(nc,np.full(nc,1/nc),size=n)
        out.append((w@sums.T)/(w@counts)[:,None])
    return np.concatenate(out),nc


class Recorder:
    def __init__(self,folder):
        self.dest=OUT/folder;self.dest.mkdir(parents=True,exist_ok=False);self.summary=[];self.index=[]
    def emit(self,test,values,rows,raw=None,details=None):
        keys=list(next(iter(values.values())));packed={n:np.column_stack([m[k] for k in keys]) for n,m in values.items()}
        labels=list(packed.items())
        labels += [('IS - '+n,packed['IS']-a) for n,a in packed.items() if n!='IS']
        matrix=np.concatenate([a for _,a in labels],axis=1);samples,nc=bootstrap_matrix(matrix,rows)
        point=matrix.mean(0)
        for j,(name,_) in enumerate(labels):
            for k,metric in enumerate(keys):
                ix=j*len(keys)+k
                self.summary.append(dict(test=test,comparison=name,metric=metric,mean=float(point[ix]),
                    ci95=np.quantile(samples[:,ix],[.025,.975]).tolist(),
                    ci99_1667=np.quantile(samples[:,ix],[.05/12,1-.05/12]).tolist(),
                    n_items=len(rows),n_source_clusters=nc,seed=42,bootstrap_draws=DRAWS))
        if 'cross' in keys:
            fi=list(packed).index('Frozen')*len(keys)+keys.index('cross')
            for i,name in enumerate(packed):
                ix=i*len(keys)+keys.index('cross')
                estimate=100*(1-point[ix]/point[fi]);bs=100*(1-samples[:,ix]/samples[:,fi])
                self.summary.append(dict(test=test,comparison=name,metric='cross_reduction_percent',mean=float(estimate),
                    ci95=np.quantile(bs,[.025,.975]).tolist(),n_items=len(rows),n_source_clusters=nc,seed=42))
        records=[]
        for name,m in values.items():
            records.extend(dict(name=name,test=test,anchor_id=r.get('anchor_id',r.get('example_id')),
                source_ids=r.get('source_ids',[r.get('source_id')]),**{k:float(a[j]) for k,a in m.items()}) for j,r in enumerate(rows))
        jsonl(self.dest/(test+'_per_example.jsonl'),records)
        if raw is not None:np.savez_compressed(self.dest/(test+'_scores.npz'),**raw)
        self.index.append(dict(test=test,anchor_ids=[r.get('anchor_id',r.get('example_id')) for r in rows],**(details or {})))
        metric='exchange_accuracy' if 'exchange_accuracy' in keys else keys[0]
        print('SCORED',test,{n:round(float(m[metric].mean())*100,3) for n,m in values.items()},flush=True)
    def finish(self):
        jsonl(self.dest/'summary.jsonl',self.summary);dump(self.dest/'score_index.json',self.index)
        dump(self.dest/'complete.json',dict(files={str(f):sha(f) for f in self.dest.iterdir() if f.is_file()}))


def primary():
    verify();torch.set_num_threads(4);rec=Recorder('primary');regs=registry(True)
    rows=lines(p.CONFIRM/'rows.jsonl')
    for color in map('-'.join,p.COLORS):
        views={}
        for view in VIEWS:
            v,t,idx=p.metrics.bank_arrays(p.OUT/'features/test','routing',color,view)
            assert [r['anchor_id'] for r in idx]==[r['anchor_id'] for r in rows]
            views[view]=(v,t)
        values,raw=scored(regs,views);rec.emit('direct_'+color,values,rows,raw)
    # Existing intermediate opacity, not a newly chosen stress level.
    views={}
    for view in VIEWS:
        v,t,rr=rt.arrays('weaker_edits','red-blue',.65,view)
        assert [r['anchor_id'] for r in rr]==[r['anchor_id'] for r in rows]
        views[view]=(v,t)
    values,raw=scored(registry(),views);rec.emit('tint65_red-blue',values,rows,raw)
    rec.finish()


def transfer():
    verify();torch.set_num_threads(4);rec=Recorder('transfer');regs=registry()
    for family in data.FAMILIES:
        verify_cache(data.OUT/'features'/family);parts=[];rawall={};condition_means=[]
        for condition in data.conditions(family):
            views={};ids=None
            for view in VIEWS:
                v,t,rows=data.arrays(family,condition,view)
                current=[r['anchor_id'] for r in rows]
                if ids is not None:assert current==ids
                ids=current;views[view]=(v,t)
            values,raw=scored(regs,views);parts.append(values)
            rawall.update({str(condition)+'/'+k:v for k,v in raw.items()})
            condition_means += [dict(condition=str(condition),name=name,**{k:float(a.mean()) for k,a in m.items()}) for name,m in values.items()]
        rec.emit(family,average(parts),rows,rawall,dict(conditions=data.conditions(family)))
        jsonl(rec.dest/(family+'_condition_means.jsonl'),condition_means)
    rows=lines(p.CONFIRM/'rows.jsonl')
    strings=read(ind.OUT/'template_index.json');text=np.load(ind.OUT/'template_features.npy');lookup={s:i for i,s in enumerate(strings)}
    for colors in p.COLORS[:2]:
        views={}
        for view in VIEWS:
            v,_,idx=p.metrics.bank_arrays(p.OUT/'features/test','routing','-'.join(colors),view)
            assert [r['anchor_id'] for r in idx]==[r['anchor_id'] for r in rows]
            indices=np.array([[lookup[s] for family in ind.FAMILIES for variants in zip(*ind.prompts(tuple(r['objects']),colors,view,family))
                              for s in variants] for r in rows]).reshape(len(rows),12,4)
            views[view]=(v,text[indices][:,ind.GROUPS['nonspatial_unseen']])
        vals,raw=scored(regs,views,True);rec.emit('unseen_wording_'+'-'.join(colors),vals,rows,raw,dict(six_individual_wordings=True))
    gallery=np.load(tt.DEST/'gallery_features.npy');entries=read(tt.DEST/'caption_gallery.json')
    pairs={tuple(e['objects']):e['pair_index'] for e in entries}
    correct=np.array([[4*pairs[tuple(r['objects'])]+j for j in range(4)] for r in rows]);parts=[];raw={}
    for view in VIEWS:
        v,_,_=p.metrics.bank_arrays(p.OUT/'features/test','routing','red-blue',view);values={}
        for reg in regs:
            x=np.einsum('nid,jd->nij',v,p.metrics.adapt(gallery,reg['checkpoint']),optimize=True)
            a,_=gallery_metrics(x,correct);values[reg['name']]=dict(gallery_top1=a[:,0],object_pair_top1=a[:,1])
            raw[view+'/'+reg['name']]=x
        parts.append(values)
    rec.emit('caption_gallery',average(parts),rows,raw,dict(captions=112));rec.finish()


def natural():
    import pandas as pd
    verify();torch.set_num_threads(4);rec=Recorder('natural');regs=registry()
    aliases={'Frozen':'Frozen','IS_current':'IS40','Ranking':'Ranking40','IS90':'IS90','Ranking90':'Ranking90'}
    selected=read(search.OUT/'selection.json')
    if selected['selected']=='IS40':aliases['IS']='IS40'
    for bench in ('sugarcrepe','aro','coco'):
        old=np.load(p.OUT/'evaluation/natural'/(bench+'_scores.npz'))
        ff=np.load((p.prior.SUGAR if bench=='sugarcrepe' else p.BASE/'preservation/features'/ind.MODEL/bench)/'features.npz')
        raw={};values={}
        if bench=='sugarcrepe':
            reference=pd.read_csv(p.prior.SUGAR/'frozen_seed0.csv');index=read(p.prior.SUGAR/'indices.json')
            vi={s:i for i,s in enumerate(index['names'])};ti={s:i for i,s in enumerate(index['prompts'])}
            v=ff['images'][[vi[s] for s in reference.filename]];pos=[ti[s] for s in reference.caption];neg=[ti[s] for s in reference.negative_caption]
            rows=[dict(example_id=str(r.subset)+'/'+str(r.example_id),source_id=r.filename,category=r.subset) for _,r in reference.iterrows()]
        else:
            rows=lines(p.BASE/'preservation/manifests'/bench/'rows.jsonl');v=ff['images']
            if bench=='aro':pos=[r['positive'] for r in rows];neg=[r['negative'] for r in rows]
            else:owner=np.array([r['image_index'] for r in rows])
        for reg in regs:
            name=reg['name'];source=aliases.get(name)
            if bench=='coco':
                if source:tr,ir=old[source+'/t2i'],old[source+'/i2t']
                else:tr,ir,_,_=retrieval_ranks(v,p.metrics.adapt(ff['texts'],reg['checkpoint']),owner,128,device='cuda')
                values[name]=(tr,ir);raw[name+'/t2i']=tr;raw[name+'/i2t']=ir
            else:
                if source:x=old[source]
                else:
                    t=p.metrics.adapt(ff['texts'],reg['checkpoint'])
                    x=np.stack([np.einsum('nd,nd->n',v,t[pos]),np.einsum('nd,nd->n',v,t[neg])],axis=1)
                values[name]=dict(accuracy=(x[:,0]>x[:,1]).astype(float));raw[name]=x
        if bench=='coco':
            for d,direction in enumerate(('t2i','i2t')):
                rr=[dict(example_id=str(j),source_id=str(owner[j] if d==0 else j)) for j in range(len(next(iter(values.values()))[d]))]
                rec.emit('coco_'+direction,{n:{'recall'+str(k):(a[d]<=k).astype(float) for k in (1,5)} for n,a in values.items()},rr)
        else:
            rec.emit(bench+'_full',values,rows)
            categories={c:np.array([r['category']==c for r in rows]) for c in sorted({r['category'] for r in rows})} if bench=='sugarcrepe' else {
                c:np.array([r[c] for r in rows]) for c in ('either_red_blue','exact_red_blue')}
            for name,mask in categories.items():
                rec.emit(bench+'_'+name,{n:{k:a[mask] for k,a in m.items()} for n,m in values.items()},[r for r,yes in zip(rows,mask) if yes])
        np.savez_compressed(rec.dest/(bench+'_raw.npz'),**raw)
        dump(rec.dest/(bench+'_rows.json'),rows)
    rec.finish()


def diagnostics():
    verify();torch.set_num_threads(4);dest=OUT/'diagnostics';dest.mkdir(exist_ok=False)
    regs=registry(True);names=[r['name'] for r in regs];rows=lines(p.CONFIRM/'rows.jsonl');n=len(rows)
    groups=np.array(['+'.join(r['objects']) for r in rows]);rng=np.random.default_rng(20260930)
    w=np.zeros((DRAWS,n))
    for group in sorted(set(groups)):
        ix=np.flatnonzero(groups==group);w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=DRAWS)
    def estimate(num,den):
        point=float(num.sum()/den.sum()) if den.sum()>0 else None
        denominator=w@den;bs=np.divide(w@num,denominator,out=np.full(DRAWS,np.nan),where=denominator>0)
        valid=bs[np.isfinite(bs)]
        return dict(mean=point,ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
                    denominator=int(den.sum()),valid_draws=len(valid)),bs
    fixes=[];single=[];directions=[];strata=[];contrasts=[];diagnosis=[];regimes_out={};distributions=[]
    for colors in p.COLORS:
        color='-'.join(colors);raw=np.load(OUT/'primary'/('direct_'+color+'_scores.npz'))
        measures={};decisions={};wordpasses={}
        for name in names:
            ss=np.stack([raw[v+'/'+name] for v in VIEWS],axis=1)
            mm=[measurements(raw[v+'/'+name]) for v in VIEWS]
            measures[name]={k:np.stack([z[k] for z in mm],axis=1) for k in mm[0]}
            decisions[name]=np.stack((ss[:,:,1,1]>ss[:,:,1,2],ss[:,:,2,2]>ss[:,:,2,1]),axis=-1)
            wordpasses[name]=np.stack([((ss[:,:,i,i]>ss[:,:,i,i^1])&(ss[:,:,i,i]>ss[:,:,i,i^2])) for i in (1,2)],axis=-1)
        f=measures['Frozen'];regimes=np.where(f['response']<=0,0,np.where(f['response']<=f['preference'],1,2))
        regimes_out[color]=regimes.tolist()
        for metric,mask in [('repair_rate',~decisions['Frozen']),('break_rate',decisions['Frozen'])]:
            den=mask.sum((1,2));draws={}
            for name in names:
                state=decisions[name] if metric=='repair_rate' else ~decisions[name]
                num=(state&mask).sum((1,2));point,bs=estimate(num,den);draws[name]=(point,bs)
                fixes.append(dict(color=color,name=name,metric=metric,**point))
            for other in ('Ranking','IS_current','IS90'):
                a,ab=draws['IS'];b,bb=draws[other]
                fixes.append(dict(color=color,name='IS - '+other,metric=metric,mean=a['mean']-b['mean'],
                     ci95=np.quantile(ab-bb,[.025,.975]).tolist(),denominator=int(den.sum())))
        mask=wordpasses['Frozen'];den=mask.sum((1,2))
        for name in names:
            point,_=estimate((~decisions[name]&mask).sum((1,2)),den)
            single.append(dict(color=color,name=name,cohort='Frozen passes both one-word alternatives for this decision',metric='assignment_error_rate',**point))
            own=wordpasses[name];point,_=estimate((~decisions[name]&own).sum((1,2)),own.sum((1,2)))
            single.append(dict(color=color,name=name,cohort='Own-model one-word-correct decisions; changing denominator, not a paired comparison',metric='assignment_error_rate',**point))
        for name in names:
            for ctx in p.metrics.all_contexts():
                key=('contrast/' if ctx['kind']=='binding' else 'absolute/')+ctx['name']
                before=f[key].mean(1);after=measures[name][key].mean(1)
                delta=after-before;oriented=delta if ctx['kind']=='binding' else -delta
                directions.append(dict(color=color,name=name,context=ctx['name'],kind=ctx['kind'],
                    frozen_mean=float(before.mean()),mean=float(after.mean()),change=float(delta.mean()),
                    directional_change=float(oriented.mean()),ci95_directional=np.quantile(w@oriented/n,[.025,.975]).tolist(),
                    improved_fraction=float((oriented>1e-5).mean()),worsened_fraction=float((oriented < -1e-5).mean())))
            for metric in CORE:
                a=measures[name][metric].mean(1)
                distributions.append(dict(color=color,name=name,metric=metric,quantiles=np.quantile(a,[0,.1,.25,.5,.75,.9,1]).tolist()))
        draws={}
        for v,layout in enumerate(('direct_to_swapped','swapped_to_direct')):
            for k,regime in enumerate(('nonpositive_response','preference_dominated','both_correct')):
                mask=regimes[:,v]==k
                for name in names:
                    for metric in CORE:
                        a=measures[name][metric][:,1-v];point,bs=estimate(a*mask,mask.astype(float));draws[layout,regime,name,metric]=(point,bs)
                        strata.append(dict(color=color,layout=layout,regime=regime,name=name,metric=metric,**point))
                for arm in data.DELETIONS:
                    for metric in CORE:
                        a,ab=draws[layout,regime,'IS',metric];b,bb=draws[layout,regime,arm,metric]
                        if a['mean'] is None:continue
                        valid=(ab-bb)[np.isfinite(ab-bb)]
                        contrasts.append(dict(color=color,layout=layout,regime=regime,comparison='IS - '+arm,metric=metric,
                            mean=a['mean']-b['mean'],ci95=np.quantile(valid,[.025,.975]).tolist(),n=int(mask.sum())))
            for arm in ('no_response','no_preference'):
                first,second=('nonpositive_response','preference_dominated') if arm=='no_response' else ('preference_dominated','nonpositive_response')
                a,ab=draws[layout,first,'IS','exchange_accuracy'];b,bb=draws[layout,first,arm,'exchange_accuracy']
                c,cb=draws[layout,second,'IS','exchange_accuracy'];d,db=draws[layout,second,arm,'exchange_accuracy']
                if a['mean'] is None or c['mean'] is None:continue
                delta=(ab-bb)-(cb-db);valid=delta[np.isfinite(delta)]
                diagnosis.append(dict(color=color,layout=layout,arm=arm,contrast=first+' minus '+second,
                    mean=(a['mean']-b['mean'])-(c['mean']-d['mean']),ci95=np.quantile(valid,[.025,.975]).tolist()))
    for name,records in [('fix_break',fixes),('single_word_conditioned',single),('all_context_directions',directions),
                         ('regime_means',strata),('regime_component_contrasts',contrasts),('diagnosis_guided',diagnosis),('distributions',distributions)]:
        jsonl(dest/(name+'.jsonl'),records)
    dump(dest/'frozen_regimes.json',dict(anchor_ids=[r['anchor_id'] for r in rows],views=VIEWS,regimes=regimes_out))
    # Opacity contrast from the original 2x2 crossed training/test pilot.
    original=np.load(p.OUT/'evaluation/routing_scores.npz');opacity=[]
    for color in map('-'.join,p.COLORS):
        low={};high={}
        for name in ('Frozen','IS40','IS90','Ranking40','Ranking90'):
            low[name]=np.mean([measurements(original['40/'+color+'/'+v+'/'+name])['cross'] for v in VIEWS],axis=0)
            high[name]=np.mean([measurements(original['90/'+color+'/'+v+'/'+name])['cross'] for v in VIEWS],axis=0)
        low_reduction=1-low['IS40'].mean()/low['Frozen'].mean();high_reduction=1-high['IS90'].mean()/high['Frozen'].mean()
        diff=100*(low_reduction-high_reduction)
        bs=100*((1-(w@low['IS40'])/(w@low['Frozen']))-(1-(w@high['IS90'])/(w@high['Frozen'])))
        opacity.append(dict(color=color,comparison='40%-trained IS on40 versus90%-trained IS on90; original multiplier',
            low_reduction_percent=100*low_reduction,high_reduction_percent=100*high_reduction,
            difference_pp=float(diff),ci95=np.quantile(bs,[.025,.975]).tolist(),
            interpretation='Paired evidence of renderer-dependent relative suppression; does not identify optimization difficulty as the unique cause'))
    jsonl(dest/'opacity_hypothesis.jsonl',opacity)
    dump(dest/'complete.json',dict(files={str(f):sha(f) for f in dest.iterdir() if f.is_file()}))


def report():
    verify();selected=read(search.OUT/'selection.json')
    primary=lines(OUT/'primary/summary.jsonl');transfer=lines(OUT/'transfer/summary.jsonl');natural=lines(OUT/'natural/summary.jsonl')
    def get(source,test,name,metric):return next(r for r in source if r['test']==test and r['comparison']==name and r['metric']==metric)
    def fmt(r,factor=1):return f'{r["mean"]*factor:.2f}'
    def ci(r,factor=100):return f'{r["mean"]*factor:+.2f} [{r["ci95"][0]*factor:+.2f}, {r["ci95"][1]*factor:+.2f}]'
    text=['# Lighter routing: complete seed-42 paper retest','',
        f'Development-selected candidate: **{selected["selected"]}**, cross-target multiplier **{selected["multiplier"]}x** relative to the first 40%-tint run. Status: `{selected["status"]}`.',
        '', 'Only seed42. All frozen encoders, captions, data sources, guards, AdamW settings and 1,944 updates are retained. '
        'Weight selection uses 888 separate development pairs. Four component deletions use the selected coefficient; '
        'the held-out transfer tests do not select it. No manuscript replacement or extra seeds. Earlier primary test outcomes were known.',
        '', '## Development grid','', '| Cross multiplier | Eligible | Exchange | Four-caption | Word1 | Word2 | Cross-effect |','|---|---|---:|---:|---:|---:|---:|']
    for r in selected['candidates']:
        a=r['macro'];text.append(f'| {r["multiplier"]}x | {r["eligible"]} | '+ ' | '.join(f'{a[k]*(1 if k=="cross" else 100):.2f}' for k in ('exchange_accuracy','caption_accuracy','word1_accuracy','word2_accuracy','cross'))+' |')
    text += ['', 'Development means give equal weight to five object pairs and two trained color pairs. Complete failed/passed eligibility clauses are saved in `weight_search/selection.json`.',
        '', '## Direct behavior and mechanism at40%','', '| Test / metric | Frozen | Ranking | Current IS | Selected IS | Selected IS minus ranking [95% CI] |','|---|---:|---:|---:|---:|---:|']
    for color in map('-'.join,p.COLORS):
        for metric in ('exchange_accuracy','caption_accuracy','word1_accuracy','word2_accuracy','binding','cross'):
            factor=100 if 'accuracy' in metric else 1;test='direct_'+color
            text.append('| '+color+' / '+metric+' | '+' | '.join(fmt(get(primary,test,n,metric),factor) for n in ('Frozen','Ranking','IS_current','IS'))+' | '+ci(get(primary,test,'IS - Ranking',metric),factor)+' |')
    text += ['', '## All six paper indirect families','', '| Test | Frozen | Ranking | Current IS | Selected IS | IS minus ranking [95% CI] |','|---|---:|---:|---:|---:|---:|']
    tests=[(f,'exchange_accuracy') for f in data.FAMILIES]+[('unseen_wording_red-blue','exchange_accuracy'),('caption_gallery','gallery_top1')]
    for test,metric in tests:
        text.append('| '+test+' | '+' | '.join(fmt(get(transfer,test,n,metric),100) for n in ('Frozen','Ranking','IS_current','IS'))+' | '+ci(get(transfer,test,'IS - Ranking',metric))+' |')
    text += ['', 'Both layouts and all originally defined conditions are included. The 2,400 noun-combination pairs are resampled as their 100 connected source groups, not as 2,400 independent examples. '
        'Unseen wording averages six individual nonspatial wordings. The 112-caption gallery reports top-1. Full per-condition values, all contexts, green/yellow wording and family-adjusted 99.1667% intervals are saved.',
        '', '## Natural-image preservation','', '| Test | Frozen | Ranking | Current IS | Selected IS | IS minus ranking [95% CI] |','|---|---:|---:|---:|---:|---:|']
    for test,metric in [('sugarcrepe_full','accuracy'),('aro_full','accuracy'),('coco_t2i','recall1'),('coco_i2t','recall1')]:
        text.append('| '+test+' | '+' | '.join(fmt(get(natural,test,n,metric),100) for n in ('Frozen','Ranking','IS_current','IS'))+' | '+ci(get(natural,test,'IS - Ranking',metric))+' |')
    text += ['', '## Matched component tests: red/blue','', '| Method | Assignment accuracy | Cross-effect | Full IS minus deletion accuracy [95% CI] |','|---|---:|---:|---:|']
    for name in ['IS',*data.DELETIONS]:
        diff='—' if name=='IS' else ci(get(primary,'direct_red-blue','IS - '+name,'exchange_accuracy'))
        text.append('| '+name+' | '+fmt(get(primary,'direct_red-blue',name,'exchange_accuracy'),100)+' | '+fmt(get(primary,'direct_red-blue',name,'cross'))+' | '+diff+' |')
    text += ['', 'No-interaction deletes binding/cross/response targets, not preference or finite-difference retention guards.',
        '', '## Diagnosis-guided component effects','', '| Color / labeling direction | Deleted term | Regime-specific advantage [pp,95% CI] |','|---|---|---:|']
    for r in lines(OUT/'diagnostics/diagnosis_guided.jsonl'):
        text.append('| '+r['color']+' / '+r['layout']+' | '+r['arm']+' | '+ci(r)+' |')
    text += ['', 'Positive effects support the original prediction: response targeting helps nonpositive-response cases more, and preference targeting helps preference-dominated cases more. Labels come from frozen scores in the opposite layout.',
        '', '## Does lighter tint permit more relative suppression?','']
    for r in lines(OUT/'diagnostics/opacity_hypothesis.jsonl'):
        text.append(f'- {r["color"]}: {r["low_reduction_percent"]:.2f}% removed at40 versus {r["high_reduction_percent"]:.2f}% at90; difference {r["difference_pp"]:+.2f} pp [{r["ci95"][0]:.2f}, {r["ci95"][1]:.2f}].')
    text += ['', 'This paired opacity comparison uses the same original IS weight at both intensities. It supports a renderer-dependent change, not an exclusive explanation of why optimization was difficult. '
        'Lower opacity also changes frozen color recognition, so the original near-perfect single-word premise must be re-established rather than copied.',
        '', 'Detailed interaction directions, per-source improvement/regression fractions, repair/break rates, single-word-conditioned assignment errors, and frozen diagnostic groups are in `diagnostics/`. '
        'All intervals are conditional on seed42; they are not estimates of variability across training seeds. Backbone ports and published-patch retests are not silently replaced by this one-backbone pilot.']
    with (OUT/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    combined=[dict(section=section,**r) for section,rs in [('primary',primary),('transfer',transfer),('natural',natural)] for r in rs]
    jsonl(OUT/'all_results.jsonl',combined)
    dump(OUT/'complete.json',dict(files={str(f):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='commands.jsonl'},
        selection_sha256=sha(search.OUT/'selection.json'),no_extra_seeds=True,no_paper_change=True))
    print('PAPER_RETEST_COMPLETE',OUT/'REPORT.md',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','primary','transfer','natural','diagnostics','report']);a=ap.parse_args()
    log(OUT,'start',stage=a.action)
    try:globals()[a.action]()
    except BaseException as exc:log(OUT,'failed',stage=a.action,error=repr(exc));raise
    log(OUT,'complete',stage=a.action)


if __name__=='__main__':main()
