"""Fixed full retests, conditional on passing the frozen development screen."""
import argparse
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from mirror.cases.color_binding import routing_tint_balance_train as train
from mirror.cases.color_binding import routing_light_suite_data as olddata
from mirror.cases.color_binding import routing_light_suite_analysis as analysis
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.cases.color_binding.rendering import captions
from mirror.core.repair import _state_hash

p=train.p;data=train.data;rt=analysis.rt;tt=analysis.tt;ind=analysis.ind
OUT=train.OUT/'retest'


def selected():
    s=read(train.OUT/'selection.json')['selected']
    assert s is not None,'No approved development candidate: do not run held-out selection'
    return s


def registry(ablations=False):
    s=selected();reg=next(r for r in read(train.OUT/'models.json')
                        if r['tint']==s['tint'] and r['recipe']==s['recipe'] and r['name']=='Ranking')
    result=[dict(name='Frozen',checkpoint=None),dict(reg,name='Ranking'),dict(s['model'],name='IS')]
    if ablations:result+=read(OUT/'ablations.json')
    verify_files({r['checkpoint']:r['sha256'] for r in result if r['checkpoint']})
    return result


def freeze():
    train.verify();s=selected()
    paths=[Path(__file__),Path(analysis.__file__),Path(olddata.__file__),train.OUT/'selection.json',
           train.OUT/'training_protocol.json',p.CONFIRM/'rows.jsonl',rt.DEST/'noun_rows.jsonl',
           ind.OUT/'template_index.json',ind.OUT/'template_features.npy',tt.DEST/'gallery_features.npy',
           tt.DEST/'caption_gallery.json']
    dump(OUT/'protocol.json',dict(inputs={str(f):sha(f) for f in paths},selected=s,models=registry(),
        primary='400 original pairs, both layouts, all three color pairs, canonical order; trained-color reverse order separately',
        families=list(olddata.FAMILIES)+['wording: reverse order and untrained attribute clauses separately','112-caption gallery'],
        no_post_score_conditions=True,component_deletions=olddata.DELETIONS,
        natural=['full SugarCrepe and all categories','full ARO and original two color subsets','COCO5000 bidirectional R@1/R@5'],
        statistics='5000 paired connected-source bootstrap draws, conditional on seed42; same all-condition averaging',
        no_manuscript_changes=True,no_other_seeds=True))


def verify():
    train.verify();verify_files(read(OUT/'protocol.json')['inputs']);analysis.OUT=OUT/'analysis'


def primary_path():
    tint=selected()['tint']
    return {40:p.OUT/'features/test',90:p.CONFIRM/'features'/p.prior.MODEL}.get(tint,OUT/'features/primary')


def conditions(family):
    tint=selected()['tint']
    if family.endswith('_shift'):return tt.conditions(family)
    return [(c,tint/100) for c,_ in olddata.ORIGINAL_CONDITIONS(family)]


def encode():
    verify();tint=selected()['tint']
    if tint in (40,90):
        verify_cache(primary_path());dump(OUT/'encoding_complete.json',dict(reuse_tint=tint));return
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info()[0]>12*1024**3
    annotations_for('train2017');annotations_for('val2017');olddata.install_strength(tint/100)
    scorer=load_subject(p.prior.MODEL,device='cuda')
    for family in ('primary',*olddata.FAMILIES):
        dest=OUT/'features'/family
        if (dest/'complete.json').exists():verify_cache(dest);continue
        rowpath=rt.DEST/'noun_rows.jsonl' if family=='noun_recombination' else p.CONFIRM/'rows.jsonl'
        rows=lines(rowpath);counter=[0];checks=[]
        def renderer(row):
            if family=='primary':result=p.render(row,p.COLORS,tint/100)
            else:
                ims,names,hashes,cc=olddata.render(row,family)
                assert all(c['outside_unchanged'] for c in cc)
                checks.append(dict(anchor_id=row['anchor_id'],outside_unchanged=True,
                                   hue_passes=sum(c['passes'] for c in cc),checks=len(cc)))
                result=ims,names,hashes
            counter[0]+=1
            if counter[0]%100==0:print('RETEST_RENDER',family,counter[0],len(rows),flush=True)
            return result
        colors=p.COLORS if family=='primary' else ([('red','blue')] if family.endswith('_shift') else [tuple(c.split('-')) for c,_ in conditions(family)])
        cache_bank(dest,rows,scorer,renderer,lambda f,n:[g for c in colors for g in captions(f,n,c)],
            registry_path=DEFAULT_REGISTRY,input_hashes={str(OUT/'protocol.json'):sha(OUT/'protocol.json'),str(rowpath):sha(rowpath)},
            image_batch_size=64,text_batch_size=128,render_workers=8,details=dict(tint=tint,family=family,no_exclusions=True))
        if checks:jsonl(OUT/'features'/(family+'_pixel_checks.jsonl'),sorted(checks,key=lambda r:r['anchor_id']))
        print('RETEST_ENCODED',family,flush=True)
    dump(OUT/'encoding_complete.json',dict(files={str(OUT/'features'/f/'complete.json'):sha(OUT/'features'/f/'complete.json') for f in ('primary',*olddata.FAMILIES)}))


def ablations():
    verify();p.prior.configure();s=selected();loaded,forms=data.load_training(s['tint'])
    cache,spec,contexts,cfg=loaded;cal=read(train.OUT/'calibration'/f'{s["tint"]}.json')
    schedule=p.old.schedule_for(42);reps=train.representation_schedule(s['recipe']);records=[]
    for name,deletions in olddata.DELETIONS.items():
        dest=OUT/'runs'/name;dest.mkdir(parents=True,exist_ok=False)
        weights=dict(read(p.prior.OUT/'normalization.json')['weights'])
        if s['name']=='IS2':weights['cross']*=2
        for key in deletions:weights[key]=0.
        model=p.make(cfg);initial=_state_hash(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        history=[];started=time.monotonic()
        for epoch,order in enumerate(schedule):
            current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),torch.tensor(np.repeat(reps[epoch],2),device='cuda')])
            model.train();torch.manual_seed(420001+epoch);torch.cuda.manual_seed_all(420001+epoch)
            for start in range(0,1280,24):
                ids=p.prior.paired_rows(torch.tensor(order[start:start+24],device='cuda'))
                terms,ce=train.parts(model,current,spec,contexts,cfg,ids,cal['m0'])
                loss=train.objective(terms,ce,weights,s['recipe'],s['name'],cal['color_weight'])
                opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
                history.append(dict(epoch=epoch+1,rows=ids.cpu().tolist(),loss=float(loss.detach()),
                    gradient_norm=float(gn),components={k:float(v.detach()) for k,v in terms.items()}))
        assert len(history)==1944
        for key,value in [('initial_state_hash',initial),('schedule_sha256',p.array_hash(schedule)),
                          ('representation_sha256',p.array_hash(reps)),('first_components',history[0]['components']),
                          ('final_rng_sha256',p.array_hash(torch.cuda.get_rng_state().cpu().numpy()))]:
            assert value==s['model'][key],key
        path=dest/'last.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            configuration=asdict(cfg),weights=weights,color_calibration=cal,deletions=deletions,updates=1944,seed=42,
            protocol_sha256=sha(OUT/'protocol.json')),path)
        jsonl(dest/'history.jsonl',history)
        record=dict(name=name,checkpoint=str(path),sha256=sha(path),seed=42,deletions=deletions,seconds=time.monotonic()-started,
                    matched_initialization_schedule_rng=True,updates=1944)
        dump(dest/'complete.json',record);records.append(record);print('ABLATION_COMPLETE',name,flush=True)
    dump(OUT/'ablations.json',records)


def arrays(family,condition,view):
    tint=selected()['tint']
    if tint==40:return olddata.arrays(family,condition,view)
    if tint==90:
        return tt.arrays(family,condition,view) if family.endswith('_shift') else rt.arrays(family,*condition,view)
    folder=OUT/'features'/family;idx=lines(folder/'index.jsonl');v=np.load(folder/'images.npy',mmap_mode='r');t=np.load(folder/'texts.npy')
    if family.endswith('_shift'):colors=('red','blue');prefix=condition;ci=0
    else:
        color,strength=condition;colors=tuple(color.split('-'));prefix=rt.condition_name(color,strength)
        ci=[c for c,_ in conditions(family)].index(color)
    names=[prefix+'/'+view+'/'+a+'_'+b for a in colors for b in colors]
    images=np.stack([v[r['image_offset']+np.array([r['state_names'].index(n) for n in names])] for r in idx])
    text=t[np.array([r['text_indices'][ci*4:ci*4+4] for r in idx])]
    rows=lines(rt.DEST/'noun_rows.jsonl' if family=='noun_recombination' else p.CONFIRM/'rows.jsonl')
    assert [r['anchor_id'] for r in rows]==[r['anchor_id'] for r in idx]
    return images,text,rows


def primary():
    verify();torch.set_num_threads(4);rec=analysis.Recorder('primary');regs=registry(True);rows=lines(p.CONFIRM/'rows.jsonl')
    for color in map('-'.join,p.COLORS):
        for order in (('canonical','reversed') if color!='purple-orange' else ('canonical',)):
            views={}
            for view in p.VIEWS:
                v,t,idx=p.metrics.bank_arrays(primary_path(),'routing',color,view)
                assert [r['anchor_id'] for r in rows]==[r['anchor_id'] for r in idx]
                if order=='reversed':t=data.text_for(rows,color,order)
                views[view]=(v,t)
            values,raw=analysis.scored(regs,views)
            rec.emit(color+'_'+order,values,rows,raw)
    rec.finish()


def transfer():
    verify();torch.set_num_threads(4);rec=analysis.Recorder('transfer');regs=registry()
    for family in olddata.FAMILIES:
        parts=[];rawall={};conditionmeans=[]
        for condition in conditions(family):
            views={}
            for view in p.VIEWS:
                v,t,rows=arrays(family,condition,view);views[view]=(v,t)
            values,raw=analysis.scored(regs,views);parts.append(values)
            rawall.update({str(condition)+'/'+k:a for k,a in raw.items()})
            conditionmeans += [dict(condition=str(condition),name=n,**{k:float(a.mean()) for k,a in m.items()}) for n,m in values.items()]
        rec.emit(family,analysis.average(parts),rows,rawall,dict(conditions=conditions(family)))
        jsonl(rec.dest/(family+'_conditions.jsonl'),conditionmeans)
    rows=lines(p.CONFIRM/'rows.jsonl');strings=read(ind.OUT/'template_index.json')
    text=np.load(ind.OUT/'template_features.npy');lookup={s:i for i,s in enumerate(strings)}
    for colors in p.COLORS[:2]:
        for family in ('reverse_order','attribute_clause'):
            views={}
            for view in p.VIEWS:
                v,_,_=p.metrics.bank_arrays(primary_path(),'routing','-'.join(colors),view)
                indices=np.array([[lookup[s] for variants in zip(*ind.prompts(tuple(r['objects']),colors,view,family)) for s in variants] for r in rows]).reshape(len(rows),3,4)
                views[view]=(v,text[indices])
            values,raw=analysis.scored(regs,views,True)
            rec.emit(family+'_'+'-'.join(colors),values,rows,raw,dict(wordings=3,exposed_in_training=(family=='reverse_order' and selected()['recipe']=='color_order')))
    gallery=np.load(tt.DEST/'gallery_features.npy');entries=read(tt.DEST/'caption_gallery.json');pairs={tuple(e['objects']):e['pair_index'] for e in entries}
    correct=np.array([[4*pairs[tuple(r['objects'])]+j for j in range(4)] for r in rows]);parts=[];raw={}
    for view in p.VIEWS:
        v,_,_=p.metrics.bank_arrays(primary_path(),'routing','red-blue',view);values={}
        for reg in regs:
            x=np.einsum('nid,jd->nij',v,p.metrics.adapt(gallery,reg['checkpoint']),optimize=True)
            a,_=analysis.gallery_metrics(x,correct);values[reg['name']]=dict(gallery_top1=a[:,0],object_pair_top1=a[:,1]);raw[view+'/'+reg['name']]=x
        parts.append(values)
    rec.emit('caption_gallery',analysis.average(parts),rows,raw,dict(captions=112));rec.finish()


def natural():
    import pandas as pd
    verify();torch.set_num_threads(4);rec=analysis.Recorder('natural');regs=registry()
    for bench in ('sugarcrepe','aro','coco'):
        ff=np.load((p.prior.SUGAR if bench=='sugarcrepe' else p.BASE/'preservation/features'/ind.MODEL/bench)/'features.npz')
        values={};raw={}
        if bench=='sugarcrepe':
            ref=pd.read_csv(p.prior.SUGAR/'frozen_seed0.csv');index=read(p.prior.SUGAR/'indices.json');vi={s:i for i,s in enumerate(index['names'])};ti={s:i for i,s in enumerate(index['prompts'])}
            v=ff['images'][[vi[s] for s in ref.filename]];pos=[ti[s] for s in ref.caption];neg=[ti[s] for s in ref.negative_caption]
            rows=[dict(example_id=str(r.subset)+'/'+str(r.example_id),source_id=r.filename,category=r.subset) for _,r in ref.iterrows()]
        else:
            rows=lines(p.BASE/'preservation/manifests'/bench/'rows.jsonl');v=ff['images']
            if bench=='aro':pos=[r['positive'] for r in rows];neg=[r['negative'] for r in rows]
            else:owner=np.array([r['image_index'] for r in rows])
        for reg in regs:
            name=reg['name'];t=p.metrics.adapt(ff['texts'],reg['checkpoint'])
            if bench=='coco':
                tr,ir,_,_=analysis.retrieval_ranks(v,t,owner,128,device='cuda');values[name]=(tr,ir);raw[name+'/t2i']=tr;raw[name+'/i2t']=ir
            else:
                x=np.stack([np.einsum('nd,nd->n',v,t[pos]),np.einsum('nd,nd->n',v,t[neg])],axis=1)
                values[name]=dict(accuracy=(x[:,0]>x[:,1]).astype(float));raw[name]=x
        if bench=='coco':
            for d,direction in enumerate(('t2i','i2t')):
                rr=[dict(example_id=str(j),source_id=str(owner[j] if d==0 else j)) for j in range(len(next(iter(values.values()))[d]))]
                rec.emit('coco_'+direction,{n:{'recall'+str(k):(a[d]<=k).astype(float) for k in (1,5)} for n,a in values.items()},rr)
        else:
            rec.emit(bench+'_full',values,rows)
            masks={c:np.array([r['category']==c for r in rows]) for c in sorted({r['category'] for r in rows})} if bench=='sugarcrepe' else {c:np.array([r[c] for r in rows]) for c in ('either_red_blue','exact_red_blue')}
            for category,mask in masks.items():rec.emit(bench+'_'+category,{n:{k:a[mask] for k,a in m.items()} for n,m in values.items()},[r for r,yes in zip(rows,mask) if yes])
        np.savez_compressed(rec.dest/(bench+'_raw.npz'),**raw);dump(rec.dest/(bench+'_rows.json'),rows)
    rec.finish()


def diagnostics():
    verify();rows=lines(p.CONFIRM/'rows.jsonl');regs=registry(True);names=[r['name'] for r in regs]
    dest=OUT/'analysis/diagnostics';dest.mkdir(exist_ok=False)
    directions=[];fixes=[];cohorts=[];regimes=[];contrasts=[]
    def estimate(num,den):
        z=np.column_stack((num,den));bs,_=analysis.bootstrap_matrix(z,rows)
        good=bs[:,1]>0;ratios=bs[good,0]/bs[good,1]
        return dict(mean=float(num.sum()/den.sum()) if den.sum() else None,
                    ci95=np.quantile(ratios,[.025,.975]).tolist() if len(ratios) else None,
                    denominator=int(den.sum())),np.divide(bs[:,0],bs[:,1],out=np.full(len(bs),np.nan),where=bs[:,1]>0)
    for color in map('-'.join,p.COLORS[:2]):
        for order in ('canonical','reversed'):
            raw=np.load(OUT/'analysis/primary'/f'{color}_{order}_scores.npz');measures={};decisions={};wordpasses={}
            for name in names:
                xx=np.stack([raw[v+'/'+name] for v in p.VIEWS],axis=1)
                mm=[analysis.measurements(raw[v+'/'+name]) for v in p.VIEWS]
                measures[name]={k:np.stack([m[k] for m in mm],axis=1) for k in mm[0]}
                decisions[name]=np.stack((xx[:,:,1,1]>xx[:,:,1,2],xx[:,:,2,2]>xx[:,:,2,1]),axis=-1)
                wordpasses[name]=np.stack([((xx[:,:,i,i]>xx[:,:,i,i^1])&(xx[:,:,i,i]>xx[:,:,i,i^2])) for i in (1,2)],axis=-1)
            frozen=measures['Frozen']
            for name in names:
                for ctx in p.metrics.all_contexts():
                    key=('contrast/' if ctx['kind']=='binding' else 'absolute/')+ctx['name']
                    delta=(measures[name][key]-frozen[key]).mean(1)*(1 if ctx['kind']=='binding' else -1)
                    bs,_=analysis.bootstrap_matrix(delta[:,None],rows)
                    directions.append(dict(color=color,order=order,name=name,context=ctx['name'],kind=ctx['kind'],
                        directional_change=float(delta.mean()),ci95=np.quantile(bs[:,0],[.025,.975]).tolist()))
                for metric,mask,state in [('repair',~decisions['Frozen'],decisions[name]),('break',decisions['Frozen'],~decisions[name])]:
                    point,_=estimate((state&mask).sum((1,2)),mask.sum((1,2)))
                    fixes.append(dict(color=color,order=order,name=name,metric=metric,**point))
                mask=wordpasses['Frozen'];point,_=estimate((~decisions[name]&mask).sum((1,2)),mask.sum((1,2)))
                cohorts.append(dict(color=color,order=order,name=name,cohort='Frozen correct on both one-word alternatives for each mixed-state decision',
                    metric='assignment_error',**point))
            strata=np.where(frozen['response']<=0,0,np.where(frozen['response']<=frozen['preference'],1,2))
            for source_view in range(2):
                draws={}
                for regime_id,regime in enumerate(('nonpositive_response','preference_dominated','both_correct')):
                    mask=strata[:,source_view]==regime_id
                    for name in names:
                        y=measures[name]['exchange_accuracy'][:,1-source_view];point,bs=estimate(y*mask,mask.astype(float))
                        regimes.append(dict(color=color,order=order,diagnosis_view=p.VIEWS[source_view],
                            measured_view=p.VIEWS[1-source_view],regime=regime,name=name,**point));draws[regime,name]=(point,bs)
                for arm in ('no_response','no_preference'):
                    first,second=('nonpositive_response','preference_dominated') if arm=='no_response' else ('preference_dominated','nonpositive_response')
                    a,ab=draws[first,'IS'];b,bb=draws[first,arm];c,cb=draws[second,'IS'];d,db=draws[second,arm]
                    if any(z['mean'] is None for z in (a,b,c,d)):continue
                    samples=(ab-bb)-(cb-db);samples=samples[np.isfinite(samples)]
                    contrasts.append(dict(color=color,order=order,diagnosis_view=p.VIEWS[source_view],arm=arm,
                        contrast=first+' minus '+second,mean=(a['mean']-b['mean'])-(c['mean']-d['mean']),
                        ci95=np.quantile(samples,[.025,.975]).tolist()))
    for name,records in [('directions',directions),('repair_break',fixes),('single_word_conditioned',cohorts),('regimes',regimes),('diagnosis_guided',contrasts)]:
        jsonl(dest/(name+'.jsonl'),records)
    dump(dest/'complete.json',dict(files={str(f):sha(f) for f in dest.iterdir() if f.is_file()}))


def report():
    verify();s=selected();text=['# Selected routing tint: complete seed42 retest','',
        f'Setting: {s["tint"]}% tint; {s["recipe"]}; {s["name"]}. Selected on888 development pairs before these scores.',
        'Frozen, ranking and IS use the same rendered test inputs. Ranking and IS share data, guards, initialization and update budget.',
        'Only seed42; original90% paper results use a different input setting. No manuscript replacement or extra seeds.',
        '', '| Test | Metric | Frozen | Ranking | IS | IS minus ranking,95%CI |', '|---|---|---:|---:|---:|---:|']
    for family in ('primary','transfer','natural'):
        rr=lines(OUT/'analysis'/family/'summary.jsonl')
        for test in dict.fromkeys(r['test'] for r in rr):
            metrics=('exchange_accuracy','caption_accuracy','word1_accuracy','word2_accuracy','binding','cross') if family=='primary' else (('gallery_top1',) if test=='caption_gallery' else (('accuracy',) if family=='natural' and not test.startswith('coco') else (('recall1',) if test.startswith('coco') else ('exchange_accuracy',))))
            for metric in metrics:
                z={r['comparison']:r for r in rr if r['test']==test and r['metric']==metric};scale=1 if metric in ('binding','cross') else 100
                d=z['IS - Ranking'];text.append(f'| {test} | {metric} | '+ ' | '.join(f'{z[n]["mean"]*scale:.2f}' for n in ('Frozen','Ranking','IS'))+f' | {d["mean"]*scale:+.2f} [{d["ci95"][0]*scale:+.2f}, {d["ci95"][1]*scale:+.2f}] |')
    text+=['','## Component controls','', '| Color/order | Arm | Assignment | Four-caption | Cross |','|---|---|---:|---:|---:|']
    rr=lines(OUT/'analysis/primary/summary.jsonl')
    for test in dict.fromkeys(r['test'] for r in rr):
        for name in ('IS',*olddata.DELETIONS):
            z={r['metric']:r['mean'] for r in rr if r['test']==test and r['comparison']==name}
            text.append(f'| {test} | {name} | {z["exchange_accuracy"]*100:.2f} | {z["caption_accuracy"]*100:.2f} | {z["cross"]:.3f} |')
    dd=lines(OUT/'analysis/diagnostics/directions.jsonl');text+=['','## Interaction directions','',
        '| Color/order | Method | Binding means increased | Cross means reduced |','|---|---|---:|---:|']
    for color in map('-'.join,p.COLORS[:2]):
        for order in ('canonical','reversed'):
            for name in ('Ranking','IS'):
                z=[r for r in dd if r['color']==color and r['order']==order and r['name']==name]
                counts=[sum(r['directional_change']>0 for r in z if r['kind']==k) for k in ('binding','unwanted')]
                text.append(f'| {color}/{order} | {name} | {counts[0]}/8 | {counts[1]}/8 |')
    text+=['','Reverse-order phrases were trained for color_order and are not unseen-wording transfer. Attribute-clause phrases remain untrained.',
           'All original indirect families and natural categories are retained. Intervals condition on one seed and preserve shared-source dependence.']
    with (OUT/'REPORT.md').open('x') as stream:stream.write('\n'.join(text)+'\n')
    dump(OUT/'complete.json',dict(report_sha256=sha(OUT/'REPORT.md'),all_retests_completed=True,seed=42))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','encode','ablations','primary','transfer','natural','diagnostics','report']);args=ap.parse_args()
    log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
