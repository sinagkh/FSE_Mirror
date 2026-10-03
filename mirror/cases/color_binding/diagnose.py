"""Paired fix/break, clause-direction and cross-layout diagnosis at 75%."""
import argparse
from pathlib import Path
import numpy as np
from mirror.cases.color_binding import evaluate as ev
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files

s=ev.stats

def run(study):
    root=ev.OUT/study;verify_files(read(root/'evaluation_complete.json')['inputs'])
    dest=root/'diagnostics';dest.mkdir(exist_ok=False)
    dump(dest/'protocol.json',dict(code_sha256=sha(__file__),input_sha256=sha(root/'evaluation_complete.json'),
        estimands='Paired source ratios; diagnosis in one physical layout, accuracy measured in the other.',
        draws=5000,seed_source='Shared crossed draws, every clause/layout retained; same as main75',
        fixed_models='Frozen and LABCLIP are single fixed models repeated only for paired comparison'))
    names=[r['name'] for r in ev.registry(study,42)];rows=ev.data.lines(ev.data.rowpath('primary'))
    results=[];directions=[]
    for test in s.PRIMARY:
        raw=[np.load(root/'evaluation'/f'seed{seed}'/'primary'/(test+'_scores.npz')) for seed in s.SEEDS]
        dec={};word={};measure={}
        for name in names:
            x=np.stack([np.stack([r[v+'/'+name] for v in ev.p.VIEWS],axis=1) for r in raw])
            dec[name]=np.stack((x[:,:,:,1,1]>x[:,:,:,1,2],x[:,:,:,2,2]>x[:,:,:,2,1]),axis=-1)
            word[name]=np.stack([((x[:,:,:,i,i]>x[:,:,:,i,i^1])&(x[:,:,:,i,i]>x[:,:,:,i,i^2])) for i in (1,2)],axis=-1)
            mm=ev.measurements(x.reshape(-1,4,4),ev.backbone(study))
            measure[name]={k:z.reshape(3,len(rows),2) for k,z in mm.items()}
        labels=[];columns=[]
        def add(label,num,den):labels.append(label);columns.extend((num,den))
        for name in names:
            for kind,mask,success in [('repair',~dec['Frozen'],dec[name]),('break',dec['Frozen'],~dec[name]),
                ('word_correct_assignment_error',word['Frozen'],~dec[name])]:
                add(dict(type=kind,name=name),(mask&success).sum((2,3)),mask.sum((2,3)))
            for context in ev.p.metrics.all_contexts():
                key=('contrast/' if context['kind']=='binding' else 'absolute/')+context['name']
                change=(measure[name][key]-measure['Frozen'][key]).mean((1,2))*(1 if context['kind']=='binding' else -1)
                directions.append(dict(test=test,model=name,context=context['name'],kind=context['kind'],
                    per_seed_directional_change=dict(zip(map(str,s.SEEDS),change.tolist())),all_seeds_expected_direction=bool((change>0).all())))
        f=measure['Frozen'];regimes=np.where(f['response']<=0,0,np.where(f['response']<=f['preference'],1,2))
        for view in (0,1):
            for ri,regime in enumerate(('nonpositive_response','preference_dominated','both_correct')):
                mask=regimes[:,:,view]==ri
                for name in names:
                    add(dict(type='regime_accuracy',name=name,diagnosis_view=ev.p.VIEWS[view],regime=regime),
                        measure[name]['exchange_accuracy'][:,:,1-view]*mask,mask.astype(float))
        array=np.stack(columns,axis=-1);bi,bh,nc=s.bootstrap(array,rows);point=array.mean(1);derived={}
        for j,label in enumerate(labels):
            ii=2*j;valid=(bi[:,ii+1]>0)&(bh[:,ii+1]>0)
            assert (point[:,ii+1]>0).all(),(test,label,'empty observed regime')
            ratios=(point[:,ii]/point[:,ii+1],bi[valid,ii]/bi[valid,ii+1],bh[valid,ii]/bh[valid,ii+1])
            results.append(dict(test=test,n_items=len(rows),n_source_clusters=nc,**label,
                **s.interval_record(*ratios),valid_bootstrap_draws=int(valid.sum()),
                denominator_per_seed={str(seed):int(array[k,:,ii+1].sum()) for k,seed in enumerate(s.SEEDS)}))
            if label['type']=='regime_accuracy':derived[label['diagnosis_view'],label['regime'],label['name']]=(ratios,valid)
        for view in ev.p.VIEWS:
            for arm in ('no_response','no_preference'):
                if arm not in names:continue
                first,second=('nonpositive_response','preference_dominated') if arm=='no_response' else ('preference_dominated','nonpositive_response')
                parts=[derived[view,reg,name] for reg,name in ((first,'IS'),(first,arm),(second,'IS'),(second,arm))]
                assert all(z[1].all() for z in parts),'Empty resampled regime needs joint-valid ratio handling'
                result=[parts[0][0][i]-parts[1][0][i]-parts[2][0][i]+parts[3][0][i] for i in range(3)]
                results.append(dict(test=test,type='diagnosis_guided',name=arm,diagnosis_view=view,
                    contrast=first+' minus '+second,n_items=len(rows),n_source_clusters=nc,**s.interval_record(*result)))
    jsonl(dest/'results.jsonl',results);jsonl(dest/'directions.jsonl',directions)
    dump(dest/'complete.json',dict(files={str(f):sha(f) for f in dest.iterdir() if f.is_file()}))

def joint_text():
    """Both models tested on exactly the same source/condition rows."""
    root=ev.OUT/'joint';dest=root/'text_comparison';dest.mkdir(exist_ok=False)
    records=[]
    for family,test,metric in [('primary',s.TEST if hasattr(s,'TEST') else 'trained_colors_both_orders','exchange_accuracy'),
                              ('transfer','color_recombination','exchange_accuracy'),('transfer','background_shift','exchange_accuracy'),
                              ('transfer','geometry_shift','exchange_accuracy'),('natural','sugarcrepe_full','accuracy'),
                              ('natural','aro_full','accuracy'),('natural','coco_t2i','recall1'),('natural','coco_i2t','recall1')]:
        tests=s.PRIMARY if family=='primary' else [test];samples=[];reference=None
        for seed,oldroot in zip(s.SEEDS,[ev.base.OLD,*[ev.base.OUT/f'seed{z}' for z in s.SEEDS[1:]]]):
            conditions=[]
            for name in tests:
                aa=ev.data.lines(root/'evaluation'/f'seed{seed}'/family/(name+'_per_example.jsonl'))
                bb=ev.data.lines(oldroot/'analysis'/family/(name+'_per_example.jsonl'))
                aa=[r for r in aa if r['name']=='IS'];bb=[r for r in bb if r['name']=='IS']
                rr=[dict(anchor_id=r['anchor_id'],source_ids=r['source_ids']) for r in aa]
                assert rr==[dict(anchor_id=r['anchor_id'],source_ids=r['source_ids']) for r in bb]
                if reference is None:reference=rr
                assert rr==reference
                conditions.append(np.array([[a[metric],b[metric]] for a,b in zip(aa,bb)]))
            samples.append(np.mean(conditions,axis=0))
        x=np.stack(samples);bi,bh,nc=s.bootstrap(x,reference);pp=x.mean(1)
        records.append(dict(family=family,test=test,metric=metric,comparison='Joint IS - Text IS',n_items=len(reference),n_source_clusters=nc,
            **s.interval_record(pp[:,0]-pp[:,1],bi[:,0]-bi[:,1],bh[:,0]-bh[:,1])))
    jsonl(dest/'results.jsonl',records);dump(dest/'complete.json',dict(files={str(f):sha(f) for f in dest.iterdir() if f.is_file()}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=ev.STUDIES,required=True);args=ap.parse_args()
    log(ev.OUT,'diagnostics_start',study=args.study);run(args.study)
    if args.study=='joint':joint_text()
    log(ev.OUT,'diagnostics_complete',study=args.study)

if __name__=='__main__':main()
