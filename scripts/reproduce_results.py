"""Independently replay the paper's decisions and interactions on CPU."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from score_views import typography_views, gallery_views

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(42,43,44)
PRIMARY=('red-blue_canonical','red-blue_reversed','green-yellow_canonical','green-yellow_reversed')
VIEWS=('canvas','swapped_canvas')
CHECKS=[]
VECTORS=[]
POINTS=[]

def read(p):return json.loads((ROOT/p).read_text())
def lines(p):return [json.loads(s) for s in (ROOT/p).read_text().splitlines() if s]
def load(p):
    with np.load(ROOT/p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvrows(p):
    with (ROOT/p).open(newline='') as f:return list(csv.DictReader(f))
def boolean(value):
    assert value in ('True','False','1','0'),value
    return value in ('True','1')
def check(label,actual,expected,tol=2e-5):
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=tol,err_msg=label)
    CHECKS.append(dict(check=label,passed=True))
def target(summary,bank,method,metric):
    rr=[r for r in summary if r.get('bank',r.get('test'))==bank and
        r.get('method',r.get('comparison'))==method and r['metric']==metric]
    assert len(rr)==1,(bank,method,metric,len(rr))
    r=rr[0]
    values=r.get('seed_values',r.get('per_seed'))
    if isinstance(values,dict):values=[values[str(s)] for s in SEEDS]
    assert values is not None,(bank,method,metric)
    return values*3 if len(values)==1 else values
def point(case,bank,method,metric,values,unit,den=None,study='main'):
    assert values.ndim==2 and values.shape[0]==3 and np.isfinite(values).all()
    den=np.ones(values.shape[1]) if den is None else np.asarray(den,float)
    assert den.sum()>0
    means=values.sum(1)/den.sum()
    p=dict(case=case,study=study,bank=bank,method=method,metric=metric,unit=unit,
           n_sources=values.shape[1],denominator=float(den.sum()),
           mean=float(means.mean()),sample_sd=float(means.std(ddof=1)),
           per_seed={str(s):float(v) for s,v in zip(SEEDS,means)})
    POINTS.append(p)
    return means

def color_binding(records):
    unit=read('config/studies.json')['color_binding']['calibration_unit']
    states=list(itertools.product((0,1),repeat=2));weights=[];own=[]
    for i,j,oi,ot in itertools.product((0,1),repeat=4):
        w=np.zeros((4,4))
        for a,b in itertools.product((0,1),repeat=2):
            im,tx=[0,0],[0,0];im[i],im[1-i]=a,oi;tx[j],tx[1-j]=b,ot
            w[states.index(tuple(im)),states.index(tuple(tx))]=1 if a==b else -1
        weights.append(w);own.append(i==j)
    weights,own=np.array(weights),np.array(own)
    for study in sorted({r['study'] for r in records if r['case']=='color_binding'}):
        selected=[r for r in records if r['case']=='color_binding' and r['study']==study and r['bank'] in PRIMARY and r['family']=='primary']
        if not selected:continue
        by={(r['seed'],r['bank']):load(r['file']) for r in selected}
        assert set(by)==set(itertools.product(SEEDS,PRIMARY)),study
        methods=sorted(set.intersection(*[set(k.split('/',1)[1] for k in z if k.startswith('canvas/')) for z in by.values()]))
        summary=read('results/reports/color_binding/'+('primary' if study=='main' else study)+'.json')
        u=unit
        if study!='main' and study!='openclip_laion_l14' and study!='joint':
            # NPZ scores are cosines; SigLIP's native logits additionally have
            # an affine scale. The frozen preflight captures the equivalent
            # cosine unit, without applying the scale twice.
            cal=read('provenance/runs/backbone_evaluation/B2_routing/'+study+'/scale_preflight.json')
            u=cal['cosine_unit']
        per_method={}
        for method in methods:
            cols={}
            for bank in PRIMARY:
                s=np.stack([np.stack([by[seed,bank][v+'/'+method] for v in VIEWS],axis=1) for seed in SEEDS]).astype(float)
                d=np.einsum('snvij,kij->snvk',s,weights)/u
                m0=(s[...,1,1]-s[...,1,2])/u
                m1=(s[...,2,2]-s[...,2,1])/u
                assignment=(s[...,1,1]-s[...,1,2]-s[...,2,1]+s[...,2,2])/u
                check(f'{study}/{method}/{bank}: assignment identity',assignment,m0+m1,tol=1e-12)
                response=assignment/2;preference=abs((m0-m1)/2)
                assert np.array_equal((m0>0)&(m1>0),response>preference)
                metrics={'binding':d[...,own].mean(-1),'cross':abs(d[...,~own]).mean(-1),
                         'response':response,'preference':preference,
                         'assignment_interaction':assignment,
                         'exchange_accuracy':((m0>0).astype(float)+(m1>0))/2}
                for metric,v in metrics.items():cols.setdefault(metric,[]).append(v.mean(2))
            per_method[method]={k:np.mean(v,axis=0) for k,v in cols.items()}
            for metric,v in per_method[method].items():
                means=point('color_binding','trained_colors_both_orders',method,metric,v,
                            'fraction' if metric=='exchange_accuracy' else 'calibrated',study=study)
                expected_metric='response' if metric=='assignment_interaction' else metric
                expected=target(summary,'trained_colors_both_orders',method,expected_metric)
                if metric=='assignment_interaction':expected=np.array(expected)*2
                check(f'{study}/{method}/{metric}',means,expected)
        if study=='main':
            rows=lines('data/color_binding/confirmation/same_rule_confirmation/rows.jsonl')
            assert len(rows)==per_method['IS']['cross'].shape[1]
            for metric in ('cross','binding','exchange_accuracy'):
                VECTORS.append(dict(case='color_binding',metric=metric,bank='trained_colors_both_orders',
                    values={n:per_method[n][metric] for n in ('Frozen','Ranking','IS')},
                    den=np.ones(len(rows)),clusters=source_clusters(rows)))

def source_clusters(rows):
    parent=list(range(len(rows)));owners={}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for i,row in enumerate(rows):
        for source in row['source_ids']:
            if source in owners:parent[find(i)]=find(owners[source])
            else:owners[source]=i
    keys=[find(i) for i in range(len(rows))]
    return np.unique(keys,return_inverse=True)[1]

def color_unit(study):
    if study in ('main','openclip_laion_l14'):
        return read('config/studies.json')['color_binding']['calibration_unit']
    return read('provenance/runs/backbone_evaluation/B2_routing/'+study+'/scale_preflight.json')['cosine_unit']

def color_transfer(records):
    """Replay every registered transfer condition, not a selected winning view."""
    states=list(itertools.product((0,1),repeat=2));weights=[];own=[]
    for i,j,oi,ot in itertools.product((0,1),repeat=4):
        w=np.zeros((4,4))
        for a,b in itertools.product((0,1),repeat=2):
            im,tx=[0,0],[0,0];im[i],im[1-i]=a,oi;tx[j],tx[1-j]=b,ot
            w[states.index(tuple(im)),states.index(tuple(tx))]=1 if a==b else -1
        weights.append(w);own.append(i==j)
    weights,own=np.asarray(weights),np.asarray(own)
    selected=[r for r in records if r['case']=='color_binding' and r['family']=='transfer' and r['format']=='score_matrix']
    for study in sorted({r['study'] for r in selected}):
        summary=read('results/reports/color_binding/'+('primary' if study=='main' else study)+'.json')
        for bank in sorted({r['bank'] for r in selected if r['study']==study}):
            rr=[r for r in selected if r['study']==study and r['bank']==bank]
            by={r['seed']:load(r['file']) for r in rr};assert set(by)==set(SEEDS)
            keys=sorted(by[42]);assert all(sorted(z)==keys for z in by.values())
            methods=sorted({k.rsplit('/',1)[1] for k in keys})
            rows=lines('data/color_binding/ranking_transfer/transfer/noun_rows.jsonl' if bank=='noun_recombination'
                       else 'data/color_binding/transfer/transfer/rows.jsonl')
            assert len({r['anchor_id'] for r in rows})==len(rows)
            for method in methods:
                parts=[]
                for key in [k for k in keys if k.endswith('/'+method)]:
                    x=np.stack([by[seed][key] for seed in SEEDS]).astype(float)
                    assert x.shape[1]==len(rows) and x.shape[-2:]==(4,4)
                    parts.append(x.reshape(3,len(rows),-1,4,4))
                s=np.concatenate(parts,axis=2);u=color_unit(study)
                d=np.einsum('snvij,kij->snvk',s,weights)/u
                m0=(s[...,1,1]-s[...,1,2])/u;m1=(s[...,2,2]-s[...,2,1])/u
                assignment=(s[...,1,1]-s[...,1,2]-s[...,2,1]+s[...,2,2])/u
                check(f'transfer/{study}/{bank}/{method}: assignment identity',assignment,m0+m1,tol=1e-12)
                metrics={'binding':d[...,own].mean(-1),'cross':abs(d[...,~own]).mean(-1),
                         'response':assignment/2,'preference':abs((m0-m1)/2),
                         'assignment_interaction':assignment,
                         'exchange_accuracy':((m0>0).astype(float)+(m1>0))/2}
                for metric,values in metrics.items():
                    means=point('color_binding',bank,method,metric,values.mean(2),
                                'fraction' if metric=='exchange_accuracy' else 'calibrated',study=study)
                    expected=np.asarray(target(summary,bank,method,'response' if metric=='assignment_interaction' else metric))
                    if metric=='assignment_interaction':expected=2*expected
                    check(f'transfer/{study}/{bank}/{method}/{metric}',means,expected)

def color_natural(records):
    """Recalculate natural-caption decisions and recall from stored ranks."""
    metadata='provenance/evaluation/color_binding/reference/midpoint75/directional_retest/analysis/natural/'
    rows={bench:read(metadata+bench+'_rows.json') for bench in ('sugarcrepe','aro','coco')}
    for seed in (43,44):
        for bench,reference in rows.items():
            assert read(f'provenance/evaluation/color_binding/main/seed{seed}/analysis/natural/{bench}_rows.json')==reference
    for study in sorted({r['study'] for r in records if r['case']=='color_binding'}):
        summary=read('results/reports/color_binding/'+('primary' if study=='main' else study)+'.json')
        for bench in ('sugarcrepe','aro','coco'):
            by={seed:load(f'results/records/color_binding/{study}/seed{seed}/natural/{bench}_raw.npz') for seed in SEEDS}
            keys=sorted(by[42]);assert all(sorted(z)==keys for z in by.values())
            if bench=='coco':
                assert len(rows[bench])==25014
                for key in keys:
                    method,direction=key.split('/');ranks=np.stack([by[seed][key] for seed in SEEDS])
                    assert ranks.shape[1]==(len(rows[bench]) if direction=='t2i' else 5000)
                    assert (ranks>=1).all()
                    for k in (1,5):
                        bank='coco_'+direction;metric='recall'+str(k)
                        means=point('color_binding',bank,method,metric,(ranks<=k).astype(float),'fraction',study=study)
                        check(f'natural/{study}/{bank}/{method}/{metric}',means,target(summary,bank,method,metric))
            else:
                masks={'full':np.ones(len(rows[bench]),bool)}
                if bench=='sugarcrepe':
                    masks.update({category:np.array([r['category']==category for r in rows[bench]])
                                  for category in sorted({r['category'] for r in rows[bench]})})
                else:
                    masks.update({category:np.array([r[category] for r in rows[bench]])
                                  for category in ('either_red_blue','exact_red_blue')})
                for method in keys:
                    scores=np.stack([by[seed][method] for seed in SEEDS])
                    assert scores.shape==(3,len(rows[bench]),2)
                    correct=(scores[...,0]>scores[...,1]).astype(float)
                    for category,mask in masks.items():
                        bank=bench+'_'+category
                        means=point('color_binding',bank,method,'accuracy',correct[:,mask],'fraction',study=study)
                        check(f'natural/{study}/{bank}/{method}/accuracy',means,target(summary,bank,method,'accuracy'))

def typography(records):
    summary=[s for r in read('results/reports/typography/main.json') for s in r['statistics']]
    vocab=read('config/typography_vocabulary.json')['evaluation_classes'];index={n:i for i,n in enumerate(vocab)}
    selected=[r for r in records if r['case']=='typography' and r['study']=='main']
    for bank in sorted({r['bank'] for r in selected}):
        rr=[r for r in selected if r['bank']==bank];mapping={(r['seed'],r['method'],r['format']):r for r in rr}
        rows=read(Path(mapping[42,'frozen','digital_scores']['file']).parent/'rows.json')
        n=len(rows);y=np.array([index[r['label']] for r in rows]);w=np.array([[index[k] for k in r['words'][1:]] for r in rows])
        frozen=load(mapping[42,'frozen','digital_scores']['file'])
        bug=(frozen['blank']>0)&(frozen['conflict']<=0)
        vectors={}
        for method in ('frozen','ranking','IS','previous_IS'):
            values={}
            for seed in SEEDS:
                z=load(mapping[seed,method,'digital_scores']['file']);all12=load(mapping[seed,method,'all12']['file'])
                margins,recognition=typography_views(z,rows,vocab)
                check(f'typography/{bank}/{method}/{seed}: recognition decisions',recognition,z['top1'],tol=0)
                check(f'typography/{bank}/{method}/{seed}: blank margin',margins[:,1],z['blank'])
                check(f'typography/{bank}/{method}/{seed}: conflict margins',margins[:,[3,4],[0,1]],z['conflict'])
                diff=np.stack([margins[:,a]-margins[:,b] for a,b in itertools.combinations(range(1,5),2)],axis=1)
                check(f'typography/{bank}/{method}/{seed}: all 12 contrasts',abs(diff),abs(all12['contrasts']))
                retained=bug&(z['conflict']>0)&(z['clean']>0)&(z['blank']>0)
                mm={'all_targets_abs':abs(diff).mean((1,2)),
                    'conflict_pair_accuracy':100*(z['conflict']>0).mean(1),
                    'attack_top1':100*z['top1'][:,3:].mean(1),'clean_top1':100*z['top1'][:,0],
                    'retained_repair':100*retained.sum(1)}
                for k,v in mm.items():values.setdefault(k,[]).append(v)
            vectors[method]={k:np.stack(v) for k,v in values.items()}
            for metric,v in vectors[method].items():
                den=bug.sum(1) if metric=='retained_repair' else np.ones(n)
                means=point('typography',bank,method,metric,v,'cosine' if metric=='all_targets_abs' else 'percent',den)
                check(f'typography/{bank}/{method}/{metric}',means,target(summary,bank,method,metric))
        if bank=='test_seen_standard':
            for metric in ('all_targets_abs','conflict_pair_accuracy','retained_repair'):
                VECTORS.append(dict(case='typography',bank=bank,metric=metric,
                    values={k:vectors[k][metric] for k in ('frozen','ranking','IS')},
                    den=bug.sum(1) if metric=='retained_repair' else np.ones(n),clusters=np.arange(n)))

def typography_public():
    summary=[s for r in read('results/reports/typography/main.json') for s in r['statistics']]
    for bank in ('SCAM','SynthSCAM','NoSCAM','RTA100'):
        for method in ('frozen','ranking','IS','previous_IS','Defense_Prefix'):
            frames=[csvrows(f'results/records/typography/main/seed{s}/external_retest/{bank}_{method}.csv') for s in SEEDS]
            ids=[r['source_id'] for r in frames[0]]
            assert all([r['source_id'] for r in f]==ids for f in frames)
            for si,f in enumerate(frames):
                margins=np.array([float(r['object_score'])-float(r['attack_score']) for r in f])
                check(f'public/{bank}/{method}/{SEEDS[si]}: margin',margins,[float(r['margin']) for r in f])
                assert np.array_equal(margins>0,[boolean(r['correct']) for r in f])
            for key,metric in [('correct','pair_accuracy')]+([('top1','official_top1')] if bank=='RTA100' else []):
                v=100*np.array([[boolean(r[key]) for r in f] for f in frames],float)
                means=point('typography',bank,method,metric,v,'percent')
                check(f'public/{bank}/{method}/{metric}',means,target(summary,bank,method,metric))
    for method in ('frozen','ranking','IS','previous_IS'):
        frames=[csvrows(f'results/records/typography/main/seed{s}/sugarcrepe/{method}.csv') for s in SEEDS]
        identity=[(r['subset'],r['example_id'],r['filename']) for r in frames[0]]
        assert all([(r['subset'],r['example_id'],r['filename']) for r in f]==identity for f in frames)
        for f in frames:
            pred=np.array([float(r['positive_score'])>float(r['negative_score']) for r in f])
            assert np.array_equal(pred,[boolean(r['correct']) for r in f])
        values=100*np.array([[boolean(r['correct']) for r in f] for f in frames],float)
        for category in ['full']+sorted({r['subset'] for r in frames[0]}):
            use=np.array([category=='full' or r['subset']==category for r in frames[0]])
            bank='SugarCrepe_'+category
            means=point('typography',bank,method,'accuracy',values[:,use],'percent')
            check(f'public/{bank}/{method}/accuracy',means,target(summary,bank,method,'accuracy'))

def typography_backbones(records):
    """Reconstruct S4 on the same sources and seeds as the selected reports."""
    summary=read('results/reports/typography/backbone_results.json')
    vocab=read('config/typography_vocabulary.json')['evaluation_classes']
    selected=[r for r in records if r['case']=='typography' and r['study'] in ('openai_clip_l14','openclip_laion_b32')]
    for study in sorted({r['study'] for r in selected}):
        for bank in sorted({r['bank'] for r in selected if r['study']==study}):
            rr=[r for r in selected if r['study']==study and r['bank']==bank]
            mapping={(r['method'],r['seed']):r for r in rr}
            assert set(mapping)=={('frozen',0),*itertools.product(('ranking','IS'),SEEDS)}
            rows=read(Path(mapping['frozen',0]['file']).parent/'rows.json');n=len(rows)
            frozen=load(mapping['frozen',0]['file']);fm,_=typography_views(frozen,rows,vocab)
            bug=(fm[:,1]>0)&(fm[:,[3,4],[0,1]]<=0)
            vectors={}
            for method in ('frozen','ranking','IS'):
                values={}
                for seed in SEEDS:
                    z=load(mapping[method,0 if method=='frozen' else seed]['file'])
                    m,recognition=typography_views(z,rows,vocab)
                    check(f'backbone/{study}/{bank}/{method}/{seed}: clean margins',m[:,0],z['clean'])
                    check(f'backbone/{study}/{bank}/{method}/{seed}: conflict margins',m[:,[3,4],[0,1]],z['conflict'])
                    d=np.stack([m[:,a]-m[:,b] for a,b in itertools.combinations(range(1,5),2)],axis=1)
                    retained=bug&(m[:,[3,4],[0,1]]>0)&(m[:,0]>0)&(m[:,1]>0)
                    check(f'backbone/{study}/{bank}/{method}/{seed}: retained decisions',retained,z['retained_repair'],tol=0)
                    mm={'all_targets_abs':abs(d).mean((1,2)),
                        'attack_pairwise':100*(m[:,[3,4],[0,1]]>0).mean(1),
                        'attack_top1':100*recognition[:,3:].mean(1),'clean_top1':100*recognition[:,0],
                        'retained_repair':100*retained.sum(1)}
                    for metric,v in mm.items():values.setdefault(metric,[]).append(v)
                vectors[method]={metric:np.stack(v) for metric,v in values.items()}
                for metric,v in vectors[method].items():
                    den=bug.sum(1) if metric=='retained_repair' else np.ones(n)
                    means=point('typography',bank,method,metric,v,'cosine' if metric=='all_targets_abs' else 'percent',den,study=study)
                    report=[r for r in summary if r['model']==study and r['category'] is None]
                    check(f'backbone/{study}/{bank}/{method}/{metric}',means,target(report,bank,method,metric))
            if bank in ('fresh_seen','fresh_heldout'):
                VECTORS.append(dict(case='typography',bank=study+'/'+bank,metric='attack_pairwise',
                    values={method:vectors[method]['attack_pairwise'] for method in ('frozen','ranking','IS')},
                    den=np.ones(n),clusters=np.arange(n)))
        for method in ('frozen','ranking','IS'):
            frames=[csvrows(f'results/records/typography/{study}/public/SugarCrepe/{method}_seed{0 if method=="frozen" else seed}.csv') for seed in SEEDS]
            identity=[(r['subset'],r['example_id'],r['filename']) for r in frames[0]]
            assert len(set(identity))==len(identity)
            assert all([(r['subset'],r['example_id'],r['filename']) for r in f]==identity for f in frames)
            for f in frames:
                assert np.array_equal([float(r['margin'])>0 for r in f],[boolean(r['correct']) for r in f])
            values=100*np.asarray([[boolean(r['correct']) for r in f] for f in frames],float)
            for category in ['full']+sorted({r['subset'] for r in frames[0]}):
                mask=np.array([category=='full' or r['subset']==category for r in frames[0]])
                means=point('typography','SugarCrepe_'+category,method,'accuracy',values[:,mask],'percent',study=study)
                report=[r for r in summary if r['model']==study and r['category']==category]
                check(f'backbone/{study}/SugarCrepe_{category}/{method}',means,target(report,'SugarCrepe',method,'accuracy'))

def typography_dyslexify():
    report=read('provenance/published_typography/summary.json')['results']
    for bank in ('SCAM','SynthSCAM','NoSCAM','RTA100'):
        rows=csvrows(f'results/records/typography/published/{bank}_Dyslexify.csv')
        reference=csvrows(f'results/records/typography/main/seed42/external_retest/{bank}_frozen.csv')
        assert [r['source_id'] for r in rows]==[r['source_id'] for r in reference]
        assert len({r['source_id'] for r in rows})==len(rows)
        margins=np.array([float(r['object_score'])-float(r['attack_score']) for r in rows])
        check(f'published/{bank}/Dyslexify: raw margin',margins,[float(r['margin']) for r in rows])
        assert np.array_equal(margins>0,[boolean(r['correct']) for r in rows])
        metric,key=('official_top1','top1') if bank=='RTA100' else ('pair_accuracy','correct')
        # One fixed published defense, broadcast for the output's seed interface;
        # this is not evidence of three independent fits.
        values=np.broadcast_to(100*np.array([boolean(r[key]) for r in rows],float),(3,len(rows)))
        means=point('typography',bank,'Dyslexify',metric,values,'percent',study='published')
        expected=[r for r in report if r['bank']==bank];assert len(expected)==1
        check(f'published/{bank}/Dyslexify/{metric}',means,expected[0]['Dyslexify'])

def typography_retention(records):
    summary=[s for r in read('results/reports/typography/retention.json') for s in r['statistics']]
    vocab=read('config/typography_vocabulary.json')['evaluation_classes'];index={n:i for i,n in enumerate(vocab)}
    selected=[r for r in records if r['case']=='typography' and r['study']=='additional_retention']
    for bank in sorted({r['bank'] for r in selected}):
        mapping={(r['seed'],r['method'],r['format']):r for r in selected if r['bank']==bank}
        rows=read(Path(mapping[42,'IS8_P','digital_scores']['file']).parent/'rows.json')
        y=np.array([index[r['label']] for r in rows]);w=np.array([[index[k] for k in r['words'][1:]] for r in rows]);n=len(rows)
        frozen=load(f'results/records/typography/main/seed42/{bank}/frozen.npz');bug=(frozen['blank']>0)&(frozen['conflict']<=0)
        for method in ('IS8_P','ranking_P'):
            values={}
            for seed in SEEDS:
                z=load(mapping[seed,method,'digital_scores']['file']);saved=load(mapping[seed,method,'all12']['file'])
                m,recognition=typography_views(z,rows,vocab)
                check(f'retention/{bank}/{method}/{seed}: recognition decisions',recognition,z['top1'],tol=0)
                d=np.stack([m[:,a]-m[:,b] for a,b in itertools.combinations(range(1,5),2)],axis=1)
                check(f'retention/{bank}/{method}/{seed}: contrasts',abs(d),abs(saved['contrasts']))
                retained=bug&(z['conflict']>0)&(z['clean']>0)&(z['blank']>0)
                mm={'all_targets_abs':abs(d).mean((1,2)),
                    'conflict_pair_accuracy':100*(z['conflict']>0).mean(1),
                    'attack_top1':100*z['top1'][:,3:].mean(1),'clean_top1':100*z['top1'][:,0],
                    'retained_repair':100*retained.sum(1)}
                for k,v in mm.items():values.setdefault(k,[]).append(v)
            for metric,arr in values.items():
                v=np.stack(arr);den=bug.sum(1) if metric=='retained_repair' else np.ones(n)
                means=point('typography',bank,method,metric,v,'cosine' if metric=='all_targets_abs' else 'percent',den,study='additional_retention')
                check(f'retention/{bank}/{method}/{metric}',means,target(summary,bank,method,metric))

def color_gallery(records):
    selected=[row for row in records if row['format']=='gallery_decision_scores']
    for study in sorted({row['study'] for row in selected}):
        rr=[row for row in selected if row['study']==study]
        mapping={row['seed']:gallery_views(load(row['file'])) for row in rr}
        assert set(mapping)==set(SEEDS)
        prefixes=set.intersection(*[set(value) for value in mapping.values()])
        methods=sorted({key.split('/',1)[1] for key in prefixes})
        summary=read('results/reports/color_binding/'+('primary' if study=='main' else study)+'.json')
        for method in methods:
            values=np.stack([np.mean([mapping[seed][view+'/'+method]['gallery_top1'].mean(1)
                                      for view in VIEWS],axis=0) for seed in SEEDS])
            means=point('color_binding','caption_gallery',method,'gallery_top1',values,'fraction',study=study)
            check(f'gallery/{study}/{method}: all 112-caption decisions',means,
                  target(summary,'caption_gallery',method,'gallery_top1'))


def backdoor(records):
    summary=[s for r in read('results/reports/backdoor/main.json') for s in r['statistics']]
    for attack in ('stripes','triangles','text'):
        for bank in ('development','banana87','banana1000','imagenetv2'):
            rr=[r for r in records if r['case']=='backdoor' and r['attack']==attack and r['bank']==bank]
            mapping={(r['method'],r['seed']):r for r in rr};ref=load(mapping['victim',0]['file'])
            y=ref['labels'];n=len(y);allmask=np.ones(n,bool)
            target_class=954 if bank in ('imagenetv2','banana1000') else 86
            states=ref['states'].tolist();fg=ref['pred']==y[:,None]
            for method in ('victim','PAR','clean_only','ranking','IS2'):
                zs=[load(mapping[method,seed if method not in ('victim','PAR') else 0]['file']) for seed in SEEDS]
                for z in zs:
                    assert np.array_equal(z['ids'],ref['ids']) and np.array_equal(z['labels'],y) and z['states'].tolist()==states
                for gi,grid in enumerate(('identity','jpeg90','jpeg70','resize168','resize112')):
                    c,a=1+4*gi,2+4*gi;bug=fg[:,c]&~fg[:,a]
                    masks={'native_clean_accuracy':allmask,'clean_accuracy':allmask,'attacked_accuracy':allmask,
                           'asr':y!=target_class,'retained_repair':bug,'trigger_induced_failure':allmask,
                           'prediction_change':allmask,'both_correct':allmask,'original_clean_regression':fg[:,c],
                           'target_interaction_abs':y!=target_class,'target_interaction_signed':y!=target_class,
                           'allclass_interaction_abs':allmask}
                    vv={}
                    for z in zs:
                        good=z['pred']==y[:,None];margin=z['target_margin'].astype(float)
                        m={'native_clean_accuracy':good[:,0],'clean_accuracy':good[:,c],'attacked_accuracy':good[:,a],
                           'asr':z['pred'][:,a]==target_class,'retained_repair':good[:,c]&good[:,a],
                           'trigger_induced_failure':good[:,c]&~good[:,a],
                           'prediction_change':z['pred'][:,c]!=z['pred'][:,a],'both_correct':good[:,c]&good[:,a],
                           'original_clean_regression':~good[:,c],
                           'target_interaction_abs':abs(margin[:,a]-margin[:,c]),
                           'target_interaction_signed':margin[:,a]-margin[:,c],
                           'allclass_interaction_abs':z['allclass_interaction_abs'][:,gi,0].astype(float)}
                        for metric,v in m.items():vv.setdefault(metric,[]).append(v)
                    for metric,arr in vv.items():
                        mask=masks[metric]
                        if not mask.any():continue
                        factor=1 if 'interaction' in metric else 100
                        v=factor*np.stack(arr)*mask;name=grid+'/'+metric
                        means=point('backdoor',attack+'/'+bank,method,name,v,'cosine' if factor==1 else 'percent',mask)
                        expected=[r['seed_values'] for r in summary if r['case']==attack and r['bank']==bank and r['method']==method and r['metric']==name]
                        assert len(expected)==1,(attack,bank,method,name)
                        check(f'backdoor/{attack}/{bank}/{method}/{name}',means,expected[0])
                        if bank=='imagenetv2' and grid=='identity' and metric=='allclass_interaction_abs':
                            point('backdoor',attack+'/'+bank,method,grid+'/allalternative_interaction_abs',v*1000/999,'cosine',mask)
                        if bank=='imagenetv2' and grid=='identity' and metric in ('attacked_accuracy','retained_repair','allclass_interaction_abs') and method in ('victim','ranking','IS2'):
                            key=(attack,bank,grid,metric)
                            existing=next((r for r in VECTORS if r.get('key')==key),None)
                            if existing is None:
                                existing=dict(key=key,case='backdoor',bank=attack+'/'+bank,metric=name,
                                    values={},den=mask.astype(float),clusters=np.arange(n));VECTORS.append(existing)
                            existing['values'][method]=v

def bootstrap(draws,seed):
    result=[];rng=np.random.default_rng(seed)
    for group in VECTORS:
        values=group['values'];names=list(values);den=group['den'];cl=group['clusters'];nc=int(cl.max())+1
        chunks=np.stack([values[n] for n in names],axis=-1)
        sums=np.stack([np.stack([np.bincount(cl,weights=chunks[s,:,j],minlength=nc) for j in range(len(names))],axis=1) for s in range(3)])
        denominator=np.bincount(cl,weights=den,minlength=nc)
        res=[]
        for start in range(0,draws,64):
            count=min(64,draws-start);w=rng.multinomial(nc,np.full(nc,1/nc),size=count)
            d=w@denominator
            while (d==0).any():
                bad=d==0;w[bad]=rng.multinomial(nc,np.full(nc,1/nc),size=int(bad.sum()));d=w@denominator
            v=(w@sums.transpose(1,0,2).reshape(nc,-1)).reshape(count,3,len(names))/d[:,None,None]
            sw=rng.multinomial(3,[1/3]*3,size=count)/3
            res.append(np.einsum('bs,bsm->bm',sw,v))
        sampled=np.concatenate(res);new=next(n for n in names if n in ('IS','IS2'))
        old=next(n for n in names if n in ('Ranking','ranking'))
        delta=sampled[:,names.index(new)]-sampled[:,names.index(old)]
        result.append(dict(case=group['case'],bank=group['bank'],metric=group['metric'],
             comparison='IS minus Ranking',draws=draws,bootstrap_seed=seed,n_source_clusters=nc,
             ci95_seed_source=np.quantile(delta,[.025,.975]).tolist()))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('reproduced'))
    p.add_argument('--bootstrap',type=int,default=0);p.add_argument('--bootstrap-seed',type=int,default=260930)
    p.add_argument('--case',choices=('all','typography','color_binding','backdoor'),default='all')
    p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    assert a.bootstrap>=0
    records=read('results/records.json')
    with threadpool_limits(limits=a.threads):
        if a.case in ('all','color_binding'):
            color_binding(records);color_gallery(records);color_transfer(records);color_natural(records)
        if a.case in ('all','typography'):
            typography(records);typography_public();typography_backbones(records);typography_dyslexify()
            if (ROOT/'results/reports/typography/retention.json').exists():typography_retention(records)
        if a.case in ('all','backdoor'):
            backdoor(records)
        intervals=bootstrap(a.bootstrap,a.bootstrap_seed) if a.bootstrap else []
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'results.json').write_text(json.dumps(POINTS,indent=2)+'\n')
    with (a.out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(POINTS[0]));w.writeheader();w.writerows(POINTS)
    (a.out/'checks.json').write_text(json.dumps(dict(checks=CHECKS,passed=len(CHECKS),n_result_rows=len(POINTS),new_training=False,new_inference=False),indent=2)+'\n')
    if intervals:(a.out/'bootstrap.json').write_text(json.dumps(intervals,indent=2)+'\n')
    print(f'Reconstructed {len(POINTS)} result rows; {len(CHECKS)} independent checks passed. Output: {a.out}')

if __name__=='__main__':main()
