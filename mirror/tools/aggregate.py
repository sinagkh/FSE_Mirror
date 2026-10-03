"""Complete paired three-seed analysis for selected-strength replication."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import mirror.tools.replicate as run
from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha

SEEDS = (42,43,44)
DEST = run.DEST


class Analysis:
    def __init__(self,case,bank,new,replicates):
        self.case,self.bank,self.new,self.B = case,bank,new,replicates
        self.groups={}
        self.stats=[]
        self.contrasts=[]
        self.perseed=[]

    def add(self,metric,values,den,ids,unit):
        ids=np.asarray(ids)
        den=np.asarray(den,dtype=float)
        assert len(set(ids.tolist()))==len(ids)
        assert den.shape==(len(ids),) and np.all(den>=0)
        if den.sum()==0:
            return
        key=hashlib.sha256(den.tobytes()+json.dumps(ids.tolist()).encode()).hexdigest()
        group=self.groups.setdefault(key,dict(den=den,entries=[],ids=ids))
        for name,num in values.items():
            num=np.asarray(num,dtype=float)
            assert num.shape==(3,len(ids)) and np.isfinite(num).all()
            point=num.sum(1)/den.sum()
            stat=dict(case=self.case,bank=self.bank,metric=metric,method=name,unit=unit,
                sources=len(ids),denominator=float(den.sum()),mean=float(point.mean()),
                sample_sd=float(point.std(ddof=1)),seed_values=point.tolist())
            self.stats.append(stat)
            for seed,value in zip(SEEDS,point):
                self.perseed.append(dict(**{k:v for k,v in stat.items() if k not in ('mean','sample_sd','seed_values')},
                    seed=seed,value=float(value)))
            group['entries'].append((metric,name,num,stat))

    def finish(self):
        rng=np.random.default_rng(646711 if self.case!='typography' else 260926)
        sw=rng.multinomial(3,[1/3]*3,size=self.B)
        for group in self.groups.values():
            den=group['den'];N=len(den);entries=group['entries'];K=len(entries)
            matrix=np.concatenate([e[2].T for e in entries],axis=1)
            item=np.empty((self.B,K));cross=np.empty_like(item)
            for start in range(0,self.B,64):
                end=min(self.B,start+64)
                counts=rng.multinomial(N,np.full(N,1/N),size=end-start).astype(float)
                denominator=counts@den
                # Very small frozen-bug cohorts can have empty resamples.
                while np.any(denominator==0):
                    bad=denominator==0
                    counts[bad]=rng.multinomial(N,np.full(N,1/N),size=int(bad.sum()))
                    denominator=counts@den
                means=(counts@matrix).reshape(end-start,K,3)/denominator[:,None,None]
                item[start:end]=means.mean(2)
                cross[start:end]=(means*sw[start:end,None,:]).sum(2)/3
            index={(e[0],e[1]):j for j,e in enumerate(entries)}
            for j,(metric,name,_,stat) in enumerate(entries):
                stat['item_ci95']=np.quantile(item[:,j],[.025,.975]).tolist()
                stat['seed_source_ci95']=np.quantile(cross[:,j],[.025,.975]).tolist()
                if name==self.new:
                    continue
                newj=index[(metric,self.new)]
                new=entries[newj][3]
                self.contrasts.append(dict(case=self.case,bank=self.bank,metric=metric,
                    new=self.new,other=name,unit=stat['unit'],sources=N,denominator=float(den.sum()),
                    mean=new['mean']-stat['mean'],
                    item_ci95=np.quantile(item[:,newj]-item[:,j],[.025,.975]).tolist(),
                    seed_source_ci95=np.quantile(cross[:,newj]-cross[:,j],[.025,.975]).tolist(),
                    resamples=self.B))
        result=dict(case=self.case,bank=self.bank,seeds=SEEDS,statistics=self.stats,
                    contrasts=self.contrasts,per_seed=self.perseed,resamples=self.B,
                    bootstrap='paired source and crossed seed/source percentile; same source draws across seeds and methods',
                    fixed_models='frozen/released deterministic references repeated only for pairing; SD zero is not three independent fits')
        dump(DEST/'analysis'/self.case/self.bank/'results.json',result)
        return result


def backdoor(case,bank):
    import mirror.cases.backdoor.report as old
    import mirror.cases.backdoor.enforcement as pilot
    paths,records=old.records(case,bank)
    ref=records['victim']
    for seed in SEEDS:
        folder=(pilot.DEST/case if seed==42 else DEST/'backdoor'/case)/'evaluation'/bank
        path=folder/f'IS2_seed{seed}_records.npz'
        receipt=run.read(folder/'complete.json')
        assert sha(path)==receipt['records_sha256'][f'IS2_seed{seed}']
        with np.load(path) as z:
            value=dict(z)
        for key in ('ids','labels','states'):
            assert np.array_equal(value[key],ref[key]),(case,bank,seed,key)
        records[f'IS2_seed{seed}']=value
        paths[f'IS2_seed{seed}']=path
    names=['victim','PAR','clean_only','ranking','IS','IS2']
    names+=['exact_filter','tolerant_filter'] if case=='stripes' else ['blend_inversion','gated_inversion','oracle_gated_inversion']
    analysis=Analysis(case,bank,'IS2',4000)
    unavailable=[]
    for gi,grid in enumerate(pilot.data.GRID):
        metrics={}
        for name in names:
            keys=[f'{name}_seed{s}' for s in SEEDS] if name in ('clean_only','ranking','IS','IS2') else [name]*3
            assert all(k in records for k in keys),(name,case,bank)
            metrics[name]=[old.metrics(records[k],ref,bank,gi) for k in keys]
        for metric,(_,mask) in metrics['IS2'][0].items():
            values={}
            for name,mm in metrics.items():
                if not all(metric in m for m in mm):
                    # These pixel-only control exports retain target margins,
                    # but not the all-class interaction. Keep their behavioral
                    # comparisons; record this mechanism measurement as absent.
                    assert metric=='allclass_interaction_abs' and name in ('gated_inversion','oracle_gated_inversion'), (case,bank,name,metric)
                    unavailable.append(dict(method=name,processing=grid,metric=metric,
                        reason='Not stored in original input-control per-example exports; not imputed.'))
                    continue
                assert all(np.array_equal(m[metric][1],mask) for m in mm)
                factor=1 if 'interaction' in metric else 100
                values[name]=factor*np.stack([m[metric][0].astype(float)*mask for m in mm])
            analysis.add(grid+'/'+metric,values,mask,ref['ids'],'cosine' if factor==1 else 'percent')
    result=analysis.finish()
    result['unavailable_measurements']=unavailable
    dump(DEST/'analysis'/case/bank/'results.json',result)
    dump(DEST/'analysis'/case/bank/'provenance.json',dict(records_sha256={str(p):sha(p) for p in paths.values()}))
    print('ANALYZED',case,bank,flush=True)
    return result


def typography_roots():
    return {42:run.ROOT/'mirror/cases/typography_strength_20260929/characterization/w4/retest',
            **{s:DEST/'typography'/f'seed{s}'/'retest' for s in (43,44)}}


def typography():
    import itertools
    import mirror.cases.typography.diagnose as d
    roots=typography_roots()
    for seed in (43,44):
        for p,h in run.read(roots[seed]/'complete.json')['files'].items():
            assert sha(roots[seed]/p)==h,p
    names=('frozen','ranking','previous_IS','IS')
    text=np.load(roots[42]/'features/IS_digital.npy')
    txt=torch_load(d.OUT/'texts.pt')
    labels=txt['vocabulary'];idx={n:i for i,n in enumerate(labels)}
    results=[]
    pairs=list(itertools.combinations(range(1,5),2))
    for first in sorted((roots[42]/'digital_retest').iterdir()):
        bank=first.name;rows=run.read(first/'rows.json');N=len(rows)
        yi=np.array([idx[r['label']] for r in rows]);wi=np.array([[idx[w] for w in r['words'][1:]] for r in rows])
        ids=[r['image_id'] for r in rows]
        zz={}
        for name in names:
            zz[name]=[]
            for seed in SEEDS:
                folder=roots[seed]/'digital_retest'/bank
                assert run.read(folder/'rows.json')==rows
                with np.load(folder/f'{name}.npz') as z:
                    zz[name].append(dict(z))
        frozen=zz['frozen'][0];bug=(frozen['blank']>0)&(frozen['conflict']<=0)
        analysis=Analysis('typography',bank,'IS',5000)
        all_values={}
        for name,zs in zz.items():
            vv={}
            for j,z in enumerate(zs):
                with np.load(roots[SEEDS[j]]/'digital_retest'/bank/f'{name}_all12.npz') as targets:
                    diff=targets['contrasts'];length=targets['caption_difference_norm']
                retained=bug&(z['conflict']>0)&(z['clean']>0)&(z['blank']>0)
                mm=dict(all_targets_abs=abs(diff).mean((1,2)),
                    directional_targets_abs=(abs(diff)/np.maximum(length[:,None,:],1e-8)).mean((1,2)),
                    conflict_pair_accuracy=100*(z['conflict']>0).mean(1),
                    attack_top1=100*z['top1'][:,3:].mean(1),clean_top1=100*z['top1'][:,0],
                    retained_repair=100*retained.sum(1))
                for metric,v in mm.items():
                    vv.setdefault(metric,[]).append(v)
            all_values[name]={metric:np.stack(v) for metric,v in vv.items()}
        for metric in all_values['IS']:
            den=bug.sum(1) if metric=='retained_repair' else np.ones(N)
            analysis.add(metric,{name:v[metric] for name,v in all_values.items()},den,ids,
                         'cosine' if 'targets' in metric else 'percent')
        results.append(analysis.finish())
    external={}
    for bank in ('SCAM','SynthSCAM','NoSCAM','RTA100'):
        frames={name:[pd.read_csv(roots[s]/'external_retest'/f'{bank}_{name}.csv').set_index('source_id').sort_index()
                      for s in SEEDS] for name in (*names,'Defense_Prefix')}
        ids=frames['IS'][0].index.to_numpy();N=len(ids)
        for ff in frames.values():
            assert all(np.array_equal(f.index,ids) for f in ff)
        external[bank]=frames
        a=Analysis('typography',bank,'IS',5000)
        for key in (('correct','top1') if bank=='RTA100' else ('correct',)):
            a.add('official_top1' if key=='top1' else 'pair_accuracy',
                  {name:100*np.stack([f[key].to_numpy(float) for f in ff]) for name,ff in frames.items()},
                  np.ones(N),ids,'percent')
        if bank in ('SCAM','SynthSCAM'):
            clean={name:[pd.read_csv(roots[s]/'external_retest'/f'NoSCAM_{name}.csv').set_index('source_id').sort_index()
                         for s in SEEDS] for name in frames}
            assert all(np.array_equal(f.index,ids) for ff in clean.values() for f in ff)
            bug=clean['frozen'][0].correct.to_numpy()&~frames['frozen'][0].correct.to_numpy()
            drift={name:np.stack([abs(c.margin.to_numpy()-v.margin.to_numpy()) for c,v in zip(clean[name],ff)]) for name,ff in frames.items()}
            repair={name:100*np.stack([bug&c.correct.to_numpy()&v.correct.to_numpy() for c,v in zip(clean[name],ff)]) for name,ff in frames.items()}
            a.add('absolute_word_removal_effect',drift,np.ones(N),ids,'cosine')
            a.add('retained_repair',repair,bug,ids,'percent')
        results.append(a.finish())
    frames={name:[pd.read_csv(roots[s]/'sugarcrepe'/f'{name}.csv') for s in SEEDS] for name in names}
    reference=frames['IS'][0]
    for ff in frames.values():
        assert all(f[['subset','example_id','filename']].equals(reference[['subset','example_id','filename']]) for f in ff)
    for category in ['full']+sorted(set(reference.subset)):
        use=np.ones(len(reference),bool) if category=='full' else reference.subset.to_numpy()==category
        ids=sorted(set(reference.loc[use,'filename']))
        values={};den=None
        for name,ff in frames.items():
            nums=[]
            for f in ff:
                grouped=f.loc[use].groupby('filename').correct.agg(['sum','count']).reindex(ids)
                nums.append(100*grouped['sum'].to_numpy())
                if den is None: den=grouped['count'].to_numpy()
                else: assert np.array_equal(den,grouped['count'])
            values[name]=np.stack(nums)
        a=Analysis('typography','SugarCrepe_'+category,'IS',5000)
        a.add('accuracy',values,den,ids,'percent')
        results.append(a.finish())
    print('ANALYZED typography',flush=True)
    return results


def torch_load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)


def report(results,scope):
    folder=DEST/'reports'/scope
    folder.mkdir(parents=True,exist_ok=True)
    stats=[r for result in results for r in result['statistics']]
    contrasts=[r for result in results for r in result['contrasts']]
    perseed=[r for result in results for r in result['per_seed']]
    pd.DataFrame(stats).to_csv(folder/'all_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(folder/'paired_contrasts.csv',index=False)
    pd.DataFrame(perseed).to_csv(folder/'per_seed.csv',index=False)
    dump(folder/'all_results.json',results)
    lines=['# Selected-strength three-seed results','',
        'Seeds 42, 43, 44. Seed 42 is reused; seeds 43 and 44 replicate fixed backdoor 2x / plain typography 4x recipes. '
        'All original results are preserved. Values are mean ± sample SD, not standard error.', '',
        'Only the interaction multiplier changes relative to original IS: same per-seed initialization, counterfactuals, '
        'stream, optimizer, guards, schedule and final-update selector. Typography has no added preservation package. '
        'The strength choices followed seed-42 development/public retests; these are seed replications, not untouched benchmark confirmation.', '',
        '## Paper-ready endpoints','']
    def table(case,bank,metrics,methods):
        lines.extend([f'### {case} / {bank}','', '| Endpoint | '+' | '.join(methods)+' |',
                      '|---|'+'---:|'*len(methods)])
        for metric in metrics:
            selected=[r for r in stats if r['case']==case and r['bank']==bank and r['metric']==metric]
            lookup={r['method']:r for r in selected}
            def cell(m):
                if m not in lookup:return '—'
                r=lookup[m];digits=5 if r['unit']=='cosine' else 2
                return f"{r['mean']:.{digits}f} ± {r['sample_sd']:.{digits}f}"
            lines.append('| '+metric+' | '+' | '.join(cell(m) for m in methods)+' |')
        lines.append('')
    if scope in ('backdoor','all'):
        for case in run.CASES:
            for bank in ('development','imagenetv2','banana87','banana1000'):
                table(case,bank,['identity/'+m for m in ('native_clean_accuracy','clean_accuracy','attacked_accuracy','retained_repair','allclass_interaction_abs','prediction_change')],
                      ['victim','PAR','clean_only','ranking','IS','IS2'])
            table(case,'imagenetv2',[g+'/attacked_accuracy' for g in ('identity','jpeg90','jpeg70','resize168','resize112')],
                  ['ranking','IS','IS2']+(['exact_filter','tolerant_filter'] if case=='stripes' else ['gated_inversion','oracle_gated_inversion']))
    if scope in ('typography','all'):
        for bank in sorted({r['bank'] for r in stats if r['case']=='typography'}):
            mm=list(dict.fromkeys(r['metric'] for r in stats if r['case']=='typography' and r['bank']==bank))
            table('typography',bank,mm,['frozen','ranking','previous_IS','IS']+(['Defense_Prefix'] if bank in ('SCAM','SynthSCAM','NoSCAM','RTA100') else []))
    lines+=['## Paired changes on primary endpoints','',
            '| Case / bank | Endpoint | Comparison | Change [95% seed/source CI] |','|---|---|---|---:|']
    for r in contrasts:
        if r['other'] not in ('ranking','IS','previous_IS'):continue
        if r['case']=='typography':
            if r['bank'] not in ('test_seen_standard','SCAM','SynthSCAM','NoSCAM','RTA100','SugarCrepe_full'):continue
        elif r['bank']!='imagenetv2' or not r['metric'].startswith('identity/'):continue
        digits=5 if r['unit']=='cosine' else 2
        lo,hi=r['seed_source_ci95']
        lines.append(f"| {r['case']} / {r['bank']} | {r['metric']} | {r['new']} − {r['other']} | {r['mean']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] |")
    lines+=['','Complete processing conditions, target-class preservation, public tests and per-seed outcomes remain in the CSV/JSON files. '
            'Bootstrap resamples are paired on source IDs, preserving all states/decisions per source; 4,000 backdoor and 5,000 typography draws. '
            'Seed and source draws are shared across methods. Deterministic frozen/released methods have zero seed variation by construction. '
            'The manuscript has not been changed automatically.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(folder/'complete.json',dict(scope=scope,seeds=SEEDS,source_sha256=sha(__file__),
        protocol_sha256=sha(DEST/'protocol.json'),outputs_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json'}))


def selftest():
    a=Analysis('selftest','paired','IS',4000)
    x=np.arange(30,dtype=float).reshape(3,10)
    a.add('identity',{'IS':x,'ranking':x},np.ones(10),np.arange(10),'percent')
    a.add('constant',{'IS':np.ones((3,10))*2,'ranking':np.ones((3,10))},np.ones(10),np.arange(10),'percent')
    r=a.finish()
    assert r['contrasts'][0]['mean']==0 and r['contrasts'][0]['seed_source_ci95']==[0,0]
    assert r['contrasts'][1]['mean']==1 and r['contrasts'][1]['seed_source_ci95']==[1,1]
    dump(DEST/'bootstrap_checks.json',dict(identity_interval_exactly_zero=True,constant_difference_exactly_one=True,source_sha256=sha(__file__)))


if __name__=='__main__':
    run.command()
    ap=argparse.ArgumentParser();ap.add_argument('scope',choices=['selftest','backdoor','typography','all']);a=ap.parse_args()
    with threadpool_limits(limits=4,user_api='blas'):
        selftest()
        if a.scope!='selftest':
            run.verify();results=[]
            if a.scope=='all':
                for scope in ('backdoor','typography'):
                    folder=DEST/'reports'/scope
                    receipt=run.read(folder/'complete.json')
                    assert sha(folder/'all_results.json')==receipt['outputs_sha256']['all_results.json']
                    results += run.read(folder/'all_results.json')
            if a.scope=='backdoor':
                results += [backdoor(case,bank) for case in run.CASES for bank in run.BANKS]
            if a.scope=='typography':
                results += typography()
            report(results,a.scope)
