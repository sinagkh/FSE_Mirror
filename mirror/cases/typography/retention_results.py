"""Paired three-seed typography comparison on every existing test bank."""
import mirror.cases.typography.retention as run
from mirror.cases.typography.diagnose import *
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.append(str(ROOT/'mirror/cases/backdoor'))
import mirror.tools.aggregate as statistics

SEEDS=(42,43,44)
METHODS=('frozen','ranking','IS4','ranking_P','IS8_P')
INPUTS={}


def track(p):
    p=Path(p);INPUTS[str(p)]=sha(p);return p


def read(p):
    return json.loads(track(p).read_text())


def array(p):
    with np.load(track(p)) as z:return dict(z)


def location(method,seed):
    current=(ROOT/'mirror/cases/typography_strength_20260929/characterization/w4/retest' if seed==42 else
             ROOT/f'clip/fse_selected_strength_replication_20260929/typography/seed{seed}/retest')
    if method in ('frozen','ranking','IS4'):
        return current, 'IS' if method=='IS4' else method
    arm='R_preserve' if method=='ranking_P' else 'IS8_preserve'
    if seed==42:
        old='fse_typographic_preservation_20260929' if method=='ranking_P' else 'fse_typographic_preservation_extension_20260929'
        return ROOT/'clip'/old/'evaluation'/arm/'retest','IS'
    return run.seed_root(seed)/'evaluation'/arm/'retest','IS'


def frame(method,seed,bank):
    root,key=location(method,seed)
    return pd.read_csv(track(root/'external_retest'/f'{bank}_{key}.csv')).set_index('source_id').sort_index()


def selftest():
    a=statistics.Analysis('selftest','paired','IS8_P',100)
    x=np.arange(30,dtype=float).reshape(3,10)
    a.add('equal',{'IS8_P':x,'ranking_P':x},np.ones(10),np.arange(10),'percent')
    a.add('constant',{'IS8_P':x+1,'ranking_P':x},np.ones(10),np.arange(10),'percent')
    r=a.finish()
    assert r['contrasts'][0]['seed_source_ci95']==[0,0]
    assert np.allclose(r['contrasts'][1]['seed_source_ci95'],[1,1])
    print('BOOTSTRAP SELFTEST PASSED',flush=True)


def analyze():
    results=[]
    for seed in (43,44):
        for arm in run.ARMS:
            out=run.seed_root(seed)/'evaluation'/arm/'retest'
            for p,h in read(out/'complete.json')['files'].items():assert sha(track(out/p))==h
    reference,_=location('IS4',42)
    for folder in sorted((reference/'digital_retest').iterdir()):
        bank=folder.name;rows=read(folder/'rows.json');ids=[r['image_id'] for r in rows]
        frozen=array(folder/'frozen.npz');bug=(frozen['blank']>0)&(frozen['conflict']<=0)
        values={}
        for method in METHODS:
            vv={}
            for seed in SEEDS:
                root,key=location(method,seed);path=root/'digital_retest'/bank
                assert read(path/'rows.json')==rows
                z=array(path/f'{key}.npz');targets=array(path/f'{key}_all12.npz')
                d=targets['contrasts'];length=targets['caption_difference_norm']
                retained=bug&(z['conflict']>0)&(z['clean']>0)&(z['blank']>0)
                mm=dict(all_targets_abs=abs(d).mean((1,2)),
                    directional_targets_abs=(abs(d)/np.maximum(length[:,None,:],1e-8)).mean((1,2)),
                    conflict_pair_accuracy=100*(z['conflict']>0).mean(1),
                    attack_top1=100*z['top1'][:,3:].mean(1),clean_top1=100*z['top1'][:,0],
                    retained_repair=100*retained.sum(1))
                for k,v in mm.items():vv.setdefault(k,[]).append(v)
            values[method]={k:np.stack(v) for k,v in vv.items()}
        a=statistics.Analysis('typography',bank,'IS8_P',5000)
        for metric in values['IS8_P']:
            a.add(metric,{m:v[metric] for m,v in values.items()},
                bug.sum(1) if metric=='retained_repair' else np.ones(len(rows)),ids,
                'cosine' if 'targets' in metric else 'percent')
        results.append(a.finish());print('ANALYZED',bank,flush=True)
    for bank in ('SCAM','SynthSCAM','NoSCAM','RTA100'):
        frames={m:[frame(m,s,bank) for s in SEEDS] for m in METHODS}
        ids=frames['IS8_P'][0].index.to_numpy();n=len(ids)
        for ff in frames.values():assert all(np.array_equal(f.index,ids) for f in ff)
        a=statistics.Analysis('typography',bank,'IS8_P',5000)
        for key in (('correct','top1') if bank=='RTA100' else ('correct',)):
            a.add('official_top1' if key=='top1' else 'pair_accuracy',
                {m:100*np.stack([f[key].to_numpy(float) for f in ff]) for m,ff in frames.items()},
                np.ones(n),ids,'percent')
        if bank in ('SCAM','SynthSCAM'):
            clean={m:[frame(m,s,'NoSCAM') for s in SEEDS] for m in METHODS}
            assert all(np.array_equal(f.index,ids) for ff in clean.values() for f in ff)
            bug=clean['frozen'][0].correct.to_numpy()&~frames['frozen'][0].correct.to_numpy()
            drift={m:np.stack([abs(c.margin.to_numpy()-v.margin.to_numpy()) for c,v in zip(clean[m],ff)]) for m,ff in frames.items()}
            repaired={m:100*np.stack([bug&c.correct.to_numpy()&v.correct.to_numpy() for c,v in zip(clean[m],ff)]) for m,ff in frames.items()}
            a.add('absolute_word_removal_effect',drift,np.ones(n),ids,'cosine')
            a.add('retained_repair',repaired,bug,ids,'percent')
        results.append(a.finish());print('ANALYZED',bank,flush=True)
    frames={}
    for m in METHODS:
        frames[m]=[]
        for s in SEEDS:
            root,key=location(m,s)
            frames[m].append(pd.read_csv(track(root/'sugarcrepe'/f'{key}.csv')))
    reference=frames['IS8_P'][0]
    for ff in frames.values():
        assert all(f[['subset','example_id','filename']].equals(reference[['subset','example_id','filename']]) for f in ff)
    for category in ['full']+sorted(set(reference.subset)):
        use=np.ones(len(reference),bool) if category=='full' else reference.subset.to_numpy()==category
        ids=sorted(set(reference.loc[use,'filename']));values={};den=None
        for m,ff in frames.items():
            nums=[]
            for f in ff:
                g=f.loc[use].groupby('filename').correct.agg(['sum','count']).reindex(ids)
                nums.append(100*g['sum'].to_numpy())
                if den is None:den=g['count'].to_numpy()
                else:assert np.array_equal(den,g['count'])
            values[m]=np.stack(nums)
        a=statistics.Analysis('typography','SugarCrepe_'+category,'IS8_P',5000)
        a.add('accuracy',values,den,ids,'percent');results.append(a.finish())
    return results


def report(results):
    out=run.RUN/'reports';out.mkdir(exist_ok=True)
    stats=[r for b in results for r in b['statistics']]
    contrasts=[r for b in results for r in b['contrasts']]
    seeds=[r for b in results for r in b['per_seed']]
    for name,data in [('all_metrics',stats),('paired_contrasts',contrasts),('per_seed',seeds)]:
        pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
    dump(out/'all_results.json',results)
    lines=['# Typography preservation: complete three-seed comparison','',
        'Seeds 42, 43, 44; mean ± sample SD. IS8_P and ranking_P share the preservation package. '
        'IS4 and ranking are the existing no-P references. Seed42 is reused; all requested seeds are retained. '
        'No automatic recipe choice or manuscript replacement.','',
        '## Full metric grid','']
    for result in results:
        bank=result['bank'];lines += ['### '+bank,'',
            '| Measure | '+' | '.join(METHODS)+' |','|---|'+'---:|'*len(METHODS)]
        for metric in dict.fromkeys(r['metric'] for r in result['statistics']):
            cells=[]
            for m in METHODS:
                r=next(r for r in result['statistics'] if r['metric']==metric and r['method']==m)
                digits=5 if r['unit']=='cosine' else 2
                cells.append(f"{r['mean']:.{digits}f} ± {r['sample_sd']:.{digits}f}")
            lines.append('| '+metric+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## Matched paired changes','',
              '| Bank | Measure | Comparator | IS8_P difference [95% seed/source CI] |','|---|---|---|---:|']
    for r in contrasts:
        if r['other']=='frozen':continue
        if r['bank'].startswith('SugarCrepe_') and r['bank']!='SugarCrepe_full':continue
        lo,hi=r['seed_source_ci95'];d=5 if r['unit']=='cosine' else 2
        lines.append(f"| {r['bank']} | {r['metric']} | {r['other']} | {r['mean']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}] |")
    lines += ['','## Per-seed preservation','',
        '| Bank / measure | Method | Seed 42 | Seed 43 | Seed 44 |','|---|---|---:|---:|---:|']
    for bank,metric in [('RTA100','official_top1'),('RTA100','pair_accuracy'),
                        ('SugarCrepe_full','accuracy'),('test_seen_standard','clean_top1')]:
        for m in METHODS:
            r=next(r for r in stats if r['bank']==bank and r['metric']==metric and r['method']==m)
            lines.append('| '+bank+'/'+metric+' | '+m+' | '+' | '.join(f'{v:.2f}' for v in r['seed_values'])+' |')
    lines += ['','RTA100 official_top1 uses all 100 classes and all 1,000 images. '
        'The separate pair_accuracy is not substituted for it. All intervals use 5,000 paired '
        'source-cluster draws and common seed resamples; all edits and decisions for one source stay together. '
        'These are developmental seed replications after previous benchmark results were known. '
        'Backbone ports await the author’s single-recipe decision.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'complete.json',dict(seeds=SEEDS,inputs=INPUTS,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'},
        protocol_sha256=sha(run.RUN/'protocol.json'),analysis_script_sha256=sha(__file__)))
    print('REPORT COMPLETE',out/'RESULTS.md',flush=True)


if __name__=='__main__':
    run.command();run.verify();statistics.DEST=run.RUN
    with threadpool_limits(limits=4,user_api='blas'):
        selftest()
        if '--selftest' not in sys.argv:report(analyze())
