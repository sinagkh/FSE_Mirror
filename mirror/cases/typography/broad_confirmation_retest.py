"""Shared prompt checkpoints, unchanged test sources/metrics and frozen selection."""
from mirror.cases.typography.diagnose import *
import mirror.cases.typography.replicate as pilot
import mirror.cases.typography.retest as digital
import mirror.cases.typography.preservation as sugar
import mirror.cases.typography.external as public
import pandas as pd

RUN = pilot.RUN

def registry():
    p=RUN/'selection.json'
    assert sha(p)==(RUN/'selection.sha256').read_text().strip()
    sel=json.loads(p.read_text())
    reg={k:sel[k] for k in ('ranking','IS')}
    reg['initial_prefix']=sel['initial_prefix']
    reg['same_scale_ablation']=next(r for r in sel['all_candidates'] if r['name']==sel['same_scale_ablation'])
    for r in reg.values():assert sha(r['checkpoint'])==r['sha256']
    return {'frozen':None,**reg}

def freeze():
    pilot.verify_prompt();public.verify_external();reg=registry()
    files=[Path(__file__),RUN/'selection.json',RUN/'protocol.json',OUT/'texts.pt',
           CODE/'retest.py',CODE/'external.py',CODE/'preservation.py',CODE/'prompt_pilot.py',
           OUT/'external_retest/protocol.json',OUT/'external_retest/cache.json',
           OUT/'external_retest/rows.json',OUT/'external_retest/prefix_tests.json',
           OUT/'sugarcrepe/cache.json',sugar.REF]
    for bank,style in digital.TESTS:files.extend([OUT/f'{bank}.json',OUT/f'{bank}_{style}_features.json'])
    cfg=dict(files={str(p):sha(p) for p in files},registry=reg,tests=digital.TESTS,
             adapter_location='prefix',prefix_position='after BOS before complete caption',seed=pilot.SEED,bootstrap_resamples=5000,
             checkpoint_selection='fixed final checkpoints; no search',
             status='developmental retest; earlier text-side public results observed',
             full_registered_populations=True)
    p=RUN/'retest_protocol_v2.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(cfg))
    else:dump(p,cfg);(RUN/'retest_protocol_v2.sha256').write_text(sha(p)+'\n')
    return reg

def encode_text_cache(reg):
    cache=Path(json.loads((OUT/'protocol.json').read_text())['cache'])/('broad_confirmation_'+RUN.name)/'prompt_retest'
    cache.mkdir(parents=True,exist_ok=True);model,tok=pilot.network()
    txt=torch.load(OUT/'texts.pt',map_location='cpu');labels=txt['vocabulary']
    sugar_meta=json.loads((OUT/'sugarcrepe/cache.json').read_text())
    public_cfg=public.verify_external();public_meta=json.loads((OUT/'external_retest/cache.json').read_text())
    public_cache=Path(public_meta['path']);allmeta={}
    for name,r in reg.items():
        if r is None:continue
        ck=torch.load(r['checkpoint'],map_location='cpu');assert ck['adapter_location']=='prefix'
        prefix=ck['prefix'].cuda()
        definitions={
            'digital':([s.format(n) for n in labels for s in TEMPLATES],len(labels),3,'norm_mean_norm'),
            'sugar':(sugar_meta['prompts'],len(sugar_meta['prompts']),1,'norm'),
        }
        for bank,templates in [('SCAM',public_cfg['scam_templates']),('RTA100',public_cfg['rta_templates'])]:
            ll=json.loads((public_cache/f'{bank}_labels.json').read_text())
            definitions[bank]=([s.format(n) for n in ll for s in templates],len(ll),len(templates),'mean_norm')
        for bank,(prompts,n,k,pool) in definitions.items():
            target=cache/f'{name}_{bank}.npy';meta_path=RUN/f'cache_{name}_{bank}.json'
            if target.exists():
                meta=json.loads(meta_path.read_text());assert meta['checkpoint_sha256']==r['sha256'] and sha(target)==meta['sha256']
                allmeta[f'{name}_{bank}']=meta;continue
            ff=[]
            with torch.no_grad():
                for start in range(0,len(prompts),128):
                    tt=pilot.tokens_with_prefix(tok,prompts[start:start+128])
                    ff.append(pilot.encode_prefix(model,tt,prefix).cpu())
            raw=torch.cat(ff)
            if pool=='norm_mean_norm':t=norm(norm(raw).reshape(n,k,-1).mean(1))
            elif pool=='mean_norm':t=norm(raw.reshape(n,k,-1).mean(1))
            else:t=norm(raw)
            np.save(target,t.numpy())
            meta=dict(path=str(target),sha256=sha(target),checkpoint_sha256=r['sha256'],n_prompts=len(prompts),pool=pool,code_sha256=sha(Path(__file__)))
            dump(meta_path,meta);allmeta[f'{name}_{bank}']=meta
            print('PREFIX TEXT',name,bank,len(prompts),flush=True)
    del model;torch.cuda.empty_cache();dump(RUN/'text_caches.json',allmeta)
    return {'frozen':None,**{name:{bank:np.load(allmeta[f'{name}_{bank}']['path']) for bank in ('digital','sugar','SCAM','RTA100')} for name,r in reg.items() if r is not None}}

def digital_score(mm):
    txt=torch.load(OUT/'texts.pt',map_location='cpu');base=txt['features'].numpy();idx={n:i for i,n in enumerate(txt['vocabulary'])}
    summaries=[];contrasts=[]
    for bank,style in digital.TESTS:
        rr=json.loads((OUT/f'{bank}.json').read_text());ids=[r['image_id'] for r in rr]
        meta=json.loads((OUT/f'{bank}_{style}_features.json').read_text());assert sha(meta['path'])==meta['sha256']
        v=np.load(meta['path']);dest=RUN/'digital_retest'/f'{bank}_{style}';dest.mkdir(parents=True,exist_ok=True);dump(dest/'rows.json',rr)
        zz={}
        for name,a in mm.items():
            scores=np.einsum('bid,cd->bic',v,base if a is None else a['digital']);z=digital.metrics(scores,rr,idx)
            if name=='frozen':frozen=z
            bug=(frozen['blank']>0)&(frozen['conflict']<=0)
            repaired=bug&(z['conflict']>0);retained=repaired&(z['blank']>0)&(z['clean']>0)
            z.update(bug=bug,repaired=repaired,retained_repaired=retained)
            zz[name]=z;np.savez_compressed(dest/f'{name}.npz',scores=scores,**z)
            row=dict(bank=bank,style=style,model=name,sources=len(rr),decisions=2*len(rr),
                     **{k:100*float((z[k]>0).mean()) for k in ('clean','blank','congruent','conflict')},
                     clean_top1=100*float(z['top1'][:,0].mean()),conflict_top1=100*float(z['top1'][:,3:].mean()),
                     interaction_abs=float(abs(z['interaction']).mean()),interaction_p90=float(np.quantile(abs(z['interaction']),.9)),
                     bug_count=int(bug.sum()),repaired=int(repaired.sum()),retained_repaired=int(retained.sum()),
                     retained_repair_rate=100*float(retained.sum()/max(1,bug.sum())),
                     clean_regression=100*float(((frozen['clean']>0)&(z['clean']<=0)).sum()/max(1,(frozen['clean']>0).sum())))
            summaries.append(row);print('DIGITAL',json.dumps(row),flush=True)
        x=zz['IS']
        for other in ('ranking','same_scale_ablation','frozen','initial_prefix'):
            y=zz[other]
            for k in ('clean','blank','conflict'):
                contrasts.append(dict(bank=bank,style=style,other=other,metric=k,**digital.interval(100*((x[k]>0).mean(1)-(y[k]>0).mean(1)),np.ones(len(ids)),ids)))
            for k in ('clean_top1','conflict_top1'):
                xi=x['top1'][:,0] if k=='clean_top1' else x['top1'][:,3:].mean(1)
                yi=y['top1'][:,0] if k=='clean_top1' else y['top1'][:,3:].mean(1)
                contrasts.append(dict(bank=bank,style=style,other=other,metric=k,**digital.interval(100*(xi.astype(float)-yi.astype(float)),np.ones(len(ids)),ids)))
            contrasts.append(dict(bank=bank,style=style,other=other,metric='interaction_abs',**digital.interval(abs(x['interaction']).mean(1)-abs(y['interaction']).mean(1),np.ones(len(ids)),ids)))
            contrasts.append(dict(bank=bank,style=style,other=other,metric='retained_repaired',**digital.interval(100*(x['retained_repaired'].sum(1)-y['retained_repaired'].sum(1)),x['bug'].sum(1),ids)))
    pd.DataFrame(summaries).to_csv(RUN/'digital_summary.csv',index=False);pd.DataFrame(contrasts).to_csv(RUN/'digital_intervals.csv',index=False)

def sugar_score(mm):
    meta=json.loads((OUT/'sugarcrepe/cache.json').read_text());assert sha(meta['path'])==meta['sha256']
    df=pd.read_csv(sugar.REF);fi={x:i for i,x in enumerate(meta['files'])};ti={x:i for i,x in enumerate(meta['prompts'])}
    z=np.load(meta['path']);pos=[ti[x] for x in df.caption];neg=[ti[x] for x in df.negative_caption];ii=[fi[x] for x in df.filename]
    dest=RUN/'sugarcrepe';dest.mkdir(exist_ok=True);rows=[];cc={};contrasts=[]
    for name,a in mm.items():
        images=z['images'][ii];t=z['texts'] if a is None else a['sugar'];ps=np.einsum('nd,nd->n',images,t[pos]);ns=np.einsum('nd,nd->n',images,t[neg]);correct=ps>ns;cc[name]=correct
        table=df[['subset','example_id','filename','caption','negative_caption']].copy();table['positive_score']=ps;table['negative_score']=ns;table['correct']=correct;table.to_csv(dest/f'{name}.csv',index=False)
        for cat in ['full']+sorted(set(df.subset)):
            use=np.ones(len(df),bool) if cat=='full' else (df.subset==cat).to_numpy()
            rows.append(dict(model=name,category=cat,n=int(use.sum()),accuracy=100*float(correct[use].mean())))
    for other in ('ranking','same_scale_ablation','frozen','initial_prefix'):
        contrasts.append(dict(other=other,**digital.interval(100*(cc['IS'].astype(float)-cc[other].astype(float)),np.ones(len(df)),df.filename)))
    pd.DataFrame(rows).to_csv(dest/'summary.csv',index=False);pd.DataFrame(contrasts).to_csv(dest/'paired_intervals.csv',index=False)
    print('SUGAR',pd.DataFrame(rows).query('category == "full"').to_dict('records'),flush=True)

def external_score(mm):
    meta=json.loads((OUT/'external_retest/cache.json').read_text());cache=Path(meta['path'])
    for p,h in meta['files'].items():assert sha(p)==h,p
    rr=json.loads((OUT/'external_retest/rows.json').read_text());v=np.load(cache/'images.npy')
    dest=RUN/'external_retest';dest.mkdir(exist_ok=True);tables=[];summaries=[];contrasts=[]
    for bank in ('SCAM','NoSCAM','SynthSCAM','RTA100'):
        ix=np.array([i for i,r in enumerate(rr) if r['bank']==bank]);rows=[rr[i] for i in ix];tb='SCAM' if bank!='RTA100' else bank
        z=np.load(cache/f'{tb}_text.npz');labels=json.loads((cache/f'{tb}_labels.json').read_text());li={x:i for i,x in enumerate(labels)}
        yi=np.array([li[r['object_label']] for r in rows]);wi=np.array([li[r['attack_word']] for r in rows])
        for name,a in {**mm,'Defense_Prefix':None}.items():
            t=z['defense_prefix'] if name=='Defense_Prefix' else (z['base'] if a is None else a[tb]);scores=v[ix]@t.T
            obj=scores[np.arange(len(rows)),yi];word=scores[np.arange(len(rows)),wi];pred=scores.argmax(1);correct=obj>word;top1=pred==yi
            table=pd.DataFrame([dict(id=r['id'],source_id=r['source_id'],bank=bank,model=name,object_label=r['object_label'],attack_word=r['attack_word'],object_score=float(obj[j]),attack_score=float(word[j]),margin=float(obj[j]-word[j]),correct=bool(correct[j]),top1=bool(top1[j]),predicted_label=labels[pred[j]]) for j,r in enumerate(rows)])
            table.to_csv(dest/f'{bank}_{name}.csv',index=False);tables.append(table)
            row=dict(bank=bank,model=name,n=len(rows),accuracy=100*float(correct.mean()),top1=100*float(top1.mean()) if bank=='RTA100' else None)
            summaries.append(row);print('EXTERNAL',json.dumps(row),flush=True)
    full=pd.concat(tables,ignore_index=True)
    def frame(model,bank):return full[(full.model==model)&(full.bank==bank)].set_index('source_id').sort_index()
    for bank in ('SCAM','NoSCAM','SynthSCAM','RTA100'):
        x=frame('IS',bank)
        for other in ('ranking','same_scale_ablation','frozen','initial_prefix','Defense_Prefix'):
            y=frame(other,bank);assert x.index.equals(y.index)
            for k in (['correct','top1'] if bank=='RTA100' else ['correct']):
                contrasts.append(dict(bank=bank,other=other,metric=k,**digital.interval(100*(x[k].astype(float)-y[k].astype(float)),np.ones(len(x)),x.index)))
    paired=[]
    for bank in ('SCAM','SynthSCAM'):
        f=frame('frozen','NoSCAM');fa=frame('frozen',bank);assert f.index.equals(fa.index);bug=f.correct.to_numpy()&~fa.correct.to_numpy();zz={}
        for name in {**mm,'Defense_Prefix':None}:
            clean=frame(name,'NoSCAM');attack=frame(name,bank);assert clean.index.equals(attack.index) and clean.index.equals(f.index)
            repaired=bug&attack.correct.to_numpy();retained=repaired&clean.correct.to_numpy();drift=clean.margin.to_numpy()-attack.margin.to_numpy();zz[name]=(retained,drift)
            paired.append(dict(bank=bank,model=name,n=len(clean),frozen_bugs=int(bug.sum()),repaired=int(repaired.sum()),retained=int(retained.sum()),retained_repair_rate=100*float(retained.sum()/max(1,bug.sum())),mean_abs_word_removal_effect=float(abs(drift).mean())))
        for other in ('ranking','same_scale_ablation','frozen','initial_prefix','Defense_Prefix'):
            x,dx=zz['IS'];y,dy=zz[other]
            contrasts.append(dict(bank=bank,other=other,metric='retained_repair',**digital.interval(100*(x.astype(float)-y.astype(float)),bug.astype(float),f.index)))
            contrasts.append(dict(bank=bank,other=other,metric='absolute_word_removal_effect',**digital.interval(abs(dx)-abs(dy),np.ones(len(f)),f.index)))
    pd.DataFrame(summaries).to_csv(dest/'summary.csv',index=False);pd.DataFrame(paired).to_csv(dest/'paired_source_audit.csv',index=False);pd.DataFrame(contrasts).to_csv(dest/'intervals.csv',index=False)

def main():
    pilot.command_log();torch.set_num_threads(4);reg=freeze();assert not (RUN/'retest_complete.json').exists();mm=encode_text_cache(reg)
    digital_score(mm);sugar_score(mm);external_score(mm)
    dump(RUN/'retest_complete.json',dict(protocol_sha256=sha(RUN/'retest_protocol_v2.json'),files={str(p.relative_to(RUN)):sha(p) for p in [*RUN.glob('*.csv'),*(RUN/'digital_retest').glob('*/*.npz'),*(RUN/'sugarcrepe').glob('*.csv'),*(RUN/'external_retest').glob('*.csv')]}))

if __name__=='__main__':main()






