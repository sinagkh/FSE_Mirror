"""Retest a development-selected strength; never writes original artifacts."""
from mirror.cases.typography.diagnose import *
import pandas as pd
import mirror.cases.typography.strength_pilot as pilot
import mirror.cases.typography.broad_duration_retest as legacy
import mirror.cases.typography.retest as digital
import mirror.cases.typography.external as public
import mirror.cases.typography.preservation as sugar

RUN = pilot.RUN
DEST = RUN / 'retest'


def development_report():
    sel=json.loads((RUN/'selection.json').read_text())
    lines=['# Typography suppression-strength pilot','',
           'Seed 42; same data, shared guards, prompt capacity, initialization, sample stream and 2,752-update budget. No manuscript or original-result changes.',
           '', '## Development-only strength decision', '',
           '| Multiplier | Bank | All-12 mean absolute interaction | Attack pair % | Attack full-class % | Clean full-class % | Retained repair % |',
           '|---|---|---:|---:|---:|---:|---:|']
    for m,r in sel['candidates'].items():
        for bank in ('original','added'):
            x=r[bank]
            lines.append(f"| {m} | {bank} | {x['all_targets_abs']:.5f} | {x['attack_accuracy']:.2f} | {x['attack_top1']:.2f} | {x['clean_top1']:.2f} | {x['retained_repair']:.2f} |")
    lines+=['',f"Selected multiplier: **{sel['selected_multiplier']}**. Eligibility and tie-breaking were frozen in STRENGTH_20260929_PROTOCOL.md before training. Selection uses only the two existing development banks.",
            '', 'The original development bank has 512 sources and 70 classification labels; the added-class development bank has 421 sources and 124 labels. They are not the paper test populations.',
            '', 'The current paper reports three-seed means; comparisons below, if present, pair the new seed-42 run with the original seed-42 checkpoints. Additional IS tuning is recorded, not described as identical historical search effort.']
    meta=json.loads((RUN/'runs/IS_w1/complete.json').read_text())
    lines+=['',f"1x replay maximum prefix difference: {meta['replay_max_prefix_error']:.3g}; maximum original development score difference: {meta['replay_max_score_error']:.3g}. Source-stream and initialization hashes match the original run."]
    if sel['selected_multiplier']==1:
        lines+=['','Neither stronger candidate met the predeclared joint repair/preservation criteria. No new public benchmark scores were used to choose a replacement.']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return sel


def text_banks(reg):
    caches=json.loads((pilot.ORIGINAL/'text_caches.json').read_text())
    mm={'frozen':None}
    for name,oldname in [('ranking','ranking'),('previous_IS','IS'),('same_scale_ablation','ranking'),('initial_prefix','initial_prefix')]:
        mm[name]={}
        for bank in ('digital','sugar','SCAM','RTA100'):
            meta=caches[f'{oldname}_{bank}']; assert sha(meta['path'])==meta['sha256']
            assert meta['checkpoint_sha256']==reg[name]['sha256']
            mm[name][bank]=np.load(meta['path'])
    model,tok=pilot.base.network()
    prefix=torch.load(reg['IS']['checkpoint'],map_location='cuda')['prefix']
    labels=torch.load(OUT/'texts.pt',map_location='cpu')['vocabulary']
    sm=json.loads((OUT/'sugarcrepe/cache.json').read_text());pc=public.verify_external()
    public_cache=Path(json.loads((OUT/'external_retest/cache.json').read_text())['path'])
    definitions={'digital':([s.format(n) for n in labels for s in TEMPLATES],len(labels),3,'norm_mean_norm'),
                 'sugar':(sm['prompts'],len(sm['prompts']),1,'norm')}
    for bank,templates in [('SCAM',pc['scam_templates']),('RTA100',pc['rta_templates'])]:
        labels=json.loads((public_cache/f'{bank}_labels.json').read_text())
        definitions[bank]=([s.format(n) for n in labels for s in templates],len(labels),len(templates),'mean_norm')
    mm['IS']={}; features=DEST/'features'; features.mkdir()
    metadata={}
    for bank,(prompts,n,k,pool) in definitions.items():
        raw=[]
        with torch.no_grad():
            for start in range(0,len(prompts),128):
                tt=pilot.base.tokens_with_prefix(tok,prompts[start:start+128])
                raw.append(pilot.base.encode_prefix(model,tt,prefix).cpu())
        raw=torch.cat(raw)
        if pool=='norm_mean_norm': t=norm(norm(raw).reshape(n,k,-1).mean(1))
        elif pool=='mean_norm': t=norm(raw.reshape(n,k,-1).mean(1))
        else:t=norm(raw)
        target=features/f'IS_{bank}.npy';np.save(target,t.numpy());mm['IS'][bank]=t.numpy()
        metadata[bank]=dict(path=str(target),sha256=sha(target),pool=pool,prompts=len(prompts))
    dump(DEST/'text_cache.json',metadata)
    del model;torch.cuda.empty_cache()
    return mm


def complete_targets(mm):
    labels=torch.load(OUT/'texts.pt',map_location='cpu')['vocabulary'];idx={n:i for i,n in enumerate(labels)}
    txt=torch.load(OUT/'texts.pt',map_location='cpu')['features'].numpy()
    all_rows=[]; contrasts=[]; target_rows=[]
    for folder in sorted((DEST/'digital_retest').iterdir()):
        rows=json.loads((folder/'rows.json').read_text());n=len(rows)
        yi=np.array([idx[r['label']] for r in rows]);wi=np.array([[idx[w] for w in r['words'][1:]] for r in rows]);ids=[r['image_id'] for r in rows]
        values={}
        for name in ('frozen','ranking','previous_IS','IS'):
            z=np.load(folder/f'{name}.npz');s=z['scores']
            m=s[np.arange(n)[:,None,None],np.arange(5)[None,:,None],yi[:,None,None]]-s[np.arange(n)[:,None,None],np.arange(5)[None,:,None],wi[:,None,:]]
            diff=np.stack([m[:,i]-m[:,j] for i,j in pilot.PAIRS],1)
            t=txt if mm[name] is None else mm[name]['digital']
            length=np.linalg.norm(t[yi,None,:]-t[wi],axis=-1)
            f=np.load(folder/'frozen.npz');bug=(f['blank']>0)&(f['conflict']<=0)
            retained=bug&(z['conflict']>0)&(z['clean']>0)&(z['blank']>0)
            measure=dict(all_targets_abs=(abs(diff).mean((1,2)),np.ones(n)),
                         directional_targets_abs=((abs(diff)/np.maximum(length[:,None,:],1e-8)).mean((1,2)),np.ones(n)),
                         conflict_pair_accuracy=(100*(z['conflict']>0).mean(1),np.ones(n)),
                         attack_top1=(100*z['top1'][:,3:].mean(1),np.ones(n)),
                         clean_top1=(100*z['top1'][:,0].astype(float),np.ones(n)),
                         retained_repair=(100*retained.sum(1),bug.sum(1)))
            values[name]=measure
            np.savez_compressed(folder/f'{name}_all12.npz',contrasts=diff,caption_difference_norm=length)
            for j,(a,b) in enumerate(pilot.PAIRS):
                for d in range(2):
                    target_rows.append(dict(bank=folder.name,model=name,state_a=a,state_b=b,distractor=d,
                                            mean_absolute=float(abs(diff[:,j,d]).mean()),
                                            rms=float(np.sqrt((diff[:,j,d]**2).mean()))))
            for metric,(num,den) in measure.items():
                all_rows.append(dict(bank=folder.name,model=name,metric=metric,mean=float(num.sum()/den.sum()),sources=n,denominator=int(den.sum())))
        for other in ('ranking','previous_IS','frozen'):
            for metric,(num,den) in values['IS'].items():
                other_num,other_den=values[other][metric];assert np.array_equal(den,other_den)
                contrasts.append(dict(bank=folder.name,other=other,metric=metric,**digital.interval(num-other_num,den,ids)))
    pd.DataFrame(all_rows).to_csv(DEST/'paper_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(DEST/'paper_metric_intervals.csv',index=False)
    pd.DataFrame(target_rows).to_csv(DEST/'per_target.csv',index=False)


def fresh_score(mm):
    fresh=ROOT/'clip/fse_pre_writing_20260926/A3_typography'
    labels=torch.load(OUT/'texts.pt',map_location='cpu')['vocabulary'];idx={n:i for i,n in enumerate(labels)}
    # Reuse exact frozen text features from the original fresh-bank evaluator.
    frozen_text=np.load(fresh/'openai_clip_b32_v2/text_features.npz')['frozen_seed0']
    for bank in ('seen','heldout'):
        meta=json.loads((fresh/'openai_clip_b32_v2'/f'{bank}_features.json').read_text())
        assert sha(meta['path'])==meta['sha256'];v=np.load(meta['path'])
        rows=json.loads((fresh/f'{bank}.json').read_text());folder=DEST/'digital_retest'/f'fresh_{bank}';folder.mkdir()
        dump(folder/'rows.json',rows)
        for name in ('frozen','ranking','previous_IS','IS'):
            text=frozen_text if mm[name] is None else mm[name]['digital']
            scores=np.einsum('bsd,cd->bsc',v,text,optimize=True)
            np.savez_compressed(folder/f'{name}.npz',scores=scores,**digital.metrics(scores,rows,idx))


def public_contrasts():
    """Supplement the reused evaluator with paired changes versus original IS."""
    rows=[]
    def frame(bank,name):
        return pd.read_csv(DEST/'external_retest'/f'{bank}_{name}.csv').set_index('source_id').sort_index()
    for bank in ('SCAM','SynthSCAM','NoSCAM','RTA100'):
        x=frame(bank,'IS')
        for other in ('ranking','previous_IS','frozen'):
            y=frame(bank,other);assert x.index.equals(y.index)
            key='top1' if bank=='RTA100' else 'correct'
            rows.append(dict(bank=bank,other=other,metric='accuracy',
                             **digital.interval(100*(x[key].astype(float)-y[key].astype(float)),np.ones(len(x)),x.index)))
            if bank in ('SCAM','SynthSCAM'):
                xc=frame('NoSCAM','IS');yc=frame('NoSCAM',other)
                assert x.index.equals(xc.index) and y.index.equals(yc.index)
                delta=abs(xc.margin-x.margin)-abs(yc.margin-y.margin)
                rows.append(dict(bank=bank,other=other,metric='absolute_word_removal_effect',
                                 **digital.interval(delta,np.ones(len(x)),x.index)))
    x=pd.read_csv(DEST/'sugarcrepe/IS.csv')
    for other in ('ranking','previous_IS','frozen'):
        y=pd.read_csv(DEST/'sugarcrepe'/f'{other}.csv')
        assert x[['filename','example_id','subset']].equals(y[['filename','example_id','subset']])
        rows.append(dict(bank='SugarCrepe',other=other,metric='accuracy',
                         **digital.interval(100*(x.correct.astype(float)-y.correct.astype(float)),np.ones(len(x)),x.filename)))
    pd.DataFrame(rows).to_csv(DEST/'public_contrasts.csv',index=False)


def extend_report(label='Selected'):
    rows=pd.read_csv(DEST/'paper_metrics.csv');cis=pd.read_csv(DEST/'paper_metric_intervals.csv')
    methods=['frozen','ranking','previous_IS','IS']
    lines=['','## Frozen-checkpoint test retest','',
           'Original tests and previously confirmed fresh-source banks are all retained. They are now developmental retests of the selected strength, not new untouched confirmation.',
           '', f'| Bank / endpoint | Frozen | Ranking seed 42 | Original IS seed 42 | {label} IS seed 42 | {label} minus original IS [95% CI] |',
           '|---|---:|---:|---:|---:|---|']
    for bank in rows.bank.unique():
        for metric in ('all_targets_abs','directional_targets_abs','conflict_pair_accuracy','attack_top1','retained_repair','clean_top1'):
            vals=[rows[(rows.bank==bank)&(rows.metric==metric)&(rows.model==m)].iloc[0]['mean'] for m in methods]
            c=cis[(cis.bank==bank)&(cis.metric==metric)&(cis.other=='previous_IS')].iloc[0]
            digits=5 if 'targets' in metric else 2
            lines.append('| '+bank+' / '+metric+' | '+' | '.join(f'{v:.{digits}f}' for v in vals)+f" | {c['mean']:+.{digits}f} [{c.ci_low:+.{digits}f}, {c.ci_high:+.{digits}f}] |")
    ext=pd.read_csv(DEST/'external_retest/summary.csv');sug=pd.read_csv(DEST/'sugarcrepe/summary.csv')
    lines+=['','## Public tests and preservation','',f'| Test | Frozen | Ranking seed 42 | Original IS seed 42 | {label} IS seed 42 |','|---|---:|---:|---:|---:|']
    for bank in ('SCAM','SynthSCAM','NoSCAM','RTA100'):
        key='top1' if bank=='RTA100' else 'accuracy'
        vals=[ext[(ext.bank==bank)&(ext.model==m)].iloc[0][key] for m in methods]
        lines.append('| '+bank+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    vals=[sug[(sug.category=='full')&(sug.model==m)].iloc[0].accuracy for m in methods]
    lines.append('| Full SugarCrepe | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    dependency=pd.read_csv(DEST/'external_retest/paired_source_audit.csv')
    lines+=['',f'| Public word dependence | Frozen | Ranking seed 42 | Original IS seed 42 | {label} IS seed 42 |','|---|---:|---:|---:|---:|']
    for bank in ('SCAM','SynthSCAM'):
        vals=[dependency[(dependency.bank==bank)&(dependency.model==m)].iloc[0].mean_abs_word_removal_effect for m in methods]
        lines.append('| '+bank+' mean absolute effect | '+' | '.join(f'{v:.5f}' for v in vals)+' |')
        lines.append('| '+bank+' influence removed % | '+' | '.join(f'{100*(1-v/vals[0]):.2f}' for v in vals)+' |')
    extra=pd.read_csv(DEST/'public_contrasts.csv')
    lines+=['',f'| Public endpoint | Comparator | {label} minus comparator [95% CI] |','|---|---|---|']
    for row in extra.itertuples():
        if row.other not in ('ranking','previous_IS'):continue
        digits=5 if row.metric=='absolute_word_removal_effect' else 2
        lines.append(f'| {row.bank} / {row.metric} | {row.other} | {row.mean:+.{digits}f} [{row.ci_low:+.{digits}f}, {row.ci_high:+.{digits}f}] |')
    lines+=['','5000 paired source-cluster bootstrap draws, conditional on seed 42. All states and distractor decisions from an image stay together. These intervals are not seed-replication intervals. No checkpoint selection or manuscript updates followed the test results.']
    with (RUN/'REPORT.md').open('a') as f:f.write('\n'.join(lines)+'\n')


def main():
    pilot.command();pilot.verify();torch.set_num_threads(2)
    assert sha(RUN/'selection.json')==(RUN/'selection.sha256').read_text().strip()
    sel=development_report()
    if sel['selected_multiplier']==1:
        dump(RUN/'complete.json',dict(selected_multiplier=1,new_test_scoring=False,
                                     reason='No stronger arm met the frozen development criteria',
                                     report_sha256=sha(RUN/'REPORT.md')))
        return
    DEST.mkdir(exist_ok=False);legacy.RUN=DEST
    oldreg=json.loads((pilot.ORIGINAL/'selection.json').read_text())
    reg={'ranking':oldreg['ranking'],'previous_IS':oldreg['IS'],
         'same_scale_ablation':oldreg['ranking'],'initial_prefix':oldreg['initial_prefix'],
         'IS':sel['candidates'][str(sel['selected_multiplier'])]}
    for r in reg.values():assert sha(r['checkpoint'])==r['sha256']
    dump(DEST/'protocol.json',dict(selection_sha256=sha(RUN/'selection.json'),registry=reg,
         code={str(p):sha(p) for p in [Path(__file__),Path(legacy.__file__),Path(digital.__file__)]},
         extra_retests='Same existing fresh seen/heldout sources; frozen before their new scores, no selection',
         intervals='5000 paired source-cluster draws; one seed',no_test_selection=True))
    mm=text_banks(reg)
    legacy.digital_score(mm);fresh_score(mm);complete_targets(mm)
    legacy.sugar_score(mm);legacy.external_score(mm);public_contrasts()
    extend_report()
    dump(RUN/'complete.json',dict(selected_multiplier=sel['selected_multiplier'],
                                 report_sha256=sha(RUN/'REPORT.md'),
                                 tables={str(p.relative_to(RUN)):sha(p) for p in DEST.rglob('*.csv')}))
    print('COMPLETE',RUN/'REPORT.md',flush=True)

if __name__=='__main__':main()
