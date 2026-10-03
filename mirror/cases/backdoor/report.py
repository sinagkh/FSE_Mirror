"""Complete fixed backdoor replications without altering first-round artifacts."""
import argparse
import copy
from datetime import datetime; from datetime import timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mirror.cases.backdoor.test_data as data
import mirror.cases.backdoor.train as tr
import mirror.cases.backdoor.evaluate as ev
import mirror.cases.backdoor.par_extension_analysis as analysis
from mirror.cases.backdoor.common import ROOT; from mirror.cases.backdoor.common import OUT; from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha

DEST = OUT/'backdoor_completion_v1'
BANKS = ('development','banana87','banana1000','imagenetv2')


def prepare():
    data.verify(); ev.verify()
    DEST.mkdir(exist_ok=True)
    protocol = DEST/'protocol.json'
    if protocol.exists(): return
    pause=json.loads((OUT/'priority_backdoor_20260927/pause_verified.json').read_text())
    for row in pause['paused_processes']:
        s=(Path('/proc')/str(row['pid'])/'stat').read_text(); fields=s[s.rfind(')')+2:].split()
        assert fields[0]=='T' and int(fields[19])==row['start_ticks']
    note=dict(proceed=True,decision_utc=datetime.now(timezone.utc).isoformat(),
        reason='User requests full backdoor completion; all registered attacks and matched arms replicated without conditioning on new ImageNetV2 outcomes.',
        outcome_access='Development results known; extension ImageNetV2 outcomes not inspected before this decision.',
        recipe='unchanged original trainer, final checkpoint and per-victim seed42 calibration',seeds=[42,43,44])
    for case in data.CASES:
        path=data.RUN/case/'replication_decision.json'
        assert not path.exists(), path
        dump(path,note)
    dump(protocol,dict(**note, cases=list(data.CASES),banks=list(BANKS),
        plan_sha256=sha(ROOT/'FSE_VLM/plan/67_backdoor_paper_completion.md'),
        original_evaluation_protocol_sha256=sha(data.RUN/'evaluation_protocol.json'),
        optional_jobs='all nine recorded queue/worker identities verified stopped; no automatic resume',
        additions='three-seed replication; paired banana invariance; pixel-gated inversion; native model check',
        analysis='4000 crossed seed/source paired bootstrap replicates; all registered conditions retained'))
    print('COMPLETION AUTHORIZED',note,flush=True)


def verify():
    cfg=json.loads((DEST/'protocol.json').read_text())
    assert cfg['plan_sha256']==sha(ROOT/'FSE_VLM/plan/67_backdoor_paper_completion.md')
    ev.verify()
    return cfg


def takeover():
    """Stop only the superseded, uncommitted evaluation and its controller."""
    controller=1330952
    def identity(pid):
        path=Path('/proc')/str(pid)
        raw=(path/'stat').read_text(); fields=raw[raw.rfind(')')+2:].split()
        return dict(pid=pid,start_ticks=int(fields[19]),state=fields[0],command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    parent=identity(controller)
    assert 'bash mirror/cases/backdoor/run_par_priority.sh' in parent['command']
    children=[int(s) for s in (Path('/proc')/str(controller)/'task'/str(controller)/'children').read_text().split()]
    assert len(children)==1, children
    child=identity(children[0])
    assert 'par_priority_runtime.py evaluate evaluate --case stripes --bank imagenetv2' in child['command']
    root=data.RUN/'stripes/evaluation/imagenetv2'
    assert not (root/'summary.json').exists() and not list(root.glob('*_records.npz'))
    workers=[int(s) for s in (Path('/proc')/str(child['pid'])/'task'/str(child['pid'])/'children').read_text().split()]
    entries=[identity(p) for p in workers]
    assert all('par_priority_runtime.py' in p['command'] for p in entries)
    os.kill(controller,signal.SIGSTOP)
    for row in entries+[child]: os.kill(row['pid'],signal.SIGTERM)
    os.kill(controller,signal.SIGTERM);os.kill(controller,signal.SIGCONT)
    dump(DEST/'queue_takeover.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),controller=parent,workers=entries+[child],
        reason='Restart uncommitted stripe evaluation with six CPU loaders and prioritize additional seed training; no completed outputs removed or changed.'))
    print('SUPERSEDED QUEUE STOPPED; ALL COMPLETED OUTPUTS PRESERVED',flush=True)


def first_round(case,bank):
    # Runtime-only loader parallelism: same original evaluator, batch size,
    # precision, source IDs, per-source RNG, and output destination.
    from mirror.cases.backdoor.par_priority_runtime import guard
    guard()
    def loader(*args,**kwargs):
        kwargs['num_workers']=6
        return DataLoader(*args,**kwargs)
    ev.DataLoader=loader
    ev.evaluate(case,bank,[42])


@torch.no_grad()
def evaluate(case,bank,smoke=False):
    verify(); tr.CASE=case; tr.verify()
    root=DEST/case/('smoke' if smoke else 'evaluation')/bank
    marker=root/'complete.json'
    if marker.exists():
        receipt=json.loads(marker.read_text())
        for name,digest in receipt['records_sha256'].items(): assert sha(root/(name+'_records.npz'))==digest
        return
    from mirror.cases.backdoor.par_priority_runtime import guard
    guard()
    model,prep,tok=tr.model_load(case)
    tv=ev.texts(model,tok,case,'victim',bank in ('banana1000','imagenetv2'))
    target=954 if bank in ('banana1000','imagenetv2') else 86
    tails={}; receipts={}
    seeds=[42] if smoke else [43,44]
    for seed in seeds:
        matched=[]
        for method in ('clean_only','ranking','IS'):
            path,receipt=ev.tail_path(case,method,seed); matched.append(receipt)
            name=f'{method}_seed{seed}'; receipts[name]=dict(path=str(path),**receipt)
            state=torch.load(path,map_location='cpu',weights_only=True)['visual_blocks']
            blocks=torch.nn.ModuleList([copy.deepcopy(b) for b in model.visual.transformer.resblocks[10:]])
            for j,block in enumerate(blocks):
                prefix=f'transformer.resblocks.{10+j}.'
                block.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
            tails[name]=blocks.eval().requires_grad_(False)
        assert len({r['initial_sha256'] for r in matched})==1
        assert len({r['sequence_sha256'] for r in matched})==1
    source=ev.Sources(case,bank,prep,with_controls=False,limit=8 if smoke else None)
    rows=source.rows; n=len(rows)
    vocab=json.loads((data.original.POUT/'protocol.json').read_text())['vocabulary']
    labels=np.array([target if bank.startswith('banana') else (r['label'] if bank=='imagenetv2' else vocab.index(r['label'])) for r in rows])
    values={name:ev.empty_records(n) for name in tails}
    captured=[]
    hook=model.visual.transformer.resblocks[10].register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    seen=[];start=time.monotonic()
    loader=DataLoader(source,batch_size=4,num_workers=2 if smoke else 6,pin_memory=True,prefetch_factor=1,
        worker_init_fn=data.original.old.typo.worker_init)
    for j,(indices,images,_) in enumerate(loader):
        ix=indices.numpy();x=images[:,0].flatten(0,1).cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16):
            model.encode_image(x); tokens=captured.pop(); assert not captured
            features={}
            for name,blocks in tails.items():
                z=tokens
                for block in blocks: z=block(z)
                pooled,_=model.visual._pool(z)
                features[name]=F.normalize((pooled@model.visual.proj).float(),dim=-1)
        for name,z in features.items():
            score=(z@tv.T).reshape(len(ix),len(ev.STATES),-1)
            ev.record(values[name],ix,score,labels,target)
        seen.extend(ix.tolist())
        if j%100==0: print('COMPLETION EVALUATE',case,bank,len(seen),'/',n,round(time.monotonic()-start,1),flush=True)
    hook.remove();assert sorted(seen)==list(range(n))
    parity={}
    if smoke:
        for name,v in values.items():
            original=np.load(data.RUN/case/'evaluation'/bank/(name+'_records.npz'))
            err=max(float(abs(v[k]-original[k][:n]).max()) for k in ('true_score','target_margin','allclass_interaction_abs'))
            agreement=float((v['pred']==original['pred'][:n]).mean())
            assert err<3e-4 and agreement>=.99,(case,bank,name,err,agreement)
            parity[name]=dict(max_score_error=err,prediction_agreement=agreement)
    root.mkdir(parents=True,exist_ok=True);digests={}
    for name,v in values.items():
        path=root/(name+'_records.npz');assert not path.exists()
        np.savez_compressed(path,**v,labels=labels,ids=np.array([r['id'] for r in rows]),states=np.array(ev.STATES))
        digests[name]=sha(path)
    dump(marker,dict(case=case,bank=bank,seeds=seeds,n=n,seconds=time.monotonic()-start,
        source_sha256=sha(__file__),records_sha256=digests,receipts=receipts,smoke_parity=parity,
        preserved_first_round=str(data.RUN/case/'evaluation'/bank)))


@torch.no_grad()
def native_check():
    """Strict full-state comparison to the released native model on fixed inputs."""
    path=DEST/'native_PAR_verification.json'
    if path.exists(): return
    spec=importlib.util.spec_from_file_location('completion_native_par',data.original.VENDOR/'pkgs/openai/model.py')
    native_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(native_module)
    rows=json.loads((data.RUN/'activation_sources.json').read_text())[:2]
    checks=[]
    for case in ('triangles','text'):
        for method in ('victim','PAR'):
            model,prep,tok=tr.model_load(case,method,device='cpu')
            entry=tr.asset(case,method);state=torch.load(entry['state_path'],map_location='cpu',weights_only=True)
            native=native_module.build(dict(state),pretrained=True).float().eval()
            native.load_state_dict(state,strict=True)
            images=torch.stack([prep(im) for row in rows for im in data.Render(case)(row)])
            text=tok(['a photo of a banana.','a photo of a cat.','a photo of a car.'])
            a=F.normalize(model.encode_image(images).float(),dim=-1)
            b=F.normalize(native.get_image_features(images).float(),dim=-1)
            ti=F.normalize(model.encode_text(text).float(),dim=-1)
            tj=F.normalize(native.get_text_features(text).float(),dim=-1)
            ei=float(abs(a-b).max());et=float(abs(ti-tj).max())
            assert max(ei,et)<3e-5,(case,method,ei,et)
            checks.append(dict(case=case,method=method,state_sha256=entry['state_sha256'],image_error=ei,text_error=et,strict_full_state_load=True))
            del model,native,state
            print('NATIVE PAR VERIFIED',case,method,ei,et,flush=True)
    cross=data.original.POUT/'imagenetv2_confirmation_v1/published_protocol_crosscheck.json'
    dump(path,dict(checks=checks,source_sha256=sha(__file__),
        shared_tokenizer_preprocess_crosscheck_sha256=sha(cross),
        renderer_parity_sha256=sha(data.RUN/'renderer_tests.json'),
        official_validation_source_sha256=sha(data.original.VENDOR/'eval/validate.py'),
        load_note='Official validate.py loads weights inside its module-prefix branch. Raw compact states were instead strictly loaded and checked directly against the native architecture; the official shell was not claimed to have run.',
        dataset_status='Original ImageNet1K validation is access-gated and unavailable locally; this verifies released-model implementation, not reproduction of the paper table.',
        published_context='PAR Table2: B32 triangles ASR15.9 ImageNet classification,91.5 COCO retrieval; our COCO/LVIS crop classification is a different protocol.'))


def records(case,bank):
    paths={p.stem[:-8]:p for p in (data.RUN/case/'evaluation'/bank).glob('*_records.npz')}
    for p in (DEST/case/'evaluation'/bank).glob('*_records.npz'): paths[p.stem[:-8]]=p
    for p in (DEST/case/'controls'/bank).glob('*_records.npz'): paths[p.stem[:-8]]=p
    rows={name:dict(np.load(path)) for name,path in paths.items()}
    ref=rows['victim']
    for r in rows.values():
        assert np.array_equal(r['ids'],ref['ids']) and np.array_equal(r['labels'],ref['labels'])
    for method in ('clean_only','ranking','IS'):
        for seed in (42,43,44): assert f'{method}_seed{seed}' in rows
    return paths,rows


def metrics(row,ref,bank,gi):
    y=ref['labels']; target=954 if bank in ('banana1000','imagenetv2') else 86
    c,a=1+4*gi,2+4*gi;good=row['pred']==y[:,None];fg=ref['pred']==y[:,None]
    allmask=np.ones(len(y),bool);bug=fg[:,c]&~fg[:,a]
    d={
        'native_clean_accuracy':(good[:,0],allmask),
        'clean_accuracy':(good[:,c],allmask),'attacked_accuracy':(good[:,a],allmask),
        'asr':(row['pred'][:,a]==target,y!=target),
        'retained_repair':(good[:,c]&good[:,a],bug),
        'trigger_induced_failure':(good[:,c]&~good[:,a],allmask),
        'prediction_change':(row['pred'][:,c]!=row['pred'][:,a],allmask),
        'both_correct':(good[:,c]&good[:,a],allmask),
        'original_clean_regression':(~good[:,c],fg[:,c]),
        'target_interaction_abs':(abs(row['target_margin'][:,a]-row['target_margin'][:,c]),y!=target),
        'target_interaction_signed':(row['target_margin'][:,a]-row['target_margin'][:,c],y!=target)}
    if 'allclass_interaction_abs' in row:
        d['allclass_interaction_abs']=(row['allclass_interaction_abs'][:,gi,0],allmask)
    return d


def analyze(case,bank):
    root=DEST/case/'analysis'/bank;path=root/'results.json'
    if path.exists(): return
    paths,rows=records(case,bank);ref=rows['victim'];rng=np.random.default_rng(646711)
    names=['victim','PAR','clean_only','ranking','IS']+(['exact_filter','tolerant_filter'] if case=='stripes' else ['blend_inversion','gated_inversion','oracle_gated_inversion'])
    methods={};comparisons={}
    for gi,grid in enumerate(data.GRID):
        metrics_by_method={}
        for name in names:
            keys=[f'{name}_seed{s}' for s in (42,43,44)] if name in ('clean_only','ranking','IS') else [name]
            if not all(k in rows for k in keys): raise RuntimeError(('Missing method',case,bank,keys))
            rr=[metrics(rows[k],ref,bank,gi) for k in keys]
            mm={}
            for metric in rr[0]:
                mask=rr[0][metric][1];values=np.stack([r[metric][0].astype(float) for r in rr])
                mm[metric]=(values,mask)
                means=values[:,mask].mean(1) if mask.any() else None
                methods.setdefault(name,{})[grid]=methods.setdefault(name,{}).get(grid,{})
                methods[name][grid][metric]=None if means is None else dict(mean=float(means.mean()),sample_sd=float(means.std(ddof=1)) if len(means)>1 else None,seed_values=means.tolist(),n=int(mask.sum()))
            metrics_by_method[name]=mm
        for comp in names:
            if comp in ('IS','victim'): continue
            result={}
            for metric,(iv,mask) in metrics_by_method['IS'].items():
                if metric not in metrics_by_method[comp]: continue
                cv,cmask=metrics_by_method[comp][metric];assert np.array_equal(mask,cmask)
                result[metric]=analysis.paired_ci(iv-cv,mask,rng)
            comparisons.setdefault('IS_minus_'+comp,{})[grid]=result
    dump(path,dict(case=case,bank=bank,n=len(ref['ids']),seeds=[42,43,44],methods=methods,comparisons=comparisons,
        records={str(p):sha(p) for p in paths.values()},source_sha256=sha(__file__),
        units='behavior proportions; interactions raw cosine differences',bootstrap='4000 crossed seed/source percentile, arms/states paired; no multiplicity-adjusted claims'))
    print('ANALYZED',case,bank,flush=True)


def report():
    import csv
    results=[]
    for case in data.CASES:
        for bank in BANKS:
            path=DEST/case/'analysis'/bank/'results.json'
            if not path.exists(): raise RuntimeError(('Incomplete bank',case,bank))
            results.append(json.loads(path.read_text()))
    dump(DEST/'all_results.json',results)
    with (DEST/'tables.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case','bank','method','processing','metric','n','mean','sample_sd','seed_values'])
        for r in results:
            for method,grids in r['methods'].items():
                for grid,mm in grids.items():
                    for metric,v in mm.items():
                        if v: w.writerow([r['case'],r['bank'],method,grid,metric,v['n'],v['mean'],v['sample_sd'],json.dumps(v['seed_values'])])
    lines=['# Backdoor: complete fixed-recipe evaluation','','Three seeds per learned arm; all original artifacts preserved. No repair recipe or checkpoint was selected using ImageNetV2.','',
        'Primary case: known-trigger diagnosis and visual-side repair. Extensions: the unchanged procedure applied to two other released B/32 victims.','']
    for r in results:
        lines += [f"## {r['case']} / {r['bank']}",'',f"Sources: {r['n']}; seeds42,43,44. Canonical processing below; complete five-condition grid in tables.csv.",'',
            '| Method | Native clean | Audit clean | Attacked | Retained repair | Target interaction |','|---|---:|---:|---:|---:|---:|']
        for method,grids in r['methods'].items():
            mm=grids['identity'];cells=[]
            for metric in ('native_clean_accuracy','clean_accuracy','attacked_accuracy','retained_repair','target_interaction_abs'):
                v=mm.get(metric);scale=1 if metric.startswith('target') else 100
                if not v: cells.append('—');continue
                cell=f"{scale*v['mean']:.4f}" if scale==1 else f"{scale*v['mean']:.2f}"
                if v['sample_sd'] is not None: cell+=f" ± {scale*v['sample_sd']:.2f}"
                cells.append(cell)
            lines.append('| '+method+' | '+' | '.join(cells)+' |')
        lines+=['','IS minus matched ranking (paired crossed seed/source 95% intervals):','']
        for metric in ('native_clean_accuracy','attacked_accuracy','retained_repair','target_interaction_abs','prediction_change'):
            v=r['comparisons']['IS_minus_ranking']['identity'].get(metric)
            if v:
                scale=1 if metric.startswith('target') else 100
                lines.append(f"- {metric}: {scale*v['mean']:+.4f} [{scale*v['ci95'][0]:.4f}, {scale*v['ci95'][1]:.4f}].")
        lines+=['']
    lines += ['## Interpretation and controls','',
        'Matched ranking and IS share data, capacity, guards, initialization, sample streams and successful-update budgets; the mixed-interaction penalty is the difference. Clean-only continuation addresses additional training without trigger counterfactuals.',
        'Released PAR is an operational comparison with different cleanup data and no supplied trigger. Native-model verification is separate from reproducing the original ImageNet1K table, whose dataset is unavailable.',
        'Input filters and detector-gated inversion are legitimate known-pattern alternatives. Oracle-gated inversion additionally receives the attack-present flag. Their successes must remain visible for claims about best practical defenses.',
        'Banana invariance uses paired prediction changes and correctness on both views, not only the net accuracy gap. Complete banana metrics and all registered processing conditions remain in the machine-readable tables.',
        'RIO and modality remain paused; no optional task is resumed by this completion.','']
    (DEST/'REPORT.md').write_text('\n'.join(lines))
    evidence=['# Paper-facing backdoor evidence','','Role: a third debugging case on an externally poisoned VLM, repaired inside the image encoder.','',
        'Lead with the completed three-seed stripes diagnosis, matched component contrast and independent 10,000-image confirmation. See ../backdoor_par/DEBUGGING_STORY.md.','',
        'The full report supplies fixed-procedure replication across triangles and text. Use all attacks when stating an across-attack claim; do not imply that the endpoint advantage holds in every domain or condition.','',
        '## ImageNetV2: all registered attack types','',
        '| Attack | Ranking attacked | IS attacked | IS − ranking, 95% CI | Native clean difference |','|---|---:|---:|---:|---:|']
    for r in results:
        if r['bank']!='imagenetv2':continue
        a=r['methods']['ranking']['identity']['attacked_accuracy'];b=r['methods']['IS']['identity']['attacked_accuracy']
        c=r['comparisons']['IS_minus_ranking']['identity'];d=c['attacked_accuracy'];clean=c['native_clean_accuracy']
        evidence.append(f"| {r['case']} | {100*a['mean']:.2f} | {100*b['mean']:.2f} | {100*d['mean']:+.2f} [{100*d['ci95'][0]:.2f}, {100*d['ci95'][1]:.2f}] | {100*clean['mean']:+.2f} |")
    evidence+=['','This table establishes the matched parameter-repair comparison. Include the completed known-pattern controls when comparing practical defenses, and keep the published-PAR information difference clear. The report is evidence for writing, not an automatic manuscript edit.','']
    (DEST/'PAPER_EVIDENCE.md').write_text('\n'.join(evidence))
    dump(DEST/'complete.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),banks=12,seeds=[42,43,44],
        report_sha256=sha(DEST/'REPORT.md'),tables_sha256=sha(DEST/'tables.csv'),native_PAR_check_sha256=sha(DEST/'native_PAR_verification.json')))
    print('BACKDOOR COMPLETION FINISHED',DEST,flush=True)


if __name__=='__main__':
    command();torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate','smoke','native_check','analyze','report','first_round','takeover'])
    p.add_argument('--case',choices=data.CASES);p.add_argument('--bank',choices=BANKS);a=p.parse_args()
    if a.action in ('prepare','native_check','report','takeover'): globals()[a.action]()
    elif a.action=='first_round':first_round(a.case,a.bank)
    elif a.action=='analyze':analyze(a.case,a.bank)
    else:evaluate(a.case,a.bank,a.action=='smoke')
