"""CPU-only complete-context re-audit of all current FSE L/14 routing models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.behavioral_pilot import CAL; from mirror.cases.color_binding.behavioral_pilot import OUT as PILOT; from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.cases.color_binding.routing_diagnostic import full_spec; from mirror.cases.color_binding.routing_diagnostic import REGIMES
from mirror.cases.color_binding.routing_adequacy import preference_spec; from mirror.cases.color_binding.routing_adequacy import adequacy
from mirror.core.specifications import compile_requirement

OUT = ROOT/'clip/interbind_adapter_reaudit_v2_20260923'
SOURCES = ROOT/'clip/interbind_source_quality_20260922/pilot/accepted_rows.jsonl'
OLD = ROOT/'clip/fse_constraint_completion_results_20260920/confirmation'
CURRENT = ROOT/'clip/fse_routing_retention_results_20260921/confirmation'
CACHE = ROOT/'clip/interbind_cache_v3_20260922/features/openclip_laion_l14/pilot/routing'
PLAN = ROOT/'FSE_VLM/plan/19_diagnosis_guided_repair_and_reaudit.md'
MODEL = 'openclip_laion_l14'
ARMS = ('frozen', 'ranking', 'full_is', 'no_cross')
SEEDS = (42, 43, 44)
SCORE_COLS = [f's_{i}_{j}' for i in range(4) for j in range(4)]


def registration():
    rows = [dict(arm='frozen', seed=0, checkpoint=None)]
    for m in read(OLD/'models.json'):
        if m['family']=='routing' and m['arm']=='ranking': rows.append({**m, 'folder':str(OLD), 'label':'ranking'})
    for m in read(CURRENT/'models.json'):
        rows.append({**m, 'arm':{'retained_is':'full_is', 'retained_no_cross':'no_cross'}[m['arm']],
                     'folder':str(CURRENT), 'label':m['arm']})
    assert len(rows)==10 and {(r['arm'],r['seed']) for r in rows[1:]} == {(a,s) for a in ARMS[1:] for s in SEEDS}
    return rows


def csv_path(bank, view, model):
    folder = OLD if model['arm']=='frozen' else Path(model['folder'])
    label = 'frozen' if model['arm']=='frozen' else model['label']
    return folder/f'audit/{bank}_{view}_{label}_seed{model["seed"]}.csv'


def freeze(out):
    models = registration()
    paths = [Path(__file__), SOURCES, PLAN, CAL, ROOT/'FSE_VLM/manuscript_v3/evidence_manifest.json', OLD/'models.json',
             CURRENT/'models.json', OLD/'training_protocol.json', CURRENT/'protocol.json',
             PILOT/'protocol.json', PILOT/'audit_rows.jsonl',
             ROOT/'clip/interbind_routing_diagnostic_20260923/protocol.json']
    paths += [Path(__file__).with_name(n+'.py') for n in ('routing_adapter_reaudit','routing_adequacy',
              'routing_diagnostic','routing_context_coverage','requirements','io')]
    paths += [ROOT/'mirror/utils/two_object_locality_lib.py', ROOT/'clip/fse_directional_pilots/train.py']
    checks = {}
    old_done=read(OLD/'audit/complete.json')['files']; current_done=read(CURRENT/'evaluation_complete.json')['audit_files']
    for m in models:
        if m['checkpoint']: checks[m['checkpoint']]=m['sha256']
        for bank in ('routing_seen','routing_heldout'):
            for view in ('direct','swapped'):
                p=csv_path(bank,view,m); expected=old_done if m['arm'] in ('frozen','ranking') else current_done
                checks[str(p)]=expected[p.name]
    for bank in ('routing_seen','routing_heldout'):
        paths += [OLD/f'audit_cache/{bank}/text.pt', OLD/f'audit_cache/{bank}/direct.npy', OLD/f'manifests/{bank}.jsonl']
    cached=read(CACHE/'complete.json')
    checks.update({str(CACHE/k):v for k,v in cached['files'].items()})
    checks.update({str(p):sha(p) for p in paths}); verify_files(checks)
    cal=read(CAL); spec,contexts=preference_spec(cal)
    out.mkdir(exist_ok=True,parents=True)
    with (out/'routing-context-preference-v3.yaml').open('x') as f: yaml.safe_dump(spec,f,sort_keys=False)
    dump(out/'models.json',models)
    dump(out/'protocol.json',dict(inputs=checks, models_sha256=sha(out/'models.json'),
        specification_sha256=sha(out/'routing-context-preference-v3.yaml'),
        subject=MODEL, historical_banks=['routing_seen','routing_heldout'], new_pilot_sources=49,
        scope='post-specified current FSE L/14 re-audit; not submitted B/32 replication',
        seeds=list(SEEDS), bootstrap=2000, statistics_seed=20260923,
        gpu=False, training=False, reserve=False, selection=False))
    dump(out/'adequacy.json',adequacy(cal,Path(__file__).with_name('requirement_library')/'routing-v1.yaml'))
    print('FROZEN',sha(out/'protocol.json'),flush=True)


def verify(out):
    p=read(out/'protocol.json'); verify_files(p['inputs'])
    assert sha(out/'models.json')==p['models_sha256']
    assert sha(out/'routing-context-preference-v3.yaml')==p['specification_sha256']
    return p


def adapted_text(text, model):
    x=torch.as_tensor(text).float()
    if model['checkpoint'] is None: return x
    ck=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
    state=ck['state_dict']; assert set(state)=={'A.weight','B.weight'}
    a,b=state['A.weight'].float(),state['B.weight'].float()
    assert a.shape==(64,768) and b.shape==(768,64)
    y=x+(x@a.T)@b.T  # alpha/rank=64/64; eval dropout is identity.
    y=y/(y.norm(dim=-1,keepdim=True)+1e-8)
    return y/(y.norm(dim=-1,keepdim=True)+1e-8)  # original forward's second unit_norm


def measurements(scores, cal):
    spec,contexts=preference_spec(cal); unit=cal['units'][MODEL]['unit']
    compiled=compile_requirement(spec,calibration_unit=unit,base_model_id=MODEL,calibration_bank_id='natural_calibration_20260922')
    x=np.asarray(scores,float); ev=compiled.evaluate(x); v=ev['contrasts']
    q0=x[:,1,1]-x[:,1,2]; q1=x[:,2,1]-x[:,2,2]
    e=(q0-q1)/2/unit; b=(q0+q1)/2/unit
    margins=np.stack((q0,-q1),1)/unit
    diag=np.diagonal(x,axis1=1,axis2=2)
    out={'response':e,'absolute_preference':abs(b),'preference':b,'surplus':e-abs(b),
         'exchange_accuracy':(margins>0).mean(1),'caption_accuracy':(x.argmax(2)==np.arange(4)).mean(1),
         'word1_accuracy':(diag>x[:,np.arange(4),np.arange(4)^2]).mean(1),
         'word2_accuracy':(diag>x[:,np.arange(4),np.arange(4)^1]).mean(1),
         'nonpositive_response':(e<=0).astype(float),
         'preference_dominated':((e>0)&(e<=abs(b))).astype(float),
         'both_exchange_correct':(e>abs(b)).astype(float),
         'preference_pass':(abs(b)<=cal['tau_fraction']).astype(float)}
    for name,val in v.items(): out['contrast/'+name]=val
    for label, choose in [('old',lambda r:r['legacy']),('added',lambda r:not r['legacy']),('all',lambda r:True)]:
        d=np.stack([v[r['contrast']] for r in contexts if choose(r) and r['kind']=='binding'],1)
        c=np.stack([v[r['contrast']] for r in contexts if choose(r) and r['kind']=='unwanted'],1)
        out[label+'_binding']=d.mean(1); out[label+'_cross_signed']=c.mean(1)
        out[label+'_cross_abs']=abs(c).mean(1)
        out[label+'_binding_nonpositive']=(d<=0).mean(1)
        positive=np.maximum(d,0).mean(1); unwanted=abs(c).mean(1)
        out[label+'_unwanted_share']=unwanted/(positive+unwanted+1e-12)
    for name,c in ev['clauses'].items(): out['clause/'+name]=np.asarray(c['passed'],float)
    return out


def score(out):
    verify(out); models=read(out/'models.json'); cal=read(CAL); torch.set_num_threads(2)
    all_rows=[]; source_records={}; max_old_error=0.; max_replay=0.
    def retain(bank,view,m,ids,sources,x):
        nonlocal all_rows
        met=measurements(x,cal)
        for j,anchor in enumerate(ids):
            source_records[(bank,anchor)] = sources[j]
            all_rows.append(dict(bank=bank,view=view,arm=m['arm'],seed=m['seed'],anchor_id=anchor,
                **{k:float(v[j]) for k,v in met.items()}, **{k:float(x[j].reshape(-1)[i]) for i,k in enumerate(SCORE_COLS)}))
    unit=cal['units'][MODEL]['unit']
    for bank in ('routing_seen','routing_heldout'):
        base_frame=pd.read_csv(csv_path(bank,'direct',models[0])); manifests=lines(OLD/f'manifests/{bank}.jsonl')
        tx=torch.load(OLD/f'audit_cache/{bank}/text.pt',map_location='cpu',weights_only=False)
        text_ids=[tx['groups'].index(tuple(r['objects'])) for r in manifests]
        ims=torch.as_tensor(np.array(np.load(OLD/f'audit_cache/{bank}/direct.npy',mmap_mode='r')[:16])).float()
        for view in ('direct','swapped'):
            for m in models:
                f=pd.read_csv(csv_path(bank,view,m))
                pd.testing.assert_frame_equal(f[['anchor_id','source_ids','objects']],base_frame[['anchor_id','source_ids','objects']])
                x=f[SCORE_COLS].to_numpy().reshape(-1,4,4); met=measurements(x,cal)
                for name in [k for k in f if k.startswith(('diag_','off_'))]:
                    val=met['contrast/'+name]*unit
                    if name.startswith('off_'): val=abs(val)
                    error=float(np.max(abs(val-f[name].to_numpy()))); max_old_error=max(max_old_error,error)
                    assert error<2e-7,(bank,view,m['arm'],name,error)
                np.testing.assert_allclose(met['caption_accuracy'],f.state_accuracy.to_numpy(),atol=1e-12)
                if view=='direct':
                    with torch.inference_mode():
                        texts=adapted_text(tx['base'],m)[text_ids[:16],:4]
                        replay=torch.einsum('bid,bjd->bij',ims,texts).numpy()
                    error=float(np.max(abs(replay-x[:16]))); max_replay=max(max_replay,error)
                    assert error<2e-6,(m['arm'],error)
                retain(bank,view,m,f.anchor_id.tolist(),[json.loads(s) for s in f.source_ids],x)
        print('REAUDIT historical',bank,flush=True)
    # Apply the same checkpoints to the different, already encoded pilot; no encoder load.
    index=lines(CACHE/'index.jsonl'); images=np.load(CACHE/'images.npy',mmap_mode='r'); texts=np.load(CACHE/'texts.npy')
    audits=[r for r in lines(PILOT/'audit_rows.jsonl') if r['model']==MODEL and r['family']=='routing'
            and r['method']=='blend90_luminance' and r['color_pair']=='red-blue']
    source_rows={r['anchor_id']:r for r in lines(SOURCES)}
    for row in audits: row['source_ids']=source_rows[row['anchor_id']]['source_ids']
    wanted={r['anchor_id']:r for r in audits}; index=[r for r in index if r['anchor_id'] in wanted]; assert len(index)==49
    frozen_regimes=[]
    for m in models:
        with torch.inference_mode(): adapted=adapted_text(texts,m).numpy()
        for view in ('canvas','swapped_canvas','in_situ'):
            xx=[]
            for r in index:
                start=r['image_offset']; ids=[i for i,n in enumerate(r['state_names']) if n.startswith('blend90_luminance/red-blue/'+view+'/')]
                assert len(ids)==4
                xx.append(images[start+np.asarray(ids)]@adapted[r['text_indices'][:4]].T)
            x=np.stack(xx)
            retain('new_pilot',view,m,[r['anchor_id'] for r in index],
                   [wanted[r['anchor_id']]['source_ids'] for r in index],x)
            if m['arm']=='frozen' and view=='canvas':
                met=measurements(x,cal)
                for j,r in enumerate(index):
                    regime=REGIMES[0] if met['response'][j]<=0 else REGIMES[1] if met['surplus'][j]<=0 else REGIMES[2]
                    frozen_regimes.append(dict(anchor_id=r['anchor_id'],source_ids=wanted[r['anchor_id']]['source_ids'],
                        model=MODEL,diagnosis_view='canvas',regime=regime,response=met['response'][j],preference=met['preference'][j]))
        print('REAUDIT new pilot',m['arm'],m['seed'],flush=True)
    # All repeated layouts/arms stay grouped; check source independence of different anchors.
    for bank in ('routing_seen','routing_heldout','new_pilot'):
        seen={}
        for (b,a),ids in source_records.items():
            if b!=bank: continue
            for source in ids:
                assert source not in seen or seen[source]==a,('cross-anchor source reuse',bank,source)
                seen[source]=a
    frame=pd.DataFrame(all_rows)
    with (out/'per_example.csv').open('x') as f: frame.to_csv(f,index=False)
    jsonl(out/'source_manifest.jsonl',[dict(bank=b,anchor_id=a,source_ids=ids) for (b,a),ids in source_records.items()])
    jsonl(out/'frozen_direct_regimes.jsonl',frozen_regimes)
    dump(out/'score_complete.json',dict(files={n:sha(out/n) for n in ('per_example.csv','source_manifest.jsonl','frozen_direct_regimes.jsonl')},
        old_contrast_max_error=max_old_error,checkpoint_cpu_replay_max_error=max_replay,
        rows=len(frame),historical_replay_lattices=320,checkpoints_verified=9,gpu=False,training=False,reserve=False))


def summary_interval(x, rng, n=2000):
    # x: seeds x unique source anchors x metrics. Fixed cohorts across seeds.
    s,a,k=x.shape; weights=rng.multinomial(a,np.full(a,1/a),size=n)/a
    seed_ids=rng.integers(s,size=(n,s)); draws=np.zeros((n,k))
    for j in range(s):
        for seed in range(s):
            ix=np.flatnonzero(seed_ids[:,j]==seed)
            draws[ix]+=weights[ix]@x[seed]/s
    paired=weights@x.mean(0)
    return np.quantile(paired,[.025,.975],axis=0).T,np.quantile(draws,[.025,.975],axis=0).T


def analyze(out):
    verify(out); complete=read(out/'score_complete.json'); verify_files({str(out/k):v for k,v in complete['files'].items()})
    frame=pd.read_csv(out/'per_example.csv'); metadata={'bank','view','arm','seed','anchor_id',*SCORE_COLS}
    metrics=[k for k in frame if k not in metadata]; summaries=[]; contrasts=[]; directions=[]
    for (bank,view),g in frame.groupby(['bank','view'],sort=True):
        arrays={}
        for arm in ARMS:
            h=g[g.arm==arm]; ids=sorted(h.anchor_id.unique()); seeds=sorted(h.seed.unique())
            x=np.stack([h[h.seed==s].set_index('anchor_id').loc[ids,metrics].to_numpy(float) for s in seeds]); arrays[arm]=x
            pc,cc=summary_interval(x,np.random.default_rng(20260923))
            for i,k in enumerate(metrics):
                per=x[:,:,i].mean(1)
                summaries.append(dict(bank=bank,view=view,arm=arm,metric=k,n_anchors=len(ids),seeds=seeds,
                    mean=float(per.mean()),sample_sd=float(per.std(ddof=1)) if len(seeds)>1 else None,per_seed=per.tolist(),
                    paired_ci95=pc[i].tolist(),crossed_ci95=cc[i].tolist()))
        for ref in ('frozen','ranking','no_cross'):
            diff=arrays['full_is']-arrays[ref]; pc,cc=summary_interval(diff,np.random.default_rng(20260923))
            for i,k in enumerate(metrics):
                contrasts.append(dict(bank=bank,view=view,contrast='full_is_minus_'+ref,metric=k,
                    mean=float(diff[:,:,i].mean()),per_seed=diff[:,:,i].mean(1).tolist(),
                    paired_ci95=pc[i].tolist(),crossed_ci95=cc[i].tolist()))
                desired=-1 if ('cross_abs' in k or 'unwanted_share' in k or k=='absolute_preference') else 1
                if k.startswith(('old_binding','added_binding','all_binding')) or k.endswith(('cross_abs','unwanted_share')) or k in ('response','surplus','absolute_preference'):
                    directions.append(dict(bank=bank,view=view,contrast='full_is_minus_'+ref,metric=k,
                        desired_sign=desired,nonworsening_per_seed=((desired*diff[:,:,i])>=-1e-10).mean(1).tolist()))
        print('SUMMARIZED',bank,view,flush=True)
    jsonl(out/'summaries.jsonl',summaries);jsonl(out/'contrasts.jsonl',contrasts);jsonl(out/'directions.jsonl',directions)
    with (out/'summary.csv').open('x') as f: pd.DataFrame(summaries).to_csv(f,index=False)
    wanted=['old_binding','old_cross_abs','added_binding','added_cross_abs','all_binding','all_cross_abs','response','absolute_preference','exchange_accuracy','caption_accuracy']
    report=['# Complete-context re-audit of current FSE L/14 adapters','',
      'All nine registered checkpoints plus frozen. Three technical seeds; frozen is one model. '
      'Historical banks and the new blend90 pilot are separate. No training/GPU/reserve scoring. '
      'All interaction quantities below are normalized by the unchanged L/14 base unit. '
      'Accuracy columns are individual decisions, not any-corner pass.','']
    for bank,view in sorted({(r['bank'],r['view']) for r in summaries}):
        report += [f'## {bank} / {view}','', '| Arm | '+' | '.join(wanted)+' |','|---|'+'---:|'*len(wanted)]
        for arm in ARMS:
            rr={r['metric']:r for r in summaries if r['bank']==bank and r['view']==view and r['arm']==arm}
            report.append('| '+arm+' | '+' | '.join(f"{rr[k]['mean']*(100 if 'accuracy' in k else 1):.4f}" for k in wanted)+' |')
        report += ['','Full IS changes: paired and crossed seed/anchor intervals in `contrasts.jsonl`.','']
    report += ['## Interpretation boundary','',
      'Old, omitted and complete-context effects must all be inspected. An old-context gain is not a complete-context gain. '
      'Conversely, incomplete original coverage alone does not prove optimization overfit. '
      'The new-pilot renderer, source composition and prompt pooling differ from historical training; '
      'that comparison is transfer, not exact replication. A separate findings note interprets these tables.',
      '', 'Every named contrast, clause, seed, direction fraction and source-level score is retained. '
      'Intervals are pointwise, source-paired/crossed, not a multiple-comparison-adjusted confirmation. '
      'The new beta=tau clause is post-specified; no checkpoints were trained or selected for it.']
    with (out/'REPORT.md').open('x') as f:f.write('\n'.join(report)+'\n')
    dump(out/'complete.json',dict(protocol_sha256=sha(out/'protocol.json'),
         files={n:sha(out/n) for n in ('summaries.jsonl','contrasts.jsonl','directions.jsonl','summary.csv','REPORT.md','score_complete.json')},
         gpu=False,training=False,reserve=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','score','analyze']);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();log(a.out,'start')
    try: {'freeze':freeze,'score':score,'analyze':analyze}[a.action](a.out)
    except BaseException as exc:log(a.out,'failed',error=repr(exc));raise
    log(a.out,'complete')


if __name__=='__main__':main()
