"""Full fixed-checkpoint retest, reusing immutable visual feature caches."""
import argparse
import itertools
from pathlib import Path
import numpy as np
import torch
from mirror.cases.typography.backbones import OUT; from mirror.cases.typography.backbones import SEEDS; from mirror.cases.typography.backbones import base
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.evaluation.typography import typography as ev
from mirror.evaluation.typography.common import summaries; from mirror.evaluation.typography.common import csvwrite; from mirror.evaluation.typography.common import save_array; from mirror.evaluation.typography.common import finish

OLD = ROOT/'clip/fse_two_case_completion_20260927/typography'
FRESH = ROOT/'clip/fse_pre_writing_20260926/A3_typography'


def complete_targets(dest):
    vocab=torch.load(ev.data.OUT/'texts.pt',map_location='cpu',weights_only=False)['vocabulary']
    pairs=list(itertools.combinations(range(1,5),2))
    results=[]
    for folder in sorted((dest/'digital').iterdir()):
        rows=read(folder/'rows.json')
        labels=vocab
        bank='digital'
        if folder.name=='added_labels':
            labels=torch.load(ROOT/'mirror/cases/typography_broad_support_20260926/filtered_texts.pt',map_location='cpu',weights_only=False)['vocabulary']
            bank='added'
        tt=dict(np.load(dest/'features'/('text_'+bank+'.npz')))
        yi=np.array([labels.index(r['label']) for r in rows])
        wi=np.array([[labels.index(w) for w in r['words'][1:]] for r in rows])
        arrays={}
        raw={}
        for file in sorted(folder.glob('*.npz')):
            key=file.stem
            s=np.load(file)['scores']
            m=s[np.arange(len(rows))[:,None,None],np.arange(5)[None,:,None],yi[:,None,None]]-s[np.arange(len(rows))[:,None,None],np.arange(5)[None,:,None],wi[:,None,:]]
            diff=np.stack([m[:,i]-m[:,j] for i,j in pairs],1)
            length=np.linalg.norm(tt[key][yi,None,:]-tt[key][wi],axis=-1)
            arrays[key]=np.column_stack([abs(diff).mean((1,2)),np.sqrt((diff**2).mean((1,2))),
                                        (abs(diff)/np.maximum(length[:,None,:],1e-8)).mean((1,2))])
            raw[key]=diff
        results += summaries(ev.packed(arrays),['all_targets_abs','per_source_target_rms','directional_targets_abs'],
                             [r['image_id'] for r in rows],dict(bank=folder.name))
        save_array(dest/'all_targets'/(folder.name+'.npz'),**raw)
    csvwrite(dest/'all_targets/summary.csv',results)
    return results


def run(model):
    dest=OUT/model/'evaluation'
    if (dest/'complete.json').exists():
        return
    assert (OUT/model/'training_complete.json').exists()
    regs=read(OUT/model/'models.json')
    verify_files({r['checkpoint']:r['sha256'] for r in regs})
    old=OLD/model
    frozen_files=read(old/'complete.json')['files']
    inputs=[Path(__file__),Path(ev.__file__),OUT/model/'models.json',OUT/model/'protocol.json',old/'complete.json',
            ev.data.OUT/'external_retest/protocol.json',ev.data.OUT/'external_retest/rows.json',
            ev.data.OUT/'sugarcrepe/cache.json',FRESH/'seen.json',FRESH/'heldout.json']
    dump(dest/'protocol.json',dict(inputs={str(p):sha(p) for p in inputs},models=regs,
         evaluation_selection=False,renderer_and_preprocessing='unchanged, hashed prior image caches',
         prefix_text_encoding='FP32, same pooling and templates',
         bootstrap='5000 paired source-cluster and crossed seed/source draws',
         audit='all twelve targeted finite differences; raw and caption-distance-normalized',
         seeds=SEEDS))
    log(dest,'start')
    ev.setup()
    subject=ev.load_subject(model,device='cuda')
    subject.model.visual.cpu()
    torch.cuda.empty_cache()
    used={}

    def images(_dest,_subject,label,_dataset,five=False):
        p=old/'features'/('image_'+label+'.npz')
        expected=frozen_files[str(p.relative_to(old))]
        assert sha(p)==expected,p
        used[str(p)]=expected
        return np.load(p)['images']

    def texts(_dest,_subject,_regs,label,strings,n,k=1,pool='norm',defense=False):
        assert not defense
        target=dest/'features'/('text_'+label+'.npz')
        if target.exists():
            return dict(np.load(target))
        p=old/'features'/('text_'+label+'.npz')
        expected=frozen_files[str(p.relative_to(old))]
        assert sha(p)==expected,p
        used[str(p)]=expected
        prev=np.load(p)
        result={key:prev[key] for key in ['frozen_seed0',*[f'ranking_seed{s}' for s in SEEDS]]}
        for r in regs:
            if r['name']!='IS':continue
            prefix=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['prefix'].cuda()
            raw=ev.encode_texts(subject,strings,prefix)
            if pool=='norm_mean_norm':
                t=ev.data.norm(ev.data.norm(raw).reshape(n,k,-1).mean(1))
            elif pool=='mean_norm':
                t=ev.data.norm(raw.reshape(n,k,-1).mean(1))
            else:t=ev.data.norm(raw)
            result[f'IS_seed{r["seed"]}']=t.numpy()
            print('TEXT',model,label,r['seed'],flush=True)
        save_array(target,**result)
        return result

    ev.image_bank=images
    ev.text_bank=texts
    stats=ev.digital(dest,subject,regs,False)
    # Fresh-source confirmation banks: identical to their saved, score-blind definitions.
    vocab=torch.load(ev.data.OUT/'texts.pt',map_location='cpu',weights_only=False)['vocabulary']
    tt=dict(np.load(dest/'features/text_digital.npz'))
    for bank in ('seen','heldout'):
        meta=read(FRESH/(model+'_v2')/(bank+'_features.json'))
        p=Path(meta['path'])
        assert sha(p)==meta['sha256']
        used[str(p)]=meta['sha256']
        v=np.load(p)
        rows=read(FRESH/(bank+'.json'))
        scores={key:np.einsum('bsd,cd->bsc',v,t,optimize=True) for key,t in tt.items()}
        stats += ev.digital_statistics(dest,'fresh_'+bank,rows,scores,vocab)
        print('FRESH_DONE',model,bank,flush=True)
    stats += ev.public_scores(dest,subject,regs)
    csvwrite(dest/'summary.csv',stats)
    targets=complete_targets(dest)
    # Frozen and reused-ranking rows must reproduce the prior full evaluation.
    import pandas as pd
    oldrows=pd.read_csv(old/'summary.csv')
    checks=[]
    for r in stats:
        if r['comparison'] not in ('frozen','ranking') or r['bank'].startswith('fresh_'):continue
        match=oldrows[(oldrows.bank==r['bank']) & (oldrows.comparison==r['comparison']) & (oldrows.metric==r['metric'])]
        if r.get('category') is not None:match=match[match.category==r['category']]
        assert len(match)==1,(r,len(match))
        error=abs(float(match.iloc[0]['mean'])-r['mean'])
        assert error<1e-5,(r,error)
        checks.append(error)
    dump(dest/'baseline_parity.json',dict(rows=len(checks),maximum_mean_error=max(checks)))
    dump(dest/'cache_inputs.json',used)
    lines=['# Typography backbone replication','',f'Backbone: {model}. Three fixed seeds; identical data, initialization, sample order, guards, optimizer and update budget for ranking and IS.',
           '', '| Test / metric | Frozen | Ranking | IS | IS − ranking [95% CI] |',
           '|---|---:|---:|---:|---|']
    endpoints=[('test_seen_standard','attack_pairwise'),('test_seen_standard','attack_top1'),
               ('test_seen_standard','retained_repair'),('test_seen_standard','clean_top1'),
               ('test_seen_serif','attack_pairwise'),('test_seen_top','attack_pairwise'),
               ('test_heldout_standard','attack_pairwise'),('fresh_seen','attack_pairwise'),
               ('fresh_heldout','attack_pairwise'),('SCAM','pairwise'),('SynthSCAM','pairwise'),
               ('NoSCAM','pairwise'),('RTA100','top1'),('SugarCrepe','accuracy'),
               ('test_seen_standard','all_targets_abs'),('test_seen_standard','directional_targets_abs')]
    for bank,metric in endpoints:
        rows=[r for r in stats+targets if r['bank']==bank and r['metric']==metric and r.get('category','full')=='full']
        def one(name):return next(r for r in rows if r['comparison']==name)
        digits=5 if 'targets' in metric else 2
        vals=[]
        for name in ('frozen','ranking','IS'):
            r=one(name);val=f'{r["mean"]:.{digits}f}'
            if r['sample_sd'] is not None:val+=f' ± {r["sample_sd"]:.{digits}f}'
            vals.append(val)
        d=one('IS - ranking')
        lines.append('| '+bank+' / '+metric+' | '+' | '.join(vals)+f' | {d["mean"]:+.{digits}f} [{d["ci95_low"]:.{digits}f}, {d["ci95_high"]:.{digits}f}] |')
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    finish(dest,all_seeds=True,no_checkpoint_selection=True,baseline_reproduction_checked=True)
    print('EVALUATION_COMPLETE',model,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=base.MODELS,required=True)
    run(ap.parse_args().model)
