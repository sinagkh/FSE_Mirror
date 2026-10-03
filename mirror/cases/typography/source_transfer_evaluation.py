"""Plan49 A3: once-only fixed-checkpoint fresh typography retest and D5."""
import argparse
import csv
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.core.encoders import load_subject

sys.path.insert(0,str(ROOT/'mirror/cases/typography'))
import mirror.cases.typography.diagnose as data
import mirror.cases.typography.objective as prompt
import retest as metric

BANK=ROOT/'clip/fse_pre_writing_20260926/A3_typography'
MODELS=('openai_clip_b32','openai_clip_l14','openclip_laion_b32')
SEEDS=(42,43,44)

def csvwrite(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def registry(model):
    if model=='openai_clip_b32':
        p=read(BANK.parent/'A_protocol.json')['typography']['checkpoints']
        return [dict(name=k,seed=int(s),checkpoint=r['checkpoint'],sha256=r['sha256']) for s,group in p.items() for k,r in group.items()]
    root=BANK.parent/'B1_typography'/model;regs=read(root/'models.json')
    for s in SEEDS:
        p=root/f'initial_prefix_seed{s}.pt';regs.append(dict(name='initial_prefix',seed=s,checkpoint=str(p),sha256=sha(p)))
    return regs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,default=MODELS[0]);a=ap.parse_args()
    root=BANK/(a.model+'_v2');root.mkdir(exist_ok=False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    visual=read(BANK/'visual_review.json');assert visual['construction_accepted']
    for n in ('seen','heldout'):assert sha(BANK/(n+'.json'))==visual[n+'_sha256']
    assert sha(BANK/'gallery_selection.json')==visual['selection_sha256']
    regs=registry(a.model);verify_files({r['checkpoint']:r['sha256'] for r in regs})
    files=[Path(__file__),Path(__file__).with_name('prewriting_fresh_typography_evaluate.py'),BANK/'visual_review.json',BANK/'construction.json',BANK/'mechanical_validation.json',
        BANK.parent/'A_protocol.json',data.OUT/'protocol.json',data.OUT/'texts.pt',
        ROOT/'mirror/cases/typography/objective.py',ROOT/'mirror/cases/typography/retest.py',
        *[BANK/(n+'.json') for n in ('seen','heldout')],*[Path(r['checkpoint']) for r in regs]]
    dump(root/'protocol.json',dict(inputs={str(p):sha(p) for p in files},model=a.model,models=regs,
        selection='fixed checkpoints before scoring',correction='v1 stopped during bootstrap on integer count division; v2 uses floating NaN output, identical models/data/metrics, reruns all arms and both banks.',draws=5000,bootstrap_rng_seed=490203,
        primary='IS minus ranking attack pairwise on fresh seen-class sources',
        missing_classes='Report counts for all original70 classes; do not claim full32/38-class empirical coverage.',
        frozen_bug='frozen blank margin>0 and frozen conflict margin<=0',
        retained_repair='frozen bug and adapted conflict, blank and clean margins>0',
        break_rate='adapted wrong conditional on frozen-correct conflicting-note decisions',
        intervals='Cross seeds with common source draw; decisions/5 states of same source kept together.',
        metrics='Pairwise attack,clean,blank;70-way attack/clean;mean absolute interaction;retained repair;attack break;occlusion and word-content flip counts.',
        no_tuning=True))
    with (root/'protocol.sha256').open('x') as f:f.write(sha(root/'protocol.json')+'\n')
    log(root,'start');scorer=load_subject(a.model,device='cuda');vocab=torch.load(data.OUT/'texts.pt',map_location='cpu',weights_only=False)['vocabulary']
    prompts=[s.format(n) for n in vocab for s in data.TEMPLATES];idx={n:i for i,n in enumerate(vocab)}
    texts={};model=scorer.model
    with torch.no_grad():
        z=model.encode_text(scorer.tokenizer(prompts).cuda());texts['frozen',0]=data.norm(data.norm(z).reshape(len(vocab),3,-1).mean(1)).cpu().numpy()
        tokens=prompt.tokens_with_prefix(scorer.tokenizer,prompts)
        for r in regs:
            prefix=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['prefix'].cuda()
            texts[r['name'],r['seed']]=prompt.prototype(model,tokens,prefix,len(vocab)).cpu().numpy()
    with (root/'text_features.npz').open('xb') as f:np.savez_compressed(f,**{n+'_seed'+str(s):t for (n,s),t in texts.items()})
    summaries=[];contrasts=[];counts=[]
    for bank in ('seen','heldout'):
        rr=read(BANK/(bank+'.json'));assert len({r['image_id'] for r in rr})==len(rr)
        for r in rr:assert sha(data.ROOT/'clip/data/coco/train2017'/r['file'])==r['image_sha256']
        start=time.monotonic();vv=[]
        with torch.no_grad():
            for batch in DataLoader(data.Images(rr,scorer.preprocess,'standard'),batch_size=32,num_workers=8,pin_memory=True,worker_init_fn=data.worker_init):
                with torch.autocast('cuda',dtype=torch.float16):v=model.encode_image(batch.flatten(0,1).cuda(non_blocking=True))
                vv.append(data.norm(v.float()).reshape(len(batch),5,-1).cpu().numpy())
        v=np.concatenate(vv);cache=Path('/external-cache/fse_plan49_typography')/(a.model+'_v2');cache.mkdir(parents=True,exist_ok=True)
        target=cache/('fresh_'+bank+'.npy')
        with target.open('xb') as f:np.save(f,v)
        dump(root/(bank+'_features.json'),dict(path=str(target),sha256=sha(target),shape=list(v.shape),seconds=time.monotonic()-start))
        dest=root/bank;dest.mkdir();zz={};metrics={}
        for key,t in texts.items():
            name,seed=key;scores=np.einsum('bsd,cd->bsc',v,t,optimize=True);z=metric.metrics(scores,rr,idx);zz[key]=z
            with (dest/(name+'_seed'+str(seed)+'.npz')).open('xb') as f:np.savez_compressed(f,scores=scores,**z)
        f=zz['frozen',0];bug=(f['blank']>0)&(f['conflict']<=0);basecorrect=f['conflict']>0
        for key,z in zz.items():
            repair=bug&(z['conflict']>0)&(z['blank']>0)&(z['clean']>0)
            br=basecorrect&(z['conflict']<=0)
            oc=(z['clean']>0)&(z['blank']<=0);word=(z['blank']>0)&(z['conflict']<=0)
            measurements={k:((z[k]>0).mean(1)*100,np.ones(len(rr))) for k in ('clean','blank','conflict')}
            measurements.update(clean_top1=(100*z['top1'][:,0].astype(float),np.ones(len(rr))),
                attack_top1=(100*z['top1'][:,3:].mean(1),np.ones(len(rr))),
                interaction_abs=(abs(z['interaction']).mean(1),np.ones(len(rr))),
                retained_repair=(100*repair.sum(1),bug.sum(1)),break_rate=(100*br.sum(1),basecorrect.sum(1)),
                occlusion_flip_rate=(100*oc.sum(1),np.full(len(rr),2)),word_flip_rate=(100*word.sum(1),np.full(len(rr),2)))
            metrics[key]=measurements
            counts.append(dict(bank=bank,name=key[0],seed=key[1],sources=len(rr),classes=len({r['label'] for r in rr}),
                decisions=2*len(rr),frozen_bugs=int(bug.sum()),frozen_correct_attacks=int(basecorrect.sum()),
                retained_repairs=int(repair.sum()),broken_attacks=int(br.sum()),occlusion_flips=int(oc.sum()),word_flips=int(word.sum())))
        rng=np.random.default_rng(490203);ii=rng.integers(len(rr),size=(5000,len(rr)));ss=rng.integers(3,size=(5000,3))
        store={}
        for name in ('frozen','initial_prefix','ranking','IS'):
            for m in metrics['frozen',0]:
                n=np.stack([metrics[name,0 if name=='frozen' else s][m][0] for s in SEEDS]);d=np.stack([metrics[name,0 if name=='frozen' else s][m][1] for s in SEEDS])
                values=n.sum(1)/d.sum(1);boot=[]
                for j in range(0,5000,100):
                    ix=ii[j:j+100];sd=ss[j:j+100];num=n[sd[:,:,None],ix[:,None,:]].sum(2);den=d[sd[:,:,None],ix[:,None,:]].sum(2)
                    boot.append(np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0).mean(1))
                boot=np.concatenate(boot);lo,hi=np.nanquantile(boot,[.025,.975]);store[name,m]=(values,boot)
                summaries.append(dict(bank=bank,name=name,metric=m,mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),
                    seed42=float(values[0]),seed43=float(values[1]),seed44=float(values[2]),ci95_low=float(lo),ci95_high=float(hi)))
        for other in ('ranking','frozen','initial_prefix'):
            for m in metrics['frozen',0]:
                x,xb=store['IS',m];y,yb=store[other,m];lo,hi=np.nanquantile(xb-yb,[.025,.975])
                contrasts.append(dict(bank=bank,comparison='IS - '+other,metric=m,mean=float((x-y).mean()),sample_sd=float((x-y).std(ddof=1)),ci95_low=float(lo),ci95_high=float(hi)))
        print('FRESH SCORED',a.model,bank,len(rr),flush=True)
    csvwrite(root/'summary.csv',summaries);csvwrite(root/'paired_contrasts.csv',contrasts);csvwrite(root/'counts.csv',counts)
    report=['# Fresh typography confirmation','',f'Model: {a.model}. Fixed3 seeds, no new checkpoint selection.','',
        'Counts reflect the outcome-independent historical exclusions; the bank is not balanced across all70 original labels. See construction.json for all absent and short classes.','',
        '| Bank | Frozen attack (%) | Ranking attack (%) | IS attack (%) | IS − ranking,95% CI |','|---|---:|---:|---:|---:|']
    for bank in ('seen','heldout'):
        rr=[next(r for r in summaries if r['bank']==bank and r['name']==n and r['metric']=='conflict') for n in ('frozen','ranking','IS')]
        c=next(r for r in contrasts if r['bank']==bank and r['comparison']=='IS - ranking' and r['metric']=='conflict')
        report.append('| '+bank+' | '+' | '.join(f"{r['mean']:.2f} ± {r['sample_sd']:.2f}" for r in rr)+f" | {c['mean']:.2f} [{c['ci95_low']:.2f}, {c['ci95_high']:.2f}] |")
    with (root/'REPORT.md').open('x') as f:f.write('\n'.join(report)+'\n')
    dump(root/'complete.json',dict(files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='commands.jsonl'},
        no_training=True,all_registered_arms=True,no_score_selection=True))
    log(root,'complete');print('\n'.join(report),flush=True)

if __name__=='__main__':main()

