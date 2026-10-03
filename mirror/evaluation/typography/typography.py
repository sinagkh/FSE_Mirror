"""Complete fixed typography ports and score published defenses on the same audit."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from mirror.evaluation.typography.common import *
from mirror.core.encoders import load_subject
from mirror.cases.typography.source_transfer_evaluation import registry

sys.path.insert(0, str(ROOT/'mirror/cases/typography'))
import mirror.cases.typography.diagnose as data
import mirror.cases.typography.objective as prompt
import retest as metric
import mirror.cases.typography.external as public
import mirror.cases.typography.preservation as sugar


def encode_texts(subject, prompts, prefix=None, defense=False):
    values = []
    model = subject.model
    with torch.inference_mode():
        for start in range(0, len(prompts), 128):
            strings = prompts[start:start+128]
            if defense:
                token = subject.tokenizer(strings).cuda()
                star = int(subject.tokenizer(['*'])[0, 1])
                z = public.prefix_encode(model, token, star, prefix.reshape(1, -1))
            elif prefix is None:
                z = model.encode_text(subject.tokenizer(strings).cuda())
            else:
                token = prompt.tokens_with_prefix(subject.tokenizer, strings)
                z = prompt.encode_prefix(model, token, prefix)
            values.append(z.float().cpu())
    return torch.cat(values)


def text_bank(dest, subject, regs, label, strings, n, k=1, pool='norm', defense=False):
    target = dest/'features'/('text_'+label+'.npz')
    if target.exists():
        return dict(np.load(target))
    texts = {}
    for r in [dict(name='frozen', seed=0, checkpoint=None), *regs]:
        prefix = None if r['checkpoint'] is None else torch.load(r['checkpoint'], map_location='cpu', weights_only=False)['prefix'].cuda()
        raw = encode_texts(subject, strings, prefix)
        if pool == 'norm_mean_norm':
            t = data.norm(data.norm(raw).reshape(n, k, -1).mean(1))
        elif pool == 'mean_norm':
            t = data.norm(raw.reshape(n, k, -1).mean(1))
        else:
            t = data.norm(raw)
        texts[r['name']+'_seed'+str(r['seed'])] = t.numpy()
        print('TYPO TEXT', subject.subject['id'], label, r['name'], r['seed'], flush=True)
    if defense:
        cfg = read(data.OUT/'external_retest/protocol.json')
        assert sha(cfg['Defense_Prefix']['path']) == cfg['Defense_Prefix']['sha256']
        prefix = torch.load(cfg['Defense_Prefix']['path'], map_location='cuda', weights_only=True)
        vocab = torch.load(data.OUT/'texts.pt', map_location='cpu', weights_only=False)['vocabulary']
        dpstrings = [s.format('* '+name) for name in vocab for s in data.TEMPLATES]
        raw = encode_texts(subject, dpstrings, prefix, defense=True)
        texts['Defense_Prefix_seed0'] = data.norm(data.norm(raw).reshape(n, k, -1).mean(1)).numpy()
        texts['Dyslexify_seed0'] = texts['frozen_seed0']
    save_array(target, **texts)
    return texts


def image_bank(dest, subject, label, dataset, five=False):
    target = dest/'features'/('image_'+label+'.npz')
    if target.exists():
        return np.load(target)['images']
    vv = []
    with torch.inference_mode():
        for j, x in enumerate(DataLoader(dataset, batch_size=16 if five else 64,
                       num_workers=8, pin_memory=True, worker_init_fn=data.worker_init)):
            original_n = len(x)
            if five:
                x = x.flatten(0, 1)
            with torch.autocast('cuda', dtype=torch.float16):
                z = subject.model.encode_image(x.cuda(non_blocking=True))
            z = data.norm(z.float()).cpu().numpy()
            vv.append(z.reshape(original_n, 5, -1) if five else z)
            if j%25 == 0:
                print('TYPO IMAGES', subject.subject['id'], label, j, flush=True)
    result = np.concatenate(vv)
    save_array(target, images=result)
    return result


def packed(values):
    names = list(dict.fromkeys(key.rsplit('_seed', 1)[0] for key in values))
    return {name:np.stack([values[name+'_seed'+str(s)] for s in
             (SEEDS if name in ('ranking','IS','initial_prefix') else (0,))]) for name in names}


def digital_statistics(dest, bank, rows, scores, vocab):
    folder = dest/'digital'/bank
    if (folder/'summary.json').exists():
        return read(folder/'summary.json')
    idx = {v:i for i,v in enumerate(vocab)}
    zz = {k:metric.metrics(s, rows, idx) for k,s in scores.items()}
    f = zz['frozen_seed0']; bug = (f['blank']>0)&(f['conflict']<=0)
    correct = f['conflict']>0
    values, denominators, records = {}, {}, []
    names = ['attack_pairwise','clean_pairwise','blank_pairwise','attack_top1','clean_top1',
             'interaction_abs','retained_repair','break_rate','occlusion_flip_rate','word_flip_rate']
    for key,z in zz.items():
        retained = bug & (z['conflict']>0) & (z['clean']>0) & (z['blank']>0)
        broken = correct & (z['conflict']<=0)
        occlusion = (z['clean']>0)&(z['blank']<=0)
        word = (z['blank']>0)&(z['conflict']<=0)
        arr = np.column_stack([100*(z['conflict']>0).mean(1),100*(z['clean']>0).mean(1),
              100*(z['blank']>0).mean(1),100*z['top1'][:,3:].mean(1),100*z['top1'][:,0],
              abs(z['interaction']).mean(1),100*retained.sum(1),100*broken.sum(1),
              100*occlusion.mean(1),100*word.mean(1)])
        den = np.ones_like(arr);den[:,6]=bug.sum(1);den[:,7]=correct.sum(1)
        values[key]=arr;denominators[key]=den
        save_array(folder/(key+'.npz'),scores=scores[key],**z,bug=bug,retained_repair=retained,broken=broken)
        records.extend(dict(model=key,source_id=r['image_id'],id=r['id'],label=r['label'],
             **dict(zip(names,arr[i].tolist())),repair_denominator=int(bug[i].sum()),
             break_denominator=int(correct[i].sum())) for i,r in enumerate(rows))
    ss = summaries(packed(values), names, [r['image_id'] for r in rows], dict(bank=bank), packed(denominators))
    jsonl(folder/'per_example.jsonl',records);dump(folder/'rows.json',rows);dump(folder/'summary.json',ss)
    return ss


def digital(dest, subject, regs, external=False):
    vocab = torch.load(data.OUT/'texts.pt',map_location='cpu',weights_only=False)['vocabulary']
    texts = text_bank(dest,subject,regs,'digital',[s.format(n) for n in vocab for s in data.TEMPLATES],len(vocab),3,'norm_mean_norm',defense=external)
    tests = [(a+'_'+b,read(data.OUT/(a+'.json')),b) for a,b in metric.TESTS]
    if external:
        fresh=ROOT/'clip/fse_pre_writing_20260926/A3_typography'
        tests += [('fresh_'+b,read(fresh/(b+'.json')),'standard') for b in ('seen','heldout')]
    cfg = read(ROOT/'data/typography/class_support/filtered_data_protocol.json')
    if not external:
        tests.append(('added_labels',read(cfg['sources']['test_new']['manifest']),'standard'))
    allsummary=[]
    # Published circuit is immutable. Base-image features always encoded without it.
    if external:
        from mirror.baselines import typography as dys
        selection=read(dys.DEST/'selection.json')
        dys.hook_model(subject.model)
    for bank,rows,style in tests:
        if (dest/'digital'/bank/'summary.json').exists():
            allsummary += read(dest/'digital'/bank/'summary.json');continue
        if external:
            dys.set_heads(subject.model,[])
        v=image_bank(dest,subject,bank,data.Images(rows,subject.preprocess,style),five=True)
        tt=texts;labels=vocab
        if bank=='added_labels':
            labels=torch.load(ROOT/'mirror/cases/typography_broad_support_20260926/filtered_texts.pt',map_location='cpu',weights_only=False)['vocabulary']
            tt=text_bank(dest,subject,regs,'added',[s.format(n) for n in labels for s in data.TEMPLATES],len(labels),3,'norm_mean_norm')
        scores={k:np.einsum('bsd,cd->bsc',v,t,optimize=True) for k,t in tt.items() if not k.startswith('Dyslexify')}
        if external:
            dys.set_heads(subject.model,selection['heads'])
            dv=image_bank(dest,subject,'Dyslexify_'+bank,data.Images(rows,subject.preprocess,style),five=True)
            scores['Dyslexify_seed0']=np.einsum('bsd,cd->bsc',dv,texts['frozen_seed0'],optimize=True)
        allsummary += digital_statistics(dest,bank,rows,scores,labels)
        print('TYPO DIGITAL DONE',subject.subject['id'],bank,flush=True)
    return allsummary


class NaturalImages(torch.utils.data.Dataset):
    def __init__(self, names, prep):self.names,self.prep=names,prep
    def __len__(self):return len(self.names)
    def __getitem__(self,i):
        with Image.open(ROOT/'clip/data/coco/val2017'/self.names[i]) as im:
            return self.prep(im.convert('RGB'))


def public_scores(dest,subject,regs):
    allsummary=[];rows=read(data.OUT/'external_retest/rows.json');cfg=read(data.OUT/'external_retest/protocol.json')
    v=image_bank(dest,subject,'public',public.ArchiveImages(rows,subject.preprocess))
    tables={}
    oldcache=Path(read(data.OUT/'external_retest/cache.json')['path'])
    for bank in ('SCAM','NoSCAM','SynthSCAM','RTA100'):
        folder=dest/'public'/bank
        if (folder/'summary.json').exists():
            allsummary+=read(folder/'summary.json');continue
        tb='RTA100' if bank=='RTA100' else 'SCAM';labels=read(oldcache/(tb+'_labels.json'))
        template=cfg['rta_templates'] if tb=='RTA100' else cfg['scam_templates']
        texts=text_bank(dest,subject,regs,tb,[s.format(n) for n in labels for s in template],len(labels),len(template),'mean_norm')
        ids=np.array([i for i,r in enumerate(rows) if r['bank']==bank]);rr=[rows[i] for i in ids]
        yi=np.array([labels.index(r['object_label']) for r in rr]);wi=np.array([labels.index(r['attack_word']) for r in rr])
        values={}
        for key,t in texts.items():
            score=v[ids]@t.T;obj=score[np.arange(len(rr)),yi];wrong=score[np.arange(len(rr)),wi]
            correct=obj>wrong;top1=score.argmax(1)==yi;values[key]=np.column_stack([100*correct,100*top1,obj-wrong])
            csvwrite(folder/(key+'.csv'),[dict(**r,object_score=float(obj[j]),attack_score=float(wrong[j]),
                     margin=float(obj[j]-wrong[j]),correct=bool(correct[j]),top1=bool(top1[j])) for j,r in enumerate(rr)])
        ss=summaries(packed(values),['pairwise','top1','margin'],[r['source_id'] for r in rr],dict(bank=bank,official_metric='top1' if bank=='RTA100' else 'pairwise'))
        dump(folder/'summary.json',ss);allsummary+=ss
        print('TYPO PUBLIC DONE',subject.subject['id'],bank,flush=True)
    folder=dest/'public'/'SugarCrepe'
    if (folder/'summary.json').exists():return allsummary+read(folder/'summary.json')
    meta=read(data.OUT/'sugarcrepe/cache.json');names=meta['files'];prompts=meta['prompts']
    vv=image_bank(dest,subject,'SugarCrepe',NaturalImages(names,subject.preprocess))
    texts=text_bank(dest,subject,regs,'SugarCrepe',prompts,len(prompts))
    frame=pd.read_csv(sugar.REF);vi={n:i for i,n in enumerate(names)};ti={n:i for i,n in enumerate(prompts)}
    image=vv[[vi[n] for n in frame.filename]];pos=[ti[n] for n in frame.caption];neg=[ti[n] for n in frame.negative_caption];values={}
    for key,t in texts.items():
        margin=np.einsum('nd,nd->n',image,t[pos]-t[neg]);values[key]=100*(margin>0)[:,None]
        records=frame[['subset','example_id','filename','caption','negative_caption']].copy()
        records['margin']=margin;records['correct']=margin>0
        csvwrite(folder/(key+'.csv'),records.to_dict('records'))
    ss=[]
    for category in ['full',*sorted(set(frame.subset))]:
        use=np.ones(len(frame),bool) if category=='full' else frame.subset.eq(category).to_numpy()
        ss+=summaries(packed({k:v[use] for k,v in values.items()}),['accuracy'],frame.filename[use],dict(bank='SugarCrepe',category=category))
    dump(folder/'summary.json',ss)
    return allsummary+ss


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['openai_clip_l14','openclip_laion_b32','openai_clip_b32'],required=True)
    args=ap.parse_args();setup();external=args.model=='openai_clip_b32'
    dest=OUT/('typography_external_diagnostic' if external else 'typography')/args.model
    if (dest/'complete.json').exists():return
    regs=registry(args.model);verify_files({r['checkpoint']:r['sha256'] for r in regs})
    inputs=[Path(__file__),ROOT/'mirror/cases/typography/diagnose.py',ROOT/'mirror/cases/typography/objective.py',
            data.OUT/'external_retest/protocol.json',data.OUT/'external_retest/rows.json',data.OUT/'sugarcrepe/cache.json',
            *[Path(r['checkpoint']) for r in regs]]
    if external:
        inputs += [ROOT/'clip/fse_published_comparisons_20260926/typography/selection.json',
                   ROOT/'mirror/baselines/typography.py']
    freeze(dest,inputs,model=args.model,models=regs,external_diagnostics=external,
        status='Fixed-checkpoint registered retest; no new training or outcome selection',
        precision='Digital images FP16 encoder, normalized FP32; text FP32; same original pooling and prefix placement',
        bootstrap='5000 source-cluster x paired-seed draws; published defenses each one checkpoint/circuit',
        digital=['original seen','original heldout','new font','new placement']+(['fresh seen','fresh heldout'] if external else ['added labels']),
        public=[] if external else ['full SCAM','full NoSCAM','full SynthSCAM','full RTA100 top1','full SugarCrepe and seven categories'])
    subject=load_subject(args.model,device='cuda')
    summary=digital(dest,subject,regs,external)
    if not external:summary+=public_scores(dest,subject,regs)
    csvwrite(dest/'summary.csv',summary)
    finish(dest,no_training=True,all_seeds=True,external_diagnostics=external)
    print('TYPOGRAPHY COMPLETE',args.model,flush=True)


if __name__=='__main__':main()
