"""Dyslexify upstream-operator reproduction and complete public comparisons."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mirror.baselines.common import OUT; from mirror.baselines.common import ROOT; from mirror.baselines.common import read; from mirror.baselines.common import dump; from mirror.baselines.common import sha; from mirror.baselines.common import log

DEST = OUT / 'typography'
UPSTREAM = ROOT / 'FSE_VLM/external_dyslexify'
CODE = ROOT / 'mirror/cases/typography'
sys.path.insert(0, str(CODE))
import mirror.cases.typography.diagnose as data
import mirror.cases.typography.external as public

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)

def freeze():
    cfg = read(ROOT / 'data/typography/class_support/filtered_data_protocol.json')
    train = read(cfg['sources']['train']['manifest'])
    selected = sorted(train, key=lambda r: hashlib.sha256(('dyslexify-train-5342:' + r['id']).encode()).hexdigest())[:512]
    development = read(data.OUT / 'development.json')
    assert not ({r['image_id'] for r in selected} & {r['image_id'] for r in development})
    dump(DEST / 'ordering_sources.json', selected)
    dump(DEST / 'selection_sources.json', development)
    files = [Path(__file__), DEST / 'ordering_sources.json', DEST / 'selection_sources.json',
             UPSTREAM / 'dyslexify/cache/multi_head_attention.py', UPSTREAM / 'dyslexify/cache/hooks.py',
             UPSTREAM / 'experiments/greedy_selection/greedy_selection.py',
             data.OUT / 'external_retest/protocol.json', data.OUT / 'external_retest/rows.json']
    dump(DEST / 'protocol.json', dict(files={str(p): sha(p) for p in files},
        upstream_revision=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
        method='Dyslexify, data/backbone-adapted reproduction; not released weights',
        backbone='OpenAI ViT-B/32, unchanged frozen text encoder',
        ordering='Upstream ratio of mean CLS attention to bottom two patch rows versus all spatial patches; conflict_1 on512 existing training sources.',
        selection='Upstream greedy algorithm on512 existing development sources; full86 training-label recognition on clean and conflict_1; eps=.001, stop_at_delta=.01, upstream >10 no-improvement stopping rule.',
        code_paper_difference='Upstream accepts an improving head before checking whether clean accuracy fell by >1pp, so its final head can cross that bound. Preserve and report this behavior, do not choose another circuit on public outcomes.',
        comparison='Official SCAM32-template raw-embedding mean then normalization; official RTA100 single template/all100classes; same images/captions for all methods.',
        additional_information='Internal attention and known note location; no public benchmark data for circuit selection',
        released_checkpoint_search='GitHub tree/releases and HF model-name search returned no circuit/checkpoint; configured smallest backbone is LAION B16.',
        frozen_weight_sha256=data.WEIGHT_SHA, training_classes=cfg['training_classes']))
    log(DEST, 'protocol_frozen')

def verify():
    cfg=read(DEST/'protocol.json')
    for p,h in cfg['files'].items():
        assert sha(p)==h,p
    return cfg

def hook_model(model):
    upstream = module(UPSTREAM/'dyslexify/cache/multi_head_attention.py','upstream_dyslexify_attention')
    for block in model.visual.transformer.resblocks:
        old=block.attn
        new=upstream.MultiheadAttentionWithWeightHook(old.embed_dim, old.num_heads, dropout=old.dropout,
              bias=old.in_proj_bias is not None, batch_first=old.batch_first).to(old.in_proj_weight.device)
        new.load_state_dict(old.state_dict(),strict=True)
        new.eval().requires_grad_(False)
        block.attn=new
    return model

def set_heads(model, heads):
    hook=module(UPSTREAM/'dyslexify/cache/hooks.py','upstream_dyslexify_hooks').create_zero_cls_attention_result_hook
    for block in model.visual.transformer.resblocks:
        block.attn.remove_all_hooks()
    for layer,head in heads:
        model.visual.transformer.resblocks[layer].attn.register_attn_result_hook(hook(head))

def pixels(rows, preprocess, states):
    result=[]
    for batch in DataLoader(data.Images(rows,preprocess,'standard'),batch_size=32,num_workers=8,
                            pin_memory=True,worker_init_fn=data.worker_init):
        result.append(batch[:,states])
    return torch.cat(result).cuda()

@torch.inference_mode()
def encoded(model, x, batch=128):
    result=[]
    for i in range(0,len(x),batch):
        with torch.autocast('cuda',dtype=torch.float16):
            z=model.encode_image(x[i:i+batch])
        result.append(data.norm(z.float()))
    return torch.cat(result)

def select():
    setup();cfg=verify();log(DEST,'select_start')
    model,preprocess,tok=data.load_model()
    tr=read(DEST/'ordering_sources.json');dev=read(DEST/'selection_sources.json')
    train_x=pixels(tr,preprocess,[3]).flatten(0,1)
    dev_x=pixels(dev,preprocess,[0,3])
    with torch.inference_mode():
        reference=model.encode_image(train_x[:4])
    hook_model(model)
    with torch.inference_mode():
        changed=model.encode_image(train_x[:4])
    error=float((reference-changed).abs().max())
    assert torch.allclose(reference,changed,atol=3e-5,rtol=3e-5),error
    sums=[torch.zeros(12,50,device='cuda') for _ in model.visual.transformer.resblocks]
    for layer,block in enumerate(model.visual.transformer.resblocks):
        def capture(pattern,q,k,v,layer=layer):
            sums[layer].add_(pattern[:,:,0,:].float().sum(0))
            return pattern
        block.attn.register_attention_pattern_hook(capture)
    encoded(model,train_x)
    patterns=torch.stack(sums)/len(train_x)
    spatial=patterns[:,:,1:].reshape(12,12,7,7)
    attention=spatial[:,:,-2:,:].sum((-1,-2))/spatial.sum((-1,-2))
    order=sorted([(float(attention[l,h]),l,h) for l in range(12) for h in range(12)],reverse=True)
    set_heads(model,[])
    labels=cfg['training_classes'];yi=torch.tensor([labels.index(r['label']) for r in dev],device='cuda')
    with torch.inference_mode():
        t=model.encode_text(tok([f'a photo of a {label}.' for label in labels]).cuda())
        t=data.norm(t.float())
    def evaluate():
        v=encoded(model,dev_x.flatten(0,1)).reshape(len(dev),2,-1)
        pred=(v@t.T).argmax(-1)
        return tuple(float((pred[:,j]==yi).float().mean()) for j in range(2))
    base=evaluate();current=base;heads=[];records=[];under=0;start=time.monotonic()
    dump(DEST/'operator_checks.json',dict(no_hook_max_error=error,upstream_attention_used_verbatim=True,
        attention_scores=attention.cpu().tolist(),baseline_development=base))
    for attention_score,layer,head in order:
        set_heads(model,heads+[(layer,head)])
        acc,attack=evaluate();skipped=attack-current[1]<.001
        row=dict(layer=layer,head=head,attention=attention_score,clean=acc,attack=attack,skipped=skipped)
        records.append(row)
        print('DYSLEXIFY',len(records),json.dumps(row),'seconds',round(time.monotonic()-start),flush=True)
        if under>10:break
        if skipped:
            under+=1
            continue
        under=0;current=(acc,attack);heads.append((layer,head))
        if base[0]-acc>.01:break
    set_heads(model,heads)
    final=evaluate()
    dump(DEST/'selection.json',dict(heads=heads,trace=records,baseline=base,selected_development=final,
        clean_drop_points=100*(base[0]-final[0]),seconds=time.monotonic()-start,protocol_sha256=sha(DEST/'protocol.json')))
    log(DEST,'select_complete',heads=heads)

def score():
    setup();verify();chosen=read(DEST/'selection.json');log(DEST,'public_score_start')
    model,prep,tok=data.load_model();hook_model(model);set_heads(model,chosen['heads'])
    rows=read(data.OUT/'external_retest/rows.json');values=[]
    with torch.inference_mode():
        for i,x in enumerate(DataLoader(public.ArchiveImages(rows,prep),batch_size=128,num_workers=8,pin_memory=True)):
            values.append(encoded(model,x.cuda(non_blocking=True)).cpu().numpy())
            if i%10==0:print('DYSLEXIFY PUBLIC',i*128,len(rows),flush=True)
    v=np.concatenate(values);cache=Path(read(data.OUT/'external_retest/cache.json')['path'])
    allrows=[]
    for bank in ('SCAM','NoSCAM','SynthSCAM','RTA100'):
        ix=np.array([i for i,r in enumerate(rows) if r['bank']==bank]);rr=[rows[i] for i in ix]
        tb='RTA100' if bank=='RTA100' else 'SCAM';labels=read(cache/(tb+'_labels.json'))
        t=np.load(cache/(tb+'_text.npz'))['base'];scores=v[ix]@t.T
        yi=np.array([labels.index(r['object_label']) for r in rr]);wi=np.array([labels.index(r['attack_word']) for r in rr])
        pred=scores.argmax(1);obj=scores[np.arange(len(rr)),yi];attack=scores[np.arange(len(rr)),wi]
        table=pd.DataFrame([dict(bank=bank,source_id=r['source_id'],id=r['id'],object_label=r['object_label'],
           attack_word=r['attack_word'],object_score=float(obj[j]),attack_score=float(attack[j]),
           margin=float(obj[j]-attack[j]),correct=bool(obj[j]>attack[j]),top1=bool(pred[j]==yi[j]),
           predicted_label=labels[pred[j]]) for j,r in enumerate(rr)])
        with (DEST/(bank+'_Dyslexify.csv')).open('x') as f:table.to_csv(f,index=False)
        allrows.append(dict(bank=bank,n=len(rr),accuracy=100*float(table['top1' if bank=='RTA100' else 'correct'].mean())))
    dump(DEST/'public_results.json',dict(selection_sha256=sha(DEST/'selection.json'),results=allrows))
    log(DEST,'public_score_complete');print(allrows,flush=True)

def report():
    from mirror.baselines.statistics import paired_interval
    sources={42:ROOT/'mirror/cases/typography_broad_support_20260926/matched_duration',
             43:ROOT/'mirror/cases/typography_broad_confirmation_20260926/seed43',
             44:ROOT/'mirror/cases/typography_broad_confirmation_20260926/seed44'}
    entries=[]
    for bank in ('SCAM','NoSCAM','SynthSCAM','RTA100'):
        metric='top1' if bank=='RTA100' else 'correct'
        dys=pd.read_csv(DEST/(bank+'_Dyslexify.csv')).set_index('source_id').sort_index()
        item=dict(bank=bank,n=len(dys),Dyslexify=100*float(dys[metric].mean()))
        for method in ['frozen','Defense_Prefix','ranking','IS']:
            arr=[]
            for seed,root in sources.items():
                frame=pd.read_csv(root/'external_retest'/f'{bank}_{method}.csv').set_index('source_id').sort_index()
                assert frame.index.equals(dys.index)
                arr.append(frame[metric].to_numpy(dtype=float)*100)
                if method in ('frozen','Defense_Prefix'):break
            a=np.stack(arr);item[method]=float(a.mean())
            if method=='IS':
                item['IS_minus_Dyslexify']=paired_interval(a-100*dys[metric].to_numpy(dtype=float)[None,:])
        entries.append(item)
    selection=read(DEST/'selection.json')
    dump(DEST/'summary.json',dict(results=entries,selection=selection))
    lines=['# Typography: published-method comparison','',
       'Fixed three-seed IS/ranking; released Defense-Prefix; Dyslexify reproduced on the same OpenAI B/32.',
       'Dyslexify uses upstream attention/hook code and greedy logic, with our existing train/development images instead of its unavailable saved circuit and original selection data. Public outcomes never select heads.','',
       '| Official test | N | Frozen | Defense-Prefix | Dyslexify reproduction | Ranking | IS | IS−Dyslexify 95% CI |',
       '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in entries:
        d=r['IS_minus_Dyslexify'];lo,hi=d['ci95']
        lines.append(f"| {r['bank']} | {r['n']} | {r['frozen']:.2f} | {r['Defense_Prefix']:.2f} | {r['Dyslexify']:.2f} | {r['ranking']:.2f} | {r['IS']:.2f} | {d['mean']:+.2f} [{lo:.2f},{hi:.2f}] |")
    lines+=['',f"Selected circuit: {selection['heads']}. Development clean accuracy change: {-selection['clean_drop_points']:+.2f}pp.",
       'Intervals resample paired sources and IS seeds (5000 draws). The published-method reproduction has one deterministic circuit; these intervals do not measure variability from repeating circuit selection.',
       'The upstream implementation accepts an improving head before its clean-drop stopping check; the recorded circuit preserves that behavior. No results were used to alter it.',
       'SCAM variants use complete official pairwise tasks; RTA100 uses official full100-class top1. NoSCAM measures word-removed preservation.']
    with (DEST/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    log(DEST,'complete');print('\n'.join(lines),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','select','score','report'])
    globals()[parser.parse_args().action]()
