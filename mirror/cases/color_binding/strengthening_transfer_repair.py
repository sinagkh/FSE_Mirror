"""Matched CPU 2x2 support/context pilot, using the final routing objective."""
import argparse
from dataclasses import replace; from dataclasses import asdict
import math
import os
from pathlib import Path
import time
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA explicitly')
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.strengthening_cpu import OUT as ROOTOUT; from mirror.cases.color_binding.strengthening_cpu import clean
from mirror.cases.color_binding.strengthening_transfer_data import order
from mirror.cases.color_binding.strengthening_labclip import intervals
from mirror.core.repair import RepairCache; from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import _state_hash
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.cases.color_binding.routing_common_noise_pilot import PairedCaptionDropout; from mirror.cases.color_binding.routing_common_noise_pilot import objective
from mirror.core.metrics import bank_arrays; from mirror.core.metrics import adapt; from mirror.core.metrics import routing
from mirror.cases.color_binding.behavioral_pilot import lines; from mirror.cases.color_binding.behavioral_pilot import CAL
from mirror.core.features import verify_cache

DATA=ROOTOUT/'routing_transfer_v3'
OUT=DATA/'repair_v1'
CELLS=('narrow_canvas','narrow_mixed','broad_canvas','broad_mixed')
NAT=ROOTOUT/'source_firewall_v3/natural_texts_clean.npy'
MODEL='openclip_laion_l14'


def make_model(seed):
    torch.manual_seed(seed);m=TextLowRankAdapter(768,64,64,.05);m.drop=PairedCaptionDropout(.05)
    return m


def load(cell):
    support,view=cell.split('_');path=DATA/'features/train';verify_cache(path)
    assignments=read(DATA/'training_assignments.json')[support];lookup={r['anchor_id']:r for r in lines(path/'index.jsonl')}
    images=np.load(path/'images.npy',mmap_mode='r');texts=np.load(path/'texts.npy');natural=np.load(NAT)
    vv=[];tt=[];nt=[];sources=[]
    views=('canvas','swapped_canvas') if view=='canvas' else ('canvas','in_situ')
    for aid in assignments:
        r=lookup[aid]
        guard=natural[int(order('guard',aid)[:12],16)%len(natural)]
        for ci,pair in enumerate((('red','blue'),('green','yellow'))):
            text=texts[np.array(r['text_indices'][ci*4:ci*4+4]+r['text_indices'][-2:])]
            for v in views:
                names=['-'.join(pair)+'/'+v+'/'+a+'_'+b for a in pair for b in pair]
                indices=[r['image_offset']+r['state_names'].index(n) for n in names]
                vv.append(images[indices]);tt.append(text);nt.append(guard);sources.append(aid)
    assert len(vv)==len(assignments)*4 and np.allclose(np.asarray(tt)[::2],np.asarray(tt)[1::2])
    old,spec,contexts,cfg=prior.load_training('cpu')
    cache=RepairCache(torch.tensor(np.asarray(vv)),torch.tensor(np.asarray(tt)),torch.tensor(np.asarray(nt)),tuple(sources),
                     MODEL,old.encoder_sha256,'transfer_'+cell)
    return cache,spec,contexts,replace(cfg,seed=42,epochs=36)


def normalize(support):
    a,spec,contexts,cfg=load(support+'_canvas');b,_,_,_=load(support+'_mixed')
    model=make_model(20260925).eval();torch.manual_seed(20260925)
    with torch.no_grad():model.B.weight.normal_(std=.001)
    norms=[];rng=np.random.default_rng(20260925);schedule=rng.permutation(len(a.images)//2)
    for j in range(8):
        blocks=torch.tensor(schedule[24*j:24*(j+1)]);idx=prior.paired_rows(blocks)
        p=prior.components(model,a if j%2==0 else b,spec,contexts,cfg,idx)
        n={}
        for k in prior.SCORE_PARTS:
            g=torch.autograd.grad(p[k],tuple(model.parameters()),retain_graph=True)
            n[k]=float(torch.sqrt(sum(x.square().sum() for x in g)))
        norms.append(n)
    med={k:float(np.median([r[k] for r in norms])) for k in prior.SCORE_PARTS}
    w={k:float(np.clip(med['binding']/max(med[k],1e-10),.1,10)) for k in prior.SCORE_PARTS}
    dump(OUT/f'{support}_normalization.json',dict(weights=w,gradient_norms=norms,mode='Training-only alternating canvas/mixed calibration, same weights in both context cells'))
    return w


def freeze():
    if not (DATA/'features/train/complete.json').exists():
        from mirror.cases.color_binding.strengthening_firewall_v3 import transfer_cache
        transfer_cache()
    for bank in ('train','development'):verify_cache(DATA/'features'/bank)
    OUT.mkdir(parents=True,exist_ok=False)
    files=[DATA/'complete.json',DATA/'training_assignments.json',DATA/'features/train/complete.json',DATA/'features/development/complete.json',
           Path(__file__),Path(prior.__file__),Path(__file__).with_name('routing_common_noise_pilot.py'),NAT,CAL]
    n=len(read(DATA/'training_assignments.json')['narrow'])
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in files},cells=CELLS,arms=['F','R','IS'],seed=42,device='cpu',threads=4,
        objective='Exact final routing_relative_pilot.components and routing_common_noise_pilot.objective; ranking is original CE+.2 anchoring',
        normalization='Per vocabulary support,8 fixed train-only batches alternating canvas/mixed; shared between its two context cells',
        budget=dict(anchors=n,color_pairs=2,lattices_per_block=2,blocks=2*n,blocks_per_update=24,epochs=36,updates=36*math.ceil(2*n/24),rank=64,alpha=64,dropout=.05,lr=.0002,weight_decay=.01,grad_clip=1.),
        selection='fixed_last',seeds_to_promote=[42,43,44],confirmation_read=False,
        eligibility='Red-blue in-situ development: response increases,binding increases,absolute cross decreases,exchange gain>=5pp; old development both single-word accuracies drop<=1pp',
        recipe_selection='Highest whole-development red-blue in-situ exchange accuracy among eligible cells, tie lexicographic cell name. No public benchmark selection.',
        comparison='Matched one-seed factorial development pilot; historical checkpoint not substituted for any cell',
        same_forwards_rng=True))
    for support in ('narrow','broad'):normalize(support)


def train(cell,seed):
    verify_files(read(OUT/'protocol.json')['inputs']);cache,spec,contexts,cfg=load(cell);cfg=replace(cfg,seed=seed)
    if seed!=42:
        decision=read(OUT/'development/selection.json')
        assert decision['selected_cell'] is not None and cell in (decision['selected_cell'],'narrow_canvas')
    support=cell.split('_')[0];weights=read(OUT/f'{support}_normalization.json')['weights']
    dest=OUT/cell/f'seed{seed}';dest.mkdir(parents=True,exist_ok=False)
    nblocks=len(cache.images)//2
    rng=np.random.default_rng(seed);schedule=[rng.permutation(nblocks).tolist() for _ in range(36)]
    dump(dest/'schedule.json',schedule);models=[];initials=[];rngs=[]
    for arm in ('IS','R'):
        folder=dest/arm;folder.mkdir();model=make_model(seed);initials.append(_state_hash(model.state_dict()))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01);history=[];begun=time.monotonic()
        for epoch,order in enumerate(schedule,1):
            model.train();torch.manual_seed(seed*10000+epoch)
            for start in range(0,nblocks,24):
                idx=prior.paired_rows(torch.tensor(order[start:start+24]))
                p=prior.components(model,cache,spec,contexts,cfg,idx)
                loss=objective(p,weights) if arm=='IS' else p['ranking']
                optimizer.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(loss)&torch.isfinite(gn);optimizer.step()
                history.append(dict(epoch=epoch,loss=float(loss.detach()),gradient_norm=float(gn),parts={k:float(v.detach()) for k,v in p.items()}))
            if epoch%12==0:print('TRANSFER_CPU_TRAIN',cell,seed,arm,epoch,'seconds',round(time.monotonic()-begun,1),flush=True)
        assert len(history)==36*math.ceil(nblocks/24);rngs.append(torch.get_rng_state().numpy().tobytes())
        torch.save(dict(state_dict=model.state_dict(),seed=seed,arm=arm,cell=cell,selection='fixed_last',weights=weights,protocol_sha256=sha(OUT/'protocol.json')),folder/'last.pt')
        jsonl(folder/'history.jsonl',history);models.append(dict(arm=arm,seed=seed,cell=cell,checkpoint=str(folder/'last.pt'),sha256=sha(folder/'last.pt'),seconds=time.monotonic()-begun))
    assert len(set(initials))==len(set(rngs))==1
    dump(dest/'models.json',models);dump(dest/'complete.json',dict(models_sha256=sha(dest/'models.json'),matched_rng=True,matched_initialization=True,gpu=False))


def evaluate():
    dest=OUT/'development';dest.mkdir(parents=True,exist_ok=False);path=DATA/'features/development'
    rows={r['anchor_id']:r for r in lines(DATA/'rows.jsonl') if r['bank']=='development'}
    entries=[dict(arm='F',seed=0,cell='F',checkpoint=None)]+[m for cell in CELLS for m in read(OUT/cell/'seed42/models.json')]
    records=[];stats=[]
    for color in ('red-blue','green-yellow'):
        for view in ('canvas','swapped_canvas','in_situ'):
            v,t,idx=bank_arrays(path,'routing',color,view);measure={};groups=np.array([rows[r['anchor_id']]['stratum'] for r in idx])
            for entry in entries:
                key=entry['cell']+'/'+entry['arm'];x=np.einsum('nid,njd->nij',v,adapt(t,entry['checkpoint']));m=routing(x,MODEL);measure[key]=m
                for j,r in enumerate(idx):records.append(dict(**entry,color=color,view=view,anchor_id=r['anchor_id'],stratum=groups[j],
                    **{k:float(a[j]) for k,a in m.items()}))
            metrics=[k for k in measure['F/F'] if not k.startswith('contrast/')]
            for group in ['all',*sorted(set(groups))]:
                mask=np.ones(len(idx),bool) if group=='all' else groups==group
                for entry in entries:
                    key=entry['cell']+'/'+entry['arm']
                    x=np.stack([measure[key][k][mask] for k in metrics],1)
                    for metric,s in zip(metrics,intervals(x[None],np.arange(mask.sum()))):stats.append(dict(cell=entry['cell'],arm=entry['arm'],color=color,view=view,stratum=group,metric=metric,**s))
            print('TRANSFER_DEV_SCORED',color,view,flush=True)
    jsonl(dest/'per_example.jsonl',records);jsonl(dest/'summary.jsonl',stats)
    bank=ROOT/'clip/interbind_routing_same_class_20260923/features'
    blocked=set(read(ROOTOUT/'source_firewall_v3/blocked_coco_ids.json'))
    allowed={r['anchor_id'] for r in lines(bank.parent/'rows.jsonl') if not set(r['source_ids'])&blocked}
    v,t,_=bank_arrays(bank,'routing','red-blue','canvas',allowed=allowed);utility=[]
    for entry in entries:
        m=routing(np.einsum('nid,njd->nij',v,adapt(t,entry['checkpoint'])),MODEL)
        utility.append(dict(cell=entry['cell'],arm=entry['arm'],word1=float(m['word1_accuracy'].mean()),word2=float(m['word2_accuracy'].mean())))
    dump(dest/'internal_utility.json',utility)
    def mean(cell,arm,metric):return next(r['mean'] for r in stats if r['cell']==cell and r['arm']==arm and r['color']=='red-blue' and r['view']=='in_situ' and r['stratum']=='all' and r['metric']==metric)
    uf=next(u for u in utility if u['cell']=='F');gates=[]
    for cell in CELLS:
        u=next(u for u in utility if u['cell']==cell and u['arm']=='IS')
        checks=dict(response=mean(cell,'IS','response')>mean('F','F','response'),binding=mean(cell,'IS','binding')>mean('F','F','binding'),
            cross=mean(cell,'IS','cross')<mean('F','F','cross'),exchange=mean(cell,'IS','exchange_accuracy')>=mean('F','F','exchange_accuracy')+.05,
            word1=u['word1']>=uf['word1']-.01,word2=u['word2']>=uf['word2']-.01)
        gates.append(dict(cell=cell,passed=all(checks.values()),checks=checks,accuracy=mean(cell,'IS','exchange_accuracy')))
    passed=sorted([r for r in gates if r['passed']],key=lambda r:(-r['accuracy'],r['cell']))
    dump(dest/'selection.json',dict(gates=gates,selected_cell=passed[0]['cell'] if passed else None,confirmation_used=False,benchmarks_used=False,
        implication='No cell passing means stop promotion and diagnose development results; do not select a favorable confirmation subset.'))
    report=['# Routing vocabulary/view-support pilot (development only)', '',
        'One seed42; four preassigned cells, original ranking and full IS in each. Equal source counts and update budgets after the metadata-only source-firewall correction (see protocol.json). A broad cell replaces half the original-vocabulary anchors with expanded-vocabulary anchors; context cells use identical sources. No confirmation or public benchmark used for selection.', '',
        '| Cell | Arm | In-situ exchange accuracy | Response | Binding | Absolute cross |',
        '|---|---|---:|---:|---:|---:|']
    for cell,arm in [('F','F')]+[(c,a) for c in CELLS for a in ('R','IS')]:
        report.append(f"| {cell} | {arm} | {100*mean(cell,arm,'exchange_accuracy'):.2f}% | {mean(cell,arm,'response'):.4f} | {mean(cell,arm,'binding'):.4f} | {mean(cell,arm,'cross'):.4f} |")
    report+=['', 'Selected by the frozen development gate: '+(passed[0]['cell'] if passed else '**none; diagnose before promotion**')+'.', '',
        'All colors, three views, vocabulary groups, per-example scores, preservation checks and source-bootstrap intervals are retained. This pilot is not three-seed confirmation. Full-scene vs canvas changes context, clutter and object scale jointly. Actual observed noun/pair support is frozen separately from declared vocabulary; use actual_support_strata.jsonl for seen/unseen claims.']
    with (dest/'REPORT.md').open('x') as f:f.write('\n'.join(report)+'\n')
    dump(dest/'complete.json',dict(files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},gpu=False))
    print('TRANSFER_DEVELOPMENT_SELECTION',passed[0]['cell'] if passed else None,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','train','evaluate']);p.add_argument('--cell',choices=CELLS);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);log(ROOTOUT,'start',stage='transfer_repair_'+a.action,cell=a.cell,seed=a.seed)
    try:
        if a.action=='train':train(a.cell,a.seed)
        else:globals()[a.action]()
    except BaseException as e:log(ROOTOUT,'failed',stage='transfer_repair_'+a.action,error=repr(e));raise
    log(ROOTOUT,'complete',stage='transfer_repair_'+a.action)


if __name__=='__main__':main()
