"""Seed42 frozen-relative, position-balanced routing pilot. Prior results untouched."""
import argparse
from dataclasses import asdict; from dataclasses import replace
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding import routing_relative_data as data
from mirror.cases.color_binding import routing_repair_pilot as prior
from mirror.cases.color_binding.routing_repair_pilot import CAL; from mirror.cases.color_binding.routing_repair_pilot import MODEL; from mirror.cases.color_binding.routing_repair_pilot import AUDIT_CACHE; from mirror.cases.color_binding.routing_repair_pilot import SUGAR; from mirror.cases.color_binding.routing_repair_pilot import ARMS; from mirror.cases.color_binding.routing_repair_pilot import lines; from mirror.cases.color_binding.routing_repair_pilot import sha_bytes
from mirror.core.features import verify_cache
from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import cache_sha256; from mirror.core.repair import historical_ranking; from mirror.core.repair import derive_scales; from mirror.core.repair import legacy_ranking_loss; from mirror.core.repair import _check; from mirror.core.repair import _state_hash; from mirror.core.repair import _contrast_values; from mirror.core.repair import _correct_margins

OUT=data.OUT
SCORE_PARTS=('binding','cross','response','preference','endpoint',
             'caption_guard','binding_keep','response_keep','object_guard')
EPS=1e-5
PRIORITY=4.


def paired_rows(blocks):
    return (blocks[:,None]*2+torch.arange(2,device=blocks.device)).flatten()


def worse_layout(x):
    """Input rows must be direct/swapped adjacent, preserving block identity."""
    assert len(x)%2==0
    return x.reshape(len(x),-1).mean(-1).reshape(-1,2).max(-1).values.mean()


def directional_hinges(d,c,e,b,df,cf,ef,bf,k,tau,beta):
    return dict(binding=F.relu(torch.maximum(df,df.new_tensor(k))-d-EPS),
        cross=F.relu(c.abs()-torch.minimum(cf.abs(),cf.new_tensor(tau))-EPS),
        response=F.relu(torch.maximum(ef,ef.new_tensor(k-tau))-e-EPS),
        preference=F.relu(b.abs()-torch.minimum(bf.abs(),bf.new_tensor(beta))-EPS))


def measurements(scores,spec,contexts):
    vals=_contrast_values(spec,scores)
    d=torch.stack([vals[r['contrast']] for r in contexts if r['kind']=='binding'],-1)
    c=torch.stack([vals[r['contrast']] for r in contexts if r['kind']=='unwanted'],-1)
    margins=torch.stack((scores[:,1,1]-scores[:,1,2],scores[:,2,2]-scores[:,2,1]),-1)/spec.unit
    e=margins.mean(-1);b=(margins[:,0]-margins[:,1])/2
    return d,c,e,b,margins


def components(adapter,cache,spec,contexts,cfg,rows):
    images=cache.images[rows];base=cache.texts[rows];natural=cache.natural_texts[rows]
    adapted=adapter(torch.cat((base,natural),1))  # one identical forward for ALL arms
    scores=images@adapted[:,:6].transpose(1,2)
    frozen=images@base.transpose(1,2)
    d,c,e,b,margin=measurements(scores[:,:,:4],spec,contexts)
    df,cf,ef,bf,mf=measurements(frozen[:,:,:4],spec,contexts)
    thresholds=spec.spec['thresholds']['fractions'];k=thresholds['K'];tau=thresholds['tau'];beta=thresholds['beta']
    raw=directional_hinges(d,c,e,b,df,cf,ef,bf,k,tau,beta)
    raw['endpoint']=F.relu(torch.maximum(mf,mf.new_tensor(k-tau-beta))-margin-EPS)
    current=_correct_margins(scores[:,:,:4],(0,1,2,3))/spec.unit
    reference=_correct_margins(frozen[:,:,:4],(0,1,2,3))/spec.unit
    raw['caption_guard']=F.relu(reference-current-EPS)*(reference>0)
    raw['binding_keep']=F.relu(df-d-EPS)*(df>0)
    raw['response_keep']=F.relu(ef-e-EPS)
    old=(frozen[:,:,4]-frozen[:,:,5])/spec.unit
    new=(scores[:,:,4]-scores[:,:,5])/spec.unit
    raw['object_guard']=F.relu(old-new-EPS)*(old>0)
    # Mean squared Euclidean distance of unit vectors (sum over feature axis).
    raw['natural']=(adapted[:,6:]-F.normalize(natural,dim=-1)).square().sum(-1)
    raw['drift']=(adapted[:,:6]-F.normalize(base,dim=-1)).square().sum(-1)
    parts={key:worse_layout(value) for key,value in raw.items()}
    ranking,terms=legacy_ranking_loss(scores,adapted[:,:6],base,cfg.legacy_ranking,rows)
    parts.update(terms);parts['ranking']=ranking
    return parts


def objective(parts,weights,arm):
    scaled={k:weights[k]*parts[k] for k in SCORE_PARTS}
    guard=PRIORITY*(sum(scaled[k] for k in ('caption_guard','binding_keep','response_keep','object_guard'))+parts['natural']+parts['drift'])
    interaction=sum(scaled[k] for k in ('binding','cross','response'))
    preference=PRIORITY*scaled['preference']
    return dict(R=parts['ranking'],G=guard,I=guard+interaction,P=guard+preference,
        IP=guard+interaction+preference,E=guard+scaled['endpoint'])[arm]


def load_training(device='cpu'):
    data.verify();enc=read(OUT/'encoding_complete.json');verify_files({str(OUT/k):v for k,v in enc['files'].items()})
    verify_cache(OUT/'swapped_features')
    cache,spec,contexts,cfg=prior.load_training()
    old=lines(prior.OUT/'features/index.jsonl');idx=lines(OUT/'swapped_features/index.jsonl')
    assert [r['anchor_id'] for r in idx]==[r['anchor_id'] for r in old]
    expected=['-'.join(pair)+'/swapped_canvas/'+a+'_'+b for pair in prior.PAIRS for a in pair for b in pair]
    assert all(r['state_names']==expected and r['image_count']==8 for r in idx)
    swapped=torch.tensor(np.load(OUT/'swapped_features/images.npy').reshape(1280,4,768))
    # Captions refer to noun identity, not screen position: exact original text is reused.
    swap_text=np.load(OUT/'swapped_features/texts.npy')[np.array([r['text_indices'] for r in idx])].reshape(1280,4,768)
    assert np.max(np.abs(swap_text-cache.texts[:,:4].numpy()))<2e-6
    cache=replace(cache,images=torch.stack((cache.images,swapped),1).reshape(2560,4,768),
        texts=cache.texts.repeat_interleave(2,0),natural_texts=cache.natural_texts.repeat_interleave(2,0),
        source_ids=tuple(i for i in cache.source_ids for _ in range(2)),bank_id='routing_position_balanced_v1')
    cfg=replace(cfg,expected_cache_sha256=cache_sha256(cache),bank_id=cache.bank_id,
        legacy_ranking=historical_ranking('routing',(4,)*2560),epochs=12,batch_size=48,
        scales=derive_scales(cache,spec,(0,1,2,3)))  # cache validator metadata; new loss uses fixed spec.unit
    _check(cfg,cache,spec)
    if device!='cpu':cache=replace(cache,images=cache.images.to(device),texts=cache.texts.to(device),natural_texts=cache.natural_texts.to(device))
    return cache,spec,contexts,cfg


def freeze():
    data.verify();cache,spec,contexts,cfg=load_training()
    paths=[Path(__file__),Path(__file__).with_name('routing_relative_evaluate.py'),
        Path(__file__).parent/'tests/test_routing_relative.py',data.PLAN,OUT/'data_protocol.json',
        OUT/'encoding_complete.json',OUT/'swapped_features/complete.json']
    checks={str(p):sha(p) for p in paths}
    checks.update({str(OUT/'swapped_features'/k):v for k,v in read(OUT/'swapped_features/complete.json')['files'].items()})
    dump(OUT/'training_config.json',asdict(cfg))
    dump(OUT/'protocol.json',dict(inputs=checks,config_sha256=sha(OUT/'training_config.json'),
        prior_protocol_sha256=sha(prior.OUT/'protocol.json'),plan_sha256=sha(data.PLAN),
        seed=42,arms=['F',*ARMS],epochs=12,updates_per_arm=648,source_pairs=640,
        blocks=1280,lattices=2560,blocks_per_update=24,layouts=['canvas','swapped_canvas'],
        selection='fixed_last',gradient_calibration_seed=20260923,gradient_calibration_batches=8,
        gradient_calibration_mode='eval',calibration_B_std=.001,weight_clip=[.1,10.],
        priority=PRIORITY,numerical_hinge_tolerance=EPS,loss_plan=str(data.PLAN),
        diagnostic_selection='All original 49 pilot IDs, all three views; full SugarCrepe; fixed natural gallery',
        automatic_seed_escalation=False,reserve=False,post_specified_development=True))
    print('FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    p=read(OUT/'protocol.json');verify_files(p['inputs']);data.verify()
    assert sha(OUT/'training_config.json')==p['config_sha256'];return p


def configure():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def norm_grads(loss,model,retain_graph=False):
    grads=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=retain_graph)
    return float(torch.sqrt(sum(g.square().sum() for g in grads)))


def preflight():
    p=verify();configure();cache,spec,contexts,cfg=load_training('cuda')
    assert json.loads(json.dumps(asdict(cfg)))==read(OUT/'training_config.json')
    torch.manual_seed(42);model=TextLowRankAdapter(768,64,64,.05).cuda().eval()
    gen=torch.Generator(device='cuda').manual_seed(p['gradient_calibration_seed'])
    with torch.no_grad():model.B.weight.copy_(torch.randn(model.B.weight.shape,device='cuda',generator=gen)*p['calibration_B_std'])
    order=np.random.default_rng(p['gradient_calibration_seed']).permutation(1280)
    records=[];cosines=[]
    for batch in range(8):
        blocks=torch.tensor(order[24*batch:24*(batch+1)],device='cuda');ids=paired_rows(blocks)
        parts=components(model,cache,spec,contexts,cfg,ids)
        gradients={k:torch.cat([g.flatten() for g in torch.autograd.grad(parts[k],tuple(model.parameters()),retain_graph=True)]) for k in SCORE_PARTS}
        norms={k:float(g.norm()) for k,g in gradients.items()}
        unit=torch.stack([F.normalize(gradients[k],dim=0) for k in SCORE_PARTS])
        cosines.append((unit@unit.T).detach().cpu().numpy())
        assert all(np.isfinite(v) for v in norms.values())
        records.append(dict(blocks=blocks.cpu().tolist(),gradient_norms=norms,losses={k:float(parts[k].detach()) for k in SCORE_PARTS}))
    means={k:float(np.mean([r['gradient_norms'][k] for r in records])) for k in SCORE_PARTS}
    ref=float(np.median([x for x in means.values() if x>0]))
    weights={k:float(np.clip(ref/max(x,ref/10),.1,10)) for k,x in means.items()}
    dump(OUT/'normalization.json',dict(batches=records,mean_norms=means,reference_norm=ref,weights=weights,
        seed=p['gradient_calibration_seed'],no_optimizer_steps=True,training_only=True,protection_priority=PRIORITY,
        gradient_cosine_order=SCORE_PARTS,mean_gradient_cosines=np.mean(cosines,axis=0).tolist()))
    torch.manual_seed(42);model=TextLowRankAdapter(768,64,64,.05).cuda().train();ids=paired_rows(torch.arange(24,device='cuda'))
    checks={};reference=None;rng=[]
    for arm in ARMS:
        torch.manual_seed(420001);torch.cuda.manual_seed_all(420001)
        parts=components(model,cache,spec,contexts,cfg,ids);vals={k:float(v.detach()) for k,v in parts.items()}
        if reference is None:reference=vals
        assert vals==reference
        total=objective(parts,weights,arm);gn=norm_grads(total,model)
        assert np.isfinite(float(total)) and np.isfinite(gn)
        if arm!='G':assert gn>0
        checks[arm]=dict(loss=float(total.detach()),gradient_norm=gn)
        rng.append(sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes()))
    assert len(set(rng))==1
    dump(OUT/'preflight.json',dict(normalization_sha256=sha(OUT/'normalization.json'),
        config_sha256=sha(OUT/'training_config.json'),shared_forward_components_equal=True,shared_rng=True,
        identity_checks=checks,no_optimizer_steps=True,cache_sha256=cfg.expected_cache_sha256))
    print('PREFLIGHT weights',weights,'identity',checks,flush=True)


@torch.inference_mode()
def train_diagnostics(model,cache,spec,contexts):
    model.eval();measure=[]
    for start in range(0,2560,128):
        scores=cache.images[start:start+128]@model(cache.texts[start:start+128,:4]).transpose(1,2)
        d,c,e,b,m=measurements(scores,spec,contexts)
        diag=scores.diagonal(dim1=-2,dim2=-1)
        wrong=scores.masked_fill(torch.eye(4,device=scores.device,dtype=bool)[None],-torch.inf).max(-1).values
        measure.append(torch.stack((d.mean(-1),c.abs().mean(-1),e,b.abs(),(m>0).float().mean(-1),
            (diag>wrong).float().mean(-1)),-1).cpu().numpy())
    x=np.concatenate(measure).reshape(1280,2,6)
    names=['binding','cross_abs','response','absolute_preference','exchange_accuracy','caption_accuracy']
    return {view:dict(zip(names,x[:,j].mean(0).astype(float).tolist())) for j,view in enumerate(('canvas','swapped_canvas'))}


def train():
    verify();configure();cache,spec,contexts,cfg=load_training('cuda')
    flight=read(OUT/'preflight.json');assert sha(OUT/'normalization.json')==flight['normalization_sha256']
    assert cache_sha256(replace(cache,images=cache.images.cpu(),texts=cache.texts.cpu(),natural_texts=cache.natural_texts.cpu()))==flight['cache_sha256']
    weights=read(OUT/'normalization.json')['weights'];rng=np.random.default_rng(42)
    schedule=[rng.permutation(1280).tolist() for _ in range(12)];dump(OUT/'batch_schedule.json',schedule)
    models=[dict(arm='F',seed=42,checkpoint=None)];initial_hashes=[];rng_hashes=[];first_batches=[];batch_hashes=[]
    for arm in ARMS:
        dest=OUT/'runs'/arm;dest.mkdir(parents=True,exist_ok=False);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=TextLowRankAdapter(768,64,64,.05).cuda();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ih=_state_hash(initial);initial_hashes.append(ih);torch.save(dict(state_dict=initial,state_hash=ih),dest/'initial.pt')
        diagnostics=[dict(epoch=0,**train_diagnostics(model,cache,spec,contexts))]
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);history=[];started=time.monotonic()
        for epoch,order in enumerate(schedule,1):
            model.train();torch.manual_seed(420000+epoch);torch.cuda.manual_seed_all(420000+epoch)
            for start in range(0,1280,24):
                ids=paired_rows(torch.tensor(order[start:start+24],device='cuda'))
                parts=components(model,cache,spec,contexts,cfg,ids);total=objective(parts,weights,arm)
                opt.zero_grad(set_to_none=True);total.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                assert torch.isfinite(total) and torch.isfinite(gn)
                opt.step();history.append(dict(epoch=epoch,rows=ids.cpu().tolist(),loss=float(total.detach()),gradient_norm=float(gn),
                    components={k:float(v.detach()) for k,v in parts.items()}))
            diagnostics.append(dict(epoch=epoch,**train_diagnostics(model,cache,spec,contexts)))
            print('TRAIN',arm,epoch,'seconds',round(time.monotonic()-started,1),'diagnostics',diagnostics[-1],flush=True)
        assert len(history)==648
        state={k:v.detach().cpu() for k,v in model.state_dict().items()};rh=sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes());rng_hashes.append(rh)
        first_batches.append(history[0]['components']);batch_hashes.append(sha_bytes(json.dumps([r['rows'] for r in history]).encode()))
        torch.save(dict(state_dict=state,arm=arm,seed=42,selection='fixed_last',epochs=12,updates=len(history),
            configuration=asdict(cfg),objective_protocol_sha256=sha(OUT/'protocol.json'),weights=weights,
            initial_state_hash=ih,cache_sha256=cfg.expected_cache_sha256,optimizer=opt.state_dict()),dest/'last.pt')
        jsonl(dest/'history.jsonl',history);jsonl(dest/'training_diagnostics.jsonl',diagnostics)
        entry=dict(arm=arm,seed=42,checkpoint=str(dest/'last.pt'),sha256=sha(dest/'last.pt'),initial_state_hash=ih,
            final_rng_hash=rh,seconds=time.monotonic()-started,updates=len(history),selection='fixed_last')
        dump(dest/'complete.json',entry);models.append(entry);del model,opt
    assert len(set(initial_hashes))==len(set(rng_hashes))==len(set(batch_hashes))==1
    assert all(r==first_batches[0] for r in first_batches)
    dump(OUT/'models.json',models)
    dump(OUT/'training_complete.json',dict(models_sha256=sha(OUT/'models.json'),schedule_sha256=sha(OUT/'batch_schedule.json'),
        normalization_sha256=sha(OUT/'normalization.json'),matched_initialization=True,matched_batches=True,
        matched_dropout_rng=True,first_batch_components_equal=True,selection='fixed_last',updates_per_arm=648))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','preflight','train','evaluate']);args=parser.parse_args()
    log(OUT,'start',stage=args.action)
    try:
        if args.action=='evaluate':
            from mirror.cases.color_binding.routing_relative_evaluate import evaluate
            evaluate()
        else:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
