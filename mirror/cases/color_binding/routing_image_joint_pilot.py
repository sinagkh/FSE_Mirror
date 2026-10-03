"""Capacity-matched image-only/joint routing on immutable cached embeddings."""
from mirror.cases.color_binding import routing_suppression_strength as base  # CPU-only environment
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import argparse
import csv
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import _state_hash; from mirror.core.repair import _correct_margins
from mirror.cases.color_binding.routing_common_noise_pilot import PairedCaptionDropout

OUT = ROOT/'clip/interbind_routing_image_joint_20260929'
PLAN = ROOT/'FSE_VLM/plan/82_routing_image_joint_pilot_20260929.md'
ARMS = ('Image_Ranking', 'Image_IS', 'Joint_Ranking', 'Joint_IS')
VIEWS = ('canvas', 'swapped_canvas')


class SideMaps(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        if mode not in ('image', 'joint', 'text'):
            raise ValueError(mode)
        state = torch.load(base.INITIAL, map_location='cpu', weights_only=False)['state_dict']
        rank = 32 if mode=='joint' else 64
        self.image_map = None; self.text_map = None
        for side in ('image', 'text'):
            if mode!=side and mode!='joint':
                continue
            adapter = TextLowRankAdapter(768, rank, rank, .05)
            adapter.drop = PairedCaptionDropout(.05)
            offset = 32 if mode=='joint' and side=='text' else 0
            with torch.no_grad():
                adapter.A.weight.copy_(state['A.weight'][offset:offset+rank])
                adapter.B.weight.zero_()
            setattr(self, side+'_map', adapter)

    def forward(self, images, texts):
        return (self.image_map(images) if self.image_map is not None else images,
                self.text_map(texts) if self.text_map is not None else texts)


def mode_for(arm):
    return 'image' if arm.startswith('Image_') else 'joint'


def freeze():
    base.verify()
    paths = [PLAN, Path(__file__), Path(__file__).with_name('routing_image_joint_report.py'),
             Path(__file__).with_name('run_routing_image_joint_pilot.sh'),
             Path(__file__).parent/'tests/test_routing_image_joint.py', base.INITIAL,
             base.OUT/'protocol.json', base.prev.OUT/'training_encoding.json',
             base.prior.OUT/'normalization.json']
    paths += [Path(m.__file__) for m in (base, base.prior, base.prev, base.cn, base.metrics)]
    refs = [read(base.OUT/'runs'/a/'complete.json') for a in ('Ranking', 'IS4')]
    verify_files({r['checkpoint']:r['sha256'] for r in refs})
    dump(OUT/'protocol.json', dict(inputs={str(p):sha(p) for p in paths}, arms=ARMS,
        seed=42, references=refs, device='cpu', updates=1944, epochs=36,
        trainable_parameters=98304, calibration_seed=20260923, calibration_batches=8,
        calibration_B_std=.001, selection='fixed_last', grad_clip=1.,
        candidate_test_scores_seen=False, historical_test_scores_seen=True,
        new_data=False, new_captions=False, internal_encoder_updated=False))
    print('FROZEN', sha(OUT/'protocol.json'), flush=True)


def verify():
    p=read(OUT/'protocol.json'); verify_files(p['inputs']); base.verify()
    verify_files({r['checkpoint']:r['sha256'] for r in p['references']})
    return p


def components(model, cache, spec, ctx, ids):
    image0=cache.images[ids]; text0=cache.texts[ids]; natural0=cache.natural_texts[ids]
    images, text = model(image0, torch.cat((text0,natural0),1))
    scores=images@text[:,:6].transpose(1,2)
    # Never compute references with the repaired images or repaired text.
    frozen=image0@text0.transpose(1,2)
    d,c,e,b,m=base.prior.measurements(scores[:,:,:4],spec,ctx)
    df,cf,ef,bf,mf=base.prior.measurements(frozen[:,:,:4],spec,ctx)
    th=spec.spec['thresholds']['fractions']; k,tau,beta=(th[n] for n in ('K','tau','beta'))
    raw=base.prior.directional_hinges(d,c,e,b,df,cf,ef,bf,k,tau,beta)
    raw['endpoint']=F.relu(torch.maximum(mf,mf.new_tensor(k-tau-beta))-m-base.prior.EPS)
    actual=_correct_margins(scores[:,:,:4],(0,1,2,3))/spec.unit
    ref=_correct_margins(frozen[:,:,:4],(0,1,2,3))/spec.unit
    raw['caption_guard']=F.relu(ref-actual-base.prior.EPS)*(ref>0)
    raw['binding_keep']=F.relu(df-d-base.prior.EPS)*(df>0)
    raw['response_keep']=F.relu(ef-e-base.prior.EPS)
    old=(frozen[:,:,4]-frozen[:,:,5])/spec.unit
    new=(scores[:,:,4]-scores[:,:,5])/spec.unit
    raw['object_guard']=F.relu(old-new-base.prior.EPS)*(old>0)
    if model.text_map is not None:
        raw['natural']=(text[:,6:]-F.normalize(natural0,dim=-1)).square().sum(-1)
        raw['drift']=(text[:,:6]-F.normalize(text0,dim=-1)).square().sum(-1)
    else:
        raw['natural']=scores[:,0,0]*0; raw['drift']=scores[:,0,0]*0
    raw['image_drift']=((images-F.normalize(image0,dim=-1)).square().sum(-1)
                        if model.image_map is not None else scores[:,0,0]*0)
    parts={key:base.prior.worse_layout(value) for key,value in raw.items()}
    parts['ce']=F.cross_entropy(100*scores[:,:,:4].reshape(-1,4),torch.arange(4).repeat(len(ids)))
    return parts, scores, frozen


def guard(parts, weights):
    names=('caption_guard','binding_keep','response_keep','object_guard')
    return 4*(sum(weights[n]*parts[n] for n in names)+parts['natural']+parts['drift']+parts['image_drift'])


def objective(parts, weights, arm):
    shared=guard(parts,weights)
    if arm.endswith('Ranking'):
        return shared+parts['ce']
    return shared+sum(weights[n]*parts[n]*mult for n,mult in
        (('binding',1),('cross',4),('response',1),('preference',4)))


def gradient(loss, model):
    grads=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                      for p,g in zip(model.parameters(),grads)])


def preflight():
    p=verify(); cache,spec,ctx,cfg=base.prior.load_training('cpu')
    forms=base.prev.template_cache(cache); weights0=read(base.prior.OUT/'normalization.json')['weights']
    ids=base.prior.paired_rows(torch.arange(24))
    assert not any(x.requires_grad for x in (cache.images,cache.texts,cache.natural_texts))
    # Old text-only objective replay establishes score indices and retention algebra.
    legacy=base.new_model(cfg).eval(); textmodel=SideMaps('text').eval()
    torch.manual_seed(9029)
    with torch.no_grad():
        delta=torch.randn_like(legacy.B.weight)*.001
        legacy.B.weight.copy_(delta); textmodel.text_map.B.weight.copy_(delta)
    old=base.prior.components(legacy,cache,spec,ctx,cfg,ids)
    new,_,_=components(textmodel,cache,spec,ctx,ids)
    for key in new:
        if key not in ('ce','image_drift'):
            torch.testing.assert_close(new[key],old[key],rtol=1e-5,atol=2e-7)
    torch.testing.assert_close(objective(new,weights0,'Text_IS'),base.cn.objective(old,weights0),rtol=1e-6,atol=2e-7)
    report={}; order=np.random.default_rng(p['calibration_seed']).permutation(1280)
    for mode in ('image','joint'):
        torch.manual_seed(42); model=SideMaps(mode).eval()
        count=sum(x.numel() for x in model.parameters()); assert count==98304
        init=_state_hash(model.state_dict()); parts,scores,frozen=components(model,cache,spec,ctx,ids)
        torch.testing.assert_close(scores,frozen,rtol=1e-5,atol=2e-7)
        gen=torch.Generator().manual_seed(p['calibration_seed'])
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                if name.endswith('B.weight'):
                    parameter.copy_(torch.randn(parameter.shape,generator=gen)*p['calibration_B_std'])
        batches=[]
        for batch in range(p['calibration_batches']):
            rows=base.prior.paired_rows(torch.tensor(order[batch*24:(batch+1)*24]))
            parts,_,_=components(model,cache,spec,ctx,rows)
            norms={n:float(gradient(parts[n],model).norm()) for n in base.prior.SCORE_PARTS}
            assert all(np.isfinite(z) for z in norms.values())
            batches.append(dict(rows=rows.tolist(),gradient_norms=norms))
        means={n:float(np.mean([r['gradient_norms'][n] for r in batches])) for n in base.prior.SCORE_PARTS}
        reference=float(np.median([v for v in means.values() if v>0]))
        weights={n:float(np.clip(reference/max(v,reference/10),.1,10)) for n,v in means.items()}
        dump(OUT/f'calibration_{mode}.json',dict(weights=weights,mean_norms=means,
            reference_norm=reference,batches=batches,training_only=True,optimizer_steps=0))
        # Perturbed-model finite-difference factorization for every context.
        model.eval(); vi,te=model(cache.images[ids],cache.texts[ids,:4]); scores=vi@te.transpose(1,2)
        for context in ctx:
            w=torch.as_tensor(spec.weights[context['contrast']],dtype=scores.dtype)
            ii=torch.where(w.abs().sum(1)>0)[0]; jj=torch.where(w.abs().sum(0)>0)[0]
            expected=w[ii[0],jj[0]]*((vi[:,ii[0]]-vi[:,ii[1]])*(te[:,jj[0]]-te[:,jj[1]])).sum(-1)/spec.unit
            observed=(scores*w).sum((1,2))/spec.unit
            torch.testing.assert_close(observed,expected,rtol=5e-4,atol=1e-5)
        d,c,e,b,m=base.prior.measurements(scores,spec,ctx)
        torch.testing.assert_close(torch.stack((e+b,e-b),-1),m,rtol=1e-5,atol=1e-6)
        # Target deletion is exactly the shared guard, not extra training.
        parts,_,_=components(model,cache,spec,ctx,ids)
        zero=dict(weights,**{n:0. for n in ('binding','cross','response','preference')})
        torch.testing.assert_close(gradient(objective(parts,zero,'IS'),model),gradient(guard(parts,weights),model),rtol=0,atol=0)
        # Matched pairs share initial state and consume the same random stream.
        records=[]
        for arm in [a for a in ARMS if mode_for(a)==mode]:
            torch.manual_seed(42); smoke=SideMaps(mode).train()
            assert _state_hash(smoke.state_dict())==init
            opt=torch.optim.AdamW(smoke.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
            torch.manual_seed(420001)
            for j in range(3):
                rows=base.prior.paired_rows(torch.tensor(order[j*24:(j+1)*24]))
                parts,_,frozen=components(smoke,cache,spec,ctx,rows)
                assert not frozen.requires_grad
                if j==0:first={n:float(v.detach()) for n,v in parts.items()}
                loss=objective(parts,weights,arm); opt.zero_grad(set_to_none=True); loss.backward()
                gn=torch.nn.utils.clip_grad_norm_(smoke.parameters(),1.)
                assert torch.isfinite(loss) and torch.isfinite(gn); opt.step()
            records.append(dict(arm=arm,first=first,rng=base.hash_array(torch.get_rng_state().numpy())))
        assert records[0]['first']==records[1]['first'] and records[0]['rng']==records[1]['rng']
        report[mode]=dict(parameters=count,initial_state_hash=init,weights=weights,
            calibration_sha256=sha(OUT/f'calibration_{mode}.json'),smoke_updates=3,
            matched_smoke_forward_rng=True,fd_factorization=True,eb_identity=True)
    dump(OUT/'preflight.json',dict(modes=report,legacy_text_components_replayed=True,
        frozen_reference_detached=True,guard_deletion_gradient_exact=True,
        checkpoints_modified=False,training_updates=0))
    print('PREFLIGHT_PASS',report,flush=True)


def train(arm):
    p=verify(); flight=read(OUT/'preflight.json'); mode=mode_for(arm)
    assert sha(OUT/f'calibration_{mode}.json')==flight['modes'][mode]['calibration_sha256']
    dest=OUT/'runs'/arm; dest.mkdir(parents=True,exist_ok=False)
    cache,spec,ctx,cfg=base.prior.load_training('cpu'); cfg=replace(cfg,epochs=36,seed=42)
    forms=base.prev.template_cache(cache); weights=read(OUT/f'calibration_{mode}.json')['weights']
    torch.manual_seed(42); model=SideMaps(mode); initial=_state_hash(model.state_dict())
    assert initial==flight['modes'][mode]['initial_state_hash']
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    schedule=base.old.schedule_for(42); reps=base.prev.representation_schedule(42)
    history=[]; start=time.monotonic()
    for ei,order in enumerate(schedule):
        current=replace(cache,texts=forms[torch.arange(2560),torch.as_tensor(np.repeat(reps[ei],2))])
        model.train(); torch.manual_seed(420000+ei+1)
        for j in range(0,1280,24):
            ids=base.prior.paired_rows(torch.tensor(order[j:j+24]))
            parts,_,_=components(model,current,spec,ctx,ids); loss=objective(parts,weights,arm)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(gn); opt.step()
            history.append(dict(epoch=ei+1,rows=ids.tolist(),loss=float(loss.detach()),
                gradient_norm=float(gn),components={n:float(v.detach()) for n,v in parts.items()}))
        if (ei+1)%6==0:print('TRAIN',arm,ei+1,round(time.monotonic()-start,1),flush=True)
    assert len(history)==1944 and not torch.cuda.is_initialized()
    checkpoint=dest/'last.pt'
    torch.save(dict(state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},
        mode=mode,configuration=asdict(cfg),seed=42,arm=arm,optimizer=opt.state_dict(),
        selection='fixed_last',updates=1944,epochs=36,weights=weights,
        calibration_sha256=sha(OUT/f'calibration_{mode}.json'),protocol_sha256=sha(OUT/'protocol.json')),
        checkpoint)
    jsonl(dest/'history.jsonl',history)
    receipt=dict(name=arm,seed=42,mode=mode,checkpoint=str(checkpoint),sha256=sha(checkpoint),
        initial_state_hash=initial,updates=1944,parameters=sum(x.numel() for x in model.parameters()),
        schedule_sha256=base.hash_array(np.asarray(schedule)),representation_sha256=base.hash_array(reps),
        final_rng_sha256=base.hash_array(torch.get_rng_state().numpy()),
        first_components=history[0]['components'],seconds=time.monotonic()-start,
        calibration_sha256=sha(OUT/f'calibration_{mode}.json'))
    dump(dest/'complete.json',receipt); print('TRAIN_COMPLETE',arm,receipt['seconds'],flush=True)


def registry():
    p=verify(); regs=[read(OUT/'runs'/a/'complete.json') for a in ARMS]
    for mode in ('image','joint'):
        pair=[r for r in regs if r['mode']==mode]
        for key in ('initial_state_hash','updates','parameters','schedule_sha256',
                    'representation_sha256','final_rng_sha256','first_components','calibration_sha256'):
            assert pair[0][key]==pair[1][key],(mode,key)
    for key in ('updates','parameters','schedule_sha256','representation_sha256'):
        assert len({r[key] for r in regs})==1
    refs=[dict(r,name='Text_Ranking' if r['name']=='Ranking' else 'Text_IS') for r in p['references']]
    regs=[dict(name='F',seed=0,checkpoint=None),*refs,*regs]
    verify_files({r['checkpoint']:r['sha256'] for r in regs if r['checkpoint']})
    return regs


def load_runner(reg):
    if 'mode' not in reg:return None
    model=SideMaps(reg['mode']).eval()
    state=torch.load(reg['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict']); return model


@torch.inference_mode()
def adapted_arrays(images,texts,reg,runner=None):
    if 'mode' not in reg:
        return np.asarray(images),base.metrics.adapt(texts,reg['checkpoint'])
    runner=runner or load_runner(reg)
    v,t=runner(torch.as_tensor(np.asarray(images)).float(),torch.as_tensor(np.asarray(texts)).float())
    return v.numpy(),t.numpy()


def evaluate():
    regs=registry(); dump(OUT/'checkpoints.json',regs)
    arrays={}; rows=[]; contexts={}; runners={r['name']:load_runner(r) for r in regs}
    features=np.load(base.prev.OUT/'training_template_features.npy')
    lookup={s:i for i,s in enumerate(read(base.prev.OUT/'training_template_strings.json'))}
    groups=base.prev.lines(base.prev.DEV/'features/text_groups.jsonl')
    alltexts=np.load(base.prev.DEV/'features/texts.npy')
    for view in VIEWS:
        images,texts,idx=base.metrics.bank_arrays(base.prev.DEV/'features','routing','red-blue',view)
        assert len(idx)==888
        arrays[view+'/ids']=np.asarray([r['anchor_id'] for r in idx])
        forms=np.stack([features[[lookup[s] for ti in r['text_indices'][:4]
            for s in groups[ti]['templates']]].reshape(4,3,768).transpose(1,0,2) for r in idx])
        objects=alltexts[np.array([r['text_indices'][8:10] for r in idx])]
        for reg in regs:
            name=reg['name']; runner=runners[name]
            v,t=adapted_arrays(images,texts,reg,runner)
            x=np.einsum('nid,njd->nij',v,t)
            mm=base.metrics.routing(x)
            _,tt=adapted_arrays(images,forms,reg,runner)
            xx=np.einsum('nid,nkjd->nkij',v,tt)
            mi=base.metrics.routing(xx.reshape(-1,4,4))
            _,ot=adapted_arrays(images,objects,reg,runner)
            om=np.einsum('nid,njd->nij',v,ot)
            arrays[view+'/'+name]=x; arrays[view+'/'+name+'/templates']=xx
            arrays[view+'/'+name+'/objects']=om
            row=dict(name=name,seed=reg['seed'],view=view,
                **{k:float(a.mean()) for k,a in mm.items() if not k.startswith('contrast/')},
                individual_exchange=float(mi['exchange_accuracy'].mean()),
                object_guard=float((om[:,:,0]>om[:,:,1]).mean()),
                image_drift=float(np.mean(np.sum((v-images)**2,axis=-1))),
                text_drift=float(np.mean(np.sum((t-texts)**2,axis=-1))))
            rows.append(row)
            contexts[view+'/'+name]={k:float(a.mean()) for k,a in mm.items() if k.startswith('contrast/')}
    assert np.array_equal(arrays['canvas/ids'],arrays['swapped_canvas/ids'])
    with (OUT/'development_scores.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    jsonl(OUT/'development.jsonl',rows); dump(OUT/'development_contexts.json',contexts)
    means=base.prev.means(rows); dump(OUT/'development_means.json',means)
    candidates=[]; direction=[]
    for name in [r['name'] for r in regs if r['name']!='F']:
        for ctx in base.metrics.all_contexts():
            key='contrast/'+ctx['name']; binding=ctx['kind']=='binding'
            fun=(lambda x:x) if binding else np.abs
            original=float(np.mean([fun(base.metrics.routing(arrays[v+'/F'])[key]) for v in VIEWS]))
            changed=float(np.mean([fun(base.metrics.routing(arrays[v+'/'+name])[key]) for v in VIEWS]))
            direction.append(dict(name=name,context=ctx['name'],kind=ctx['kind'],frozen=original,
                adapted=changed,change=changed-original,intended=(changed>=original if binding else changed<=original)))
    for arm in ('Image_IS','Joint_IS'):
        a=means[arm]; ranking=arm.replace('_IS','_Ranking')
        checks=dict(suppression_20percent=a['cross']<=.8*means['F']['cross'],
            all_context_directions=all(r['intended'] for r in direction if r['name']==arm),
            exchange_vs_text=a['exchange_accuracy']>=means['Text_IS']['exchange_accuracy'],
            exchange_vs_ranking=a['exchange_accuracy']>=means[ranking]['exchange_accuracy'],
            individual_vs_text=a['individual_exchange']>=means['Text_IS']['individual_exchange'])
        for view in VIEWS:
            aa=next(r for r in rows if r['name']==arm and r['view']==view)
            ff=next(r for r in rows if r['name']=='F' and r['view']==view)
            checks[view+'/words']=min(aa['word1_accuracy'],aa['word2_accuracy'])>=.99
            checks[view+'/object']=aa['object_guard']>=ff['object_guard']-.01
        candidates.append(dict(name=arm,ranking=ranking,eligible=all(checks.values()),checks=checks))
    dump(OUT/'all_context_directions.json',direction)
    dump(OUT/'selection.json',dict(candidates=candidates,
        selected=[r['name'] for r in candidates if r['eligible']],candidate_test_scores_seen=False))
    print('DEVELOPMENT',{n:{k:round(a[k],6) for k in ('binding','cross','exchange_accuracy','individual_exchange')} for n,a in means.items()},flush=True)
    print('SELECTION',candidates,flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=('freeze','preflight','train','evaluate'))
    ap.add_argument('--arm',choices=ARMS); args=ap.parse_args(); log(OUT,'start',action=args.action,arm=args.arm)
    try:
        base.configure()
        if args.action=='train':train(args.arm)
        else:globals()[args.action]()
    except BaseException as exc:
        log(OUT,'failed',action=args.action,arm=args.arm,error=repr(exc)); raise
    log(OUT,'complete',action=args.action,arm=args.arm)


if __name__=='__main__':main()
