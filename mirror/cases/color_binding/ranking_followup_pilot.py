"""Bounded development-only continuations and template-exposure pilot (plan 38)."""
import argparse
from dataclasses import asdict; from dataclasses import replace
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding import ranking_comparison as rc
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.cases.color_binding import routing_common_noise_pilot as cn
from mirror.cases.color_binding import routing_budget_replication as old
from mirror.core.repair import _state_hash; from mirror.core.repair import _correct_margins; from mirror.core.repair import legacy_ranking_loss
from mirror.core.metrics import adapt; from mirror.core.metrics import bank_arrays; from mirror.core.metrics import routing
from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.core.encoders import load_subject; from mirror.core.encoders import legacy_unit

OUT = ROOT / 'clip/interbind_ranking_followup_20260925'
PLAN = ROOT / 'FSE_VLM/plan/38_ranking_transfer_and_repair_followup.md'
PROTOCOL = OUT / 'training_protocol_v2.json'
SEEDS = (42, 43, 44)
STAGE_ARMS = ('R_to_R', 'R_to_G', 'R_to_IS')
TEMPLATE_ARMS = ('R_templates', 'IS_templates')
TRAIN = ROOT / 'clip/interbind_routing_repair_pilot_20260923'
DEV = rc.DEV


def configure():
    prior.configure()
    cn.available_gpu()


def old_registry():
    regs = read(rc.OUT / 'checkpoints_frozen.json')['models']
    verify_files({r['checkpoint']: r['sha256'] for r in regs})
    return [dict(name='F', seed=0, checkpoint=None), *regs]


def freeze():
    paths = [PLAN, Path(__file__), Path(__file__).with_name('ranking_transfer_extension.py'),
             Path(__file__).with_name('ranking_followup_evaluate.py'),
             Path(__file__).parent/'tests/test_ranking_followup.py',
             rc.OUT/'checkpoints_frozen.json', rc.OUT/'development/selection.json',
             prior.OUT/'normalization.json', prior.OUT/'training_config.json',
             TRAIN/'training_rows.jsonl', TRAIN/'features/text_groups.jsonl',
             DEV/'rows.jsonl', DEV/'features/complete.json']
    paths += [Path(__file__).with_name(n+'.py') for n in
              ('ranking_comparison', 'routing_relative_pilot', 'routing_common_noise_pilot',
               'routing_budget_replication', 'completion_metrics', 'repair', 'strengthening_statistics')]
    dump(PROTOCOL, dict(inputs={str(p):sha(p) for p in paths},
        supersedes_sha256=sha(OUT/'training_protocol.json'),
        pretraining_correction='Transfer provenance check: the eight audit nouns are a subset of ten training nouns; all 24 proposed pairs remain untrained. No model outcomes accessed.',
        original_models=old_registry(), plan_sha256=sha(PLAN), seeds=SEEDS,
        stage_arms=STAGE_ARMS, template_arms=TEMPLATE_ARMS, epochs=36, updates=1944,
        checkpoint_selection='fixed_last', development=888, training_anchors=640,
        stage_reference='incoming ranking and frozen; binding/response never targeted downward',
        template_forms=['pooled', 'template0', 'template1', 'template2'],
        new_template_strings=False, new_test_outcomes_seen=False,
        manuscript_modified=False, pilot_only_until_development_gate=True))
    print('TRAINING_PROTOCOL_FROZEN', sha(PROTOCOL), flush=True)


def verify():
    p = read(PROTOCOL)
    verify_files(p['inputs'])
    verify_files({r['checkpoint']:r['sha256'] for r in p['original_models'] if r['checkpoint']})
    return p


def encode():
    """Only existing training/dev strings, never new transfer test inputs."""
    verify(); configure()
    groups = lines(TRAIN/'features/text_groups.jsonl')
    strings = list(dict.fromkeys(t for g in groups for t in g['templates']))
    scorer = load_subject(rc.MODEL, device='cuda')
    features = scorer.encode_texts(strings, batch_size=128).numpy()
    ix = {s:i for i,s in enumerate(strings)}
    errors=[]
    original=np.load(TRAIN/'features/texts.npy')
    for g in groups:
        pooled=legacy_unit(torch.from_numpy(features[[ix[t] for t in g['templates']]]).mean(0)).numpy()
        errors.append(float(np.max(np.abs(pooled-original[g['index']]))))
    assert max(errors)<2e-6
    with (OUT/'training_template_features.npy').open('xb') as f:np.save(f,features)
    dump(OUT/'training_template_strings.json',strings)
    dump(OUT/'training_encoding.json',dict(files={str(OUT/n):sha(OUT/n) for n in
        ('training_template_features.npy','training_template_strings.json')},
        max_pool_replay_error=max(errors), strings=len(strings), new_training_strings=False))
    print('TEMPLATE_ENCODING_COMPLETE',len(strings),max(errors),flush=True)


def template_cache(cache):
    verify_files(read(OUT/'training_encoding.json')['files'])
    features=np.load(OUT/'training_template_features.npy')
    lookup={s:i for i,s in enumerate(read(OUT/'training_template_strings.json'))}
    groups=lines(TRAIN/'features/text_groups.jsonl')
    index=lines(TRAIN/'features/index.jsonl')
    forms=cache.texts[:,None].repeat(1,4,1,1)
    for j,row in enumerate(index):
        for color in range(2):
            for state in range(4):
                templates=groups[row['text_indices'][color*4+state]]['templates']
                assert len(templates)==3
                for form,text in enumerate(templates,1):
                    forms[j*4+color*2:j*4+color*2+2,form,state] = torch.tensor(
                        features[lookup[text]],device=cache.texts.device)
    assert torch.equal(forms[:,0],cache.texts)
    assert torch.equal(forms[::2],forms[1::2])
    return forms


def representation_schedule(seed):
    offsets=np.random.default_rng(seed+380000).integers(0,4,size=1280)
    schedule=np.stack([(offsets+epoch)%4 for epoch in range(36)])
    assert all(np.all((schedule==k).sum(0)==9) for k in range(4))
    return schedule


def incoming_constraints(d,c,e,b,df,cf,ef,bf,dr,cr,er,br,k,tau,beta):
    """The incoming positive interaction cannot be reduced to a low absolute target."""
    binding_floor=torch.maximum(torch.maximum(df,dr),df.new_tensor(k))
    response_floor=torch.maximum(torch.maximum(ef,er),ef.new_tensor(k-tau))
    cross_cap=torch.minimum(torch.minimum(cf.abs(),cr.abs()),cf.new_tensor(tau))
    preference_cap=torch.minimum(torch.minimum(bf.abs(),br.abs()),bf.new_tensor(beta))
    return dict(binding=F.relu(binding_floor-d-prior.EPS),
                cross=F.relu(c.abs()-cross_cap-prior.EPS),
                response=F.relu(response_floor-e-prior.EPS),
                preference=F.relu(b.abs()-preference_cap-prior.EPS))


def stage_components(model,cache,spec,contexts,cfg,ids,incoming):
    images=cache.images[ids];base=cache.texts[ids];natural=cache.natural_texts[ids]
    changed=model(torch.cat((base,natural),1))
    scores=images@changed[:,:6].transpose(1,2)
    frozen=images@base.transpose(1,2)
    ranked=images@incoming[ids].transpose(1,2)
    d,c,e,b,m=prior.measurements(scores[:,:,:4],spec,contexts)
    df,cf,ef,bf,mf=prior.measurements(frozen[:,:,:4],spec,contexts)
    dr,cr,er,br,mr=prior.measurements(ranked[:,:,:4],spec,contexts)
    th=spec.spec['thresholds']['fractions'];k,tau,beta=(th[n] for n in ('K','tau','beta'))
    raw=incoming_constraints(d,c,e,b,df,cf,ef,bf,dr,cr,er,br,k,tau,beta)
    raw['endpoint']=F.relu(torch.maximum(torch.maximum(mf,mr),mf.new_tensor(k-tau-beta))-m-prior.EPS)
    current=_correct_margins(scores[:,:,:4],(0,1,2,3))/spec.unit
    fm=_correct_margins(frozen[:,:,:4],(0,1,2,3))/spec.unit
    rm=_correct_margins(ranked[:,:,:4],(0,1,2,3))/spec.unit
    ref=torch.maximum(fm,rm)
    raw['caption_guard']=F.relu(ref-current-prior.EPS)*(ref>0)
    refd=torch.maximum(df,dr)
    raw['binding_keep']=F.relu(refd-d-prior.EPS)*(refd>0)
    raw['response_keep']=F.relu(torch.maximum(ef,er)-e-prior.EPS)
    fo=(frozen[:,:,4]-frozen[:,:,5])/spec.unit
    ro=(ranked[:,:,4]-ranked[:,:,5])/spec.unit
    refobj=torch.maximum(fo,ro)
    raw['object_guard']=F.relu(refobj-(scores[:,:,4]-scores[:,:,5])/spec.unit-prior.EPS)*(refobj>0)
    raw['natural']=(changed[:,6:]-F.normalize(natural,dim=-1)).square().sum(-1)
    raw['drift']=(changed[:,:6]-F.normalize(base,dim=-1)).square().sum(-1)
    parts={key:prior.worse_layout(value) for key,value in raw.items()}
    rank,terms=legacy_ranking_loss(scores,changed[:,:6],base,cfg.legacy_ranking,ids)
    parts.update(terms);parts['ranking']=rank
    ce=F.cross_entropy(100*scores[:,:,:4].reshape(-1,4),torch.arange(4,device=scores.device).repeat(len(ids)))
    return parts,ce


def train(arm,seed,loaded,forms):
    dest=OUT/'runs'/f'seed{seed}'/arm
    if (dest/'complete.json').exists():
        done=read(dest/'complete.json');verify_files(done['files']);return done
    dest.mkdir(parents=True,exist_ok=False)
    cache,spec,contexts,cfg=loaded;cfg=replace(cfg,seed=seed,epochs=36)
    weights=read(prior.OUT/'normalization.json')['weights']
    stage=arm in STAGE_ARMS
    start=next(r for r in old_registry() if r['name']=='R_s100' and r['seed']==seed)['checkpoint'] if stage else str(old.OUT/f'seed{seed}/IS/initial.pt')
    initial=torch.load(start,map_location='cpu',weights_only=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=cn.make_model(cfg,'cuda');model.load_state_dict(initial['state_dict'])
    ih=_state_hash(model.state_dict())
    torch.save(dict(state_dict=initial['state_dict'],state_hash=ih,source=start,sha256=sha(start)),dest/'initial.pt')
    with torch.inference_mode():
        model.eval()
        incoming=torch.cat([model(cache.texts[i:i+128]) for i in range(0,len(cache.texts),128)]) if stage else None
    # Clone outside inference mode: reference tensors may be saved for backward.
    if incoming is not None:incoming=incoming.clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    schedules=old.schedule_for(seed,72)[36:] if stage else old.schedule_for(seed)
    rep=representation_schedule(seed)
    history=[];begun=time.monotonic()
    for ei,order in enumerate(schedules):
        epoch=ei+37 if stage else ei+1
        if not stage:
            choice=torch.as_tensor(np.repeat(rep[ei],2),device='cuda')
            current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),choice])
        model.train();torch.manual_seed(seed*10000+epoch);torch.cuda.manual_seed_all(seed*10000+epoch)
        for start_ix in range(0,1280,24):
            ids=prior.paired_rows(torch.tensor(order[start_ix:start_ix+24],device='cuda'))
            if stage:
                parts,ce=stage_components(model,cache,spec,contexts,cfg,ids,incoming)
            else:
                captured=[]
                hook=model.register_forward_hook(lambda _m,_a,z:captured.append(z))
                try:parts=prior.components(model,current,spec,contexts,cfg,ids)
                finally:hook.remove()
                assert len(captured)==1
                scores=(cache.images[ids]@captured[0][:,:6].transpose(1,2))[:,:,:4]
                ce=F.cross_entropy(100*scores.reshape(-1,4),torch.arange(4,device='cuda').repeat(len(ids)))
            guard=prior.objective(parts,weights,'G')
            loss=(ce+cfg.legacy_ranking.anchor_weight*parts['ranking_anchor'] if arm in ('R_to_R','R_templates')
                  else guard if arm=='R_to_G' else cn.objective(parts,weights))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(grad)
            optimizer.step()
            history.append(dict(epoch=epoch,rows=ids.tolist(),loss=float(loss.detach()),
                gradient_norm=float(grad),ce=float(ce.detach()),guard=float(guard.detach()),
                components={k:float(v.detach()) for k,v in parts.items()}))
        if (ei+1)%6==0:print('TRAIN',seed,arm,ei+1,'seconds',round(time.monotonic()-begun,1),flush=True)
    assert len(history)==1944
    rh=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        seed=seed,arm=arm,configuration=asdict(cfg),optimizer=optimizer.state_dict(),
        initial_state_hash=ih,epochs=36,total_epochs=72 if stage else 36,updates=1944,
        selection='fixed_last',protocol_sha256=sha(PROTOCOL)),dest/'last.pt')
    jsonl(dest/'history.jsonl',history)
    done=dict(name=arm,seed=seed,checkpoint=str(dest/'last.pt'),sha256=sha(dest/'last.pt'),
        incoming_checkpoint=start,incoming_sha256=sha(start),initial_state_hash=ih,
        final_rng_sha256=rh,updates=1944,total_epochs=72 if stage else 36,
        schedule_sha256=hashlib.sha256(np.array(schedules,dtype=np.int64).tobytes()).hexdigest(),
        representation_sha256=None if stage else hashlib.sha256(rep.tobytes()).hexdigest(),
        first_components=history[0]['components'],seconds=time.monotonic()-begun,
        files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    dump(dest/'complete.json',done)
    return done


def development(regs, destination):
    """Fixed red/blue development gate; test data is not opened here."""
    strings=read(OUT/'training_template_strings.json');lookup={s:i for i,s in enumerate(strings)}
    feature=np.load(OUT/'training_template_features.npy');groups=lines(DEV/'features/text_groups.jsonl')
    alltexts=np.load(DEV/'features/texts.npy')
    results=[]
    for view in ('canvas','swapped_canvas'):
        v,t,idx=bank_arrays(DEV/'features','routing','red-blue',view)
        assert len(idx)==888
        forms=np.stack([feature[[lookup[s] for ti in r['text_indices'][:4] for s in groups[ti]['templates']]].reshape(4,3,768).transpose(1,0,2) for r in idx])
        objects=alltexts[np.array([r['text_indices'][8:10] for r in idx])]
        for reg in regs:
            x=np.einsum('nid,njd->nij',v,adapt(t,reg['checkpoint']))
            mm=routing(x)
            individual=np.einsum('nid,nkjd->nkij',v,adapt(forms,reg['checkpoint']))
            mi=routing(individual.reshape(-1,4,4))
            om=np.einsum('nid,njd->nij',v,adapt(objects,reg['checkpoint']))
            results.append(dict(name=reg['name'],seed=reg['seed'],view=view,
                **{k:float(a.mean()) for k,a in mm.items() if not k.startswith('contrast/')},
                individual_exchange=float(mi['exchange_accuracy'].mean()),
                object_guard=float((om[:,:,0]>om[:,:,1]).mean())))
    jsonl(destination,results)
    return results


def means(rows):
    out={}
    for name in dict.fromkeys(r['name'] for r in rows):
        group=[r for r in rows if r['name']==name]
        out[name]={k:float(np.mean([r[k] for r in group])) for k in group[0] if k not in ('name','seed','view')}
    return out


def gates(rows):
    m=means(rows);a=m['R_to_IS'];r=m['R_s100'];rr=m['R_to_R']
    stage=dict(exchange_vs_incoming=a['exchange_accuracy']>=r['exchange_accuracy']-.01,
        exchange_vs_continued=a['exchange_accuracy']>=rr['exchange_accuracy']-.01,
        cross_vs_incoming=a['cross']<=.9*r['cross'],cross_vs_continued=a['cross']<=.9*rr['cross'],
        incoming_binding=a['binding']>=.99*r['binding'],
        word_accuracy=min(a['word1_accuracy'],a['word2_accuracy'])>=.98,
        object_guard=a['object_guard']>=r['object_guard']-.01)
    t=m['IS_templates'];oldis=m['IS']
    template=dict(individual_gain=t['individual_exchange']>=oldis['individual_exchange']+.02,
        pooled_retention=t['exchange_accuracy']>=oldis['exchange_accuracy']-.02,
        word_accuracy=min(t['word1_accuracy'],t['word2_accuracy'])>=.98,
        object_guard=t['object_guard']>=oldis['object_guard']-.01)
    return dict(stage=dict(passed=all(stage.values()),checks=stage),
                templates=dict(passed=all(template.values()),checks=template),means=m)


def check_matches(regs):
    for seed in sorted({r['seed'] for r in regs}):
        for family in (STAGE_ARMS,TEMPLATE_ARMS):
            group=[r for r in regs if r['seed']==seed and r['name'] in family]
            if not group:continue
            for key in ('initial_state_hash','final_rng_sha256','schedule_sha256','updates','representation_sha256'):
                assert len({r[key] for r in group})==1,(seed,family,key)
            assert all(r['first_components']==group[0]['first_components'] for r in group)


def pilot():
    verify();configure();loaded=prior.load_training('cuda');forms=template_cache(loaded[0])
    regs=[train(a,42,loaded,forms) for a in (*STAGE_ARMS,*TEMPLATE_ARMS)]
    check_matches(regs)
    references=[r for r in old_registry() if r['seed'] in (0,42)]
    rows=development(references+regs,OUT/'development_seed42.jsonl')
    gate=gates(rows)
    dump(OUT/'seed42_decision.json',dict(**gate,checkpoint_selection='fixed_last',
        new_test_scores_seen=False,training_protocol_sha256=sha(PROTOCOL)))
    dump(OUT/'pilot_models.json',regs)
    print('PILOT_DECISION',gate,flush=True)


def expand():
    verify();configure();decision=read(OUT/'seed42_decision.json');regs=read(OUT/'pilot_models.json')
    arms=(*STAGE_ARMS,) if decision['stage']['passed'] else ()
    if decision['templates']['passed']:arms=(*arms,*TEMPLATE_ARMS)
    if arms:
        loaded=prior.load_training('cuda');forms=template_cache(loaded[0])
        for seed in (43,44):
            regs.extend(train(arm,seed,loaded,forms) for arm in arms)
    check_matches(regs)
    dump(OUT/'checkpoints_frozen.json',dict(models=regs,original_models=old_registry(),
        decision_sha256=sha(OUT/'seed42_decision.json'),training_protocol_sha256=sha(PROTOCOL),
        new_test_scores_seen=False,matched_updates=True,matched_rng=True,matched_initialization=True,
        completed_seeds={a:[r['seed'] for r in regs if r['name']==a] for a in (*STAGE_ARMS,*TEMPLATE_ARMS)}))
    print('CHECKPOINTS_FROZEN',len(regs),sha(OUT/'checkpoints_frozen.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','encode','pilot','expand'))
    args=p.parse_args();log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as e:log(OUT,'failed',stage=args.action,error=repr(e));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
