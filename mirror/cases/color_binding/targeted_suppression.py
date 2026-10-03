"""Plan 39: matched retention and incoming-relative cross-only repair."""
import argparse
from dataclasses import asdict; from dataclasses import replace
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding import ranking_followup_pilot as prev
from mirror.cases.color_binding import ranking_comparison as rc
from mirror.cases.color_binding import routing_relative_pilot as prior
from mirror.cases.color_binding import routing_common_noise_pilot as cn
from mirror.cases.color_binding import routing_budget_replication as old
from mirror.core.repair import _state_hash; from mirror.core.repair import _correct_margins; from mirror.core.repair import legacy_ranking_loss
from mirror.cases.color_binding.behavioral_pilot import lines

OUT = ROOT/'clip/interbind_targeted_suppression_20260925'
PLAN = ROOT/'FSE_VLM/plan/39_fair_constraints_and_targeted_suppression.md'
SEEDS = (42,43,44)
SCALES = (1,10,100)
MULTIPLIERS = (1,4,16)
STAGE_CONTROLS = ('continue_R','continue_RG','retain_only')


def configure():
    prior.configure(); cn.available_gpu()


def hash_array(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def prior_models():
    r=read(prev.OUT/'checkpoints_frozen.json')['models']
    return [dict(x) for x in r if x['name'] in prev.TEMPLATE_ARMS]


def freeze():
    prev.verify()
    paths=[PLAN,Path(__file__),Path(__file__).parent/'tests/test_targeted_suppression.py',
        prev.OUT/'checkpoints_frozen.json',prev.PROTOCOL,
        prior.OUT/'normalization.json',prior.OUT/'training_config.json',
        prev.DEV/'rows.jsonl',prev.DEV/'features/complete.json']
    paths += [Path(__file__).with_name(n+'.py') for n in
        ('ranking_followup_pilot','ranking_comparison','routing_relative_pilot',
         'routing_common_noise_pilot','routing_budget_replication','repair')]
    for r in prior_models():
        paths.extend([Path(r['checkpoint']),Path(r['checkpoint']).with_name('complete.json')])
    dump(OUT/'protocol.json',dict(inputs={str(p):sha(p) for p in paths},
        scales=SCALES,multipliers=MULTIPLIERS,seeds=SEEDS,epochs_per_stage=36,
        updates_per_stage=1944,checkpoint_selection='fixed_last',
        stage_reference='incoming selected template ranking only',
        stage_target='absolute complete-context unwanted cross-effect only',
        new_training_images=False,new_training_caption_strings=False,
        confirmation_previously_observed=True,new_test_conditions_scored=False,
        manuscript_modified=False))
    # Verify immutable matching evidence rather than equating a shared forward to loss use.
    checks=[]
    for seed in SEEDS:
        pair=[r for r in prior_models() if r['seed']==seed]
        assert len(pair)==2
        for key in ('initial_state_hash','final_rng_sha256','schedule_sha256',
                    'representation_sha256','updates'):
            assert len({r[key] for r in pair})==1,(seed,key)
        histories=[lines(Path(r['checkpoint']).with_name('history.jsonl')) for r in pair]
        assert [x['rows'] for x in histories[0]]==[x['rows'] for x in histories[1]]
        assert len(histories[0])==1944
        checks.append(dict(seed=seed,matched_initialization=True,matched_rows=True,
            matched_representation=True,matched_dropout=True,matched_updates=True))
    dump(OUT/'fairness_audit.json',dict(checks=checks,
        identical_training_inputs=True,identical_optimization_budget=True,
        identical_loss_constraints=False,
        plain_ranking_uses=['four-way captions and CE labels','original task embedding anchor'],
        is_additional_loss_use=['object-caption positive-margin retention',
            'natural-caption embedding retention','binding and response retention',
            'correct-caption margin retention','specified interaction and preference targets'],
        exact_control='CE plus the full unchanged IS retention objective',
        historical_hyperparameter_search_equal=False))
    print('PROTOCOL_FROZEN',sha(OUT/'protocol.json'),flush=True)


def verify():
    p=read(OUT/'protocol.json');verify_files(p['inputs']);return p


def cycle(seed,stage=False):
    # 36 is divisible by four: continuation covers every form nine times as well.
    offsets=np.random.default_rng(seed+380000).integers(0,4,size=1280)
    return np.stack([(offsets+e)%4 for e in range(36 if stage else 0,72 if stage else 36)])


def incoming_hinges(d,e,b,m,o,dr,er,br,mr,orr):
    """No frozen max, no positive target, no binding-growth term."""
    return dict(binding_keep=F.relu(dr-d-prior.EPS),
        response_keep=F.relu(er-e-prior.EPS),
        preference_keep=F.relu(b.abs()-br.abs()-prior.EPS),
        caption_guard=F.relu(mr-m-prior.EPS)*(mr>0),
        object_guard=F.relu(orr-o-prior.EPS)*(orr>0))


def stage_components(model,images,base,natural,reference,spec,contexts,cfg,ids):
    changed=model(torch.cat((base,natural),1))
    scores=images@changed[:,:6].transpose(1,2)
    ranked=images@reference[:,:6].transpose(1,2)
    d,c,e,b,_=prior.measurements(scores[:,:,:4],spec,contexts)
    dr,cr,er,br,_=prior.measurements(ranked[:,:,:4],spec,contexts)
    m=_correct_margins(scores[:,:,:4],(0,1,2,3))/spec.unit
    mr=_correct_margins(ranked[:,:,:4],(0,1,2,3))/spec.unit
    o=(scores[:,:,4]-scores[:,:,5])/spec.unit
    ro=(ranked[:,:,4]-ranked[:,:,5])/spec.unit
    raw=incoming_hinges(d,e,b,m,o,dr,er,br,mr,ro)
    raw.update(cross=c.abs(),natural=(changed[:,6:]-reference[:,6:]).square().sum(-1),
        drift=(changed[:,:6]-reference[:,:6]).square().sum(-1))
    parts={k:prior.worse_layout(v) for k,v in raw.items()}
    _,terms=legacy_ranking_loss(scores,changed[:,:6],base,cfg.legacy_ranking,ids)
    parts.update(terms)
    return parts,scores


def stage_guard(parts,weights):
    keys=('binding_keep','response_keep','caption_guard','object_guard')
    return 4*(sum(weights[k]*parts[k] for k in keys)+
        weights['preference']*parts['preference_keep']+parts['natural']+parts['drift'])


def stage_objective(parts,ce,weights,arm,anchor_weight=.2,multiplier=0):
    guard=stage_guard(parts,weights)
    if arm=='continue_R':return ce+anchor_weight*parts['ranking_anchor']
    if arm=='continue_RG':return ce+guard
    if arm=='retain_only':return guard
    assert arm.startswith('suppress_') and multiplier>0
    return guard+multiplier*weights['cross']*parts['cross']


def candidate(name,scale=100,multiplier=0,stage=False):
    return dict(name=name,scale=scale,multiplier=multiplier,stage=stage)


def selected_start(seed):
    p=read(OUT/'fairness_models.json')
    return next(r for r in p['models'] if r['name']=='R_selected' and r['seed']==seed)


def train(cand,seed,loaded,forms):
    name=cand['name'];dest=OUT/'runs'/f'seed{seed}'/name
    if (dest/'complete.json').exists():
        done=read(dest/'complete.json');verify_files(done['files']);return done
    dest.mkdir(parents=True,exist_ok=False)
    cache,spec,contexts,cfg=loaded;cfg=replace(cfg,seed=seed,epochs=36)
    weights=read(prior.OUT/'normalization.json')['weights'];stage=cand['stage']
    initial_path=Path(selected_start(seed)['checkpoint']) if stage else old.OUT/f'seed{seed}/IS/initial.pt'
    initial=torch.load(initial_path,map_location='cpu',weights_only=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=cn.make_model(cfg,'cuda');model.load_state_dict(initial['state_dict'])
    ih=_state_hash(model.state_dict())
    torch.save(dict(state_dict=initial['state_dict'],source=str(initial_path),
        source_sha256=sha(initial_path),state_hash=ih),dest/'initial.pt')
    reference=None
    if stage:
        model.eval();nat=cache.natural_texts[:,None].expand(-1,4,-1,-1)
        incoming_input=torch.cat((forms,nat),2)
        with torch.inference_mode():
            reference=torch.cat([model(incoming_input[j:j+64]) for j in range(0,len(forms),64)])
        reference=reference.clone()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    schedule=old.schedule_for(seed,72)[36:] if stage else old.schedule_for(seed)
    reps=cycle(seed,stage);history=[];begun=time.monotonic()
    for ei,order in enumerate(schedule):
        epoch=ei+37 if stage else ei+1
        choice=torch.as_tensor(np.repeat(reps[ei],2),device='cuda')
        current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),choice])
        model.train();torch.manual_seed(seed*10000+epoch);torch.cuda.manual_seed_all(seed*10000+epoch)
        for j in range(0,1280,24):
            ids=prior.paired_rows(torch.as_tensor(order[j:j+24],device='cuda'))
            if stage:
                parts,scores=stage_components(model,cache.images[ids],current.texts[ids],
                    cache.natural_texts[ids],reference[ids,choice[ids]],spec,contexts,cfg,ids)
                ce=F.cross_entropy(cand['scale']*scores[:,:,:4].reshape(-1,4),
                    torch.arange(4,device='cuda').repeat(len(ids)))
                guard=stage_guard(parts,weights)
                loss=stage_objective(parts,ce,weights,name,cfg.legacy_ranking.anchor_weight,cand['multiplier'])
            else:
                captured=[];hook=model.register_forward_hook(lambda _m,_a,z:captured.append(z))
                try:parts=prior.components(model,current,spec,contexts,cfg,ids)
                finally:hook.remove()
                assert len(captured)==1
                scores=cache.images[ids]@captured[0][:,:4].transpose(1,2)
                ce=F.cross_entropy(cand['scale']*scores.reshape(-1,4),torch.arange(4,device='cuda').repeat(len(ids)))
                guard=prior.objective(parts,weights,'G')
                loss=ce+guard if name.startswith('RG_') else ce+cfg.legacy_ranking.anchor_weight*parts['ranking_anchor']
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(grad)
            opt.step();history.append(dict(epoch=epoch,rows=ids.tolist(),loss=float(loss.detach()),
                ce=float(ce.detach()),guard=float(guard.detach()),gradient_norm=float(grad),
                components={k:float(v.detach()) for k,v in parts.items()}))
        if (ei+1)%6==0:print('TRAIN',seed,name,ei+1,'seconds',round(time.monotonic()-begun,1),flush=True)
    assert len(history)==1944
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        configuration=asdict(cfg),candidate=cand,optimizer=opt.state_dict(),seed=seed,
        initial_state_hash=ih,epochs=36,total_epochs=72 if stage else 36,updates=1944,
        selection='fixed_last',protocol_sha256=sha(OUT/'protocol.json')),dest/'last.pt')
    jsonl(dest/'history.jsonl',history)
    done=dict(**cand,seed=seed,checkpoint=str(dest/'last.pt'),sha256=sha(dest/'last.pt'),
        initial_state_hash=ih,initial_path=str(initial_path),initial_sha256=sha(initial_path),
        final_rng_sha256=hash_array(torch.cuda.get_rng_state().cpu().numpy()),
        schedule_sha256=hash_array(np.asarray(schedule,dtype=np.int64)),
        representation_sha256=hash_array(reps),updates=1944,total_epochs=72 if stage else 36,
        first_components=history[0]['components'],seconds=time.monotonic()-begun,
        files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    dump(dest/'complete.json',done);return done


def matched(rows):
    for seed in SEEDS:
        group=[r for r in rows if r['seed']==seed]
        if not group:continue
        for key in ('initial_state_hash','final_rng_sha256','schedule_sha256','representation_sha256','updates'):
            assert len({r[key] for r in group})==1,(seed,key)
        assert all(r['first_components']==group[0]['first_components'] for r in group)


def select_temperature(records,family):
    opts=[r for r in records if r['family']==family]
    eligible=[r for r in opts if r['eligible']]
    best=sorted(eligible or opts,key=lambda r:(-r['exchange_accuracy'],r['scale']))[0]
    return dict(**best,any_eligible=bool(eligible),fallback=not bool(eligible))


def fairness():
    verify();configure();loaded=prior.load_training('cuda');forms=prev.template_cache(loaded[0])
    regs=[]
    for scale in SCALES:
        if scale==100:
            reg=next(r for r in prior_models() if r['seed']==42 and r['name']=='R_templates')
            regs.append(dict(reg,name='R_s100',scale=100))
        else:regs.append(train(candidate(f'R_s{scale}',scale),42,loaded,forms))
        regs.append(train(candidate(f'RG_s{scale}',scale),42,loaded,forms))
    is42=next(r for r in prior_models() if r['seed']==42 and r['name']=='IS_templates')
    matched([*regs,is42])
    dev=prev.development([dict(name='F',seed=0,checkpoint=None),*regs,is42],OUT/'fairness_development.jsonl')
    means=prev.means(dev);frozen={r['view']:r for r in dev if r['name']=='F'};choices=[]
    for reg in regs:
        m=means[reg['name']];views=[r for r in dev if r['name']==reg['name']]
        eligible=all(min(r['word1_accuracy'],r['word2_accuracy'])>=.99 and
            r['object_guard']>=frozen[r['view']]['object_guard']-.01 for r in views)
        choices.append(dict(name=reg['name'],family=reg['name'].split('_')[0],scale=reg['scale'],
            eligible=eligible,exchange_accuracy=m['exchange_accuracy']))
    selected={f:select_temperature(choices,f) for f in ('R','RG')}
    dump(OUT/'fairness_selection.json',dict(selected=selected,candidates=choices,
        new_checkpoint_test_scores_seen=False,protocol_sha256=sha(OUT/'protocol.json')))
    models=[dict(name='F',seed=0,checkpoint=None)]
    for seed in SEEDS:
        si=next(r for r in prior_models() if r['seed']==seed and r['name']=='IS_templates')
        group=[si];models.append(si)
        for family in ('R','RG'):
            c=selected[family];name=c['name']
            if seed==42:reg=next(r for r in regs if r['name']==name)
            elif family=='R' and c['scale']==100:
                reg=next(r for r in prior_models() if r['seed']==seed and r['name']=='R_templates')
            else:reg=train(candidate(name,c['scale']),seed,loaded,forms)
            group.append(reg);models.append(dict(reg,name=family+'_selected',scale=c['scale'],source_name=name))
        matched(group)
    dump(OUT/'fairness_models.json',dict(models=models,development_candidates=regs,
        selection_sha256=sha(OUT/'fairness_selection.json'),matched_constraints_for_RG=True,
        paper_not_modified=True))
    print('FAIRNESS_SELECTED',selected,flush=True)


def stage_gate(means,name):
    a=means[name];r=means['R_selected'];rr=means['continue_R']
    checks=dict(cross_incoming=a['cross']<=.9*r['cross'],cross_continued=a['cross']<=.9*rr['cross'],
        binding=a['binding']>=.99*r['binding'],exchange_incoming=a['exchange_accuracy']>=r['exchange_accuracy']-.01,
        exchange_continued=a['exchange_accuracy']>=rr['exchange_accuracy']-.01,
        word=min(a['word1_accuracy'],a['word2_accuracy'])>=.98,
        object=a['object_guard']>=r['object_guard']-.01)
    return dict(name=name,passed=all(checks.values()),checks=checks,cross=a['cross'])


def stage():
    verify();configure();loaded=prior.load_training('cuda');forms=prev.template_cache(loaded[0])
    scale=read(OUT/'fairness_selection.json')['selected']['R']['scale']
    cands=[candidate(n,scale,stage=True) for n in STAGE_CONTROLS]+[
        candidate(f'suppress_{m}',scale,m,True) for m in MULTIPLIERS]
    regs=[train(c,42,loaded,forms) for c in cands];matched(regs)
    refs=[r for r in read(OUT/'fairness_models.json')['models'] if r['seed'] in (0,42)]
    dev=prev.development(refs+regs,OUT/'stage_development.jsonl');means=prev.means(dev)
    gates=[stage_gate(means,f'suppress_{m}') for m in MULTIPLIERS]
    eligible=[g for g in gates if g['passed']]
    chosen=sorted(eligible or gates,key=lambda g:(g['cross'],int(g['name'].split('_')[1])))[0]
    dump(OUT/'stage_selection.json',dict(candidates=gates,chosen=chosen,replicate=bool(eligible),
        means=means,new_checkpoint_test_scores_seen=False,
        protocol_sha256=sha(OUT/'protocol.json')))
    dump(OUT/'stage_pilot_models.json',regs)
    if eligible:
        for seed in (43,44):
            regs.extend(train(c,seed,loaded,forms) for c in cands if c['name'] in (*STAGE_CONTROLS,chosen['name']))
    matched(regs)
    chosen_regs=[dict(r,name='suppression_selected' if r['name']==chosen['name'] else r['name'],
        source_name=r['name']) for r in regs if r['name'] in (*STAGE_CONTROLS,chosen['name'])]
    base=read(OUT/'fairness_models.json')['models']
    dump(OUT/'checkpoints_frozen.json',dict(models=base+chosen_regs,
        all_stage_candidates=regs,selection_sha256=sha(OUT/'stage_selection.json'),
        fairness_selection_sha256=sha(OUT/'fairness_selection.json'),
        stage_replication=bool(eligible),new_checkpoint_test_scores_seen=False))
    print('STAGE_SELECTED',chosen,'REPLICATE',bool(eligible),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','fairness','stage'))
    args=p.parse_args();log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as e:log(OUT,'failed',stage=args.action,error=repr(e));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
