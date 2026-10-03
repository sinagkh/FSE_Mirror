"""Backbone-generic implementation of the fixed primary template recipe."""
import argparse
from dataclasses import replace; from dataclasses import asdict
from pathlib import Path
import hashlib
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.cases.color_binding.repair_trainbank import source; from mirror.cases.color_binding.repair_trainbank import annotations_for; from mirror.cases.color_binding.repair_trainbank import lines
from mirror.cases.color_binding.rendering import captions
from mirror.cases.color_binding.completion_confirmation_encode import render
from mirror.cases.color_binding import routing_repair_pilot as original
from mirror.cases.color_binding import routing_relative_pilot as relative
from mirror.cases.color_binding import routing_common_noise_pilot as cn
from mirror.cases.color_binding import routing_budget_replication as budget
from mirror.cases.color_binding import ranking_followup_pilot as templates
from mirror.cases.color_binding import targeted_suppression as selection
from mirror.cases.color_binding.routing_adequacy import preference_spec
from mirror.core.specifications import compile_requirement
from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import RepairCache; from mirror.core.repair import cache_sha256; from mirror.core.repair import derive_scales; from mirror.core.repair import historical_ranking; from mirror.core.repair import _check; from mirror.core.repair import _state_hash
from mirror.core.metrics import adapt; from mirror.core.metrics import routing

OUT=ROOT/'clip/fse_pre_writing_20260926/B2_routing'
MODELS=('openai_clip_l14','google_siglip_b16_224')
SEEDS=(42,43,44)
TRAIN=original.OUT/'training_rows.jsonl'
DEV=ROOT/'data/color_binding/object_pairs/rows.jsonl'
TEST=ROOT/'data/color_binding/confirmation/same_rule_confirmation/rows.jsonl'
CAL=ROOT/'data/models/calibration/calibration_frozen.json'

def ah(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def freeze(model):
    dest=OUT/model;assert not (dest/'protocol.json').exists()
    paths=[Path(__file__),TRAIN,DEV,TEST,CAL,relative.OUT/'training_config.json',relative.OUT/'normalization.json',
        templates.PROTOCOL,selection.OUT/'fairness_selection.json',DEFAULT_REGISTRY,DEFAULT_REGISTRY.with_name('activation_addendum.json'),
        ROOT/'FSE_VLM/plan/49_pre_writing_experiments.md',ROOT/'FSE_VLM/plan/49a_execution_review.md']
    paths.extend(Path(__file__).with_name(n+'.py') for n in ('routing_relative_pilot','routing_common_noise_pilot',
        'routing_repair_pilot','ranking_followup_pilot','routing_budget_replication','targeted_suppression',
        'completion_confirmation_encode'))
    paths.extend(ROOT/'mirror/core'/(n+'.py') for n in
                 ('repair','specifications','encoders','features','metrics'))
    row=next(r for r in read(DEFAULT_REGISTRY)['subjects'] if r['id']==model)
    paths.extend(Path(f['path']) for f in row['files'])
    dump(dest/'protocol.json',dict(study='B2 fixed primary routing recipe, new backbone',model=model,
        inputs={str(p):sha(p) for p in paths},seeds=SEEDS,updates=1944,epochs=36,selection='fixed_last',
        adapter=dict(rank=64,alpha=64,dropout=.05,dimension='base joint embedding width',dropout_rule='same common noise shared across captions and both layouts'),
        inputs_matched='same640 source pairs,2 trained color pairs,2 layouts,4 template forms,5 natural captions and2 object captions',
        objective='Unchanged relative.components and cn.objective; original AdamW, LR, clipping, weight decay, priorities and schedule.',
        scaling='All training scores are cosine; for SigLIP convert frozen native calibration unit to cosine by dividing by exp(logit_scale). Native bias cancels in contrasts and margins. CE grid remains1/10/100 on cosine, not on temperature-scaled logits.',
        normalization='Same original training-only8-batch,seed20260923 gradient-norm rule; eval mode, B Gaussian std.001, median reference, weights clipped[.1,10]. Shared across seeds.',
        ranking=dict(grid=[1,10,100],selector='Unchanged targeted_suppression.select_temperature; each view word1/word2>=.99 and object guard>=frozen-.01; highest exchange, tie smaller scale; original explicit fallback.',
            pilot_seed=42,remaining_seeds='same selected scale for43/44',arms=['R','RG']),
        fixed_is=True,additional_is_search=False,
        evaluation=dict(primary=['IS minus RG exchange','IS minus R cross-effect'],
            secondary=['IS minus R exchange','binding/response/preference','one-word','fix/break',
                'noun recombinations','new color combinations','SugarCrepe','ARO Attribution','COCO retrieval'],
            source_bank='Original400-anchor bank; A2 fresh COCO is infeasible under registered exclusions.',
            uncertainty='paired source x seed10000 draws',new_source_claim=False),
        expected_preference='OpenAI L14 has larger frozen preference; optional no-preference test remains separate, not a tuning gate.',
        training_checkpoints='Only fixed last; every scale candidate retained and selection logged.',
        training_cache_only_before_grid=True))
    with (dest/'protocol.sha256').open('x') as f:f.write(sha(dest/'protocol.json')+'\n')
    log(dest,'freeze')

def verify(model):
    dest=OUT/model;p=read(dest/'protocol.json')
    assert sha(dest/'protocol.json')==(dest/'protocol.sha256').read_text().strip();verify_files(p['inputs']);return p

def encode_bank(subject,name,path):
    dest=OUT/subject.subject['id']/name
    if (dest/'complete.json').exists():verify_cache(dest);return dest
    rows=lines(path);vocab=sorted({tuple(r['objects']) for r in lines(TRAIN)})
    def renderer(r):
        ims,nn,hh,cc=render(r)
        assert all(c['edit']['outside_unchanged'] for c in cc)
        if name=='train':
            expected=oldpixels[r['anchor_id']];direct=hh[:4]+hh[8:12];swapped=hh[4:8]+hh[12:16]
            assert direct==expected[0] and swapped==expected[1],'Historical training pixels changed'
        return ims[:16],nn[:16],hh[:16]
    def prompts(family,pair):
        pair=tuple(pair);wrong=vocab[(vocab.index(pair)+1)%len(vocab)]
        return [g for color in original.PAIRS for g in captions('routing',pair,color)]+[original.objects(pair),original.objects(wrong)]
    oldpixels={}
    if name=='train':
        direct=lines(original.OUT/'features/index.jsonl');swapped=lines(relative.OUT/'swapped_features/index.jsonl')
        assert [r['anchor_id'] for r in direct]==[r['anchor_id'] for r in swapped]
        oldpixels={d['anchor_id']:(d['pixel_sha256'],s['pixel_sha256']) for d,s in zip(direct,swapped)}
    cache_bank(dest,rows,subject,renderer,prompts,registry_path=DEFAULT_REGISTRY,
        input_hashes={str(path):sha(path),str(OUT/subject.subject['id']/'protocol.json'):sha(OUT/subject.subject['id']/'protocol.json')},
        image_batch_size=32,text_batch_size=64,render_workers=8,
        details=dict(colors=original.PAIRS,layouts=['canvas','swapped_canvas'],score_scale=1.,score_bias=0.,training_pixel_replay=name=='train'))
    print('ENCODED',subject.subject['id'],name,len(rows),flush=True);return dest

def training(subject):
    dest=OUT/subject.subject['id'];path=encode_bank(subject,'train',TRAIN);rows=lines(TRAIN);index=lines(path/'index.jsonl')
    texts=np.load(path/'texts.npy');dim=texts.shape[-1]
    tt=texts[np.array([r['text_indices'] for r in index])];assert tt.shape==(640,10,dim)
    t=np.stack([np.concatenate((tt[:,4*c:4*c+4],tt[:,8:]),1) for c in range(2)],1).repeat(2,axis=1).reshape(2560,6,dim)
    image=np.load(path/'images.npy').reshape(2560,4,dim)
    groups=lines(path/'text_groups.jsonl');strings=list(dict.fromkeys(s for g in groups for s in g['templates']))
    natural=[[c['text'] for c in r['natural_guard']['captions']] for r in rows]
    allstrings=list(dict.fromkeys(strings+[s for ss in natural for s in ss]));lookup={s:i for i,s in enumerate(allstrings)}
    feat=subject.encode_texts(allstrings,batch_size=64).numpy()
    nat=feat[np.array([[lookup[s] for s in ss] for ss in natural])].repeat(4,axis=0)
    forms=np.repeat(t[:,None],4,axis=1)
    for j,r in enumerate(index):
        for c in range(2):
            for s in range(4):
                prompts=groups[r['text_indices'][4*c+s]]['templates'];assert len(prompts)==3
                for f,string in enumerate(prompts,1):forms[j*4+c*2:j*4+c*2+2,f,s]=feat[lookup[string]]
    assert np.array_equal(forms[:,0],t) and np.array_equal(forms[::2],forms[1::2])
    cal=read(CAL);native_unit=cal['units'][subject.subject['id']]['unit'];scale=1.;bias=0.
    if subject.subject['backend']=='siglip_transformers':
        scale=float(subject.model.logit_scale.exp());bias=float(subject.model.logit_bias)
    unit=native_unit/scale
    spec,contexts=preference_spec(cal)
    compiled=compile_requirement(spec,calibration_unit=unit,base_model_id=subject.subject['id'],calibration_bank_id='natural_calibration_20260922')
    # All declared contrasts have zero coefficient sum, including first-order b.
    cosine=image[0].astype(float)@t[0,:4].astype(float).T;native=cosine*scale+bias;errors=[]
    for w in compiled.weights.values():
        assert abs(w.sum())<1e-12
        errors.append(abs((cosine*w).sum()/unit-(native*w).sum()/native_unit))
    assert max(errors)<1e-10
    cache=RepairCache(torch.tensor(image),torch.tensor(t),torch.tensor(nat),
        tuple(r['source_ids'][0] for r in rows for _ in range(4)),subject.subject['id'],subject.subject['files'][0]['sha256'],
        'plan49_same_primary_bank_'+subject.subject['id'])
    # Reuse the primary configuration; replace only model-bound provenance and scales.
    _,_,_,cfg=relative.load_training()
    cfg=replace(cfg,base_model_id=cache.base_model_id,encoder_sha256=cache.encoder_sha256,bank_id=cache.bank_id,
        expected_cache_sha256=cache_sha256(cache),scales=derive_scales(cache,compiled,(0,1,2,3)),epochs=36,
        legacy_ranking=historical_ranking('routing',(4,)*2560))
    _check(cfg,cache,compiled)
    parity=dict(native_unit=native_unit,cosine_unit=unit,native_scale=scale,native_bias=bias,
        max_normalized_contrast_error=max(errors),all_training_pixels_match_original=True,
        same_template_cycle=True,cache_sha256=cfg.expected_cache_sha256,dimension=dim)
    if not (dest/'scale_preflight.json').exists():dump(dest/'scale_preflight.json',parity)
    else:assert read(dest/'scale_preflight.json')==parity
    if not (dest/'training_config.json').exists():dump(dest/'training_config.json',asdict(cfg))
    cache=replace(cache,images=cache.images.cuda(),texts=cache.texts.cuda(),natural_texts=cache.natural_texts.cuda())
    return cache,compiled,contexts,cfg,torch.tensor(forms,device='cuda')

def make(cfg,dim,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=TextLowRankAdapter(dim,cfg.rank,cfg.alpha,cfg.dropout).cuda();model.drop=cn.PairedCaptionDropout(cfg.dropout)
    return model

def calibrate(model_id,loaded):
    dest=OUT/model_id
    if (dest/'normalization.json').exists():return read(dest/'normalization.json')['weights']
    cache,spec,contexts,cfg,forms=loaded;model=make(cfg,cache.images.shape[-1],42).eval()
    gen=torch.Generator(device='cuda').manual_seed(20260923)
    with torch.no_grad():model.B.weight.copy_(torch.randn(model.B.weight.shape,device='cuda',generator=gen)*.001)
    order=np.random.default_rng(20260923).permutation(1280);records=[]
    for j in range(8):
        ids=relative.paired_rows(torch.tensor(order[j*24:(j+1)*24],device='cuda'))
        parts=relative.components(model,cache,spec,contexts,cfg,ids)
        norms={k:relative.norm_grads(parts[k],model,retain_graph=True) for k in relative.SCORE_PARTS}
        records.append(norms)
    means={k:float(np.mean([r[k] for r in records])) for k in relative.SCORE_PARTS}
    ref=float(np.median([v for v in means.values() if v>0]));weights={k:float(np.clip(ref/max(v,ref/10),.1,10.)) for k,v in means.items()}
    assert all(np.isfinite(v) for v in weights.values())
    dump(dest/'normalization.json',dict(weights=weights,batches=records,mean_norms=means,reference_norm=ref,training_only=True,shared_across_seeds=True))
    print('NORMALIZED',model_id,weights,flush=True);return weights

def fit(model_id,loaded,weights,seed,family,scale):
    name='IS' if family=='IS' else family+'_s'+str(scale);dest=OUT/model_id/'runs'/f'seed{seed}'/name
    if (dest/'complete.json').exists():
        r=read(dest/'complete.json');verify_files(r['files']);return r
    dest.mkdir(parents=True,exist_ok=False);cache,spec,contexts,cfg,forms=loaded;cfg=replace(cfg,seed=seed)
    model=make(cfg,cache.images.shape[-1],seed);initial=_state_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    schedule=budget.schedule_for(seed);reps=templates.representation_schedule(seed);history=[];started=time.monotonic()
    for ei,order in enumerate(schedule):
        choice=torch.as_tensor(np.repeat(reps[ei],2),device='cuda');current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),choice])
        model.train();torch.manual_seed(seed*10000+ei+1);torch.cuda.manual_seed_all(seed*10000+ei+1)
        for j in range(0,1280,24):
            ids=relative.paired_rows(torch.tensor(order[j:j+24],device='cuda'))
            captured=[];hook=model.register_forward_hook(lambda _m,_a,z:captured.append(z))
            try:parts=relative.components(model,current,spec,contexts,cfg,ids)
            finally:hook.remove()
            assert len(captured)==1
            scores=cache.images[ids]@captured[0][:,:4].transpose(1,2)
            ce=F.cross_entropy(scale*scores.reshape(-1,4),torch.arange(4,device='cuda').repeat(len(ids)))
            guard=relative.objective(parts,weights,'G')
            loss=cn.objective(parts,weights) if family=='IS' else ce+guard if family=='RG' else ce+cfg.legacy_ranking.anchor_weight*parts['ranking_anchor']
            opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
            history.append(dict(epoch=ei+1,rows=ids.tolist(),loss=float(loss.detach()),gradient_norm=float(gn),components={k:float(v.detach()) for k,v in parts.items()}))
        if (ei+1)%6==0:print('TRAIN',model_id,seed,name,ei+1,'seconds',round(time.monotonic()-started,1),flush=True)
    assert len(history)==1944
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},configuration=asdict(cfg),
        seed=seed,family=family,scale=scale,updates=1944,selection='fixed_last',weights=weights,initial_state_hash=initial,
        protocol_sha256=sha(OUT/model_id/'protocol.json')),dest/'last.pt')
    jsonl(dest/'history.jsonl',history)
    r=dict(name=name,family=family,scale=scale,seed=seed,checkpoint=str(dest/'last.pt'),sha256=sha(dest/'last.pt'),
        initial_state_hash=initial,final_rng_sha256=ah(torch.cuda.get_rng_state().cpu().numpy()),
        schedule_sha256=ah(np.asarray(schedule,dtype=np.int64)),representation_sha256=ah(reps),updates=1944,
        first_components=history[0]['components'],seconds=time.monotonic()-started,
        files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    dump(dest/'complete.json',r);return r

def development(model_id,regs):
    root=OUT/model_id;path=root/'development';idx=lines(path/'index.jsonl');image=np.load(path/'images.npy').reshape(len(idx),16,-1)
    text=np.load(path/'texts.npy');text=text[np.array([r['text_indices'] for r in idx])];aff=read(root/'scale_preflight.json');out=[]
    for view in range(2):
        v=image[:,view*4:view*4+4]
        for r in [dict(name='F',seed=0,checkpoint=None),*regs]:
            t=adapt(text,r['checkpoint']);cos=np.einsum('nid,njd->nij',v,t,optimize=True)
            scores=cos[:,:,:4].astype(float)*aff['native_scale']+aff['native_bias'];z=routing(scores,model_id)
            out.append(dict(name=r['name'],seed=r['seed'],view=('canvas','swapped_canvas')[view],
                exchange_accuracy=float(z['exchange_accuracy'].mean()),word1_accuracy=float(z['word1_accuracy'].mean()),
                word2_accuracy=float(z['word2_accuracy'].mean()),object_guard=float((cos[:,:,8]>cos[:,:,9]).mean())))
    return out

def matched(rows):
    for s in SEEDS:
        group=[r for r in rows if r['seed']==s]
        for k in ('initial_state_hash','final_rng_sha256','schedule_sha256','representation_sha256','updates'):
            assert len({r[k] for r in group})<=1,(s,k)
        assert all(r['first_components']==group[0]['first_components'] for r in group)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('freeze','run'));ap.add_argument('--model',choices=MODELS,required=True);a=ap.parse_args()
    if a.action=='freeze':freeze(a.model);return
    verify(a.model);dest=OUT/a.model;log(dest,'start');relative.configure();cv2.setNumThreads(1)
    annotations_for('train2017');annotations_for('val2017');subject=load_subject(a.model,device='cuda')
    loaded=training(subject);encode_bank(subject,'development',DEV)
    del subject;torch.cuda.empty_cache();weights=calibrate(a.model,loaded);grid=[]
    for family in ('R','RG'):
        for scale in (1,10,100):grid.append(fit(a.model,loaded,weights,42,family,scale))
    full=fit(a.model,loaded,weights,42,'IS',100);matched([*grid,full])
    dev=development(a.model,grid);frozen={r['view']:r for r in dev if r['name']=='F'};choices=[]
    for r in grid:
        views=[v for v in dev if v['name']==r['name']]
        eligible=all(min(v['word1_accuracy'],v['word2_accuracy'])>=.99 and v['object_guard']>=frozen[v['view']]['object_guard']-.01 for v in views)
        choices.append(dict(name=r['name'],family=r['family'],scale=r['scale'],eligible=eligible,exchange_accuracy=float(np.mean([v['exchange_accuracy'] for v in views]))))
    selected={f:selection.select_temperature(choices,f) for f in ('R','RG')}
    if not (dest/'selection.json').exists():
        jsonl(dest/'development_results.jsonl',dev);dump(dest/'selection.json',dict(selected=selected,candidates=choices,test_scores_used=False))
    else:assert read(dest/'selection.json')['selected']==selected
    models=[]
    for s in SEEDS:
        group=[full if s==42 else fit(a.model,loaded,weights,s,'IS',100)]
        for f in ('R','RG'):
            r=fit(a.model,loaded,weights,s,f,selected[f]['scale']);group.append(r)
        matched(group);models.extend(dict(r,name='IS' if r['family']=='IS' else r['family']+'_selected') for r in group)
    dump(dest/'models.json',models);dump(dest/'training_complete.json',dict(models_sha256=sha(dest/'models.json'),
        fits=13,selected_models=9,all_matching_checks_passed=True,selection_sha256=sha(dest/'selection.json'),evaluation_pending=True))
    log(dest,'complete')

if __name__=='__main__':main()
