"""One-seed, complete-context routing repair; explicit create-only stages.

The historical engine/specifications remain unchanged. The hook below reuses
its single shared text forward, so every arm consumes identical dropout draws.
"""
import argparse
from dataclasses import asdict; from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import OUT as BANK; from mirror.cases.color_binding.repair_trainbank import render; from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.core.encoders import load_subject; from mirror.core.encoders import read_registry; from mirror.core.encoders import DEFAULT_REGISTRY
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.cases.color_binding.rendering import captions
from mirror.cases.color_binding.routing_adapter_reaudit_v2 import OUT as REAUDIT; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import CAL; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import CACHE as AUDIT_CACHE; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import MODEL; from mirror.cases.color_binding.routing_adapter_reaudit_v2 import OLD
from mirror.cases.color_binding.routing_adequacy import preference_spec; from mirror.cases.color_binding.routing_adequacy import pilot_losses
from mirror.core.specifications import compile_requirement
from mirror.core.repair import RepairCache; from mirror.core.repair import RepairConfig; from mirror.core.repair import LossScales; from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import GUARDS; from mirror.core.repair import derive_scales; from mirror.core.repair import cache_sha256; from mirror.core.repair import exclusion_sha256; from mirror.core.repair import historical_ranking; from mirror.core.repair import loss_components; from mirror.core.repair import _check; from mirror.core.repair import _state_hash

OUT = ROOT/'clip/interbind_routing_repair_pilot_20260923'
DESIGN = ROOT/'clip/interbind_diagnosis_guided_prep_20260923/pilot_design.json'
SUGAR = OLD/'sugarcrepe/routing'
PAIRS = (('red','blue'), ('green','yellow'))
ARMS = ('R','G','I','P','IP','E')
PILOT_SOURCES=ROOT/'clip/interbind_source_quality_20260922/pilot/accepted_rows.jsonl'


def objects(pair):
    a,b=pair
    return [f'a photo of a {a} and a {b}', f'a {a} next to a {b}']


def freeze():
    design=read(DESIGN); verify_files(design['inputs'])
    bank=read(BANK/'protocol.json');verify_files(bank['inputs']);done=read(BANK/'complete.json')
    assert sha(BANK/'rows.jsonl')==done['rows_sha256'] and sha(BANK/'excluded_source_ids.json')==done['excluded_sha256']
    rows=[r for r in lines(BANK/'rows.jsonl') if r['family']=='routing'];assert len(rows)==640
    excluded=set(read(BANK/'excluded_source_ids.json'))
    assert not excluded.intersection(i for r in rows for i in r['source_ids'])
    assert not excluded.intersection(r['natural_guard']['image_id'] for r in rows)
    checks={str(p):sha(p) for p in [DESIGN,BANK/'protocol.json',BANK/'complete.json',BANK/'rows.jsonl',BANK/'excluded_source_ids.json',CAL,
        DEFAULT_REGISTRY,DEFAULT_REGISTRY.with_name('activation_addendum.json'),REAUDIT/'frozen_direct_regimes.jsonl',PILOT_SOURCES]}
    for name in ('routing_repair_pilot','routing_repair_evaluate','repair','routing_adequacy','diagnosis_guided_protocol',
                 'routing_adapter_reaudit_v2','feature_cache','scorers','repair_trainbank','rendering','rendering_v3','io'):
        path=Path(__file__).with_name(name+'.py');checks[str(path)]=sha(path)
    for r in rows:checks.update(r['source_image_sha256'])
    for path in [AUDIT_CACHE,*[AUDIT_CACHE.parent/f'natural_{v}' for v in ('foreground_context','full_image')]]:
        checks[str(path/'complete.json')]=sha(path/'complete.json')
        checks.update({str(path/k):v for k,v in read(path/'complete.json')['files'].items()})
    for p in [SUGAR/'features.npz',SUGAR/'indices.json',SUGAR/'frozen_seed0.csv',SUGAR/'complete.json',
              ROOT/'clip/interbind_natural_localizer_quality_20260922/pilot/accepted_rows.jsonl']:
        checks[str(p)]=sha(p)
    assert sha(SUGAR/'frozen_seed0.csv')==read(SUGAR/'complete.json')['files']['frozen_seed0.csv']
    verify_files(checks)
    jsonl(OUT/'training_rows.jsonl',rows)
    wanted={r['anchor_id'] for r in lines(REAUDIT/'frozen_direct_regimes.jsonl')}
    pilot=[r for r in lines(PILOT_SOURCES) if r['anchor_id'] in wanted];assert len(pilot)==49
    eval_pairs=sorted({tuple(r['objects']) for r in pilot})
    object_mapping=[]
    for r in pilot:
        pair=tuple(r['objects']);j=eval_pairs.index(pair)
        wrong=next(eval_pairs[(j+k)%len(eval_pairs)] for k in range(1,len(eval_pairs)) if set(eval_pairs[(j+k)%len(eval_pairs)])!=set(pair))
        object_mapping.append(dict(anchor_id=r['anchor_id'],correct=objects(pair),distractor=objects(wrong),objects=pair,distractor_objects=wrong))
    jsonl(OUT/'evaluation_object_mapping.jsonl',object_mapping)
    dump(OUT/'protocol.json',dict(inputs=checks,design_sha256=sha(DESIGN),training_rows_sha256=sha(OUT/'training_rows.jsonl'),
        author_authorization='2026-09-23: ok gpu is free continue',model=MODEL,seed=42,arms=['F',*ARMS],
        configuration=design,training_lattices=1280,training_sources=640,colors=PAIRS,layout='canvas',
        captions='Existing pooled three-template color captions; historical two-template object captions; cyclic next training noun pair as object distractor; all five assigned natural TRAINING captions separately',
        object_guard='Each of four states: correct color-free object-pair caption at index4 versus cyclic distractor at index5; same in all guarded arms',
        encoder_precision='FP32, TF32 disabled; exact pinned scorer, template pooling and preprocessing',
        gpu_encoding_batch=32,text_batch=64,render_threads=8,cpu_threads=4,
        gpu_memory_gate_gib=12,gate_interval_seconds=45,gate_stable_checks=2,
        storage='Frozen features and initial/last only; no per-epoch checkpoint selection',
        evaluation='All 49 frozen pilot IDs, blend90 red-blue, direct/swapped/in-situ; full seven-category SugarCrepe; both fixed natural-gallery views',
        evaluation_object_mapping_sha256=sha(OUT/'evaluation_object_mapping.jsonl'),
        selection='fixed_last',reserve=False,automatic_seed_escalation=False,all_prior_results_preserved=True))
    # Bounded construction smoke; no model and no score-dependent rejection.
    smoke=[]
    for r in rows[:4]:
        for pair in PAIRS:
            im,names,hashes,checks_=render(r,pair)
            assert len(im)==4 and all(c['edit']['outside_unchanged'] for c in checks_)
            smoke.append(dict(anchor_id=r['anchor_id'],colors=pair,pixels=hashes,checks=checks_))
    jsonl(OUT/'construction_smoke.jsonl',smoke)
    print('FROZEN one-seed pilot',sha(OUT/'protocol.json'),flush=True)


def verify():
    p=read(OUT/'protocol.json');verify_files(p['inputs'])
    assert sha(OUT/'training_rows.jsonl')==p['training_rows_sha256']
    assert sha(OUT/'evaluation_object_mapping.jsonl')==p['evaluation_object_mapping_sha256']
    return p


def wait_gpu():
    p=read(OUT/'protocol.json');need=p['gpu_memory_gate_gib']*1024**3;stable=0
    while stable<p['gate_stable_checks']:
        free,total=torch.cuda.mem_get_info();stable=stable+1 if free>=need else 0
        print('GPU gate free_GiB',round(free/1024**3,2),'needed',p['gpu_memory_gate_gib'],'stable',stable,flush=True)
        if stable<p['gate_stable_checks']:time.sleep(p['gate_interval_seconds'])


def encode():
    p=verify();wait_gpu();torch.set_num_threads(p['cpu_threads']);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=lines(OUT/'training_rows.jsonl');vocab=sorted({tuple(r['objects']) for r in rows})
    annotations_for('train2017')  # warm the shared annotation cache before render threads
    scorer=load_subject(MODEL,device='cuda');qualification=[]
    def renderer(row):
        ims=[];names=[];hashes=[];checks=[]
        for pair in PAIRS:
            ii,nn,hh,cc=render(row,pair);ims+=ii;names+=['-'.join(pair)+'/'+n for n in nn];hashes+=hh;checks+=cc
        assert all(c['edit']['outside_unchanged'] for c in checks)
        qualification.append(dict(anchor_id=row['anchor_id'],checks=checks))
        return ims,names,hashes
    def prompts(family,pair):
        pair=tuple(pair);wrong=vocab[(vocab.index(pair)+1)%len(vocab)]
        return [g for colors in PAIRS for g in captions(family,pair,colors)]+[objects(pair),objects(wrong)]
    cache_bank(OUT/'features',rows,scorer,renderer,prompts,registry_path=DEFAULT_REGISTRY,
        input_hashes={str(OUT/'protocol.json'):sha(OUT/'protocol.json'),str(OUT/'training_rows.jsonl'):sha(OUT/'training_rows.jsonl')},
        image_batch_size=p['gpu_encoding_batch'],text_batch_size=p['text_batch'],render_workers=p['render_threads'],
        details=dict(family='routing',colors=PAIRS,layout='canvas',score_scale=1,score_bias=0))
    natural=[[c['text'] for c in r['natural_guard']['captions']] for r in rows];assert all(len(x)==5 for x in natural)
    unique=list(dict.fromkeys(t for group in natural for t in group));mapping={t:i for i,t in enumerate(unique)}
    embeddings=scorer.encode_texts(unique,batch_size=p['text_batch']).numpy()
    array=embeddings[np.array([[mapping[t] for t in group] for group in natural])]
    with (OUT/'features/natural_texts.npy').open('xb') as f:np.save(f,array)
    object_mapping=lines(OUT/'evaluation_object_mapping.jsonl')
    evaluation_objects=scorer.encode_texts([r[k] for r in object_mapping for k in ('correct','distractor')]).numpy().reshape(49,2,768)
    with (OUT/'evaluation_object_texts.npy').open('xb') as f:np.save(f,evaluation_objects)
    dump(OUT/'features/natural_templates.json',natural)
    jsonl(OUT/'features/pixel_checks.jsonl',sorted(qualification,key=lambda x:x['anchor_id']))
    checks=[c for r in qualification for c in r['checks']]
    dump(OUT/'encoding_complete.json',dict(files={str(Path('features')/n):sha(OUT/'features'/n) for n in
        ('complete.json','natural_texts.npy','natural_templates.json','pixel_checks.jsonl')}|{'evaluation_object_texts.npy':sha(OUT/'evaluation_object_texts.npy')},
        pixel_checks=len(checks),pixel_qualified=sum(c['edit']['passes'] for c in checks),
        no_score_based_exclusions=True,gpu=True,training=False))
    print('ENCODING complete; pixel qualification',sum(c['edit']['passes'] for c in checks),'/',len(checks),flush=True)


def load_training(device='cpu'):
    p=verify();done=read(OUT/'encoding_complete.json');verify_files({str(OUT/k):v for k,v in done['files'].items()})
    torch.set_num_threads(p['cpu_threads'])
    meta,_=verify_cache(OUT/'features');rows=lines(OUT/'training_rows.jsonl');index=lines(OUT/'features/index.jsonl')
    assert [r['anchor_id'] for r in rows]==[r['anchor_id'] for r in index]
    expected=[ '-'.join(pair)+'/canvas/'+a+'_'+b for pair in PAIRS for a in pair for b in pair ]
    assert all(r['state_names']==expected and r['image_count']==8 for r in index)
    v=np.load(OUT/'features/images.npy').reshape(640,8,768)
    t=np.load(OUT/'features/texts.npy')[np.array([r['text_indices'] for r in index])]
    assert t.shape==(640,10,768)
    texts=np.stack([np.concatenate([t[:,:4],t[:,8:]],1),np.concatenate([t[:,4:8],t[:,8:]],1)],1).reshape(1280,6,768)
    natural=np.repeat(np.load(OUT/'features/natural_texts.npy'),2,axis=0)
    cache=RepairCache(torch.tensor(v.reshape(1280,4,768)),torch.tensor(texts),torch.tensor(natural),
        tuple(i for r in rows for i in [r['source_ids'][0]]*2),MODEL,
        meta['model_binding']['checkpoint_files'][0]['sha256'],'interbind_routing_four_color_v1')
    spec,contexts=preference_spec(read(CAL));cal=read(CAL)
    compiled=compile_requirement(spec,calibration_unit=cal['units'][MODEL]['unit'],base_model_id=MODEL,calibration_bank_id='natural_calibration_20260922')
    design=p['configuration'];scales=derive_scales(cache,compiled,(0,1,2,3));excluded=tuple(read(BANK/'excluded_source_ids.json'))
    config=RepairConfig(arm='full_is',seed=42,expected_cache_sha256=cache_sha256(cache),exclusion_sha256=exclusion_sha256(excluded),
        encoder_sha256=cache.encoder_sha256,base_model_id=MODEL,bank_id=cache.bank_id,excluded_source_ids=excluded,
        correct_captions=(0,1,2,3),object_pairs=tuple((i,4,5) for i in range(4)),
        binding_contrasts=tuple(r['contrast'] for r in contexts if r['kind']=='binding'),
        clause_weights={c['name']:0. for c in compiled.spec['clauses']},guard_weights=design['shared_guard_weights'],scales=scales,
        legacy_ranking=historical_ranking('routing',(4,)*1280),natural_tolerance=0,drift_tolerance=0)
    _check(config,cache,compiled)
    if device!='cpu':cache=replace(cache,images=cache.images.to(device),texts=cache.texts.to(device),natural_texts=cache.natural_texts.to(device))
    return cache,compiled,contexts,config


def objective(adapter,cache,spec,contexts,cfg,rows,arm):
    if arm not in ARMS:raise ValueError(arm)
    outputs=[];hook=adapter.register_forward_hook(lambda module,args,result: outputs.append(result))
    try:guards,components=loss_components(adapter,cache,spec,cfg,rows)
    finally:hook.remove()
    assert len(outputs)==1
    scores=cache.images[rows].float()@outputs[0][:,:4].transpose(1,2)
    extra=pilot_losses(scores,spec,contexts);components.update({f'pilot:{k}':v for k,v in extra.items()})
    totals=dict(R=components['ranking_endpoint']+.2*components['ranking_anchor'],G=guards,
        I=guards+extra['interaction'],P=guards+extra['preference'],IP=guards+extra['interaction']+extra['preference'],E=guards+extra['endpoint'])
    return totals[arm],components


def preflight():
    cache,spec,contexts,cfg=load_training();torch.set_num_threads(4);torch.manual_seed(42)
    model=TextLowRankAdapter(768,64,64,.05);ids=torch.arange(24);records={};component_reference=None
    for arm in ARMS:
        torch.manual_seed(420001);total,components=objective(model,cache,spec,contexts,cfg,ids,arm)
        vals={k:float(v.detach()) for k,v in components.items()}
        if component_reference is None:component_reference=vals
        assert vals==component_reference
        grads=torch.autograd.grad(total,tuple(model.parameters()));norm=float(torch.sqrt(sum(g.square().sum() for g in grads)))
        assert np.isfinite(norm) and np.isfinite(float(total))
        if arm not in ('G','P'):assert norm>1e-9,(arm,'no gradient')
        records[arm]=dict(loss=float(total.detach()),gradient_norm=norm)
    # Guards may have zero gradient at the identity. Check their response to a
    # deterministic parameter perturbation, not an optimizer step or a fitted model.
    with torch.no_grad():model.B.weight.copy_(torch.randn_like(model.B.weight)*.001)
    torch.manual_seed(420001);_,parts=objective(model,cache,spec,contexts,cfg,ids,'G')
    guard_records={}
    for k in GUARDS:
        gg=torch.autograd.grad(parts[k],tuple(model.parameters()),retain_graph=True)
        norm=float(torch.sqrt(sum(g.square().sum() for g in gg)))
        assert np.isfinite(norm) and norm>0,(k,norm)
        guard_records[k]=dict(loss=float(parts[k].detach()),gradient_norm=norm)
    dump(OUT/'training_config.json',asdict(cfg))
    dump(OUT/'preflight.json',dict(config_sha256=sha(OUT/'training_config.json'),shared_forward_components_equal=True,
        losses_at_identity=records,perturbed_guard_checks=guard_records,training_scales=asdict(cfg.scales),
        cache_sha256=cfg.expected_cache_sha256,no_optimizer_steps=True,all_source_ids_and_natural_guard_ids_checked=True))
    print('PREFLIGHT',records,'scales',asdict(cfg.scales),flush=True)


def train():
    cache,spec,contexts,cfg=load_training('cuda');assert asdict(cfg)==read(OUT/'training_config.json') or json.loads(json.dumps(asdict(cfg)))==read(OUT/'training_config.json')
    assert sha(OUT/'training_config.json')==read(OUT/'preflight.json')['config_sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(42);schedule=[rng.permutation(1280).tolist() for _ in range(6)]
    dump(OUT/'batch_schedule.json',schedule);models=[dict(arm='F',seed=42,checkpoint=None)]
    initial_hashes=[];rng_hashes=[];histories=[]
    for arm in ARMS:
        dest=OUT/'runs'/arm;dest.mkdir(parents=True,exist_ok=False);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=TextLowRankAdapter(768,64,64,.05).cuda();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ih=_state_hash(initial);initial_hashes.append(ih);torch.save(dict(state_dict=initial,state_hash=ih),dest/'initial.pt')
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);history=[];started=time.monotonic()
        for epoch,order in enumerate(schedule,1):
            model.train();torch.manual_seed(420000+epoch);torch.cuda.manual_seed_all(420000+epoch)
            for start in range(0,len(order),24):
                ids=torch.tensor(order[start:start+24],device='cuda');total,components=objective(model,cache,spec,contexts,cfg,ids,arm)
                opt.zero_grad(set_to_none=True);total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(total) and torch.isfinite(norm)
                opt.step();history.append(dict(epoch=epoch,rows=ids.cpu().tolist(),loss=float(total.detach()),
                    gradient_norm=float(norm),components={k:float(v.detach()) for k,v in components.items()}))
            print('TRAIN',arm,'epoch',epoch,'loss',history[-1]['loss'],'seconds',round(time.monotonic()-started,1),flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        rh=sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes());rng_hashes.append(rh)
        torch.save(dict(state_dict=state,arm=arm,seed=42,selection='fixed_last',epochs=6,updates=len(history),
            shared_guard_engine_config=asdict(cfg),objective=read(DESIGN)['arms'][arm],
            initial_state_hash=ih,cache_sha256=cfg.expected_cache_sha256,optimizer=opt.state_dict()),dest/'last.pt')
        jsonl(dest/'history.jsonl',history);histories.append(history)
        entry=dict(arm=arm,seed=42,checkpoint=str(dest/'last.pt'),sha256=sha(dest/'last.pt'),initial_state_hash=ih,
            final_rng_hash=rh,seconds=time.monotonic()-started,updates=len(history),selection='fixed_last')
        dump(dest/'complete.json',entry);models.append(entry)
        del model,opt
    assert len(set(initial_hashes))==len(set(rng_hashes))==1
    assert all([r['rows'] for r in h]==[r['rows'] for r in histories[0]] for h in histories)
    assert all(h[0]['components']==histories[0][0]['components'] for h in histories)
    dump(OUT/'models.json',models)
    dump(OUT/'training_complete.json',dict(models_sha256=sha(OUT/'models.json'),schedule_sha256=sha(OUT/'batch_schedule.json'),
        matched_initialization=True,matched_batches=True,matched_dropout_rng=True,first_batch_components_equal=True,selection='fixed_last',updates_per_arm=324))


def sha_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def run():
    # Separate processes release encoder memory before optimization/evaluation.
    for action in ('encode','preflight','train','evaluate'):
        command=[sys.executable,'-u','-m','mirror.cases.color_binding.routing_repair_pilot',action]
        print('STAGE',action,flush=True);subprocess.run(command,cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','encode','preflight','train','evaluate','run']);args=parser.parse_args()
    log(OUT,'start',stage=args.action)
    try:
        if args.action=='evaluate':
            from mirror.cases.color_binding.routing_repair_evaluate import evaluate
            evaluate()
        else:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
