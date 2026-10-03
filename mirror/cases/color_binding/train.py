"""Fixed 75% text-side backbone ports and ordinary-ranking control."""
import argparse
from dataclasses import asdict; from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mirror.cases.color_binding import test_data as data
from mirror.core.repair import RepairCache; from mirror.core.repair import TextLowRankAdapter; from mirror.core.repair import cache_sha256; from mirror.core.repair import derive_scales; from mirror.core.repair import historical_ranking; from mirror.core.repair import _check; from mirror.core.repair import _state_hash
from mirror.cases.color_binding.routing_adequacy import preference_spec
from mirror.core.specifications import compile_requirement
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.core.io import dump_or_verify

OUT=data.OUT
base=data.base;p=data.p;breadth=data.breadth
MODELS=('openclip_laion_l14',*data.MODELS)

def load_training(model,device='cuda'):
    if model=='openclip_laion_l14':return base.mid.load_training(75,device)
    folder=data.bank(model,'train');data.verify_cache(folder)
    index=data.lines(folder/'index.jsonl');rows=data.lines(data.rowpath('train'))
    assert [r['anchor_id'] for r in rows]==[r['anchor_id'] for r in index]
    extra=OUT/model/'extra_text';verify_files(read(extra/'complete.json')['files'])
    features=np.load(extra/'features.npy');lookup={s:i for i,s in enumerate(read(extra/'strings.json'))}
    def single(strings):return features[[lookup[s] for s in strings]]
    def pooled(strings):
        a=single(strings).mean(0);return a/np.linalg.norm(a)
    values=[];forms=[];natural=[];v=np.load(folder/'images.npy');vocab=sorted({tuple(r['objects']) for r in rows})
    for row,idx in zip(rows,index):
        nouns=tuple(row['objects']);wrong=vocab[(vocab.index(nouns)+1)%len(vocab)]
        objects=np.stack([pooled(breadth.original.objects(pair)) for pair in (nouns,wrong)])
        for colors in p.COLORS[:2]:
            representations=[]
            for family in ('train_templates','reverse_order'):
                strings=data.ind.prompts(nouns,colors,'canvas',family)
                matrix=np.stack([single(s) for s in strings])
                mean=matrix.mean(1);mean/=np.linalg.norm(mean,axis=-1,keepdims=True)
                representations += [np.concatenate((t,objects),axis=0) for t in (mean,*matrix.transpose(1,0,2))]
            for view in p.VIEWS:
                names=['-'.join(colors)+'/'+view+'/'+a+'_'+b for a in colors for b in colors]
                values.append(v[idx['image_offset']+np.array([idx['state_names'].index(n) for n in names])])
                forms.append(representations);natural.append(single([c['text'] for c in row['natural_guard']['captions']]))
    forms=np.asarray(forms);dim=forms.shape[-1]
    assert forms.shape==(2560,8,6,dim) and np.array_equal(forms[::2],forms[1::2])
    # Pooled canonical captions must reproduce the feature-bank caption encoder.
    encoded=np.load(folder/'texts.npy')
    for ci in range(2):
        expected=encoded[np.array([r['text_indices'][ci*4:ci*4+4] for r in index])]
        assert np.max(abs(forms[ci*2::4,0,:4]-expected))<2e-6
    aff=read(breadth.OUT/model/'scale_preflight.json')
    spec,ctx=preference_spec(read(breadth.CAL))
    compiled=compile_requirement(spec,calibration_unit=aff['cosine_unit'],base_model_id=model,calibration_bank_id='natural_calibration_20260922')
    subject=next(r for r in read(data.DEFAULT_REGISTRY)['subjects'] if r['id']==model)
    cache=RepairCache(torch.tensor(np.asarray(values)),torch.tensor(forms[:,0]),torch.tensor(np.asarray(natural)),
        tuple(r['source_ids'][0] for r in rows for _ in range(4)),model,subject['files'][0]['sha256'],'routing75_complete_'+model)
    _,_,_,cfg=p.prior.load_training('cpu')
    cfg=replace(cfg,base_model_id=model,encoder_sha256=cache.encoder_sha256,bank_id=cache.bank_id,
        expected_cache_sha256=cache_sha256(cache),scales=derive_scales(cache,compiled,(0,1,2,3)),epochs=36,
        legacy_ranking=historical_ranking('routing',(4,)*2560))
    _check(cfg,cache,compiled)
    cache=replace(cache,images=cache.images.to(device),texts=cache.texts.to(device),natural_texts=cache.natural_texts.to(device))
    return (cache,compiled,ctx,cfg),torch.tensor(forms,device=device)

def make(cfg,dim,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=TextLowRankAdapter(dim,cfg.rank,cfg.alpha,cfg.dropout).cuda();model.drop=p.cn.PairedCaptionDropout(cfg.dropout)
    if cfg.base_model_id=='openclip_laion_l14':
        saved=torch.load(base.initial(seed),map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict']);assert _state_hash(model.state_dict())==saved['state_hash']
    return model

def normalization(model):
    return read(p.prior.OUT/'normalization.json' if model=='openclip_laion_l14' else breadth.OUT/model/'normalization.json')

def calibrate(model,loaded,forms):
    path=OUT/model/'color_calibration.json'
    if path.exists():return read(path)
    if model=='openclip_laion_l14':
        cal=read(base.mid.OUT/'calibration/75.json');dump(path,cal);return cal
    cache,spec,ctx,cfg=loaded
    with torch.no_grad():
        mm=torch.cat([base.engine.color_margins(cache.images@forms[:,i,:4].transpose(1,2),spec.unit).flatten() for i in range(4)])
        m0=float(.25*mm[mm>0].median())
    adapter=make(cfg,cache.images.shape[-1],42).eval();gen=torch.Generator(device='cuda').manual_seed(20260923)
    with torch.no_grad():adapter.B.weight.copy_(torch.randn(adapter.B.weight.shape,device='cuda',generator=gen)*.001)
    order=np.random.default_rng(20260923).permutation(1280);records=[]
    for j in range(8):
        ids=p.prior.paired_rows(torch.tensor(order[j*24:(j+1)*24],device='cuda'))
        terms,_=base.engine.parts(adapter,cache,spec,ctx,cfg,ids,m0)
        records.append(dict(rows=ids.cpu().tolist(),gradient_norm=p.prior.norm_grads(terms['color_floor'],adapter,retain_graph=True)))
    norm=float(np.mean([r['gradient_norm'] for r in records]));reference=normalization(model)['reference_norm']
    cal=dict(m0=m0,color_weight=float(np.clip(reference/max(norm,reference/10),.1,10)),
        reference_norm=reference,gradient_norm=norm,batches=records,training_only=True,shared_across_seeds=True)
    dump(path,cal);return cal

def objective(terms,ce,weights,arm,cal,cfg):
    if arm=='OrdinaryRanking':return ce+cfg.legacy_ranking.anchor_weight*terms['ranking_anchor']
    return base.engine.objective(terms,ce,weights,'color_order',arm,cal['color_weight'])

def prepare(model):
    data.verify();dest=OUT/model
    paths=[data.PLAN,Path(__file__),Path(base.__file__),Path(base.engine.__file__),Path(breadth.__file__),
           data.OUT/'data_protocol.json',base.OUT/'protocol.json',
           p.prior.OUT/'normalization.json' if model=='openclip_laion_l14' else breadth.OUT/model/'normalization.json']
    if model!='openclip_laion_l14':paths += [dest/'encoding_complete.json',dest/'extra_text/complete.json',breadth.OUT/model/'scale_preflight.json']
    if not (dest/'training_protocol.json').exists():
        dump(dest/'training_protocol.json',dict(inputs={str(f):sha(f) for f in paths},seeds=[42,43,44],tint=.75,
            arms=['OrdinaryRanking'] if model=='openclip_laion_l14' else ['Ranking','IS'],
            updates=1944,epochs=36,selection='fixed_last',ranking_cosine_temperature=100,temperature_search=False,
            shared_color_floor=True,existing_weights_unchanged=True,no_score_selection=True,
            ordinary_control='Historical CE plus anchor drift only; same75 inputs, templates, schedule and budget'))
    else:verify_files(read(dest/'training_protocol.json')['inputs'])
    loaded,forms=load_training(model);cal=calibrate(model,loaded,forms)
    dump_or_verify(dest/'training_inputs.json',dict(configuration=asdict(loaded[-1]),color_calibration_sha256=sha(dest/'color_calibration.json'),
        forms_shape=list(forms.shape),forms_sha256=p.array_hash(forms.cpu().numpy())))
    return loaded,forms,cal

def train_one(model,seed,arm,loaded,forms,cal):
    dest=OUT/model/f'seed{seed}'/arm
    if (dest/'complete.json').exists():
        r=read(dest/'complete.json');assert sha(r['checkpoint'])==r['sha256'];return r
    dest.mkdir(parents=True,exist_ok=False)
    cache,spec,ctx,cfg=loaded;cfg=replace(cfg,seed=seed);weights=normalization(model)['weights']
    adapter=make(cfg,cache.images.shape[-1],seed);initial=_state_hash(adapter.state_dict())
    schedule=p.old.schedule_for(seed);reps=base.representations(seed)
    opt=torch.optim.AdamW(adapter.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    history=[];started=time.monotonic()
    for epoch,order in enumerate(schedule):
        current=replace(cache,texts=forms[torch.arange(2560,device='cuda'),torch.tensor(np.repeat(reps[epoch],2),device='cuda')])
        adapter.train();torch.manual_seed(seed*10000+epoch+1);torch.cuda.manual_seed_all(seed*10000+epoch+1)
        for start in range(0,1280,24):
            ids=p.prior.paired_rows(torch.tensor(order[start:start+24],device='cuda'))
            terms,ce=base.engine.parts(adapter,current,spec,ctx,cfg,ids,cal['m0'])
            loss=objective(terms,ce,weights,arm,cal,cfg)
            opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(adapter.parameters(),cfg.grad_clip)
            assert torch.isfinite(loss) and torch.isfinite(gn);opt.step()
            history.append(dict(epoch=epoch+1,rows=ids.cpu().tolist(),loss=float(loss.detach()),ce=float(ce.detach()),
                gradient_norm=float(gn),components={k:float(v.detach()) for k,v in terms.items()}))
        if (epoch+1)%12==0:print('TEXT_TRAIN',model,seed,arm,epoch+1,round(time.monotonic()-started,1),flush=True)
    assert len(history)==1944
    path=dest/'last.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in adapter.state_dict().items()},configuration=asdict(cfg),
        seed=seed,arm=arm,tint=75,recipe='color_order',weights=weights,color_calibration=cal,updates=1944,
        selection='fixed_last',protocol_sha256=sha(OUT/model/'training_protocol.json')),path)
    jsonl(dest/'history.jsonl',history)
    result=dict(name=arm,seed=seed,checkpoint=str(path),sha256=sha(path),initial_state_hash=initial,
        schedule_sha256=p.array_hash(schedule),representation_sha256=p.array_hash(reps),
        final_rng_sha256=p.array_hash(torch.cuda.get_rng_state().cpu().numpy()),first_components=history[0]['components'],
        seconds=time.monotonic()-started,updates=1944)
    dump(dest/'complete.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);a=ap.parse_args()
    log(OUT,'training_start',model=a.model);p.prior.configure();loaded,forms,cal=prepare(a.model)
    results=[]
    for seed in (42,43,44):
        rr=[train_one(a.model,seed,arm,loaded,forms,cal) for arm in (('OrdinaryRanking',) if a.model=='openclip_laion_l14' else ('Ranking','IS'))]
        for key in ('initial_state_hash','schedule_sha256','representation_sha256','final_rng_sha256','first_components','updates'):
            assert all(r[key]==rr[0][key] for r in rr),key
        if a.model=='openclip_laion_l14':
            original=read(base.OLD/'choice.json')['candidate']['model'] if seed==42 else next(r for r in read(base.OUT/f'seed{seed}/models.json') if r['name']=='IS')
            for key in ('initial_state_hash','schedule_sha256','representation_sha256','final_rng_sha256','first_components','updates'):
                assert rr[0][key]==original[key],('ordinary_vs_main',seed,key)
        results.extend(rr)
    dump_or_verify(OUT/a.model/'models.json',results)
    dump_or_verify(OUT/a.model/'training_complete.json',dict(models_sha256=sha(OUT/a.model/'models.json'),all_matched=True))
    log(OUT,'training_complete',model=a.model)

if __name__=='__main__':main()
