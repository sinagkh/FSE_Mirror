"""Official natural-image retests, with a verified Defense-Prefix port."""
from mirror.cases.typography.diagnose import *
from mirror.cases.typography.retest import registry
import ast; import io; import re; import zipfile; import importlib.util; import types
import pandas as pd
sys.path.insert(0,str(ROOT/'clip/fse_color_worker_scale'))
from analyze import interval
sys.path.insert(0,str(ROOT/'mirror/utils'))
from mirror.utils.two_object_locality_lib import TextLoRAAdapter

EX=OUT/'external_retest'

def manifest():
    verify();meta=json.loads((OUT/'external_downloads.json').read_text());root=Path(meta['storage'])
    for r in meta['records']:assert sha(r['path'])==r['sha256']
    with zipfile.ZipFile(root/'SCAM/images.zip') as z:
        rr=[json.loads(line) for line in z.read('images/train/metadata.jsonl').decode().splitlines() if line.strip()]
        for r in rr:
            r.update(archive=str(root/'SCAM/images.zip'),member='images/train/'+r['file_name'],bank=r['type'],source_id=r['id'].removeprefix(r['type']+'_'))
            assert r['member'] in z.namelist()
    assert len(rr)==3486
    paired={}
    for r in rr:paired.setdefault(r['source_id'],{})[r['type']]=(r['object_label'],r['attack_word'])
    assert len(paired)==1162 and all(set(g)=={'SCAM','NoSCAM','SynthSCAM'} and len(set(g.values()))==1 for g in paired.values())
    with zipfile.ZipFile(root/'Defense-Prefix/rta100.zip') as z:
        for filename in sorted(z.namelist()):
            if Path(filename).suffix.lower() not in ('.jpg','.png','.jpeg'):continue
            match=re.fullmatch(r'label=(.+)_text=(.+)\.(?:jpg|png|jpeg)',Path(filename).name);assert match,filename
            label,word=match.groups();rr.append(dict(id=Path(filename).name,source_id=Path(filename).name,bank='RTA100',object_label=label,attack_word=word,archive=str(root/'Defense-Prefix/rta100.zip'),member=filename))
    assert len(rr)==4486
    code=root/'SCAM/code/main_vlm_openclip.py';parsed=ast.parse(code.read_text())
    templates=next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TEMPLATES' for t in n.targets));assert len(templates)==32
    EX.mkdir(exist_ok=True);dump(EX/'rows.json',rr)
    # Snapshot additional official RTA task definition before scores.
    import requests
    rta_code=EX/'official_rta100_source.txt'
    if not rta_code.exists():
        url=f"https://raw.githubusercontent.com/azuma164/Defense-Prefix/{meta['defense_prefix_revision']}/datasets/rta100.py"
        response=requests.get(url,timeout=30);response.raise_for_status();rta_code.write_text(response.text)
    plan=dict(models=registry(),Defense_Prefix=dict(path=str(root/'Defense-Prefix/learned_token/dp_vit-b32.pt'),sha256=sha(root/'Defense-Prefix/learned_token/dp_vit-b32.pt'),upstream_sha256=sha(root/'Defense-Prefix/utils/non_nv.py')),scam_templates=templates,rta_templates=['a photo of a {}.'],files={str(p):sha(p) for p in [Path(__file__),CODE/'EXTERNAL_PROTOCOL.md',OUT/'selection.json',OUT/'external_downloads.json',EX/'rows.json',code,rta_code]},image_counts={bank:sum(r['bank']==bank for r in rr) for bank in ('SCAM','NoSCAM','SynthSCAM','RTA100')},rta_vocabulary=sorted(set(x for r in rr if r['bank']=='RTA100' for x in (r['object_label'],r['attack_word']))))
    assert not (EX/'protocol.json').exists();dump(EX/'protocol.json',plan);(EX/'protocol.sha256').write_text(sha(EX/'protocol.json')+'\n')
    print('FROZEN EXTERNAL',plan['image_counts'],'RTA classes',len(plan['rta_vocabulary']),flush=True)

def verify_external():
    cfg=json.loads((EX/'protocol.json').read_text());assert sha(EX/'protocol.json')==(EX/'protocol.sha256').read_text().strip()
    for p,h in cfg['files'].items():assert sha(p)==h,p
    for r in cfg['models'].values():
        if r:assert sha(r['checkpoint'])==r['sha256']
    assert sha(cfg['Defense_Prefix']['path'])==cfg['Defense_Prefix']['sha256']
    return cfg

def prefix_encode(model,tokens,star,prefix):
    cast=model.transformer.get_cast_dtype();x=model.token_embedding(tokens).to(cast)
    ii,jj=torch.where(tokens==star);assert len(ii)==len(tokens) and len(set(ii.tolist()))==len(tokens)
    x[ii,jj]=prefix[0].to(cast)
    x=model.transformer(x+model.positional_embedding.to(cast),attn_mask=model.attn_mask)
    x=model.ln_final(x)
    return x[torch.arange(len(tokens),device=tokens.device),tokens.argmax(-1)]@model.text_projection

def check_prefix(model,tok,cfg):
    prefix=torch.load(cfg['Defense_Prefix']['path'],map_location='cuda',weights_only=True);assert prefix.shape==(1,512)
    root=Path(json.loads((OUT/'external_downloads.json').read_text())['storage']);path=root/'Defense-Prefix/utils/non_nv.py'
    assert sha(path)==cfg['Defense_Prefix']['upstream_sha256']
    spec=importlib.util.spec_from_file_location('verified_official_defense_prefix',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    facade=types.SimpleNamespace(dtype=torch.float32,token_embedding=model.token_embedding,positional_embedding=model.positional_embedding,ln_final=model.ln_final,text_projection=model.text_projection,transformer=lambda x:model.transformer(x.permute(1,0,2),attn_mask=model.attn_mask).permute(1,0,2))
    tokens=tok(['a photo of a * cat.','a photo of a * traffic light.','a sculpture of a * apple.']).cuda();star=int(tok(['*'])[0,1])
    with torch.no_grad():
        x=prefix_encode(model,tokens,star,prefix);y=module.encode_text_with_learnt_tokens(facade,tokens,star,prefix.unsqueeze(0),is_emb=False)
        assert torch.allclose(x,y,atol=1e-5,rtol=1e-5)
        ordinary=model.encode_text(tokens);identity=prefix_encode(model,tokens,star,model.token_embedding.weight[star:star+1])
        assert torch.allclose(ordinary,identity,atol=1e-5,rtol=1e-5)
    dump(EX/'prefix_tests.json',dict(upstream_equivalence_max_error=float((x-y).abs().max()),identity_equivalence_max_error=float((ordinary-identity).abs().max()),official_function_sha256=sha(path),prefix_sha256=cfg['Defense_Prefix']['sha256']))
    return prefix,star

class ArchiveImages(Dataset):
    def __init__(self,rr,prep):self.rr,self.prep,self.handles=rr,prep,{}
    def __len__(self):return len(self.rr)
    def __getitem__(self,i):
        r=self.rr[i];path=r['archive']
        if path not in self.handles:self.handles[path]=zipfile.ZipFile(path)
        with Image.open(io.BytesIO(self.handles[path].read(r['member']))) as im:return self.prep(im.convert('RGB'))

def encode():
    cfg=verify_external();torch.set_num_threads(4);rr=json.loads((EX/'rows.json').read_text());model,prep,tok=load_model();prefix,star=check_prefix(model,tok,cfg)
    cache=Path(json.loads((OUT/'protocol.json').read_text())['cache'])/'external';cache.mkdir(exist_ok=True)
    p=cache/'images.npy'
    if not p.exists():
        ff=[]
        with torch.no_grad():
            for im in DataLoader(ArchiveImages(rr,prep),batch_size=128,num_workers=8,pin_memory=True,worker_init_fn=worker_init):
                with torch.autocast('cuda',dtype=torch.float16):v=model.encode_image(im.cuda(non_blocking=True))
                ff.append(norm(v.float()).cpu().numpy())
        np.save(p,np.concatenate(ff))
    for bank,templates in [('SCAM',cfg['scam_templates']),('RTA100',cfg['rta_templates'])]:
        target=cache/f'{bank}_text.npz'
        if target.exists():continue
        sub=[r for r in rr if (r['bank']!='RTA100')==(bank=='SCAM')]
        labels=sorted(set(x for r in sub for x in (r['object_label'],r['attack_word'])))
        prompts=[t.format(label) for label in labels for t in templates];prefix_prompts=[t.format('* '+label) for label in labels for t in templates]
        base=[];defended=[]
        with torch.no_grad():
            for i in range(0,len(prompts),128):
                base.append(model.encode_text(tok(prompts[i:i+128]).cuda()).cpu())
                defended.append(prefix_encode(model,tok(prefix_prompts[i:i+128]).cuda(),star,prefix).cpu())
        b=norm(torch.cat(base).reshape(len(labels),len(templates),-1).mean(1)).numpy()
        d=norm(torch.cat(defended).reshape(len(labels),len(templates),-1).mean(1)).numpy()
        np.savez_compressed(target,base=b,defense_prefix=d);dump(cache/f'{bank}_labels.json',labels)
        print('ENCODED TEXT',bank,len(labels),flush=True)
    dump(EX/'cache.json',dict(path=str(cache),files={str(p):sha(p) for p in cache.glob('*') if p.is_file()},rows_sha256=sha(EX/'rows.json')))

def score():
    cfg=verify_external();assert not (EX/'complete.json').exists();torch.set_num_threads(4)
    meta=json.loads((EX/'cache.json').read_text());cache=Path(meta['path'])
    for p,h in meta['files'].items():assert sha(p)==h,p
    rr=json.loads((EX/'rows.json').read_text());v=np.load(cache/'images.npy');allrows=[];summaries=[];comparisons=[]
    for bank in ('SCAM','NoSCAM','SynthSCAM','RTA100'):
        ix=np.array([i for i,r in enumerate(rr) if r['bank']==bank]);rows=[rr[i] for i in ix];tbank='SCAM' if bank!='RTA100' else bank
        z=np.load(cache/f'{tbank}_text.npz');labels=json.loads((cache/f'{tbank}_labels.json').read_text());labelids={n:i for i,n in enumerate(labels)}
        yi=np.array([labelids[r['object_label']] for r in rows]);wi=np.array([labelids[r['attack_word']] for r in rows]);vv=v[ix];models={**cfg['models'],'Defense_Prefix':None}
        for name,r in models.items():
            t=torch.tensor(z['defense_prefix'] if name=='Defense_Prefix' else z['base'],device='cuda')
            if r:
                a=TextLoRAAdapter(512,r=64,alpha=64,dropout=.05).cuda();a.load_state_dict(torch.load(r['checkpoint'],map_location='cpu')['state_dict']);a.eval()
                with torch.no_grad():t=norm(a(t))
            scores=vv@t.cpu().numpy().T;obj=scores[np.arange(len(rows)),yi];attack=scores[np.arange(len(rows)),wi]
            pred=scores.argmax(1);correct=obj>attack;top1=pred==yi
            table=pd.DataFrame([dict(id=r['id'],source_id=r['source_id'],bank=bank,model=name,object_label=r['object_label'],attack_word=r['attack_word'],object_score=float(obj[j]),attack_score=float(attack[j]),margin=float(obj[j]-attack[j]),correct=bool(correct[j]),top1=bool(top1[j]),predicted_label=labels[pred[j]]) for j,r in enumerate(rows)])
            table.to_csv(EX/f'{bank}_{name}.csv',index=False);allrows.append(table)
            row=dict(bank=bank,model=name,n=len(rows),accuracy=100*float(correct.mean()),mean_margin=float((obj-attack).mean()),primary_metric='top1' if bank=='RTA100' else 'pairwise',top1=100*float(top1.mean()) if bank=='RTA100' else None)
            summaries.append(row);print('EXTERNAL',json.dumps(row),flush=True)
    full=pd.concat(allrows,ignore_index=True)
    for bank in ('SCAM','NoSCAM','SynthSCAM','RTA100'):
        x=full[(full.bank==bank)&(full.model=='IS')].set_index('source_id').sort_index()
        for other in ('ranking','same_scale_ablation','frozen','Defense_Prefix'):
            y=full[(full.bank==bank)&(full.model==other)].set_index('source_id').sort_index();assert x.index.equals(y.index)
            for metric in (['correct','top1'] if bank=='RTA100' else ['correct']):
                comparisons.append(dict(bank=bank,other=other,metric=metric,**interval(100*(x[metric].astype(float)-y[metric].astype(float)),np.ones(len(x)),x.index)))
    # Actual matched-source word-removal contrasts, not fabricated paired objects.
    matched=[];matched_arrays={}
    def paired(model,bank):return full[(full.model==model)&(full.bank==bank)].set_index('source_id').sort_index()
    frozen_clean=paired('frozen','NoSCAM')
    for bank in ('SCAM','SynthSCAM'):
        frozen_attack=paired('frozen',bank);assert frozen_clean.index.equals(frozen_attack.index)
        bug=frozen_clean.correct.to_numpy()&~frozen_attack.correct.to_numpy()
        for name in models:
            clean=paired(name,'NoSCAM');attack=paired(name,bank);assert clean.index.equals(attack.index)
            repaired=bug&attack.correct.to_numpy();retained=repaired&clean.correct.to_numpy();drift=clean.margin.to_numpy()-attack.margin.to_numpy()
            row=dict(bank=bank,model=name,n=len(clean),frozen_bugs=int(bug.sum()),repaired=int(repaired.sum()),retained=int(retained.sum()),repair_rate=100*float(repaired.sum()/max(1,bug.sum())),retained_repair_rate=100*float(retained.sum()/max(1,bug.sum())),mean_abs_word_removal_effect=float(abs(drift).mean()))
            matched.append(row);matched_arrays[(bank,name)]=dict(ids=clean.index.to_numpy(),bug=bug,retained=retained,drift=drift)
        x=matched_arrays[(bank,'IS')]
        for other in ('ranking','same_scale_ablation','frozen','Defense_Prefix'):
            y=matched_arrays[(bank,other)]
            comparisons.append(dict(bank=bank,other=other,metric='retained_repair',**interval(100*(x['retained'].astype(float)-y['retained'].astype(float)),x['bug'].astype(float),x['ids'])))
            comparisons.append(dict(bank=bank,other=other,metric='absolute_word_removal_effect',**interval(abs(x['drift'])-abs(y['drift']),np.ones(len(x['ids'])),x['ids'])))
    pd.DataFrame(summaries).to_csv(EX/'summary.csv',index=False);pd.DataFrame(matched).to_csv(EX/'paired_source_audit.csv',index=False);pd.DataFrame(comparisons).to_csv(EX/'intervals.csv',index=False)
    lines=['# External typographic retest: fixed seed42 checkpoints','',
           'No benchmark-specific training or checkpoint choice. OpenAI ViT-B/32 throughout. Defense-Prefix is an external released defense with different original training, not a matched-budget local arm.','',
           '| Benchmark | Model | N | Pairwise % | Official RTA top1 % |','|---|---|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['bank']} | {r['model']} | {r['n']} | {r['accuracy']:.2f} | "+(f"{r['top1']:.2f}" if r['top1'] is not None else '—')+' |')
    lines+=['','SCAM variants use the official32-template raw-embedding mean then normalization. RTA uses the official single-template, full-label-union classifier; its pairwise column is secondary. Defense-Prefix insertion was checked against upstream code and identity substitution.','',
           '## Actual paired source audit','','| Variant | Model | Frozen bugs | Retained repaired | Retained repair % | Mean absolute word-removal effect |','|---|---|---:|---:|---:|---:|']
    for r in matched:lines.append(f"| {r['bank']} | {r['model']} | {r['frozen_bugs']} | {r['retained']} | {r['retained_repair_rate']:.2f} | {r['mean_abs_word_removal_effect']:.5f} |")
    lines+=['','NoSCAM is the digitally blanked note, not an independently photographed clean scene. The measured difference is the actual word-removal edit, not the unobserved congruent-word interaction from the training bank.','',
           '## IS minus selected ranking','','| Benchmark | Metric | Difference | Paired95% interval |','|---|---|---:|---:|']
    for r in comparisons:
        if r['other']=='ranking':lines.append(f"| {r['bank']} | {r['metric']} | {r['mean']:+.5f} | [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] |")
    lines+=['','5000 paired source-cluster percentile bootstrap samples, conditional on one training seed. All sources retained; no color/vocabulary/score-defined subset selection. Full per-example margins and predictions are saved.','']
    (EX/'REPORT.md').write_text('\n'.join(lines));dump(EX/'complete.json',dict(protocol_sha256=sha(EX/'protocol.json'),files={str(p.relative_to(EX)):sha(p) for p in EX.glob('*.csv')},prefix_tests=json.loads((EX/'prefix_tests.json').read_text())))

def main():
    log();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['manifest','encode','score']);a=ap.parse_args();globals()[a.action]()

if __name__=='__main__':main()
