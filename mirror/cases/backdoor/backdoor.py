"""Published BadNet victim and CleanCLIP: faithful trigger, paired diagnosis."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import open_clip
from PIL import Image; from PIL import ImageDraw
from torch.utils.data import Dataset; from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor; from torchvision.transforms.functional import to_pil_image
from mirror.cases.backdoor.common import ROOT; from mirror.cases.backdoor.common import OUT; from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha; from mirror.cases.backdoor.common import temp_root

sys.path.insert(0,str(ROOT/'mirror/cases/typography'))
import mirror.cases.typography.diagnose as typo

BOUT=OUT/'backdoor'
VENDOR=ROOT/'mirror/cases/backdoor/vendor/CleanCLIP'
STATES=['clean','trigger_1','trigger_2','mean_patch_sham']


def native_module():
    path=VENDOR/'pkgs/openai/model.py'
    spec=importlib.util.spec_from_file_location('official_cleanclip_model',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def trigger_function():
    # Execute only the inspected official trigger function, not module-level
    # imports of tracking services or unrelated training/data helpers.
    path=VENDOR/'backdoor/utils.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_trigger')
    scope=dict(torch=torch,transforms=transforms,Image=Image,np=np,random=random,os=os,
               F=torch.nn.functional)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    return scope['apply_trigger']


def prepare():
    BOUT.mkdir(parents=True,exist_ok=True)
    assert not (BOUT/'protocol.json').exists()
    cfgpath=ROOT/'data/typography/class_support/filtered_data_protocol.json'
    cfg=json.loads(cfgpath.read_text());vocab=cfg['training_classes']+['banana']
    files=[Path(__file__),cfgpath,VENDOR/'backdoor/utils.py',VENDOR/'pkgs/openai/model.py']
    manifests={}
    for bank in ('train','development'):
        originals=[Path(cfg['sources']['train']['manifest'])] if bank=='train' else [typo.OUT/'development.json',Path(cfg['sources']['development_new']['manifest'])]
        support=OUT/'target_support'/('banana_'+bank+'.json');files.extend(originals+[support])
        rows=[r for p in originals+[support] for r in json.loads(p.read_text())]
        assert len({r['image_id'] for r in rows})==len(rows)
        dump(BOUT/(bank+'.json'),rows);manifests[bank]=sha(BOUT/(bank+'.json'))
    models={}
    for name,suffix in [('victim','006'),('CleanCLIP','004')]:
        path=OUT/'assets'/f'cleanclip_badnet_{suffix}.json';files.append(path)
        models[name]=json.loads(path.read_text())
    pp=dict(status='development diagnosis; published weights, not new training',
            model='RN50, CC3M3M/1500BadNet',models=models,vocabulary=vocab,target='banana',states=STATES,
            training_data='existing COCO/LVIS train sources, with annotation-only banana support',
            trigger='Official apply_trigger: patch_type=random, patch_size=16, patch_location=random; fixed per-source seeds for paired methods',
            preprocessing='COCO pilot uses existing224crop; native ImageNet evaluation uses official original-image resize224 before trigger, then CLIP preprocessing',
            sham='same16x16location as trigger1, constant per-channel image mean',
            seed_rule='SHA256(sourceID+realization), first8hex, modulo2^31',
            pairing='same underlying clean image, trigger realization, captions and vocabulary',
            comparison='Released native CleanCLIP used100kcleanCC pairs; our future repairs know trigger and labels: operational, not information-matched, comparison',
            cache=str(temp_root()/'backdoor'),files={str(p):sha(p) for p in files},manifests=manifests)
    dump(BOUT/'protocol.json',pp);dump(BOUT/'protocol_hash.json',dict(sha256=sha(BOUT/'protocol.json')))
    print('BACKDOOR PROTOCOL FROZEN',flush=True)


def verify():
    p=BOUT/'protocol.json';assert sha(p)==json.loads((BOUT/'protocol_hash.json').read_text())['sha256']
    cfg=json.loads(p.read_text())
    for path,h in cfg['files'].items():assert sha(path)==h,path
    for bank,h in cfg['manifests'].items():assert sha(BOUT/(bank+'.json'))==h
    return cfg


def seed_for(row,variant):
    return int(hashlib.sha256((row['id']+f':badnet:{variant}').encode()).hexdigest()[:8],16)%(2**31)


class Render:
    def __init__(self):self.trigger=trigger_function()
    def __call__(self,row):
        clean=typo.render(row)[0];images=[clean]
        old_py=random.getstate()
        try:
            for variant in (1,2):
                seed=seed_for(row,variant)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed);random.seed(seed)
                    changed=self.trigger(clean,patch_size=16,patch_type='random',patch_location='random')
                images.append(changed)
        finally:random.setstate(old_py)
        rng=random.Random(seed_for(row,1));h,w=rng.randint(0,207),rng.randint(0,207)
        x=to_tensor(clean);x[:,h:h+16,w:w+16]=x.mean((1,2),keepdim=True)
        images.append(to_pil_image(x))
        outside=np.ones((224,224),bool);outside[h:h+16,w:w+16]=False
        for im in (images[1],images[3]):
            assert np.array_equal(np.asarray(im)[outside],np.asarray(clean)[outside])
        return images


class Sources(Dataset):
    def __init__(self,rows,prep):self.rows,self.prep,self.render=rows,prep,Render()
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return torch.stack([self.prep(im) for im in self.render(self.rows[i])])


def model_load(name,device='cuda'):
    cfg=verify();entry=cfg['models'][name]
    assert sha(entry['state_path'])==entry['state_sha256']
    state=torch.load(entry['state_path'],map_location='cpu',weights_only=True)
    model=open_clip.create_model('RN50',pretrained=None,force_quick_gelu=True)
    model.load_state_dict(state,strict=True)
    model=model.to(device).float().eval().requires_grad_(False)
    from open_clip.transform import image_transform
    prep=image_transform(224,is_train=False,mean=(.48145466,.4578275,.40821073),std=(.26862954,.26130258,.27577711))
    return model,prep,open_clip.get_tokenizer('RN50'),state


@torch.no_grad()
def equivalence():
    torch.set_num_threads(4);model,prep,tok,state=model_load('victim')
    native=native_module().build(dict(state),pretrained=True).float()
    # Native constructor may round while initializing; reload the original
    # float tensors as official main.py does when restoring a checkpoint.
    native.load_state_dict(state,strict=True);native=native.cuda().eval().requires_grad_(False)
    rows=json.loads((BOUT/'development.json').read_text())[:2]
    images=torch.stack([prep(im) for r in rows for im in Render()(r)]).cuda()
    text=tok(['a photo of a banana.','a photo of an elephant.']).cuda()
    a=model.encode_image(images);b=native.get_image_features(images)
    x=model.encode_text(text);y=native.get_text_features(text)
    vi=float(abs(typo.norm(a)-typo.norm(b)).max());te=float(abs(typo.norm(x)-typo.norm(y)).max())
    assert vi<3e-5 and te<3e-5,(vi,te)
    dump(BOUT/'implementation_tests.json',dict(native_image_normalized_error=vi,native_text_normalized_error=te,
        strict_state_load=True,native_trigger_verbatim=True,code_sha256=sha(__file__)))
    print('NATIVE PARITY',vi,te,flush=True)


def gallery():
    verify();rows=json.loads((BOUT/'development.json').read_text())
    selected=sorted(rows,key=lambda r:hashlib.sha256(('backdoor-gallery:'+r['id']).encode()).hexdigest())[:8]
    out=Image.new('RGB',(4*224,8*250),'white');d=ImageDraw.Draw(out);render=Render()
    for i,r in enumerate(selected):
        for j,im in enumerate(render(r)):
            out.paste(im,(224*j,250*i));d.text((224*j,250*i+226),r['label']+' / '+STATES[j],fill='black')
    out.save(BOUT/'gallery.jpg');dump(BOUT/'gallery_ids.json',[r['id'] for r in selected])


@torch.no_grad()
def encode(name,bank):
    cfg=verify();torch.set_num_threads(4);assert (BOUT/'implementation_tests.json').exists()
    dest=Path(cfg['cache'])/name;dest.mkdir(parents=True,exist_ok=True)
    target=dest/(bank+'.npy');assert not target.exists()
    model,prep,tok,state=model_load(name);del state
    textfile=BOUT/(name+'_texts.pt')
    if not textfile.exists():
        prompts=[p.format(n) for n in cfg['vocabulary'] for p in typo.TEMPLATES];tt=[]
        for j in range(0,len(prompts),128):tt.append(typo.norm(model.encode_text(tok(prompts[j:j+128]).cuda())).cpu())
        text=typo.norm(torch.cat(tt).reshape(len(cfg['vocabulary']),3,-1).mean(1))
        torch.save(dict(features=text,vocabulary=cfg['vocabulary']),textfile)
    rows=json.loads((BOUT/(bank+'.json')).read_text());ff=[];clock=time.monotonic()
    for j,images in enumerate(DataLoader(Sources(rows,prep),batch_size=32,num_workers=8,pin_memory=True,worker_init_fn=typo.worker_init)):
        with torch.autocast('cuda',dtype=torch.float16):v=model.encode_image(images.flatten(0,1).cuda(non_blocking=True))
        ff.append(typo.norm(v.float()).reshape(len(images),4,-1).cpu().numpy())
        if j%25==0:print('BACKDOOR ENCODING',name,bank,j*32,'/',len(rows),flush=True)
    values=np.concatenate(ff);np.save(target,values)
    dump(BOUT/(name+'_'+bank+'_features.json'),dict(path=str(target),sha256=sha(target),shape=list(values.shape),seconds=time.monotonic()-clock,
        manifest_sha256=sha(BOUT/(bank+'.json')),state_sha256=cfg['models'][name]['state_sha256']))


def diagnose():
    cfg=verify();rows=json.loads((BOUT/'development.json').read_text())
    y=np.array([cfg['vocabulary'].index(r['label']) for r in rows]);target=cfg['vocabulary'].index('banana');nt=y!=target
    summaries={}
    for name in ('victim','CleanCLIP'):
        meta=json.loads((BOUT/(name+'_development_features.json')).read_text());assert sha(meta['path'])==meta['sha256']
        v=np.load(meta['path']);t=torch.load(BOUT/(name+'_texts.pt'),map_location='cpu',weights_only=True)['features'].numpy()
        s=np.einsum('nsd,cd->nsc',v,t);pred=s.argmax(-1);correct=pred==y[:,None]
        m=s[np.arange(len(y))[:,None],np.arange(4)[None,:],y[:,None]]-s[:,:,target]
        stats={state+'_top1':float(correct[:,j].mean()) for j,state in enumerate(STATES)}
        stats.update(n=len(y),non_target_n=int(nt.sum()),target_n=int((~nt).sum()),
            asr=float((pred[nt,1]==target).mean()),
            clean_correct_trigger_wrong=int((correct[:,0]&~correct[:,1]).sum()),
            clean_correct_sham_wrong=int((correct[:,0]&~correct[:,3]).sum()),
            target_clean_top1=float(correct[~nt,0].mean()),
            target_trigger_top1=float(correct[~nt,1].mean()),
            target_interaction_signed=float((m[nt,1]-m[nt,0]).mean()),
            target_interaction_abs=float(abs(m[nt,1]-m[nt,0]).mean()))
        np.savez_compressed(BOUT/(name+'_development_scores.npz'),scores=s,labels=y,margins=m,ids=np.array([r['id'] for r in rows]))
        summaries[name]=stats
    dump(BOUT/'development_diagnosis.json',summaries)
    print('BACKDOOR DIAGNOSIS',json.dumps(summaries),flush=True)


if __name__=='__main__':
    command();p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','gallery','equivalence','encode','diagnose'])
    p.add_argument('--model',choices=['victim','CleanCLIP'],default='victim')
    p.add_argument('--bank',choices=['train','development'],default='development');a=p.parse_args()
    if a.action=='encode':encode(a.model,a.bank)
    else:globals()[a.action]()
