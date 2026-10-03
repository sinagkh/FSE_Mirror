"""Official strong-backbone BadNet-Stripes victim and published PAR defense."""
import argparse
import ast
import importlib.util
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import open_clip
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset; from torch.utils.data import DataLoader
import mirror.cases.backdoor.backdoor as old
from mirror.cases.backdoor.common import ROOT; from mirror.cases.backdoor.common import OUT; from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha; from mirror.cases.backdoor.common import temp_root

POUT=OUT/'backdoor_par'
VENDOR=ROOT/'mirror/cases/backdoor/vendor/PerturbAndRecover'
STATES=old.STATES


def prepare():
    oldcfg=old.verify();POUT.mkdir(parents=True,exist_ok=True);assert not (POUT/'protocol.json').exists()
    models={name:json.loads((OUT/'assets'/('PAR_'+name+'.json')).read_text()) for name in ('victim','PAR')}
    for bank in ('train','development'):dump(POUT/(bank+'.json'),json.loads((old.BOUT/(bank+'.json')).read_text()))
    files=[Path(__file__),VENDOR/'backdoor/utils.py',VENDOR/'pkgs/openai/model.py',VENDOR/'eval/validate.py',VENDOR/'eval/eval_data.py']
    p=dict(stage='development diagnostic; published victim and defense, no IS outcomes',models=models,
      rationale='CC3M3M victim has0%genuine-target recall on development, making preservation uninformative; use released poisoned OpenAIB32 and paired PAR defense with the exact published stripe trigger.',
      backbone='OpenAI ViT-B/32',target='banana',vocabulary=oldcfg['vocabulary'],states=STATES,
      rendering='verbatim PAR apply_trigger with badnet_rs, random position,16x16, image_size224; same per-source seeds and equal-support mean-patch sham as original pilot',
      data='same COCO/LVIS development and training IDs as CC3M diagnostic; no scores used to choose new sources',
      comparison='PAR released cleaned weights: operational comparison; trigger-informed IS/ranking receive extra trigger and class-label knowledge',
      cache=str(temp_root()/'backdoor_par'),files={str(p):sha(p) for p in files},
      manifests={bank:sha(POUT/(bank+'.json')) for bank in ('train','development')})
    dump(POUT/'protocol.json',p);dump(POUT/'protocol_hash.json',dict(sha256=sha(POUT/'protocol.json')))


def verify():
    p=POUT/'protocol.json';assert sha(p)==json.loads((POUT/'protocol_hash.json').read_text())['sha256']
    cfg=json.loads(p.read_text())
    for path,h in cfg['files'].items():assert sha(path)==h,path
    for bank,h in cfg['manifests'].items():assert sha(POUT/(bank+'.json'))==h
    return cfg


def trigger_function():
    path=VENDOR/'backdoor/utils.py';tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('random_choice','get_init_patch','apply_trigger')]
    assert len(selected)==3
    scope=dict(torch=torch,transforms=transforms,Image=Image,np=np,random=random,os=os,math=math,F=torch.nn.functional)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
    return scope['apply_trigger']


class Render(old.Render):
    def __init__(self):self.native=trigger_function()
    def trigger(self,image,**kwargs):
        kwargs['patch_type']='badnet_rs';return self.native(image,**kwargs)


class Sources(Dataset):
    def __init__(self,rows,prep):self.rows,self.prep,self.render=rows,prep,Render()
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return torch.stack([self.prep(im) for im in self.render(self.rows[i])])


def model_load(name):
    cfg=verify();entry=cfg['models'][name];assert sha(entry['state_path'])==entry['state_sha256']
    state=torch.load(entry['state_path'],map_location='cpu',weights_only=True)
    model=open_clip.create_model('ViT-B-32',pretrained=None,force_quick_gelu=True)
    model.load_state_dict(state,strict=True);model=model.cuda().float().eval().requires_grad_(False)
    from open_clip.transform import image_transform
    prep=image_transform(224,is_train=False,mean=(.48145466,.4578275,.40821073),std=(.26862954,.26130258,.27577711))
    return model,prep,open_clip.get_tokenizer('ViT-B-32'),state


@torch.no_grad()
def encode(name,bank):
    cfg=verify();torch.set_num_threads(4);dest=Path(cfg['cache'])/name;dest.mkdir(parents=True,exist_ok=True)
    target=dest/(bank+'.npy');assert not target.exists()
    model,prep,tok,state=model_load(name)
    spec=importlib.util.spec_from_file_location('native_par_model',VENDOR/'pkgs/openai/model.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    native=native.build(dict(state),pretrained=True).float().cuda().eval();native.load_state_dict(state,strict=True)
    rows=json.loads((POUT/(bank+'.json')).read_text());xx=torch.stack([prep(im) for im in Render()(rows[0])]).cuda()
    tt=tok(['a photo of a banana.','a photo of a cat.']).cuda()
    error_i=float(abs(old.typo.norm(model.encode_image(xx))-old.typo.norm(native.get_image_features(xx))).max())
    error_t=float(abs(old.typo.norm(model.encode_text(tt))-old.typo.norm(native.get_text_features(tt))).max())
    assert error_i<3e-5 and error_t<3e-5,(error_i,error_t)
    dump(POUT/(name+'_parity.json'),dict(image_error=error_i,text_error=error_t,strict_state_load=True))
    del native,state
    textfile=POUT/(name+'_texts.pt')
    if not textfile.exists():
        pp=[p.format(n) for n in cfg['vocabulary'] for p in old.typo.TEMPLATES]
        zz=[old.typo.norm(model.encode_text(tok(pp[j:j+128]).cuda())).cpu() for j in range(0,len(pp),128)]
        text=old.typo.norm(torch.cat(zz).reshape(len(cfg['vocabulary']),3,-1).mean(1))
        torch.save(dict(features=text,vocabulary=cfg['vocabulary']),textfile)
    ff=[]
    for j,x in enumerate(DataLoader(Sources(rows,prep),batch_size=32,num_workers=8,pin_memory=True,worker_init_fn=old.typo.worker_init)):
        with torch.autocast('cuda',dtype=torch.float16):v=model.encode_image(x.flatten(0,1).cuda(non_blocking=True))
        ff.append(old.typo.norm(v.float()).reshape(len(x),4,-1).cpu().numpy())
        if j%25==0:print('PAR ENCODE',name,bank,32*j,'/',len(rows),flush=True)
    arr=np.concatenate(ff);np.save(target,arr)
    dump(POUT/(name+'_'+bank+'_features.json'),dict(path=str(target),sha256=sha(target),shape=list(arr.shape),manifest_sha256=sha(POUT/(bank+'.json'))))


def diagnose():
    cfg=verify();rows=json.loads((POUT/'development.json').read_text());y=np.array([cfg['vocabulary'].index(r['label']) for r in rows])
    nt=y!=len(cfg['vocabulary'])-1;summary={}
    for name in ('victim','PAR'):
        m=json.loads((POUT/(name+'_development_features.json')).read_text());assert sha(m['path'])==m['sha256']
        v=np.load(m['path']);t=torch.load(POUT/(name+'_texts.pt'),map_location='cpu',weights_only=True)['features'].numpy()
        scores=v@t.T;pred=scores.argmax(-1);good=pred==y[:,None]
        margins=scores[np.arange(len(y))[:,None],np.arange(4)[None,:],y[:,None]]-scores[:,:,-1]
        stats={state+'_top1':float(good[:,j].mean()) for j,state in enumerate(STATES)}
        stats.update(n=len(y),asr=float((pred[nt,1]==len(t)-1).mean()),genuine_target_clean=float(good[~nt,0].mean()),
          genuine_target_attack=float(good[~nt,1].mean()),clean_correct_trigger_wrong=int((good[:,0]&~good[:,1]).sum()),
          clean_correct_sham_wrong=int((good[:,0]&~good[:,3]).sum()),interaction_abs=float(abs(margins[nt,1]-margins[nt,0]).mean()))
        np.savez_compressed(POUT/(name+'_development_scores.npz'),scores=scores,labels=y,ids=np.array([r['id'] for r in rows]))
        summary[name]=stats
    dump(POUT/'development_diagnosis.json',summary);print('PAR DIAGNOSIS',summary,flush=True)


if __name__=='__main__':
    command();p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','encode','diagnose'])
    p.add_argument('--model',choices=['victim','PAR'],default='victim');p.add_argument('--bank',choices=['train','development'],default='development')
    a=p.parse_args()
    if a.action=='encode':encode(a.model,a.bank)
    else:globals()[a.action]()
