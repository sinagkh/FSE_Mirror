"""Frozen-recipe ImageNetV2 transfer, with published PAR and paired audits."""
import argparse
import copy
import hashlib
import io
import json
import random
import tarfile
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset; from torch.utils.data import DataLoader; from torch.utils.data import get_worker_info
from torchvision.transforms.functional import to_tensor; from torchvision.transforms.functional import to_pil_image
import mirror.cases.backdoor.par_diagnostic as data
import mirror.cases.backdoor.par_visual_blocks_v3 as repair
from mirror.cases.backdoor.common import OUT; from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha; from mirror.cases.backdoor.common import temp_root

RUN=OUT/'backdoor_par/imagenetv2_confirmation_v1'
STATES=['native_clean','audit_clean','trigger_1','trigger_2','mean_patch_sham']
ARCHIVE=temp_root()/'downloads/imagenetv2-matched-frequency.tar.gz'


def prepare():
    repair.verify();RUN.mkdir(parents=True,exist_ok=True);assert not (RUN/'protocol.json').exists()
    meta=json.loads((OUT/'assets/downloads/imagenetv2-matched-frequency.tar.gz.json').read_text())
    members=[]
    with tarfile.open(ARCHIVE,'r|*') as archive:
        for item in archive:
            if item.isfile() and item.name.lower().endswith(('.jpg','.jpeg','.png')):
                members.append(dict(id=item.name,label=int(Path(item.name).parent.name)))
    assert len(members)==10000 and len(set(r['label'] for r in members))==1000
    dump(RUN/'manifest.json',members)
    files={str(ARCHIVE):sha(ARCHIVE),str(Path(__file__)):sha(__file__),
      str(data.VENDOR/'asset/imagenet/classes.py'):sha(data.VENDOR/'asset/imagenet/classes.py'),
      str(repair.RUN/'protocol.json'):sha(repair.RUN/'protocol.json')}
    for seed in (42,43,44):
        a=[]
        for method in ('ranking','IS'):
            directory=repair.RUN/'runs'/f'{method}_seed{seed}'
            receipt=json.loads((directory/'complete.json').read_text());assert receipt['steps']==1536
            path=directory/'last.pt';assert sha(path)==receipt['checkpoint_sha256']
            files[str(path)]=sha(path);a.append(receipt)
        assert a[0]['initial_sha256']==a[1]['initial_sha256'] and a[0]['sequence_sha256']==a[1]['sequence_sha256']
    dump(RUN/'protocol.json',dict(stage='independent frozen-recipe confirmation; no model scores inspected during registration',
      data='complete official ImageNetV2 matched-frequency,10000images/1000classes; NOT original50000image ImageNet1K',
      asset=meta,manifest_sha256=sha(RUN/'manifest.json'),files=files,
      methods=['victim','PAR']+[f'{m}_seed{s}' for s in (42,43,44) for m in ('ranking','IS')],
      checkpoints='fixed final1536successfulupdates from visual_blocks_v3; no test selection',
      protocol='official PAR1000class names and80template normalized ensemble; exact official badnet_rs16x16 random position',
      states=STATES,geometry='native_clean uses official resize-short-edge/center-crop; audit_clean resizes224square matching native trigger preprocessing; interactions compare trigger against audit_clean, never against native_clean',
      trigger='SHA256(imageID+:badnet:+realization), same rule as development; two independent realizations and equal-support mean sham',
      metrics='native clean Top1/5; audit clean and each attack/sham Top1; banana recall; nonbanana ASR; true-minus-banana interaction; mean allclass interaction; clean-correct-to-attackwrong; behavior/mechanism perimage',
      precision='float32 parameters and scoring; FP16 visual autocast shared across all methods; text ensemble float32',
      comparison='ranking versus IS information/budget matched; released PAR is an operational comparison with different cleanup data and no supplied trigger',
      uncertainty='paired hierarchical bootstrap: resample3trainingseeds and10000sourceimages, keep all states and methods paired,4000replicates,seed646401; descriptive SD across3seeds',
      native_replication_limit='ImageNetV2 secondary transfer evaluation; not claimed as exact reproduction of published ImageNet1K table'))
    dump(RUN/'protocol_hash.json',dict(sha256=sha(RUN/'protocol.json')))


def verify():
    p=RUN/'protocol.json';assert sha(p)==json.loads((RUN/'protocol_hash.json').read_text())['sha256'];cfg=json.loads(p.read_text())
    for file,h in cfg['files'].items():assert sha(file)==h,file
    assert sha(RUN/'manifest.json')==cfg['manifest_sha256']
    return cfg


class Sources(IterableDataset):
    def __init__(self,prep,limit=None):self.prep,self.limit=prep,limit
    def __iter__(self):
        info=get_worker_info();wid,nw=(info.id,info.num_workers) if info else (0,1)
        fn=data.trigger_function();index=0
        with tarfile.open(ARCHIVE,'r|*') as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(('.jpg','.jpeg','.png')):continue
                i=index;index+=1
                if self.limit is not None and i>=self.limit:break
                if i%nw!=wid:continue
                image=Image.open(io.BytesIO(archive.extractfile(member).read())).convert('RGB')
                clean=image.resize((224,224));states=[image,clean];row=dict(id=member.name)
                oldrandom=random.getstate()
                try:
                    for variant in (1,2):
                        seed=data.old.seed_for(row,variant)
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(seed);random.seed(seed)
                            states.append(fn(image,patch_size=16,patch_type='badnet_rs',patch_location='random'))
                finally:random.setstate(oldrandom)
                rng=random.Random(data.old.seed_for(row,1));h,w=rng.randint(0,207),rng.randint(0,207)
                x=to_tensor(clean);x[:,h:h+16,w:w+16]=x.mean((1,2),keepdim=True);states.append(to_pil_image(x))
                outside=np.ones((224,224),bool);outside[h:h+16,w:w+16]=False
                assert np.array_equal(np.asarray(states[2])[outside],np.asarray(clean)[outside])
                yield i,torch.stack([self.prep(im) for im in states])


@torch.no_grad()
def texts(model,tok,name):
    path=RUN/(name+'_text_prototypes.pt')
    if path.exists():return torch.load(path,map_location='cuda',weights_only=True)['features']
    # Official checked-in expression contains only class strings and formatting lambdas.
    config=eval((data.VENDOR/'asset/imagenet/classes.py').read_text(),{'__builtins__':{}})
    assert len(config['classes'])==1000 and config['classes'].index('banana')==954
    rows=[]
    for i,c in enumerate(config['classes']):
        z=F.normalize(model.encode_text(tok([template(c) for template in config['templates']]).cuda()).float(),dim=-1)
        rows.append(F.normalize(z.mean(0),dim=-1))
        if i%200==0:print('TEXT ENSEMBLE',name,i,flush=True)
    t=torch.stack(rows);torch.save(dict(features=t.cpu(),classes=config['classes'],n_templates=len(config['templates'])),path)
    return t


@torch.no_grad()
def evaluate(smoke=False):
    cfg=verify();destination=RUN/('smoke_summary.json' if smoke else 'summary.json');assert not destination.exists()
    model,prep,tok,state=data.model_load('victim');del state
    par,_,_,state=data.model_load('PAR');del state
    tv=texts(model,tok,'victim');tp=texts(par,tok,'PAR')
    tails={}
    for method in cfg['methods'][2:]:
        trained=torch.load(repair.RUN/'runs'/method/'last.pt',map_location='cpu',weights_only=True)['visual_blocks']
        blocks=torch.nn.ModuleList([copy.deepcopy(b) for b in model.visual.transformer.resblocks[10:]])
        for j,block in enumerate(blocks):
            prefix=f'transformer.resblocks.{10+j}.'
            block.load_state_dict({k[len(prefix):]:v for k,v in trained.items() if k.startswith(prefix)},strict=True)
        tails[method]=blocks.eval().requires_grad_(False)
    rows=json.loads((RUN/'manifest.json').read_text());N=16 if smoke else len(rows);labels=np.array([r['label'] for r in rows[:N]])
    # Compact per-example endpoints and exact continuous audit summaries. Full
    # 1000class vectors are saved in float16; predictions/metrics use float32.
    values={m:dict(pred=np.zeros((N,5),np.int32),top5=np.zeros((N,5),bool),
                  target_margin=np.zeros((N,5),np.float32),mean_abs_interaction=np.zeros((N,3),np.float32),
                  true_score=np.zeros((N,5),np.float32)) for m in cfg['methods']}
    full={m:np.zeros((N,5,1000),np.float16) for m in cfg['methods']}
    captured=[]
    handle=model.visual.transformer.resblocks[10].register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    start=time.monotonic();seen=[]
    loader=DataLoader(Sources(prep,N if smoke else None),batch_size=16,num_workers=4,pin_memory=True,prefetch_factor=2)
    for j,(indices,images) in enumerate(loader):
        ix=indices.numpy();x=images.flatten(0,1).cuda(non_blocking=True);y=torch.tensor(labels[ix],device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            features=F.normalize(model.encode_image(x).float(),dim=-1);tokens=captured.pop();assert not captured
            allfeatures={'victim':features,'PAR':F.normalize(par.encode_image(x).float(),dim=-1)}
            for name,blocks in tails.items():
                z=tokens
                for block in blocks:z=block(z)
                pooled,_=model.visual._pool(z);allfeatures[name]=F.normalize((pooled@model.visual.proj).float(),dim=-1)
        for name,v in allfeatures.items():
            scores=(v@(tp if name=='PAR' else tv).T).reshape(len(ix),5,1000)
            true=scores.gather(-1,y[:,None,None].expand(-1,5,1)).squeeze(-1);m=true[:,:,None]-scores
            values[name]['pred'][ix]=scores.argmax(-1).cpu().numpy()
            values[name]['top5'][ix]=(scores.topk(5,-1).indices==y[:,None,None]).any(-1).cpu().numpy()
            values[name]['target_margin'][ix]=(true-scores[:,:,954]).cpu().numpy()
            values[name]['true_score'][ix]=true.cpu().numpy()
            values[name]['mean_abs_interaction'][ix]=abs(m[:,2:]-m[:,1:2]).mean(-1).cpu().numpy()
            full[name][ix]=scores.cpu().numpy().astype(np.float16)
        seen.extend(ix.tolist())
        if j%30==0:print('PAR V2 EVAL',len(seen),'/',N,round(time.monotonic()-start,1),flush=True)
    handle.remove();assert sorted(seen)==list(range(N));nt=labels!=954;summary={}
    for name,v in values.items():
        good=v['pred']==labels[:,None]
        stats={state+'_top1':float(good[:,i].mean()) for i,state in enumerate(STATES)}
        stats.update(native_clean_top5=float(v['top5'][:,0].mean()),asr=float((v['pred'][nt,2]==954).mean()),
          native_asr_all_sources=float((v['pred'][:,2]==954).mean()),banana_clean_top1=float(good[~nt,0].mean()) if (~nt).any() else None,
          banana_attack_top1=float(good[~nt,2].mean()) if (~nt).any() else None,
          trigger_induced_failure_rate=float((good[:,1]&~good[:,2]).mean()),
          target_interaction_abs=float(abs(v['target_margin'][nt,2]-v['target_margin'][nt,1]).mean()),
          allclass_interaction_abs=float(v['mean_abs_interaction'][:,0].mean()))
        summary[name]=stats
        if not smoke:
            np.savez_compressed(RUN/(name+'_records.npz'),**v,labels=labels,ids=np.array([r['id'] for r in rows]),scores=full[name])
    dump(destination,dict(n=N,methods=summary,seconds=time.monotonic()-start,protocol_sha256=sha(RUN/'protocol.json')))
    print('PAR V2 SUMMARY',summary,flush=True)


if __name__=='__main__':
    command();torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','evaluate']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:evaluate(a.action=='smoke')
