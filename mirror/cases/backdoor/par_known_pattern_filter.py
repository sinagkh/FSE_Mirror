"""Known-pattern input control; uses no ground-truth patch coordinates or labels."""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mirror.cases.backdoor.par_v2_confirmation as evaluation
import mirror.cases.backdoor.par_diagnostic as data
from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha

RUN=evaluation.RUN/'known_pattern_control'


def windows(a,h,w):
    s=np.pad(a.astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0)))
    return s[h:,w:]-s[:-h,w:]-s[h:,:-w]+s[:-h,:-w]


def filter_pixels(image):
    # Verbatim public renderer converts stripe values-1/+1 to uint8 values1/255.
    # A patch is16x16, constant vertically, and has horizontal color variation.
    extreme=np.isin(image,[1,255]).all(-1)
    vertical=(image[1:]==image[:-1]).all(-1)
    horizontal=(image[:,1:]!=image[:,:-1]).any(-1)
    candidates=(windows(extreme,16,16)==256)&(windows(vertical,15,16)==240)&(windows(horizontal,16,15)>0)
    mask=np.zeros(image.shape[:2],bool)
    for y,x in zip(*np.where(candidates)):mask[y:y+16,x:x+16]=True
    result=image.copy()
    if mask.any() and (~mask).any():result[mask]=np.rint(image[~mask].mean(0)).astype(np.uint8)
    return result,int(mask.sum())


def prepare():
    evaluation.verify();RUN.mkdir(parents=True,exist_ok=True);assert not (RUN/'protocol.json').exists()
    dump(RUN/'protocol.json',dict(stage='post-specified baseline-completeness check; main ImageNetV2 outcomes already known, no model or detector tuning',
      source_sha256=sha(__file__),parent_sha256=sha(evaluation.RUN/'protocol.json'),
      operation='detect16x16exact binary vertical stripe windows using public renderer pixel alphabet; fill union with mean outside detected region; abstain if wholeimage masked',
      access='trigger family/size/quantization known, as in supplied intervention; no seed, sourceID, position annotation, class label or model scores used by filter',
      weights='unchanged poisoned victim, official1000class/80template ensemble',
      evaluation='all10000sources and all5registeredstates, same endpoints; clean false detections retained',
      comparison='input sanitization with exact known-pattern structure, versus an internal parameter repair; report rather than select by outcome'))
    # Score-free positive/negative renderer tests.
    rng=np.random.default_rng(646611);image=rng.integers(10,240,(224,224,3),dtype=np.uint8)
    clean,count=filter_pixels(image);assert count==0 and np.array_equal(clean,image)
    attacked=image.copy();attacked[30:46,90:106]=rng.choice([1,255],size=(1,16,3)).astype(np.uint8)
    fixed,count=filter_pixels(attacked);assert count==256 and np.array_equal(fixed[:30],image[:30])
    dump(RUN/'unit_test.json',dict(random_clean_unchanged=True,known_patch_detected_pixels=count))


@torch.no_grad()
def evaluate():
    cfg=json.loads((RUN/'protocol.json').read_text());assert sha(__file__)==cfg['source_sha256'];evaluation.verify()
    assert not (RUN/'summary.json').exists();model,prep,tok,state=data.model_load('victim');del state
    text=torch.load(evaluation.RUN/'victim_text_prototypes.pt',map_location='cuda',weights_only=True)['features']
    rows=json.loads((evaluation.RUN/'manifest.json').read_text());labels=np.array([r['label'] for r in rows]);N=len(rows)
    predictions=np.zeros((N,5),np.int32);areas=np.zeros((N,5),np.int32);margins=np.zeros((N,5),np.float32)
    mean=np.array([.48145466,.4578275,.40821073]);std=np.array([.26862954,.26130258,.27577711]);start=time.monotonic();seen=[]
    loader=DataLoader(evaluation.Sources(prep),batch_size=16,num_workers=4,pin_memory=True)
    for j,(indices,images) in enumerate(loader):
        ix=indices.numpy();a=images.numpy().transpose(0,1,3,4,2);pixels=np.clip(np.rint((a*std+mean)*255),0,255).astype(np.uint8)
        for n in range(len(ix)):
            for k in range(5):pixels[n,k],areas[ix[n],k]=filter_pixels(pixels[n,k])
        tensor=torch.from_numpy(((pixels/255.-mean)/std).transpose(0,1,4,2,3).astype(np.float32)).flatten(0,1).cuda()
        with torch.autocast('cuda',dtype=torch.float16):v=F.normalize(model.encode_image(tensor).float(),dim=-1)
        scores=(v@text.T).reshape(len(ix),5,1000);y=torch.tensor(labels[ix],device='cuda')
        predictions[ix]=scores.argmax(-1).cpu().numpy();margins[ix]=(scores.gather(-1,y[:,None,None].expand(-1,5,1)).squeeze(-1)-scores[:,:,954]).cpu().numpy()
        seen.extend(ix.tolist())
        if j%50==0:print('KNOWN PATTERN CONTROL',len(seen),'/',N,flush=True)
    assert sorted(seen)==list(range(N));good=predictions==labels[:,None];nt=labels!=954
    summary={state+'_top1':float(good[:,k].mean()) for k,state in enumerate(evaluation.STATES)}
    summary.update(asr_nonbanana=float((predictions[nt,2]==954).mean()),native_clean_detection_rate=float((areas[:,0]>0).mean()),
      trigger_detection_rate=float((areas[:,2]>0).mean()),trigger_induced_failure=float((good[:,1]&~good[:,2]).mean()),
      target_interaction_abs=float(abs(margins[nt,2]-margins[nt,1]).mean()),seconds=time.monotonic()-start)
    np.savez_compressed(RUN/'records.npz',pred=predictions,mask_area=areas,target_margin=margins,labels=labels,ids=np.array([r['id'] for r in rows]))
    dump(RUN/'summary.json',summary);print('KNOWN PATTERN SUMMARY',summary,flush=True)


if __name__=='__main__':
    command();torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate']);globals()[p.parse_args().action]()
