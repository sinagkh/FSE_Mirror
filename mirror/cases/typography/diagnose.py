"""Outcome-independent typographic corpus and frozen development diagnosis."""
from mirror.paths import ARTIFACT_ROOT
from pathlib import Path
import os; import sys; import json; import hashlib; import shlex; import argparse; import time
from datetime import datetime; from datetime import timezone
ROOT=ARTIFACT_ROOT;CODE=Path(__file__).resolve().parent
OUT=ROOT/'mirror/cases/typography_20260926'
CACHE=Path(os.environ.get('FSE_TYPO_CACHE','/external-cache/fse_typographic_20260926'))
for key,ending in [('HF_HOME',''),('HF_HUB_CACHE','/hub'),('TRANSFORMERS_CACHE','/transformers')]:
    os.environ[key]=str(ROOT/'clip/fse_color_joint_repair_20260926/hf_cache')+ending
import numpy as np
import torch
from PIL import Image; from PIL import ImageDraw; from PIL import ImageFont; from PIL import ImageOps
from torch.utils.data import Dataset; from torch.utils.data import DataLoader
import open_clip

WEIGHTS=ROOT/'clip/interbind_subjects_20260922/weights/openai_b32/ViT-B-32.pt'
WEIGHT_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
SERIF=Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf')
STATES=['clean','blank','congruent','conflict_1','conflict_2']
TEMPLATES=['a photo of a {}.','a {}.','an image of a {}.']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def key(s):return hashlib.sha256(s.encode()).hexdigest()
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def log():
    OUT.mkdir(exist_ok=True)
    with (OUT/'commands.log').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+' '+shlex.join([sys.executable,*sys.argv])+'\n')
def norm(v):return torch.nn.functional.normalize(v,dim=-1)

def prepare():
    assert not (OUT/'protocol.json').exists()
    path=ROOT/'clip/data/coco/annotations/instances_train2017.json';data=json.loads(path.read_text())
    names={c['id']:c['name'] for c in data['categories']};images={r['id']:r for r in data['images']};pool={}
    for a in data['annotations']:
        name=names[a['category_id']];im=images[a['image_id']];x,y,w,h=a['bbox']
        if name=='person' or a.get('iscrowd') or min(w,h)<64 or a['area']/(im['width']*im['height'])<.08:continue
        group=pool.setdefault(name,{})
        if a['image_id'] not in group or a['area']>group[a['image_id']]['area']:group[a['image_id']]=a
    eligible=sorted([n for n,v in pool.items() if len(v)>=160],key=lambda n:key('typographic-class-v1:'+n))
    seen=eligible[:32];assert len(seen)==32
    held=sorted([n for n,v in pool.items() if n not in seen and len(v)>=64])
    counts={n:len(v) for n,v in pool.items()};used=set();banks={k:[] for k in ('train','development','test_seen','test_heldout')};shortages={}
    for name in sorted(seen+held,key=lambda n:(counts[n],n)):
        want=112 if name in seen else 32
        available=sorted(pool[name].values(),key=lambda a:key(f"typographic-source-v1:{name}:{a['image_id']}"))
        selected=[a for a in available if a['image_id'] not in used][:want]
        if len(selected)<want:
            assert name not in seen,f'Training class shortage {name}: {len(selected)}'
            shortages[name]=len(selected);continue
        for j,a in enumerate(selected):
            im=images[a['image_id']];used.add(a['image_id'])
            split=('train' if j<64 else 'development' if j<80 else 'test_seen') if name in seen else 'test_heldout'
            vocab=seen if name in seen else held;ci=vocab.index(name)
            offset1=1+j%(len(vocab)-1);offset2=1+(j+len(vocab)//2)%(len(vocab)-1)
            wrong=[vocab[(ci+offset1)%len(vocab)],vocab[(ci+offset2)%len(vocab)]]
            assert len(set([name,*wrong]))==3
            banks[split].append(dict(id=f"{split}:{a['id']}",image_id=a['image_id'],ann_id=a['id'],file=im['file_name'],label=name,words=[name,*wrong],bbox=a['bbox'],width=im['width'],height=im['height'],coco_split='train2017'))
    for bank,rr in banks.items():
        rr.sort(key=lambda r:key('typographic-row-v1:'+r['id']));dump(OUT/f'{bank}.json',rr)
    assert len(used)==sum(len(v) for v in banks.values())
    paths=[Path(__file__),CODE/'DIAGNOSTIC_PROTOCOL.md',FONT,SERIF,path,*[OUT/f'{b}.json' for b in banks]]
    assert sha(WEIGHTS)==WEIGHT_SHA
    cfg=dict(states=STATES,templates=TEMPLATES,seen_classes=seen,heldout_classes=held,excluded_shortages=shortages,eligible_counts=counts,counts={b:len(r) for b,r in banks.items()},weights=dict(path=str(WEIGHTS),sha256=WEIGHT_SHA),files={str(p):sha(p) for p in paths},selection_score_blind=True,cache=str(CACHE))
    dump(OUT/'protocol.json',cfg);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    print('FROZEN',cfg['counts'],len(seen),len(held),'shortages',shortages,flush=True)
    gallery(banks['development'][:12])

def verify():
    assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    cfg=json.loads((OUT/'protocol.json').read_text())
    for p,h in cfg['files'].items():assert sha(p)==h,p
    assert sha(WEIGHTS)==WEIGHT_SHA
    return cfg

def render(row,style='standard'):
    with Image.open(ROOT/'clip/data/coco/train2017'/row['file']) as im:
        im=im.convert('RGB');x,y,w,h=row['bbox'];box=(max(0,int(x-.1*w)),max(0,int(y-.1*h)),min(im.width,int(np.ceil(x+1.1*w))),min(im.height,int(np.ceil(y+1.1*h))))
        crop=ImageOps.pad(im.crop(box),(224,224),method=Image.Resampling.BICUBIC,color=(128,128,128))
    rectangle=(11,174,212,212) if style!='top' else (11,11,212,49)
    fontpath=SERIF if style=='serif' else FONT
    probe=ImageDraw.Draw(crop);size=27
    while True:
        font=ImageFont.truetype(str(fontpath),size)
        boxes=[probe.textbbox((0,0),s.upper(),font=font) for s in row['words']]
        if max(b[2]-b[0] for b in boxes)<=190 and max(b[3]-b[1] for b in boxes)<=31:break
        size-=1;assert size>=10
    blank=crop.copy();ImageDraw.Draw(blank).rectangle(rectangle,fill='white')
    ims=[crop,blank]
    for word,b in zip(row['words'],boxes):
        img=blank.copy();draw=ImageDraw.Draw(img);cx=(rectangle[0]+rectangle[2])/2;cy=(rectangle[1]+rectangle[3])/2
        draw.text((cx-(b[2]-b[0])/2-b[0],cy-(b[3]-b[1])/2-b[1]),word.upper(),font=font,fill='black')
        ims.append(img)
    outside=np.ones((224,224),bool);outside[rectangle[1]:rectangle[3]+1,rectangle[0]:rectangle[2]+1]=False
    for im in ims[1:]:assert np.array_equal(np.asarray(im)[outside],np.asarray(crop)[outside])
    return ims

def gallery(rr):
    sheet=Image.new('RGB',(5*224,len(rr)*252),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rr):
        for j,im in enumerate(render(r)):
            sheet.paste(im,(j*224,i*252));draw.text((j*224,i*252+224),r['label']+' / '+STATES[j],fill='black')
    sheet.save(OUT/'gallery.jpg');dump(OUT/'gallery_ids.json',[r['id'] for r in rr])

class Images(Dataset):
    def __init__(self,rr,prep,style):self.rr,self.prep,self.style=rr,prep,style
    def __len__(self):return len(self.rr)
    def __getitem__(self,i):return torch.stack([self.prep(im) for im in render(self.rr[i],self.style)])
def worker_init(i):torch.set_num_threads(1)

def load_model():
    model=open_clip.load_openai_model(str(WEIGHTS),precision='fp32',device='cuda');model.eval().requires_grad_(False)
    from open_clip.transform import image_transform
    prep=image_transform(224,is_train=False,mean=model.visual.image_mean,std=model.visual.image_std)
    return model,prep,open_clip.get_tokenizer('ViT-B-32')

def encode(bank,style='standard'):
    cfg=verify();torch.set_num_threads(4);cache=Path(cfg['cache']);cache.mkdir(parents=True,exist_ok=True)
    target=cache/f'{bank}_{style}.npy'
    if target.exists():
        meta=json.loads((OUT/f'{bank}_{style}_features.json').read_text());assert sha(target)==meta['sha256'];return
    rr=json.loads((OUT/f'{bank}.json').read_text());model,prep,tok=load_model()
    if not (OUT/'texts.pt').exists():
        vocab=cfg['seen_classes']+cfg['heldout_classes'];features=[]
        with torch.no_grad():
            for name in vocab:
                emb=model.encode_text(tok([s.format(name) for s in TEMPLATES]).cuda());features.append(norm(norm(emb).mean(0)))
        torch.save(dict(vocabulary=vocab,features=torch.stack(features).cpu(),templates=TEMPLATES),OUT/'texts.pt')
    ff=[];start=time.monotonic()
    with torch.no_grad():
        for im in DataLoader(Images(rr,prep,style),batch_size=32,num_workers=8,pin_memory=True,worker_init_fn=worker_init):
            with torch.autocast('cuda',dtype=torch.float16):v=model.encode_image(im.flatten(0,1).cuda(non_blocking=True))
            ff.append(norm(v.float()).reshape(len(im),5,-1).cpu().numpy())
    values=np.concatenate(ff);np.save(target,values)
    dump(OUT/f'{bank}_{style}_features.json',dict(path=str(target),sha256=sha(target),manifest_sha256=sha(OUT/f'{bank}.json'),texts_sha256=sha(OUT/'texts.pt'),shape=list(values.shape),seconds=time.monotonic()-start,pixel_checks=True))
    print('ENCODED',bank,style,values.shape,flush=True)

def diagnose():
    cfg=verify();meta=json.loads((OUT/'development_standard_features.json').read_text());assert sha(meta['path'])==meta['sha256']
    v=np.load(meta['path']);rr=json.loads((OUT/'development.json').read_text());txt=torch.load(OUT/'texts.pt',map_location='cpu');t=txt['features'].numpy();idx={n:i for i,n in enumerate(txt['vocabulary'])}
    scores=np.einsum('bsd,cd->bsc',v,t);records=[];geometries=[]
    for i,r in enumerate(rr):
        a=idx[r['label']]
        for k,wrong in enumerate(r['words'][1:]):
            d=idx[wrong];m=scores[i,:,a]-scores[i,:,d];dv=v[i,2]-v[i,3+k];dt=t[a]-t[d]
            interaction=float(m[2]-m[3+k]);assert np.isclose(interaction,dv@dt,atol=2e-6)
            records.append(dict(id=r['id'],image_id=r['image_id'],label=r['label'],wrong=wrong,clean=float(m[0]),blank=float(m[1]),congruent=float(m[2]),conflict=float(m[3+k]),interaction=interaction,blank_to_conflict=float(m[1]-m[3+k]),bug=bool(m[1]>0 and m[3+k]<=0),occlusion_flip=bool(m[0]>0 and m[1]<=0),image_displacement=float(np.linalg.norm(dv)),text_displacement=float(np.linalg.norm(dt)),alignment=float((dv@dt)/max(1e-12,np.linalg.norm(dv)*np.linalg.norm(dt)))))
    import pandas as pd
    frame=pd.DataFrame(records);frame.to_csv(OUT/'frozen_development_rows.csv',index=False)
    # Full-class recognition is separate from pairwise supplied-label decisions.
    yi=np.array([idx[r['label']] for r in rr]);pred=scores.argmax(-1)
    summary=dict(sources=len(rr),paired_decisions=len(frame),pairwise_accuracy={k:float((frame[k]>0).mean()) for k in ('clean','blank','congruent','conflict')},full_vocabulary_size=len(idx),full_class_accuracy={state:float((pred[:,j]==yi).mean()) for j,state in enumerate(STATES)},word_induced_bugs=int(frame.bug.sum()),blank_correct=int((frame.blank>0).sum()),bug_rate_on_blank_correct=float(frame.bug.sum()/max(1,(frame.blank>0).sum())),occlusion_flips=int(frame.occlusion_flip.sum()),interaction_mean=float(frame.interaction.mean()),interaction_abs_mean=float(abs(frame.interaction).mean()),interaction_p90=float(np.quantile(abs(frame.interaction),.9)),exact_finite_difference_identity=True)
    np.savez_compressed(OUT/'frozen_development_scores.npz',scores=scores,labels=yi)
    dump(OUT/'frozen_diagnosis.json',summary);print('DIAGNOSIS',json.dumps(summary),flush=True)

def main():
    log();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','encode','diagnose']);ap.add_argument('--bank',default='development');ap.add_argument('--style',default='standard',choices=['standard','serif','top']);a=ap.parse_args()
    if a.action=='encode':encode(a.bank,a.style)
    else:globals()[a.action]()

if __name__=='__main__':main()
