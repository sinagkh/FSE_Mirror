"""Bounded opacity/order development banks. No repair scores in construction."""
import argparse
from pathlib import Path
from dataclasses import replace
import cv2
import numpy as np
import torch
from PIL import Image; from PIL import ImageDraw
from mirror.cases.color_binding import routing_light_tint as p
from mirror.cases.color_binding import routing_light_search as oldsearch
from mirror.cases.color_binding import indirect_generalization as ind
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files
from mirror.cases.color_binding.repair_trainbank import lines; from mirror.cases.color_binding.repair_trainbank import annotations_for
from mirror.core.features import cache_bank; from mirror.core.features import verify_cache
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY; from mirror.core.encoders import legacy_unit
from mirror.cases.color_binding.rendering import captions
from mirror.core.repair import cache_sha256; from mirror.core.repair import derive_scales

OUT=p.ROOT/'clip/interbind_routing_tint_balance_20260930'
PLAN=p.ROOT/'FSE_VLM/plan/84_routing_tint_color_order.md'
TINTS=(40,65,90)
COLORS=p.COLORS[:2]


def freeze():
    paths=[PLAN,Path(__file__),Path(p.__file__),Path(p.prior.__file__),Path(p.cn.__file__),
           Path(p.templates.__file__),Path(ind.__file__),p.INITIAL,
           p.TRAIN/'training_rows.jsonl',oldsearch.DEV/'rows.jsonl',p.CONFIRM/'rows.jsonl',
           p.prior.OUT/'normalization.json',p.OUT/'protocol.json',oldsearch.OUT/'protocol.json']
    rr=[lines(path) for path in paths if path.name.endswith('rows.jsonl')]
    sources=[{str(s) for r in rows for s in r['source_ids']} for rows in rr]
    assert all(not sources[i]&sources[j] for i in range(3) for j in range(i+1,3))
    inputs={str(f):sha(f) for f in paths}
    inputs.update({f:h for rows in rr for r in rows for f,h in r['source_image_sha256'].items()})
    dump(OUT/'data_protocol.json',dict(inputs=inputs,tints=TINTS,seed=42,train_pairs=640,development_pairs=888,
        known_previous_test_outcomes=True,no_new_model_scores=True,caption_orders=['canonical','reversed'],
        model=p.prior.MODEL,selection='Existing source banks, no filtering',plan_sha256=sha(PLAN)))
    dump(OUT/'data_protocol_hash.json',dict(sha256=sha(OUT/'data_protocol.json')))
    print('DATA_FROZEN',flush=True)


def verify():
    assert sha(OUT/'data_protocol.json')==read(OUT/'data_protocol_hash.json')['sha256']
    verify_files(read(OUT/'data_protocol.json')['inputs'])


def preview():
    verify();cv2.setNumThreads(1);annotations_for('train2017');annotations_for('val2017')
    rows=lines(p.TRAIN/'training_rows.jsonl')
    chosen=sorted([r for r in rows if r['objects']==['car','boat']],key=lambda r:r['anchor_id'])[:2]
    chosen += [next(r for r in rows if r['objects']==['person','bicycle'])]
    records=[]
    for j,row in enumerate(chosen):
        page=Image.new('RGB',(1024,3*284),'white');draw=ImageDraw.Draw(page)
        for y,tint in enumerate(TINTS):
            ims,names,hashes=p.render(row,COLORS[:1],tint/100)
            for state in range(4):
                page.paste(Image.fromarray(ims[state]),(state*256,y*284+24))
                draw.text((state*256+5,y*284+5),f'{tint}% / '+['RR','RB','BR','BB'][state],fill='black')
        dest=OUT/'preview'/f'{j:02d}.png';dest.parent.mkdir(parents=True,exist_ok=True);page.save(dest)
        records.append(dict(anchor_id=row['anchor_id'],objects=row['objects'],file=str(dest),sha256=sha(dest)))
    jsonl(OUT/'preview/selection.jsonl',records)
    print('PREVIEWS_READY',flush=True)


def encode():
    verify();assert (OUT/'preview/review.json').exists()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info()[0]>12*1024**3
    annotations_for('train2017');annotations_for('val2017')
    scorer=load_subject(p.prior.MODEL,device='cuda')
    # Only intermediate opacity requires new training/development image passes.
    for bank,rowpath in [('train',p.TRAIN/'training_rows.jsonl'),('development',oldsearch.DEV/'rows.jsonl')]:
        dest=OUT/'features/65'/bank
        if (dest/'complete.json').exists():verify_cache(dest);continue
        rows=lines(rowpath);counter=[0]
        def renderer(row):
            result=p.render(row,COLORS,.65);counter[0]+=1
            if counter[0]%100==0:print('RENDERED65',bank,counter[0],len(rows),flush=True)
            return result
        cache_bank(dest,rows,scorer,renderer,lambda f,n:[g for c in COLORS for g in captions(f,n,c)],
            registry_path=DEFAULT_REGISTRY,input_hashes={str(OUT/'data_protocol.json'):sha(OUT/'data_protocol.json'),str(rowpath):sha(rowpath)},
            image_batch_size=64,text_batch_size=128,render_workers=8,details=dict(alpha=.65,no_exclusions=True))
        print('ENCODED65',bank,flush=True)
    # Only training-noun/caption strings. Reversing order preserves noun-color assignment.
    pairs=sorted({tuple(r['objects']) for r in lines(p.TRAIN/'training_rows.jsonl')})
    groups=[]
    for nouns in pairs:
        for colors in COLORS:
            for order,family in [('canonical','train_templates'),('reversed','reverse_order')]:
                forms=ind.prompts(nouns,colors,'canvas',family)
                if order=='reversed':
                    expected=ind.prompts(nouns[::-1],colors,'canvas','train_templates')
                    assert forms==[expected[i] for i in (0,2,1,3)]
                groups += [dict(objects=nouns,color='-'.join(colors),order=order,state=i,templates=ts) for i,ts in enumerate(forms)]
    strings=list(dict.fromkeys(s for g in groups for s in g['templates']))
    features=scorer.encode_texts(strings,batch_size=128).numpy();lookup={s:i for i,s in enumerate(strings)}
    for g in groups:g['indices']=[lookup[s] for s in g['templates']]
    dest=OUT/'text';dest.mkdir(exist_ok=False)
    np.save(dest/'features.npy',features);dump(dest/'strings.json',strings);dump(dest/'groups.json',groups)
    dump(dest/'complete.json',dict(files={str(f):sha(f) for f in dest.iterdir() if f.is_file()},semantic_order_check=True))
    dump(OUT/'encoding_complete.json',dict(files={str(f):sha(f) for f in [OUT/'features/65/train/complete.json',OUT/'features/65/development/complete.json',dest/'complete.json']}))
    print('ENCODING_COMPLETE',flush=True)


def devpath(tint):
    return {40:oldsearch.OUT/'features',65:OUT/'features/65/development',90:oldsearch.DEV/'features'}[int(tint)]


def text_for(rows,color,order='canonical',individual=False):
    verify_files(read(OUT/'text/complete.json')['files'])
    features=np.load(OUT/'text/features.npy');groups=read(OUT/'text/groups.json')
    lookup={(tuple(g['objects']),g['color'],g['order'],g['state']):g['indices'] for g in groups}
    indices=np.array([[lookup[tuple(r['objects']),color,order,i] for i in range(4)] for r in rows])
    x=features[indices]  # N, 4 states, 3 wordings, D
    if individual:return x.transpose(0,2,1,3)
    return legacy_unit(torch.from_numpy(x.mean(2))).numpy()


def load_training(tint,device='cuda'):
    cache,spec,contexts,cfg=p.prior.load_training('cpu')
    if tint!=90:
        folder=p.OUT/'features/train' if tint==40 else OUT/'features/65/train'
        verify_cache(folder);idx=lines(folder/'index.jsonl');v=np.load(folder/'images.npy');values=[]
        for row in idx:
            for colors in COLORS:
                for view in p.VIEWS:
                    names=['-'.join(colors)+'/'+view+'/'+a+'_'+b for a in colors for b in colors]
                    values.append(v[row['image_offset']+np.array([row['state_names'].index(n) for n in names])])
        cache=replace(cache,images=torch.tensor(np.stack(values)))
    cache=replace(cache,bank_id='routing_tint_balance_'+str(tint))
    cfg=replace(cfg,bank_id=cache.bank_id,expected_cache_sha256=cache_sha256(cache),scales=derive_scales(cache,spec,(0,1,2,3)),seed=42,epochs=36)
    cache=replace(cache,images=cache.images.to(device),texts=cache.texts.to(device),natural_texts=cache.natural_texts.to(device))
    forms=p.templates.template_cache(cache);reverse=forms.clone()
    rows=lines(p.TRAIN/'training_rows.jsonl')
    for ci,colors in enumerate(COLORS):
        color='-'.join(colors);pooled=text_for(rows,color,'reversed');single=text_for(rows,color,'reversed',True)
        for j in range(len(rows)):
            start=j*4+ci*2
            reverse[start:start+2,0,:4]=torch.tensor(pooled[j],device=device)
            reverse[start:start+2,1:,:4]=torch.tensor(single[j],device=device)
        original=text_for(rows,color,'canonical')
        assert np.max(abs(original-forms[ci*2::4,0,:4].cpu().numpy()))<2e-6
    return (cache,spec,contexts,cfg),torch.cat((forms,reverse),dim=1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','preview','encode']);args=ap.parse_args()
    log(OUT,'start',stage=args.action)
    try:globals()[args.action]()
    except BaseException as exc:log(OUT,'failed',stage=args.action,error=repr(exc));raise
    log(OUT,'complete',stage=args.action)


if __name__=='__main__':main()
