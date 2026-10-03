"""Freeze and encode the score-blind broader lexical-support bank."""
from mirror.cases.typography.diagnose import *
import mirror.cases.typography.broad_support_inventory as inv
DEST=inv.DEST
TEMP=inv.TEMP

def log_run():inv.log_run()

def prepare():
    cfg=verify();assert not (DEST/'data_protocol.json').exists()
    detail=json.loads((DEST/'inventory_v2.json').read_text());new=detail['new_classes']
    train_labels=cfg['seen_classes']+new;vocabulary=train_labels+cfg['heldout_classes']
    original=json.loads((OUT/'train.json').read_text())
    banks={}
    for bank in ('train_new','development_new','test_new'):
        rows=json.loads((DEST/f'{bank}.json').read_text())
        for r in rows:
            wrong=sorted([label for label in train_labels if label!=r['label']],key=lambda s:key('typographic-broad-union-word-v1:'+r['id']+':'+s))[:2]
            r['words']=[r['label'],*wrong]
        banks[bank]=rows;dump(DEST/f'{bank}_final.json',rows)
    train=original+banks['train_new'];dump(DEST/'train.json',train)
    assert len(train)==3840 and len(train_labels)==88
    original_ids=set()
    for bank in ('train','development','test_seen','test_heldout'):original_ids.update(r['image_id'] for r in json.loads((OUT/f'{bank}.json').read_text()))
    new_ids=[r['image_id'] for rows in banks.values() for r in rows]
    assert len(new_ids)==len(set(new_ids)) and not (set(new_ids)&original_ids)
    files=[Path(__file__),CODE/'BROAD_SUPPORT_PROTOCOL.md',CODE/'broad_support_inventory.py',CODE/'broad_support_freeze_sources.py',
         OUT/'protocol.json',OUT/'texts.pt',OUT/'train.json',DEST/'inventory.json',DEST/'inventory_v2.json',
         DEST/'lvis_v1_train.json.zip.download.json',DEST/'wordnet.zip.download.json',DEST/'train.json',
         *[DEST/f'{bank}_final.json' for bank in banks]]
    dump(DEST/'data_protocol.json',dict(files={str(p):sha(p) for p in files},training_classes=train_labels,new_classes=new,
        heldout_classes=cfg['heldout_classes'],vocabulary=vocabulary,counts={k:len(v) for k,v in banks.items()},
        unique_training_sources=len(train),training_score_inputs=False,renderer_sha256=sha(CODE/'diagnose.py')))
    (DEST/'data_protocol.sha256').write_text(sha(DEST/'data_protocol.json')+'\n')
    rows=sorted(banks['development_new'],key=lambda r:key('typographic-broad-gallery-v1:'+r['id']))[:12]
    sheet=Image.new('RGB',(5*224,len(rows)*252),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows):
        for j,im in enumerate(render(r)):
            sheet.paste(im,(j*224,i*252));draw.text((j*224,i*252+224),r['label']+' / '+STATES[j],fill='black')
    sheet.save(DEST/'gallery.jpg');dump(DEST/'gallery_ids.json',[r['id'] for r in rows])
    print('DATA FROZEN',len(train_labels),'classes',len(train),'unique train sources',flush=True)

def verify_data():
    verify();p=DEST/'data_protocol.json';assert sha(p)==(DEST/'data_protocol.sha256').read_text().strip()
    cfg=json.loads(p.read_text())
    for f,h in cfg['files'].items():assert sha(f)==h,f
    assert sha(CODE/'diagnose.py')==cfg['renderer_sha256']
    return cfg

def encode():
    cfg=verify_data();torch.set_num_threads(4);model,prep,tok=load_model();TEMP.mkdir(exist_ok=True)
    path=DEST/'texts.pt'
    if not path.exists():
        ff=[]
        with torch.no_grad():
            for label in cfg['vocabulary']:
                emb=model.encode_text(tok([s.format(label) for s in TEMPLATES]).cuda())
                ff.append(norm(norm(emb).mean(0)).cpu())
        torch.save(dict(vocabulary=cfg['vocabulary'],features=torch.stack(ff),templates=TEMPLATES),path)
    for bank in ('train_new','development_new','test_new'):
        target=TEMP/f'{bank}_standard.npy';meta=DEST/f'{bank}_standard_features.json'
        if target.exists():
            assert sha(target)==json.loads(meta.read_text())['sha256'];continue
        rows=json.loads((DEST/f'{bank}_final.json').read_text());ff=[];start=time.monotonic()
        with torch.no_grad():
            for im in DataLoader(Images(rows,prep,'standard'),batch_size=32,num_workers=8,pin_memory=True,worker_init_fn=worker_init):
                with torch.autocast('cuda',dtype=torch.float16):v=model.encode_image(im.flatten(0,1).cuda(non_blocking=True))
                ff.append(norm(v.float()).reshape(len(im),5,-1).cpu().numpy())
        values=np.concatenate(ff);np.save(target,values)
        dump(meta,dict(path=str(target),sha256=sha(target),manifest_sha256=sha(DEST/f'{bank}_final.json'),shape=list(values.shape),seconds=time.monotonic()-start,pixel_checks=True))
        print('ENCODED',bank,values.shape,flush=True)
    oldmeta=json.loads((OUT/'train_standard_features.json').read_text());assert sha(oldmeta['path'])==oldmeta['sha256']
    newmeta=json.loads((DEST/'train_new_standard_features.json').read_text())
    target=TEMP/'train_standard.npy'
    if not target.exists():
        values=np.concatenate([np.load(oldmeta['path']),np.load(newmeta['path'])]);np.save(target,values)
        dump(DEST/'train_standard_features.json',dict(path=str(target),sha256=sha(target),manifest_sha256=sha(DEST/'train.json'),shape=list(values.shape),original_feature_sha256=oldmeta['sha256']))
    dump(DEST/'encoding_complete.json',dict(texts_sha256=sha(path),files={str(p):sha(p) for p in DEST.glob('*features.json')}))
if __name__=='__main__':
    log_run();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','encode']);a=ap.parse_args();globals()[a.action]()
