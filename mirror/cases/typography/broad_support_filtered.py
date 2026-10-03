"""Final metadata-only overlap filtering; no score-dependent exclusions."""
from mirror.cases.typography.diagnose import *
import mirror.cases.typography.broad_support_data as broad
DEST=broad.DEST
EXCLUDE={'motor scooter','coffee table'}

def freeze():
    broad.verify_data();assert not (DEST/'filtered_data_protocol.json').exists()
    cfg=json.loads((DEST/'data_protocol.json').read_text())
    train_labels=[x for x in cfg['training_classes'] if x not in EXCLUDE]
    vocab=train_labels+cfg['heldout_classes'];sources={};counts={}
    for bank,manifest in [('train','train.json'),('development_new','development_new_final.json'),('test_new','test_new_final.json')]:
        rows=json.loads((DEST/manifest).read_text())
        keep=[i for i,r in enumerate(rows) if not set(r['words'])&EXCLUDE]
        selected=[rows[i] for i in keep];dump(DEST/f'{bank}_filtered.json',selected)
        meta=DEST/f'{bank}_standard_features.json'
        sources[bank]=dict(manifest=str(DEST/f'{bank}_filtered.json'),indices=keep,feature_meta=str(meta))
        counts[bank]=dict(unique_sources=len(selected),labels=len(set(r['label'] for r in selected)))
    original_ids=[r['id'] for r in json.loads((OUT/'train.json').read_text())]
    selected=json.loads((DEST/'train_filtered.json').read_text())
    assert [r['id'] for r in selected[:len(original_ids)]]==original_ids
    texts=torch.load(DEST/'texts.pt',map_location='cpu');idx={x:i for i,x in enumerate(texts['vocabulary'])}
    torch.save(dict(vocabulary=vocab,features=texts['features'][[idx[x] for x in vocab]],templates=TEMPLATES),DEST/'filtered_texts.pt')
    files=[Path(__file__),CODE/'BROAD_SUPPORT_AMENDMENT.md',DEST/'data_protocol.json',DEST/'encoding_complete.json',
           DEST/'filtered_texts.pt',*[Path(x['manifest']) for x in sources.values()],*[Path(x['feature_meta']) for x in sources.values()]]
    pp=dict(files={str(p):sha(p) for p in files},sources=sources,training_classes=train_labels,
       new_classes=[x for x in cfg['new_classes'] if x not in EXCLUDE],heldout_classes=cfg['heldout_classes'],
       vocabulary=vocab,counts=counts,excluded_labels=sorted(EXCLUDE),score_inputs=False)
    dump(DEST/'filtered_data_protocol.json',pp);(DEST/'filtered_data_protocol.sha256').write_text(sha(DEST/'filtered_data_protocol.json')+'\n')
    print('FINAL FILTERED COUNTS',counts,'train classes',len(train_labels),flush=True)

def verify_data():
    broad.verify_data();p=DEST/'filtered_data_protocol.json';assert sha(p)==(DEST/'filtered_data_protocol.sha256').read_text().strip()
    cfg=json.loads(p.read_text())
    for p,h in cfg['files'].items():assert sha(p)==h,p
    return cfg

def data(bank):
    cfg=verify_data();source=cfg['sources'][bank];rows=json.loads(Path(source['manifest']).read_text())
    meta=json.loads(Path(source['feature_meta']).read_text());assert sha(meta['path'])==meta['sha256']
    v=torch.tensor(np.load(meta['path'])[source['indices']],device='cuda')
    txt=torch.load(DEST/'filtered_texts.pt',map_location='cpu');idx={x:i for i,x in enumerate(txt['vocabulary'])}
    y=torch.tensor([idx[r['label']] for r in rows],device='cuda')
    w=torch.tensor([[idx[x] for x in r['words'][1:]] for r in rows],device='cuda')
    return v,y,w,txt['features'].cuda(),rows

def stream(y,updates=1024):
    labels=y.detach().cpu().numpy();pools=[np.flatnonzero(labels==j) for j in range(int(labels.max())+1)]
    assert all(len(p)>0 for p in pools)
    sequence=[];cycle=1
    while len(sequence)<updates*32:
        rng=np.random.default_rng(4200+cycle);entries=np.concatenate([np.resize(rng.permutation(p),64) for p in pools])
        sequence.extend(rng.permutation(entries).tolist());cycle+=1
    return np.array(sequence[:updates*32],dtype=np.int64)

if __name__=='__main__':
    broad.log_run();freeze()

