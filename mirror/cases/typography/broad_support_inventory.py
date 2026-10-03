"""Outcome-independent inventory for broader lexical training support."""
from mirror.cases.typography.diagnose import *
import requests; import zipfile; import re
from collections import Counter
import nltk
from functools import lru_cache

DEST=ROOT/'mirror/cases/typography_broad_support_20260926'
TEMP=Path('/external-cache/fse_typographic_broad_support_20260926')
LVIS_URL='https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip'
WORDNET_URL='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip'

def log_run():
    DEST.mkdir(exist_ok=True);TEMP.mkdir(exist_ok=True)
    with (DEST/'commands.log').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+' '+shlex.join([sys.executable,*sys.argv])+'\n')

def download(url,path,max_bytes):
    meta=DEST/(path.name+'.download.json')
    if path.exists():
        assert meta.exists() and sha(path)==json.loads(meta.read_text())['sha256'];return
    path.parent.mkdir(parents=True,exist_ok=True);response=requests.get(url,stream=True,timeout=60);response.raise_for_status()
    total=0;tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('wb') as f:
        for chunk in response.iter_content(1<<20):
            total+=len(chunk);assert total<=max_bytes;f.write(chunk)
    tmp.replace(path);dump(meta,dict(url=url,final_url=response.url,bytes=total,sha256=sha(path),headers=dict(response.headers)))
    print('DOWNLOADED',path,total,flush=True)

def wordnet():
    path=TEMP/'nltk_data/corpora/wordnet.zip';download(WORDNET_URL,path,20<<20)
    nltk.data.path.insert(0,str(TEMP/'nltk_data'))
    from nltk.corpus import wordnet as wn
    wn.ensure_loaded();return wn

def surface(s):return re.sub(r'\s+',' ',s.lower().replace('_',' ')).strip()

def inventory():
    cfg=verify();download(LVIS_URL,TEMP/'lvis_v1_train.json.zip',400<<20);wn=wordnet()
    path=TEMP/'lvis_v1_train.json.zip'
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('/lvis_v1_train.json') or n=='lvis_v1_train.json'];assert len(names)==1
        with z.open(names[0]) as f:data=json.load(f)
    categories=data['categories'];images={i['id']:i for i in data['images']}
    original=set(cfg['seen_classes']+cfg['heldout_classes'])
    original_synsets=set();unresolved=[]
    for label in sorted(original):
        synsets=wn.synsets(label.replace(' ','_'),pos=wn.NOUN)
        if not synsets:unresolved.append(label)
        original_synsets.update(s.name() for s in synsets)
    # Include exact LVIS aliases for these nouns, independently of model outputs.
    for cat in categories:
        if {surface(x) for x in cat['synonyms']} & original:original_synsets.add(cat['synset'])
    @lru_cache(None)
    def ancestors(name):
        try:return {s.name() for s in wn.synset(name).closure(lambda s:s.hypernyms())}
        except Exception as e:
            if type(e).__name__=='WordNetError':return None
            raise
    unresolved_synsets=sorted(s for s in original_synsets if ancestors(s) is None)
    original_synsets={s for s in original_synsets if ancestors(s) is not None}
    def related(a,b):
        return a==b or b in ancestors(a) or a in ancestors(b)
    forbidden_images=set()
    for bank in ('train','development','test_seen','test_heldout'):
        forbidden_images.update(r['image_id'] for r in json.loads((OUT/f'{bank}.json').read_text()))
    pool={};counts=Counter()
    for ann in data['annotations']:
        im=images[ann['image_id']];url=im['coco_url']
        if '/train2017/' not in url:continue
        iid=im['id'];x,y,w,h=ann['bbox']
        if iid in forbidden_images or min(w,h)<64 or ann['area']/(im['width']*im['height'])<.08:continue
        filename=url.rsplit('/',1)[-1]
        if not (ROOT/'clip/data/coco/train2017'/filename).exists():continue
        group=pool.setdefault(ann['category_id'],{})
        row=dict(id='lvis:'+str(ann['id']),ann_id=ann['id'],image_id=iid,file=filename,bbox=ann['bbox'],area=ann['area'],width=im['width'],height=im['height'],coco_split='train2017')
        if iid not in group or ann['area']>group[iid]['area']:group[iid]=row
    candidates=[];exclusions=[]
    for cat in categories:
        label=surface(cat['name']);aliases={surface(x) for x in cat['synonyms']}
        reason=None
        if len(pool.get(cat['id'],{}))<56:reason='fewer than56 eligible sources'
        elif '(' in label or len(label.split())>3:reason='qualified or long printed label'
        elif aliases & original:reason='original class synonym'
        elif ancestors(cat['synset']) is None:reason='unresolved WordNet synset'
        else:
            overlap=[s for s in sorted(original_synsets) if related(cat['synset'],s)]
            if overlap:reason='WordNet related to original class'
        if reason:exclusions.append(dict(id=cat['id'],name=label,reason=reason));continue
        candidates.append(dict(id=cat['id'],label=label,synset=cat['synset'],synonyms=sorted(aliases),eligible_sources=len(pool[cat['id']])))
    candidates.sort(key=lambda c:key('typographic-broad-class-v1:'+c['label']))
    chosen=[]
    for cat in candidates:
        if any(related(cat['synset'],other['synset']) or set(cat['synonyms'])&set(other['synonyms']) for other in chosen):continue
        chosen.append(cat)
    dump(DEST/'inventory.json',dict(lvis_archive_sha256=sha(path),wordnet_sha256=sha(TEMP/'nltk_data/corpora/wordnet.zip'),
          original_unresolved_wordnet_labels=unresolved,original_unresolved_lvis_synsets=unresolved_synsets,original_synsets=sorted(original_synsets),
          exclusions=exclusions,candidates=candidates,antichain_candidates=chosen,
          excluded_original_images=len(forbidden_images),min_sources=56,score_inputs=False,
          code_sha256=sha(Path(__file__))))
    # Save only candidate boxes/metadata, no segmentation polygons or images.
    dump(DEST/'candidate_sources.json',{str(c['id']):list(pool[c['id']].values()) for c in chosen})
    print('CANDIDATES',len(candidates),'ANTICHAIN',len(chosen),'unresolved',unresolved,flush=True)
    print([(c['label'],c['eligible_sources']) for c in chosen],flush=True)

if __name__=='__main__':
    log_run();inventory()
