"""Create-only artifacts and temporary download provenance."""
from pathlib import Path
import tempfile
import requests
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import sha; from mirror.core.io import log; from mirror.core.io import jsonl

OUT = ROOT / 'clip/fse_published_comparisons_20260926'

def storage():
    path = OUT / 'storage.json'
    if not path.exists():
        dump(path, {'directory': tempfile.mkdtemp(prefix='fse_published_', dir='/external-cache')})
    return Path(read(path)['directory'])

def download(url, name):
    dest = storage() / name
    meta = OUT / 'downloads' / (name.replace('/', '_') + '.json')
    if meta.exists():
        assert sha(dest) == read(meta)['sha256']
        return dest
    assert not dest.exists(), dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(OUT, 'download_start', url=url, destination=str(dest))
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with dest.open('xb') as stream:
            for block in response.iter_content(1 << 20):
                stream.write(block)
    dump(meta, dict(url=url, path=str(dest), sha256=sha(dest), bytes=dest.stat().st_size))
    log(OUT, 'download_complete', name=name)
    return dest

def acquire_negationclip():
    result = requests.get('https://huggingface.co/api/models/jerryray/negationclip', timeout=30)
    result.raise_for_status()
    revision = result.json()['sha']
    path = download(f'https://huggingface.co/jerryray/negationclip/resolve/{revision}/negationclip_ViT-B32.pth',
                    'negationclip/negationclip_ViT-B32.pth')
    dump(OUT / 'negationclip_source.json', dict(revision=revision, checkpoint=str(path), sha256=sha(path),
         source='https://github.com/parkquasar/NegationCLIP', backbone='OpenAI ViT-B/32',
         training='COCO2014 negation-inclusive captions; text encoder fine-tuned',
         evaluation='Official NegBench; exclude custom COCO-train task as independent evidence'))

if __name__ == '__main__':
    log(OUT, 'acquire_negationclip')
    acquire_negationclip()
