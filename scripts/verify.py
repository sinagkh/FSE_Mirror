"""Standard-library verification of the unpacked submission payload."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'provenance/package_manifest.json'
assert manifest.is_file(),'Package seal is missing.'
records=json.loads(manifest.read_text())['files']
bad=[]
for name,expected in records.items():
    p=ROOT/name
    if not p.is_file():bad.append((name,'missing'));continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected['sha256']:bad.append((name,'checksum'))
# A local virtual environment (README) is not part of the payload; its .pth path files are not checkpoints.
weights=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in ('.pt','.pth','.ckpt','.safetensors','.bin')
         and not {'.venv','site-packages'}&set(p.relative_to(ROOT).parts)]
assert not weights,weights
assert not bad,bad
assert (ROOT/'mirror/core/specifications.py').is_file()
assert not (ROOT/'source').exists()
assert all((ROOT/name).is_file() for name in json.loads(
    (ROOT/'provenance/code_entrypoints.json').read_text()))
cfg=json.loads((ROOT/'config/studies.json').read_text())
assert cfg['seeds']==[42,43,44] and cfg['color_binding']['tint']==.75
raw=json.loads((ROOT/'results/records.json').read_text())
assert {r['case'] for r in raw}=={'typography','color_binding','backdoor'}
for case in ('typography','color_binding'):
    assert {r['seed'] for r in raw if r['case']==case and r['study']=='main'}=={42,43,44}
for attack in ('stripes','triangles','text'):
    for method in ('ranking','IS2','clean_only'):
        assert {r['seed'] for r in raw if r['case']=='backdoor' and r['attack']==attack and r['method']==method}=={42,43,44}
        assert {r['bank'] for r in raw if r['case']=='backdoor' and r['attack']==attack and r['method']==method}=={'development','banana87','banana1000','imagenetv2'}
print(f'Verified {len(records)} files; no checkpoints; all three paper studies and seeds retained.')
