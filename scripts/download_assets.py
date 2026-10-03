"""Download a declared public archive, verifying its size and SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
assets=json.loads((ROOT/'config/downloads.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--list',action='store_true');p.add_argument('--asset')
p.add_argument('--out-dir',type=Path);a=p.parse_args()
if a.list:
    for r in assets:print(r['name'],r['bytes'],'bytes',r['url'])
else:
    if not a.asset or a.out_dir is None:p.error('Use --list, or --asset NAME --out-dir DIRECTORY.')
    rr=[r for r in assets if r['name']==a.asset]
    if len(rr)!=1:p.error('Unknown asset: '+a.asset)
    r=rr[0];a.out_dir.mkdir(parents=True,exist_ok=True);dest=a.out_dir/r['filename']
    if dest.exists():
        h=hashlib.sha256()
        with dest.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==r['sha256'],'Existing file has a different hash; not replaced.'
        print('Verified existing',dest)
    else:
        partial=dest.with_suffix(dest.suffix+'.partial');assert not partial.exists(),'Existing partial download retained.'
        h=hashlib.sha256();count=0
        with urllib.request.urlopen(r['url'],timeout=120) as response,partial.open('xb') as f:
            for chunk in iter(lambda:response.read(1024*1024),b''):
                f.write(chunk);h.update(chunk);count+=len(chunk)
        assert count==r['bytes'] and h.hexdigest()==r['sha256'],'Download failed integrity check; partial retained.'
        partial.rename(dest);print('Downloaded and verified',dest)
