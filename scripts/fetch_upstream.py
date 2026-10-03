"""Fetch the exact third-party revisions; retain upstream ownership/licensing."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
entries=json.loads((ROOT/'config/upstream.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--list',action='store_true');p.add_argument('--name');p.add_argument('--out-dir',type=Path)
a=p.parse_args()
if a.list:
    for r in entries:print(r['name'],r['revision'],r['url'])
else:
    if not a.name or a.out_dir is None:p.error('Use --list, or --name NAME --out-dir DIRECTORY.')
    rr=[r for r in entries if r['name']==a.name]
    if len(rr)!=1:p.error('Unknown upstream: '+a.name)
    r=rr[0];dest=a.out_dir/r['name'];assert not dest.exists(),'Existing checkout retained.'
    a.out_dir.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','clone','--no-checkout',r['url'],str(dest)],check=True)
    subprocess.run(['git','-C',str(dest),'checkout','--detach',r['revision']],check=True)
    got=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    assert got==r['revision'];print('Fetched',r['name'],got)
