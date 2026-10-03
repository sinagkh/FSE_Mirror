"""Durable provenance helpers for the four-case campaign."""
from mirror.paths import ARTIFACT_ROOT
from pathlib import Path
from datetime import datetime; from datetime import timezone
import hashlib
import json
import os
import shlex
import sys

ROOT = ARTIFACT_ROOT
OUT = ROOT / 'clip/fse_four_case_20260927'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def log(event, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {'time_utc': datetime.now(timezone.utc).isoformat(),
           'event': event, **fields}
    with (OUT / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


def command():
    log('command', argv=[sys.executable, *sys.argv], cwd=os.getcwd())


def temp_root():
    path = Path(os.environ.get('FSE_FOUR_TEMP', '/external-cache/fse_four_case_20260927'))
    path.mkdir(parents=True, exist_ok=True)
    return path
