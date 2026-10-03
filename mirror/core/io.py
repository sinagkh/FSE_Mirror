"""Append-only execution records and create-only scientific artifacts."""
from mirror.paths import ARTIFACT_ROOT
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import sys

ROOT = ARTIFACT_ROOT
CODE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "runs/foundations"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def dump_or_verify(path, value):
    """Create a receipt, or verify that an existing receipt is identical."""
    path = Path(path)
    if path.exists():
        expected = json.loads(json.dumps(value, allow_nan=False))
        if read(path) != expected:
            raise ValueError(f"Existing receipt differs: {path}")
    else:
        dump(path, value)


def jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def log(out, event, **details):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            command=shlex.join([sys.executable, *sys.argv]), event=event, **details)) + "\n")


def verify_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Immutable input changed: {path}")


def code_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(CODE.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json", ".md"}}
