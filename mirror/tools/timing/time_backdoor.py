"""Inference-only timing of the backdoor test generator and frozen (poisoned) victim audit on ImageNetV2.

Reuses mirror/cases/backdoor: par_extension_evaluate (manifest, archive streaming, record logic),
par_extension_data.Render / process (PAR renderer, identity grid), par_extension_train.model_load.
Only the RQ1 states are generated: identity clean, trigger 1, trigger 2, plain-color control.
Writes only results/timing_backdoor_<case>.json in this folder.
"""
from mirror.paths import ARTIFACT_ROOT
import argparse; import io; import json; import platform; import sys; import tarfile; import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import IterableDataset; from torch.utils.data import Dataset; from torch.utils.data import DataLoader; from torch.utils.data import get_worker_info

ROOT = ARTIFACT_ROOT
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mirror/cases/backdoor'))
import mirror.cases.backdoor.evaluate as ev   # noqa: E402
data, training = ev.data, ev.training

WORKERS = 2                 # existing evaluator: DataLoader(num_workers=2, prefetch_factor=1)
BATCH_SOURCES = 21          # 21 sources x 4 states = 84 images = existing 4 sources x 21 states
TARGET = 954


def sync(): torch.cuda.synchronize()


class Stream(IterableDataset):
    """par_extension_evaluate.Sources.__iter__ restricted to the identity grid."""
    def __init__(self, case, rows, prep=None, limit=None):
        self.case, self.rows, self.prep = case, rows[:limit] if limit else rows, prep

    def __iter__(self):
        info = get_worker_info(); wid, nw = (info.id, info.num_workers) if info else (0, 1)
        renderer = data.Render(self.case); index = 0
        with tarfile.open(ev.ARCHIVE, 'r|*') as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                i = index; index += 1
                if i >= len(self.rows): break
                assert member.name == self.rows[i]['id']
                if i % nw != wid: continue
                image = Image.open(io.BytesIO(archive.extractfile(member).read())).convert('RGB')
                states = [data.process(im, 'identity') for im in renderer.images(image, self.rows[i])]
                if self.prep is None:
                    yield i, torch.from_numpy(np.stack([np.asarray(im) for im in states]))
                else:
                    yield i, torch.stack([self.prep(im) for im in states])


class Prep(Dataset):
    def __init__(self, arrays, prep): self.arrays, self.prep = arrays, prep
    def __len__(self): return len(self.arrays)
    def __getitem__(self, i): return i, torch.stack([self.prep(Image.fromarray(a)) for a in self.arrays[i]])


def worker_init(_): torch.set_num_threads(1)


@torch.no_grad()
def class_texts(model, tok):
    """par_extension_evaluate.texts compute branch (1,000 classes x 80 templates, FP32)."""
    cfg = eval((data.original.VENDOR / 'asset/imagenet/classes.py').read_text(), {'__builtins__': {}})
    feats = []
    for name in cfg['classes']:
        z = F.normalize(model.encode_text(tok([p(name) for p in cfg['templates']]).cuda()).float(), dim=-1)
        feats.append(F.normalize(z.mean(0), dim=-1))
    return torch.stack(feats), len(cfg['classes']) * len(cfg['templates'])


class Records:
    def __init__(self, n):
        self.pred = np.zeros((n, 4), np.int32); self.true = np.zeros((n, 4), np.float32)
        self.inter_abs = np.zeros((n, 3), np.float32); self.inter_signed = np.zeros((n, 3), np.float32)
        self.score_s = 0.

    @torch.no_grad()
    def add(self, ix, z, tv, labels):
        t = time.perf_counter()
        scores = (z @ tv.T).reshape(len(ix), 4, -1)              # FP32 GPU scores (record())
        y = torch.tensor(labels[ix], device='cuda')
        true = scores.gather(-1, y[:, None, None].expand(-1, 4, 1)).squeeze(-1)
        margins = true[:, :, None] - scores
        d = margins[:, 1:4] - margins[:, 0:1]                     # 3 edits x 1000 classes (true class contributes 0)
        self.pred[ix] = scores.argmax(-1).cpu().numpy(); self.true[ix] = true.cpu().numpy()
        self.inter_abs[ix] = d.abs().mean(-1).cpu().numpy(); self.inter_signed[ix] = d.mean(-1).cpu().numpy()
        sync(); self.score_s += time.perf_counter() - t


@torch.no_grad()
def encode(model, x):
    with torch.autocast('cuda', dtype=torch.float16):
        return F.normalize(model.encode_image(x.flatten(0, 1).cuda(non_blocking=True)).float(), dim=-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--case', required=True, choices=data.CASES); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(4)
    rec = dict(scenario='backdoor', case=a.case, bank='ImageNetV2 matched-frequency (10,000 images)', render_workers=WORKERS,
               batch_sources=BATCH_SOURCES, batch_images=4 * BATCH_SOURCES,
               precision='image FP16 autocast; text FP32 (80-template ensemble); scores FP32 GPU', gpu=torch.cuda.get_device_name(0),
               cpu=platform.processor() or 'AMD EPYC 7R32', torch=torch.__version__, setup={}, timing_s={})
    rows = ev.rows_for('imagenetv2'); rows = rows[:a.limit] if a.limit else rows; n = len(rows)
    labels = np.array([r['label'] for r in rows])
    t = time.perf_counter(); model, prep, tok = training.model_load(a.case); sync(); rec['setup']['model_load_and_hash_s'] = time.perf_counter() - t
    # warm-up (untimed): first 84 images through every phase
    for _ in DataLoader(Stream(a.case, rows, None, 84), batch_size=4, num_workers=WORKERS, prefetch_factor=1, worker_init_fn=worker_init): pass
    tv, n_strings = class_texts(model, tok); w = Records(84); wl = labels
    for ix, x in DataLoader(Stream(a.case, rows, prep, 84), batch_size=BATCH_SOURCES, num_workers=WORKERS, prefetch_factor=1, worker_init_fn=worker_init):
        w.add(ix.numpy(), encode(model, x), tv, wl)
    sync()
    # G: generation only, arrays kept in RAM
    arrays = np.zeros((n, 4, 224, 224, 3), np.uint8); t = time.perf_counter()
    for ix, x in DataLoader(Stream(a.case, rows), batch_size=4, num_workers=WORKERS, prefetch_factor=1, worker_init_fn=worker_init):
        arrays[ix.numpy()] = x.numpy()
    rec['timing_s']['generation'] = time.perf_counter() - t
    # E_txt
    t = time.perf_counter(); tv, n_strings = class_texts(model, tok); sync(); rec['timing_s']['exec_text_encode'] = time.perf_counter() - t
    # E_img (+ E_score accumulated inside)
    r = Records(n); t = time.perf_counter(); forwards = 0
    for ix, x in DataLoader(Prep(arrays, prep), batch_size=BATCH_SOURCES, num_workers=WORKERS, pin_memory=True, worker_init_fn=worker_init):
        r.add(ix.numpy(), encode(model, x), tv, labels); forwards += 1
    sync(); total = time.perf_counter() - t
    rec['timing_s']['exec_score_and_checks'] = r.score_s; rec['timing_s']['exec_image_prep_encode'] = total - r.score_s
    rec['timing_s']['execution_total'] = total + rec['timing_s']['exec_text_encode']
    del arrays
    # pipelined end to end (existing structure: workers stream+render+prep, main thread encodes and scores)
    rp = Records(n); t = time.perf_counter(); tvp, _ = class_texts(model, tok)
    for ix, x in DataLoader(Stream(a.case, rows, prep), batch_size=BATCH_SOURCES, num_workers=WORKERS, pin_memory=True, prefetch_factor=1, worker_init_fn=worker_init):
        rp.add(ix.numpy(), encode(model, x), tvp, labels)
    sync(); rec['timing_s']['end_to_end_pipelined'] = time.perf_counter() - t
    good = r.pred == labels[:, None]
    fail = good[:, 0] & ~good[:, 1]
    rec['work'] = dict(sources=n, images_rendered=4 * n, follow_up_images=3 * n, image_forwards=forwards, caption_strings_encoded=n_strings,
                       scores=4 * n * 1000, interaction_tests_trigger1=n * 999, interaction_tests_all_edits=3 * n * 999,
                       decision_checks=n, failing_decisions=int(fail.sum()),
                       failing_trigger2=int((good[:, 0] & ~good[:, 2]).sum()), failing_control=int((good[:, 0] & ~good[:, 3]).sum()))
    # fidelity versus saved victim records
    z = np.load(ROOT / f'clip/fse_four_case_20260927/backdoor_extension_v1/{a.case}/evaluation/imagenetv2/victim_records.npz')
    assert list(z['ids'][:n]) == [r_['id'] for r_ in rows]
    states = list(z['states']); cols = [states.index('identity_' + k) for k in ('clean', 'trigger_1', 'trigger_2', 'mean_patch_sham')]
    sp = z['pred'][:n][:, cols]; sgood = sp == labels[:, None]; sfail = sgood[:, 0] & ~sgood[:, 1]
    cached = ev.V2 / 'victim_text_prototypes.pt' if a.case == 'stripes' else training.directory(a.case) / 'victim_1000_texts.pt'
    ct = torch.load(cached, map_location='cuda', weights_only=True)['features'].float()
    rec['fidelity'] = dict(
        max_abs_text_prototype_diff=float((tv - ct).abs().max()),
        prediction_agreement=float((r.pred == sp).mean()), prediction_disagreements=int((r.pred != sp).sum()),
        prediction_agreement_pipelined=float((rp.pred == sp).mean()),
        max_abs_true_score_diff=float(abs(r.true - z['true_score'][:n][:, cols]).max()),
        max_abs_interaction_diff=float(abs(r.inter_abs - z['allclass_interaction_abs'][:n][:, 0, :]).max()),
        failing_recomputed=int(fail.sum()), failing_saved=int(sfail.sum()), failing_set_identical=bool(np.array_equal(fail, sfail)),
        failing_pipelined=int(((rp.pred[:, 0] == labels) & (rp.pred[:, 1] != labels)).sum()),
        interaction_mean_abs_over_999=float(r.inter_abs[:, 0].mean() * 1000 / 999))
    name = f'timing_backdoor_{a.case}.json' if not a.limit else f'smoke_backdoor_{a.case}.json'
    (HERE / 'results' / name).write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps({k: rec[k] for k in ('timing_s', 'work', 'fidelity', 'setup')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
