'Inference-only timing of the typography test generator and frozen OpenAI B/32 audit.\n\nReuses mirror/cases/typography/diagnose.py (render, Images, load_model, TEMPLATES)\nand the metric code of retest.metrics / strength_retest.complete_targets.\nWrites only results/timing_typography_<bank>.json in this folder.\n'
from mirror.paths import ARTIFACT_ROOT
import argparse; import itertools; import json; import os; import platform; import statistics; import sys; import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader; from torch.utils.data import Dataset

ROOT = ARTIFACT_ROOT
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mirror/cases/typography'))
import mirror.cases.typography.diagnose as dg  # noqa: E402

PAIRS = list(itertools.combinations(range(1, 5), 2))  # strength_pilot.PAIRS
BATCH, WORKERS = 32, 8                                 # diagnostic.encode settings


def sync():
    torch.cuda.synchronize()


class RenderOnly(Dataset):
    """Generation only: source crop + 4 note states, as uint8 arrays."""
    def __init__(self, rr, style): self.rr, self.style = rr, style
    def __len__(self): return len(self.rr)
    def __getitem__(self, i): return torch.from_numpy(np.stack([np.asarray(im) for im in dg.render(self.rr[i], self.style)]))


class PrepOnly(Dataset):
    """Execution-side preprocessing of already generated images."""
    def __init__(self, arrays, prep): self.arrays, self.prep = arrays, prep
    def __len__(self): return len(self.arrays)
    def __getitem__(self, i):
        from PIL import Image
        return torch.stack([self.prep(Image.fromarray(a)) for a in self.arrays[i]])


def generate(rr, style):
    out = []
    for batch in DataLoader(RenderOnly(rr, style), batch_size=BATCH, num_workers=WORKERS, worker_init_fn=dg.worker_init):
        out.append(batch.numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_images(arrays, model, prep):
    ff = []; forwards = 0
    for im in DataLoader(PrepOnly(arrays, prep), batch_size=BATCH, num_workers=WORKERS, pin_memory=True, worker_init_fn=dg.worker_init):
        with torch.autocast('cuda', dtype=torch.float16):
            v = model.encode_image(im.flatten(0, 1).cuda(non_blocking=True))
        ff.append(dg.norm(v.float()).reshape(len(im), 5, -1).cpu().numpy()); forwards += 1
    return np.concatenate(ff), forwards


@torch.no_grad()
def pipelined_images(rr, style, model, prep):
    """Exactly the diagnostic.encode loop (render+prep in workers, GPU in main)."""
    ff = []
    for im in DataLoader(dg.Images(rr, prep, style), batch_size=BATCH, num_workers=WORKERS, pin_memory=True, worker_init_fn=dg.worker_init):
        with torch.autocast('cuda', dtype=torch.float16):
            v = model.encode_image(im.flatten(0, 1).cuda(non_blocking=True))
        ff.append(dg.norm(v.float()).reshape(len(im), 5, -1).cpu().numpy())
    return np.concatenate(ff)


@torch.no_grad()
def encode_texts(model, tok, vocab):
    feats = []
    for name in vocab:  # diagnostic.encode text loop
        emb = model.encode_text(tok([s.format(name) for s in dg.TEMPLATES]).cuda()); feats.append(dg.norm(dg.norm(emb).mean(0)))
    return torch.stack(feats).cpu().numpy()


def checks(scores, rr, idx):
    n = len(rr); y = np.array([idx[r['label']] for r in rr]); w = np.array([[idx[s] for s in r['words'][1:]] for r in rr])
    m = scores[np.arange(n)[:, None, None], np.arange(5)[None, :, None], y[:, None, None]] - \
        scores[np.arange(n)[:, None, None], np.arange(5)[None, :, None], w[:, None, :]]
    conflict = np.stack([m[:, 3, 0], m[:, 4, 1]], 1)
    all12 = np.stack([m[:, i] - m[:, j] for i, j in PAIRS], 1)
    bug = (m[:, 1] > 0) & (conflict <= 0)
    top1 = scores.argmax(-1) == y[:, None]
    return dict(scores=scores, clean=m[:, 0], blank=m[:, 1], conflict=conflict, all12=all12, bug=bug, top1=top1)


def score(v, t, rr, idx):
    return checks(np.einsum('bid,cd->bic', v, t), rr, idx)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bank', default='test_seen'); ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args(); style = 'standard'
    torch.set_num_threads(4)
    rec = dict(scenario='typography', bank=a.bank, style=style, batch_sources=BATCH, batch_images=5 * BATCH, workers=WORKERS,
               precision='image FP16 autocast; text FP32; scores CPU float32 einsum', gpu=torch.cuda.get_device_name(0),
               cpu=platform.processor() or 'AMD EPYC 7R32', torch=torch.__version__, setup={})
    t = time.perf_counter(); cfg = dg.verify(); rec['setup']['integrity_verify_s'] = time.perf_counter() - t
    rr = json.loads((dg.OUT / f'{a.bank}.json').read_text())
    vocab = cfg['seen_classes'] + cfg['heldout_classes']; idx = {c: i for i, c in enumerate(vocab)}
    t = time.perf_counter(); model, prep, tok = dg.load_model(); sync(); rec['setup']['model_load_s'] = time.perf_counter() - t
    # warm-up (untimed): 64 sources through every phase
    w = rr[:64]; arr = generate(w, style); v, _ = encode_images(arr, model, prep); tt = encode_texts(model, tok, vocab); score(v, tt, w, idx)
    pipelined_images(w, style, model, prep); sync(); del arr
    G, EI, ET, ES, P = [], [], [], [], []
    for rep in range(a.repeats):
        t = time.perf_counter(); arr = generate(rr, style); G.append(time.perf_counter() - t)
        t = time.perf_counter(); v, forwards = encode_images(arr, model, prep); sync(); EI.append(time.perf_counter() - t)
        t = time.perf_counter(); tt = encode_texts(model, tok, vocab); sync(); ET.append(time.perf_counter() - t)
        t = time.perf_counter(); res = score(v, tt, rr, idx); ES.append(time.perf_counter() - t)
        t = time.perf_counter(); vp = pipelined_images(rr, style, model, prep); tp = encode_texts(model, tok, vocab); sync(); rp = score(vp, tp, rr, idx)
        P.append(time.perf_counter() - t)
        print(f'REP {rep}: G={G[-1]:.2f} E_img={EI[-1]:.2f} E_txt={ET[-1]:.2f} E_score={ES[-1]:.3f} pipelined={P[-1]:.2f}', flush=True)
    def summ(x): return dict(median=statistics.median(x), min=min(x), max=max(x), runs=x)
    rec['timing_s'] = dict(generation=summ(G), exec_image_prep_encode=summ(EI), exec_text_encode=summ(ET), exec_score_and_checks=summ(ES),
                           execution_total=summ([a_ + b + c for a_, b, c in zip(EI, ET, ES)]),
                           end_to_end_pipelined=summ(P))
    rec['work'] = dict(sources=len(rr), images_rendered=int(arr.shape[0] * arr.shape[1]), follow_up_images=4 * len(rr),
                       image_forwards=forwards, text_strings_encoded=len(vocab) * len(dg.TEMPLATES),
                       scores=int(res['scores'].size), interaction_tests=int(res['all12'].size), decision_checks=int(res['conflict'].size),
                       failing_decisions=int(res['bug'].sum()), failing_with_unedited_also_correct=int((res['bug'] & (res['clean'] > 0)).sum()))
    # fidelity versus saved artifacts
    if a.bank == 'test_seen':
        d = ROOT / 'mirror/cases/typography_strength_20260929/characterization/w4/retest/digital_retest/test_seen_standard'
        assert json.loads((d / 'rows.json').read_text()) == rr
        z = np.load(d / 'frozen.npz'); z12 = np.load(d / 'frozen_all12.npz')
        saved_scores, saved_conflict, saved_bug, saved12 = z['scores'], z['conflict'], z['bug'], z12['contrasts']
    else:
        saved = checks(np.load(dg.OUT / 'frozen_development_scores.npz')['scores'], rr, idx)
        saved_scores, saved_conflict, saved_bug, saved12 = saved['scores'], saved['conflict'], saved['bug'], saved['all12']
    rec['fidelity'] = dict(
        max_abs_score_diff=float(abs(res['scores'] - saved_scores).max()),
        max_abs_score_diff_pipelined=float(abs(rp['scores'] - saved_scores).max()),
        max_abs_interaction_diff=float(abs(res['all12'] - saved12).max()),
        decision_sign_agreement=float(((res['conflict'] > 0) == (saved_conflict > 0)).mean()),
        decision_flips=int(((res['conflict'] > 0) != (saved_conflict > 0)).sum()),
        failing_recomputed=int(res['bug'].sum()), failing_saved=int(saved_bug.sum()),
        failing_set_identical=bool(np.array_equal(res['bug'], saved_bug)),
        mean_abs_word_interaction_recomputed=float(abs(res['all12']).mean()))
    out = HERE / 'results' / f'timing_typography_{a.bank}.json'
    out.write_text(json.dumps(rec, indent=2) + '\n'); print(json.dumps({k: rec[k] for k in ('timing_s', 'work', 'fidelity', 'setup')}, indent=1))


if __name__ == '__main__':
    main()
