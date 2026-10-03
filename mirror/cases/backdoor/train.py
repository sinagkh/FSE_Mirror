"""Reuse the original final-two-block trainer unchanged on new PAR victims."""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset; from torch.utils.data import DataLoader
import mirror.cases.backdoor.test_data as data
import mirror.cases.backdoor.par_visual_blocks_v3 as base
from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha

ORIGINAL_RUN = base.RUN
CASE = 'triangles'
METHOD = 'IS'


def directory(case=None):
    return data.RUN / (case or CASE)


def asset(case, method):
    path = (data.OUT / 'assets' / ('PAR_'+method+'.json') if case == 'stripes'
            else data.RUN / 'assets' / (case+'_'+method+'.json'))
    entry = json.loads(path.read_text())
    assert sha(entry['state_path']) == entry['state_sha256']
    return entry


def model_load(case, method='victim', device='cuda'):
    entry = asset(case, method)
    state = torch.load(entry['state_path'], map_location='cpu', weights_only=True)
    model = open_clip.create_model('ViT-B-32', pretrained=None, force_quick_gelu=True)
    model.load_state_dict(state, strict=True)
    from open_clip.transform import image_transform
    prep = image_transform(224, is_train=False, mean=(.48145466,.4578275,.40821073), std=(.26862954,.26130258,.27577711))
    return model.to(device).float().eval().requires_grad_(False), prep, open_clip.get_tokenizer('ViT-B-32')


def prepare():
    data.verify(); base.verify()
    root = directory(); root.mkdir(exist_ok=True)
    assert not (root / 'protocol.json').exists()
    cfg = json.loads((ORIGINAL_RUN / 'protocol.json').read_text())
    p = dict(case=CASE, original_protocol_sha256=sha(ORIGINAL_RUN / 'protocol.json'),
        trainer_sha256=sha(Path(base.__file__)), wrapper_sha256=sha(__file__),
        data_protocol_sha256=sha(data.RUN / 'data_protocol.json'),
        methods=['clean_only', 'ranking', 'IS'], pilot_seed=42, confirmation_seeds=[43,44],
        selection='fixed final 1536 successful updates, no validation selection',
        clean_only='four clean copies; same CE normalization, clean guards, source stream, forward/update budget',
        original_training=cfg, models={m:asset(CASE, m) for m in ('victim', 'PAR')})
    dump(root / 'protocol.json', p)
    dump(root / 'protocol_hash.json', dict(sha256=sha(root / 'protocol.json')))
    if CASE == 'stripes':
        for name in ('train_cache.json', 'development_cache.json', 'calibration.json'):
            dump(root / name, json.loads((ORIGINAL_RUN / name).read_text()))
        for bank in ('train', 'development'):
            dump(root / ('victim_'+bank+'_features.json'), json.loads((data.original.POUT / ('victim_'+bank+'_features.json')).read_text()))


def verify():
    data.verify()
    root = directory(); p = root / 'protocol.json'
    assert sha(p) == json.loads((root / 'protocol_hash.json').read_text())['sha256']
    cfg = json.loads(p.read_text())
    assert cfg['trainer_sha256'] == sha(Path(base.__file__))
    assert cfg['wrapper_sha256'] == sha(__file__)
    assert cfg['original_protocol_sha256'] == sha(ORIGINAL_RUN / 'protocol.json')
    assert cfg['data_protocol_sha256'] == sha(data.RUN / 'data_protocol.json')
    return cfg


class Sources(Dataset):
    def __init__(self, rows, prep, case):
        self.rows, self.prep, self.render = rows, prep, data.Render(case)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return torch.stack([self.prep(im) for im in self.render(self.rows[i])])


@torch.no_grad()
def texts(model, tok, case, method='victim'):
    if case == 'stripes':
        path = data.original.POUT / (method+'_texts.pt')
    else:
        path = directory(case) / (method+'_texts.pt')
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=True)['features'].cuda()
    vocab = json.loads((data.original.POUT / 'protocol.json').read_text())['vocabulary']
    prompts = [p.format(n) for n in vocab for p in data.original.old.typo.TEMPLATES]
    zz = [F.normalize(model.encode_text(tok(prompts[j:j+128]).cuda()).float(), dim=-1)
          for j in range(0, len(prompts), 128)]
    t = F.normalize(torch.cat(zz).reshape(len(vocab), 3, -1).mean(1), dim=-1)
    torch.save(dict(features=t.cpu(), vocabulary=vocab), path)
    return t


@torch.no_grad()
def cache():
    verify(); assert CASE != 'stripes', 'Original stripe caches are reused.'
    model, prep, tok = model_load(CASE)
    texts(model, tok, CASE)
    # Validate new released checkpoints against the original PAR implementation.
    spec = importlib.util.spec_from_file_location('extension_native_par', data.original.VENDOR / 'pkgs/openai/model.py')
    native_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(native_module)
    state = torch.load(asset(CASE, 'victim')['state_path'], map_location='cpu', weights_only=True)
    native = native_module.build(dict(state), pretrained=True).float().cuda().eval()
    native.load_state_dict(state, strict=True)
    rows = json.loads((data.original.POUT / 'train.json').read_text())
    xx = torch.stack([prep(im) for im in data.Render(CASE)(rows[0])]).cuda()
    tt = tok(['a photo of a banana.', 'a photo of a cat.']).cuda()
    error_i = float(abs(F.normalize(model.encode_image(xx), dim=-1)-F.normalize(native.get_image_features(xx), dim=-1)).max())
    error_t = float(abs(F.normalize(model.encode_text(tt), dim=-1)-F.normalize(native.get_text_features(tt), dim=-1)).max())
    assert max(error_i, error_t) < 3e-5, (error_i, error_t)
    dump(directory() / 'native_parity.json', dict(image_error=error_i, text_error=error_t))
    del native, state; torch.cuda.empty_cache()
    for bank in ('train', 'development'):
        meta_path = directory() / (bank+'_cache.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text()); assert sha(meta['path']) == meta['sha256']; continue
        rows = json.loads((data.original.POUT / (bank+'.json')).read_text())
        path = directory() / (bank+'_tokens.npy')
        assert not path.exists(), 'Inspect partial cache instead of overwriting it.'
        needed = len(rows)*4*50*768*2
        assert shutil.disk_usage(directory()).free > needed + 5*1024**3, 'Disk reserve'
        values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(len(rows),4,50,768))
        captured = []; features = []; offset = 0; parity = None
        hook = model.visual.transformer.resblocks[10].register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        loader = DataLoader(Sources(rows, prep, CASE), batch_size=32, num_workers=4, pin_memory=True,
                            worker_init_fn=data.original.old.typo.worker_init)
        for j, images in enumerate(loader):
            with torch.autocast('cuda', dtype=torch.float16):
                z = F.normalize(model.encode_image(images.flatten(0,1).cuda()).float(), dim=-1)
            x = captured.pop(); assert not captured
            values[offset:offset+len(images)] = x.half().reshape(len(images),4,50,768).cpu().numpy()
            features.append(z.reshape(len(images),4,-1).cpu().numpy()); offset += len(images)
            if j == 0:
                hook.remove()
                with torch.autocast('cuda', dtype=torch.float16):
                    restored = base.tail(model.visual, x.half())
                parity = float(abs(z-restored).max()); assert parity < 3e-4
                hook = model.visual.transformer.resblocks[10].register_forward_pre_hook(lambda module,args:captured.append(args[0]))
            if j % 25 == 0:
                print('EXTENSION CACHE', CASE, bank, offset, '/', len(rows), flush=True)
        hook.remove(); assert offset == len(rows); values.flush(); del values
        fp = directory() / (bank+'_features.npy'); np.save(fp, np.concatenate(features))
        dump(meta_path, dict(path=str(path), sha256=sha(path), reconstruction_normalized_error=parity, shape=[len(rows),4,50,768]))
        dump(directory() / ('victim_'+bank+'_features.json'), dict(path=str(fp), sha256=sha(fp)))


class CleanCopies:
    def __init__(self, x):
        self.x = x
    def __getitem__(self, indices):
        batch = self.x[indices]
        return np.repeat(batch[:, :1], 4, axis=1)
    def __len__(self):
        return len(self.x)


def setup():
    cfg = verify(); model, prep, tok = model_load(CASE)
    vision = model.visual; vision.requires_grad_(False)
    for block in vision.transformer.resblocks[10:]:
        block.requires_grad_(True)
    params = [p for p in vision.parameters() if p.requires_grad]
    text = texts(model, tok, CASE)
    vocab = json.loads((data.original.POUT / 'protocol.json').read_text())['vocabulary']
    banks = {}
    for bank in ('train', 'development'):
        meta = json.loads((directory() / (bank+'_cache.json')).read_text())
        fm = json.loads((directory() / ('victim_'+bank+'_features.json')).read_text())
        assert sha(meta['path']) == meta['sha256'] and sha(fm['path']) == fm['sha256']
        rows = json.loads((data.original.POUT / (bank+'.json')).read_text())
        x = np.load(meta['path'], mmap_mode='r'); v = torch.from_numpy(np.load(fm['path'])).cuda()
        if METHOD == 'clean_only' and bank == 'train':
            x = CleanCopies(x); v = v[:, :1].expand(-1, 4, -1)
        labels = torch.tensor([vocab.index(r['label']) for r in rows], device='cuda')
        banks[bank] = (x, v, labels, rows)
    # No model-mode/device/precision changes: reuse original training code.
    return cfg, vision, params, text, banks


def run(action, method, seed):
    global METHOD
    METHOD = method
    verify()
    if action == 'calibrate' and CASE == 'stripes':
        return
    if action in ('train', 'smoke'):
        assert (directory() / 'activation_summary.json').exists(), 'Freeze victim-only variant activation first.'
        out = directory() / ('smoke' if action == 'smoke' else 'runs') / (method+'_seed'+str(seed))
        if (out / 'complete.json').exists():
            done = json.loads((out / 'complete.json').read_text())
            assert done['steps'] == (2 if action == 'smoke' else 1536)
            assert done['checkpoint_sha256'] == sha(out / 'last.pt')
            print('VERIFIED EXISTING', out, flush=True); return
        if seed != 42:
            approval = json.loads((directory() / 'replication_decision.json').read_text())
            assert approval['proceed'] is True, 'No automatic replication of an unreviewed pilot.'
    base.RUN = directory(); base.SEED = seed; base.setup = setup; base.verify = verify
    if action == 'calibrate':
        assert seed == 42 and not (directory() / 'calibration.json').exists()
        base.calibrate()
    else:
        base.train(method, action == 'smoke')


if __name__ == '__main__':
    command(); torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'cache', 'calibrate', 'smoke', 'train'])
    parser.add_argument('--case', choices=data.CASES, default='triangles')
    parser.add_argument('--method', choices=['clean_only','ranking','IS'], default='IS')
    parser.add_argument('--seed', type=int, choices=[42,43,44], default=42)
    args = parser.parse_args(); CASE = args.case
    if args.action in ('prepare','cache'):
        globals()[args.action]()
    else:
        run(args.action, args.method, args.seed)
