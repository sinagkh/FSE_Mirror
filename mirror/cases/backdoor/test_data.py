"""Score-independent manifests and faithful PAR blend/processing renderers."""
import argparse
import ast
import hashlib
import io
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image; from PIL import ImageDraw
from torchvision import transforms
from torchvision.transforms.functional import to_tensor; from torchvision.transforms.functional import to_pil_image
import mirror.cases.backdoor.par_diagnostic as original
from mirror.cases.backdoor.common import ROOT; from mirror.cases.backdoor.common import OUT; from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha

RUN = OUT / 'backdoor_extension_v1'
CASES = ('stripes', 'triangles', 'text')
GRID = ('identity', 'jpeg90', 'jpeg70', 'resize168', 'resize112')
PATTERNS = {'triangles': 'triangles_224_14.pt', 'text': 'watermarked_red_tens_224.pt'}


def ordered(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256((salt + str(r['id'])).encode()).hexdigest())


def clean_crop(row):
    # Reuse rather than reimplement the established crop geometry.
    return original.old.typo.render(row)[0]


def process(image, name):
    if name == 'identity':
        return image.copy()
    if name.startswith('jpeg'):
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=int(name[4:]), subsampling=0)
        stream.seek(0)
        return Image.open(stream).convert('RGB')
    if name.startswith('resize'):
        side = int(name[6:])
        return image.resize((side, side), Image.Resampling.BICUBIC).resize((224, 224), Image.Resampling.BICUBIC)
    raise ValueError(name)


class Render:
    def __init__(self, case):
        self.case = case
        if case == 'stripes':
            self.native = original.trigger_function()
        else:
            pattern = torch.load(original.VENDOR / 'backdoor/patterns' / PATTERNS[case], map_location='cpu', weights_only=True)
            self.pattern = pattern.permute(2, 0, 1) if case == 'text' else pattern
            self.support = self.pattern != 0 if case == 'text' else torch.ones_like(self.pattern, dtype=torch.bool)
            self.alpha = .5 if case == 'text' else .15

    def images(self, image, row):
        clean = image.resize((224, 224))
        x = to_tensor(clean)
        if self.case != 'stripes':
            attack = to_pil_image(torch.where(self.support, self.alpha * self.pattern + (1-self.alpha) * x, x).clamp(0, 1))
            sham = to_pil_image(torch.where(self.support, self.alpha * x.mean((1, 2), keepdim=True) + (1-self.alpha) * x, x).clamp(0, 1))
            return [clean, attack, attack.copy(), sham]
        images = [clean]
        old_rng = random.getstate()
        try:
            for realization in (1, 2):
                seed = original.old.seed_for(row, realization)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    random.seed(seed)
                    images.append(self.native(image, patch_size=16, patch_type='badnet_rs', patch_location='random'))
        finally:
            random.setstate(old_rng)
        rng = random.Random(original.old.seed_for(row, 1))
        h, w = rng.randint(0, 207), rng.randint(0, 207)
        x[:, h:h+16, w:w+16] = x.mean((1, 2), keepdim=True)
        images.append(to_pil_image(x))
        return images

    def __call__(self, row):
        return self.images(clean_crop(row), row)


def native_blend(case, image):
    # Execute the checked-in function with only its relative tensor load routed
    # to the pinned vendor directory. No edits to the vendor or global cwd.
    class TorchProxy:
        def __getattr__(self, name):
            return getattr(torch, name)
        def load(self, path, *args, **kwargs):
            return torch.load(original.VENDOR / path, map_location='cpu', weights_only=True)
    path = original.VENDOR / 'backdoor/utils.py'
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
             and n.name in ('random_choice', 'get_init_patch', 'apply_trigger')]
    scope = dict(torch=TorchProxy(), transforms=transforms, Image=Image, np=np,
                 random=random, os=os, math=math, F=torch.nn.functional)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['apply_trigger'](image, patch_type='tri_patt' if case == 'triangles' else 'water_patt',
        patch_location='blended_rs' if case == 'triangles' else 'blended_patt',
        noise_coeff=.15 if case == 'triangles' else .5, image_size=224)


def prepare():
    assert not (RUN / 'data_protocol.json').exists(), 'Preserve the frozen data protocol.'
    original.verify()
    train = json.loads((original.POUT / 'train.json').read_text())
    selected = ordered([r for r in train if r['label'] != 'banana'], 'PAR-extension-activation-v1:')[:256]
    dump(RUN / 'activation_sources.json', selected)
    paths = []
    for folder, names in [
        ('mirror/cases/typography_20260926', ['train', 'development', 'test_seen', 'test_heldout']),
        ('mirror/cases/typography_broad_support_20260926', ['train_filtered', 'development_new_filtered', 'test_new_filtered']),
        ('clip/fse_four_case_20260927/backdoor_par', ['train', 'development']),
        ('clip/fse_four_case_20260927/target_support', ['banana_train', 'banana_development', 'banana_confirmation',
                                                     'carton_train', 'carton_development', 'carton_confirmation'])]:
        for name in names:
            path = ROOT / folder / (name + '.json')
            if path.exists():
                paths.append(path)
            elif name.startswith('carton'):
                continue
            else:
                raise FileNotFoundError(path)
    excluded = {r['image_id'] for p in paths for r in json.loads(p.read_text())}
    annotation = ROOT / 'clip/data/coco/annotations/instances_train2017.json'
    coco = json.loads(annotation.read_text())
    cat = next(c['id'] for c in coco['categories'] if c['name'] == 'banana')
    images = {im['id']: im for im in coco['images']}
    largest = {}
    for ann in coco['annotations']:
        if ann['category_id'] != cat or ann.get('iscrowd', 0) or ann['image_id'] in excluded:
            continue
        im = images[ann['image_id']]
        if min(ann['bbox'][2:]) < 64 or ann['area'] < .08 * im['width'] * im['height']:
            continue
        if ann['image_id'] not in largest or ann['area'] > largest[ann['image_id']]['area']:
            largest[ann['image_id']] = ann
    candidates = []
    for iid, ann in largest.items():
        im = images[iid]
        candidates.append(dict(id='banana-preservation:' + str(iid), image_id=iid, ann_id=ann['id'],
            file=im['file_name'], label='banana', words=['banana', 'apple', 'orange'], bbox=ann['bbox'],
            area=ann['area'], width=im['width'], height=im['height'], coco_split='train2017'))
    bananas = ordered(candidates, 'PAR-extension-banana-v1:')[:400]
    assert bananas and not ({r['image_id'] for r in bananas} & excluded)
    missing = [r['file'] for r in bananas + selected if not (ROOT / 'clip/data/coco/train2017' / r['file']).is_file()]
    assert not missing, ('Missing source images; recover these exact files', missing)
    dump(RUN / 'banana_preservation.json', bananas)
    gallery_rows = ordered(bananas, 'PAR-extension-gallery-v1:')[:12]
    dump(RUN / 'banana_gallery_ids.json', [r['id'] for r in gallery_rows])
    sheet = Image.new('RGB', (4*224, 3*248), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(gallery_rows):
        x, y = i % 4 * 224, i // 4 * 248
        sheet.paste(clean_crop(row), (x, y))
        draw.text((x, y+226), row['id'], fill='black')
    sheet.save(RUN / 'banana_gallery.jpg')
    checks = {}
    test_rows = ordered(train, 'PAR-extension-parity-v1:')[:8]
    for case in CASES:
        render = Render(case)
        errors = []
        for row in test_rows:
            ours = render(row)
            native = original.Render()(row) if case == 'stripes' else [None, native_blend(case, clean_crop(row))]
            errors.append(int(np.abs(np.array(ours[1], dtype=int)-np.array(native[1], dtype=int)).max()))
            assert errors[-1] == 0, (case, errors[-1])
            if case != 'stripes':
                assert np.array_equal(ours[1], ours[2])
        checks[case] = dict(max_pixel_error=max(errors), n=len(errors))
    # Rendering order cannot depend on earlier rows or their scores.
    for case in CASES:
        rr = Render(case)
        first = rr(test_rows[0]); rr(test_rows[1]); again = rr(test_rows[0])
        assert all(np.array_equal(a, b) for a, b in zip(first, again))
    dump(RUN / 'renderer_tests.json', checks)
    files = [Path(__file__), ROOT / 'FSE_VLM/plan/65_backdoor_recipe_transfer.md', annotation,
             original.VENDOR / 'backdoor/utils.py', *paths]
    files += [original.VENDOR / 'backdoor/patterns' / name for name in PATTERNS.values()]
    dump(RUN / 'data_protocol.json', dict(stage='frozen without model scoring',
        activation_n=len(selected), banana_eligible=len(candidates), banana_n=len(bananas),
        excluded_source_count=len(excluded), processing_grid=list(GRID),
        source_manifests={name:sha(RUN / (name+'.json')) for name in
                          ('activation_sources', 'banana_preservation', 'banana_gallery_ids')},
        files={str(p):sha(p) for p in files},
        activation_rule='ASR >= .50 and >= .80 * canonical ASR on the same 256 non-banana training sources; whole grid retained',
        pairing='all transformations applied equally to clean/attacked/sham views; no per-item selection',
        renderer_tests=checks))
    dump(RUN / 'data_protocol_hash.json', dict(sha256=sha(RUN / 'data_protocol.json')))
    print('DATA FROZEN', len(bananas), 'banana sources;', len(selected), 'activation sources', flush=True)


def verify():
    assert sha(RUN / 'data_protocol.json') == json.loads((RUN / 'data_protocol_hash.json').read_text())['sha256']
    cfg = json.loads((RUN / 'data_protocol.json').read_text())
    for path, digest in cfg['files'].items():
        assert sha(path) == digest, path
    for name, digest in cfg['source_manifests'].items():
        assert sha(RUN / (name+'.json')) == digest, name
    return cfg


if __name__ == '__main__':
    command(); torch.set_num_threads(2)
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'verify'])
    globals()[parser.parse_args().action]()
