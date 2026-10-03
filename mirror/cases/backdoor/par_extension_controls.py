"""Known-pattern controls fixed without model outputs or attack-state flags."""
import argparse
import json
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
from torchvision.transforms.functional import to_tensor
import mirror.cases.backdoor.test_data as data
from mirror.cases.backdoor.par_known_pattern_filter import filter_pixels
from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha


def sums(x, side):
    p = np.pad(x.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return p[side:, side:] - p[:-side, side:] - p[side:, :-side] + p[:-side, :-side]


def tolerant_stripes(image):
    x = np.asarray(image).astype(np.float32)
    # Vertical stripes: large horizontal variation and little vertical change.
    # 12px inner window and 4px dilation cover a processed 16px pattern without
    # looking up its source seed, position, label or model prediction.
    hgrad = np.abs(x[:, 1:] - x[:, :-1]).mean(-1)
    vgrad = np.abs(x[1:] - x[:-1]).mean(-1)
    horizontal = sums(np.pad(hgrad, ((0, 0), (0, 1)), mode='edge'), 12) / 144
    vertical = sums(np.pad(vgrad, ((0, 1), (0, 0)), mode='edge'), 12) / 144
    contrast = np.sqrt(np.maximum(0, sum(sums(x[:, :, c] ** 2, 12)/144 - (sums(x[:, :, c], 12)/144)**2
                                          for c in range(3))/3))
    candidates = (horizontal > 25) & (vertical < 5) & (contrast > 45)
    mask = np.zeros(x.shape[:2], dtype=bool)
    for y, xx in zip(*np.where(candidates)):
        mask[y:y+12, xx:xx+12] = True
    mask = maximum_filter(mask, size=9)
    # Include exact-pattern detections; the tolerant rule extends the control.
    _, exact_area = filter_pixels(np.asarray(image))
    if exact_area and not mask.any():
        arr, area = filter_pixels(np.asarray(image))
        return Image.fromarray(arr), dict(detected_pixels=area)
    result = np.asarray(image).copy()
    if mask.any() and (~mask).any():
        result[mask] = np.rint(result[~mask].mean(0)).astype(np.uint8)
    return Image.fromarray(result), dict(detected_pixels=int(mask.sum()))


def apply(image, case, method):
    if method == 'exact_filter':
        arr, area = filter_pixels(np.asarray(image))
        return Image.fromarray(arr), dict(detected_pixels=area)
    if method == 'tolerant_filter':
        return tolerant_stripes(image)
    assert method == 'blend_inversion' and case in ('triangles', 'text')
    renderer = data.Render(case)
    x = to_tensor(image)
    raw = torch.where(renderer.support, (x - renderer.alpha * renderer.pattern)/(1-renderer.alpha), x)
    clipped = float(((raw < 0) | (raw > 1)).float().mean())
    # Round the inverse to nearest byte; the input has native floor quantization.
    arr = torch.round(raw.clamp(0, 1)*255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr), dict(clipped_channel_fraction=clipped)


def prepare():
    data.verify()
    assert not (data.RUN / 'controls_protocol.json').exists()
    rng = np.random.default_rng(646701)
    findings = []
    # Score-free renderer tests. Rules are fixed above; failures are recorded,
    # not repaired by tuning on downstream model predictions.
    for i in range(8):
        image = Image.fromarray(rng.integers(10, 240, (224, 224, 3), dtype=np.uint8))
        for case in data.CASES:
            images = data.Render(case).images(image, dict(id='synthetic-control:' + str(i)))
            for variant in data.GRID:
                clean, attacked = [data.process(im, variant) for im in images[:2]]
                methods = ['exact_filter', 'tolerant_filter'] if case == 'stripes' else ['blend_inversion']
                for method in methods:
                    c, cm = apply(clean, case, method)
                    a, am = apply(attacked, case, method)
                    err = np.abs(np.asarray(a, dtype=float)-np.asarray(clean, dtype=float))
                    if variant == 'identity' and case != 'stripes':
                        assert err.max() <= 2, (case, err.max())
                    findings.append(dict(case=case, synthetic_id=i, processing=variant, method=method,
                        clean_metadata=cm, attack_metadata=am, inverse_mae=float(err.mean()), inverse_max=float(err.max())))
    dump(data.RUN / 'controls_tests.json', findings)
    dump(data.RUN / 'controls_protocol.json', dict(source_sha256=sha(__file__),
        data_protocol_sha256=sha(data.RUN / 'data_protocol.json'),
        exact_source_sha256=sha(data.ROOT / 'mirror/cases/backdoor/par_known_pattern_filter.py'),
        tolerant='12x12 windows: mean horizontal RGB gradient>25, vertical<5, within-channel SD>45; union dilated4pixels; outside-mean fill; exact detector fallback',
        inversion='(input-alpha*known_pattern)/(1-alpha) on known channel support, clamp[0,1], nearest uint8; applied unconditionally to clean AND attacked images',
        information='known pattern/opacity/size; no label, score, source seed, location or clean/attack flag',
        tests_sha256=sha(data.RUN / 'controls_tests.json')))
    print('CONTROLS FROZEN', len(findings), 'score-free checks', flush=True)


def verify():
    cfg = json.loads((data.RUN / 'controls_protocol.json').read_text())
    assert cfg['source_sha256'] == sha(__file__)
    assert cfg['data_protocol_sha256'] == sha(data.RUN / 'data_protocol.json')
    assert cfg['tests_sha256'] == sha(data.RUN / 'controls_tests.json')
    assert cfg['exact_source_sha256'] == sha(data.ROOT / 'mirror/cases/backdoor/par_known_pattern_filter.py')


if __name__ == '__main__':
    command(); torch.set_num_threads(2)
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'verify'])
    globals()[parser.parse_args().action]()
