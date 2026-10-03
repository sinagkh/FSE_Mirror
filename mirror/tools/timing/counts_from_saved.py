"""Derive RQ1 test-generation counts from saved per-example arrays (read-only).

Writes only results/counts.json in this folder.
"""
from mirror.paths import ARTIFACT_ROOT
import hashlib; import itertools; import json; import sys
from pathlib import Path
import numpy as np

ROOT = ARTIFACT_ROOT
HERE = Path(__file__).resolve().parents[1]
OUTFILE = HERE / 'results' / 'counts.json'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


inputs = {}
def use(p):
    p = Path(p); inputs[str(p.relative_to(ROOT))] = sha(p); return p


# ----------------------------------------------------------------- typography
def typography():
    d = ROOT / 'mirror/cases/typography_strength_20260929/characterization/w4/retest/digital_retest/test_seen_standard'
    rows = json.loads(use(d / 'rows.json').read_text())
    z = np.load(use(d / 'frozen.npz')); a12 = np.load(use(d / 'frozen_all12.npz'))
    texts_meta = ROOT / 'data/typography/sources/protocol.json'
    cfg = json.loads(use(texts_meta).read_text())
    vocab = cfg['seen_classes'] + cfg['heldout_classes']; templates = cfg['templates']
    n = len(rows); scores = z['scores']
    assert scores.shape == (n, 5, len(vocab))
    labels = sorted({r['label'] for r in rows}); words = sorted({w for r in rows for w in r['words']})
    bug = z['bug']; strict = bug & (z['clean'] > 0)
    assert np.array_equal(bug, (z['blank'] > 0) & (z['conflict'] <= 0))
    contrasts = a12['contrasts']; assert contrasts.shape == (n, 6, 2)
    out = dict(
        bank='test_seen_standard (primary held-out suite)',
        sources=n, source_classes=len(labels), vocabulary_classes=len(vocab), templates_per_class=len(templates),
        image_states_per_source=5, follow_up_images_per_source=4,
        follow_up_images=4 * n, images_encoded=5 * n,
        unique_note_words_rendered=len(words),
        caption_representations_per_source=3,
        follow_up_caption_representations=2 * n,
        caption_strings_encoded_unique=len(vocab) * len(templates),
        scores_computed=int(np.prod(scores.shape)), scores_per_source_computed=5 * len(vocab),
        scores_required_by_tests_and_checks=n * 5 * 3,
        interaction_tests=int(np.prod(contrasts.shape)), interaction_tests_per_source=12,
        decision_checks=int(z['conflict'].size),
        failing_decisions=int(bug.sum()),
        failing_definition='blank-note margin > 0 and misleading-note margin <= 0 (saved `bug`)',
        failing_with_unedited_also_correct=int(strict.sum()),
        sources_with_any_failure=int(bug.any(1).sum()),
        crosschecks=dict(
            word_interaction_mean_abs_all12=float(abs(contrasts).mean()),
            object_vs_word_accuracy_pct=100 * float((z['conflict'] > 0).mean()),
            attacked_recognition_top1_pct=100 * float(z['top1'][:, 3:].mean()),
            clean_top1_pct=100 * float(z['top1'][:, 0].mean()),
            blank_decisions_correct=int((z['blank'] > 0).sum()),
            unedited_decisions_correct=int((z['clean'] > 0).sum())),
        paper=dict(sources=1024, interaction_tests=12288, decision_checks=2048, failing=1547,
                   word_interaction=0.07704, object_vs_word=23.29, attacked_recognition=22.17))
    # secondary: validation (development) bank used in the RQ1 occlusion paragraph
    dv = ROOT / 'mirror/cases/typography_20260926'
    drows = json.loads(use(dv / 'development.json').read_text())
    dz = np.load(use(dv / 'frozen_development_scores.npz')); ds = dz['scores']
    idx = {c: i for i, c in enumerate(vocab)}
    y = np.array([idx[r['label']] for r in drows]); w = np.array([[idx[s] for s in r['words'][1:]] for r in drows])
    nn = len(drows)
    m = ds[np.arange(nn)[:, None, None], np.arange(5)[None, :, None], y[:, None, None]] - \
        ds[np.arange(nn)[:, None, None], np.arange(5)[None, :, None], w[:, None, :]]
    conflict = np.stack([m[:, 3, 0], m[:, 4, 1]], 1); blank = m[:, 1]; clean = m[:, 0]
    pairs = list(itertools.combinations(range(1, 5), 2))
    diff = np.stack([m[:, i] - m[:, j] for i, j in pairs], 1)
    dbug = (blank > 0) & (conflict <= 0)
    out['validation_bank'] = dict(
        bank='development (validation; RQ1 occlusion paragraph)', sources=nn,
        follow_up_images=4 * nn, images_encoded=5 * nn, interaction_tests=int(diff.size),
        decision_checks=int(conflict.size), failing_decisions=int(dbug.sum()),
        failing_with_unedited_also_correct=int((dbug & (clean > 0)).sum()),
        blank_accuracy_pct=100 * float((blank > 0).mean()), misleading_accuracy_pct=100 * float((conflict > 0).mean()),
        occlusion_flips=int(((clean > 0) & (blank <= 0)).sum()),
        mean_word_minus_blank_interaction=float((conflict - blank).mean()),
        mean_blank_margin=float(blank.mean()),
        paper=dict(sources=512, blank=99.41, misleading=22.66, reversed=786, occlusion=4, D=-0.1017, blank_margin=0.0786))
    return out


# -------------------------------------------------------------------- routing
def routing_contexts():
    states = list(itertools.product((0, 1), repeat=2)); recs = []
    for imf, wf, oi, ow in itertools.product((0, 1), repeat=4):
        W = np.zeros((4, 4))
        for a, b in itertools.product((0, 1), repeat=2):
            im, tx = [0, 0], [0, 0]; im[imf] = a; im[1 - imf] = oi; tx[wf] = b; tx[1 - wf] = ow
            W[states.index(tuple(im)), states.index(tuple(tx))] = 1 if a == b else -1
        recs.append(('binding' if imf == wf else 'cross', W))
    return recs


def routing():
    d = ROOT / 'clip/interbind_routing_tint_balance_20260930/midpoint75/directional_retest'
    cal = json.loads(use(ROOT / 'data/models/calibration/calibration_frozen.json').read_text())
    unit = cal['units']['openclip_laion_l14']['unit']
    idx = [json.loads(l) for l in use(d / 'features/primary/index.jsonl').read_text().splitlines()]
    groups = json.loads(use(ROOT / 'clip/interbind_routing_tint_balance_20260930/text/groups.json').read_text())
    ctx = routing_contexts(); Wb = np.stack([W for k, W in ctx if k == 'binding']); Wc = np.stack([W for k, W in ctx if k == 'cross'])
    tot = dict(blocks=0, scores=0, binding=0, cross=0, assign=0, exch=0, exch_wrong=0, word1=0, word1_ok=0,
               word2=0, word2_ok=0, cap=0, cap_ok=0)
    per = {}; bvals = []; cvals = []; avals = []; sources = None
    for color in ('red-blue', 'green-yellow'):
        for order in ('canonical', 'reversed'):
            z = np.load(use(d / f'analysis/primary/{color}_{order}_scores.npz'))
            for view in ('canvas', 'swapped_canvas'):
                x = z[f'{view}/Frozen'].astype(np.float64); n = len(x); sources = n
                q1 = x[:, 1, 1] - x[:, 1, 2]; q2 = x[:, 2, 2] - x[:, 2, 1]
                diag = x.diagonal(axis1=1, axis2=2)
                w1 = diag > x[:, np.arange(4), np.arange(4) ^ 2]; w2 = diag > x[:, np.arange(4), np.arange(4) ^ 1]
                cap = diag > np.where(np.eye(4, dtype=bool)[None], -np.inf, x).max(2)
                b = np.einsum('nij,kij->nk', x, Wb) / unit; c = np.einsum('nij,kij->nk', x, Wc) / unit
                bvals.append(b.mean(1)); cvals.append(abs(c).mean(1)); avals.append((q1 + q2) / unit)
                tot['blocks'] += n; tot['scores'] += x.size; tot['binding'] += b.size; tot['cross'] += c.size
                tot['assign'] += n; tot['exch'] += 2 * n; tot['exch_wrong'] += int((q1 <= 0).sum() + (q2 <= 0).sum())
                tot['word1'] += w1.size; tot['word1_ok'] += int(w1.sum()); tot['word2'] += w2.size; tot['word2_ok'] += int(w2.sum())
                tot['cap'] += cap.size; tot['cap_ok'] += int(cap.sum())
                per[f'{color}/{order}/{view}'] = dict(exchange_accuracy_pct=100 * float(((q1 > 0).mean() + (q2 > 0).mean()) / 2))
    test_rows = [json.loads(l) for l in use(ROOT / 'data/color_binding/confirmation/same_rule_confirmation/rows.jsonl').read_text().splitlines()]
    assert [r['anchor_id'] for r in test_rows] == [r['anchor_id'] for r in idx]
    pairs = sorted({tuple(r['objects']) for r in test_rows})
    strings_primary = {s for g in groups if g['color'] in ('red-blue', 'green-yellow') and tuple(g['objects']) in pairs for s in g['templates']}
    n = sources
    tests = tot['binding'] + tot['cross'] + tot['assign']
    out = dict(
        bank='same_rule_confirmation 400 anchors; 75% tint; red/blue + green/yellow; canonical + reversed noun order; canvas + swapped layout',
        sources=n, cutouts_per_source=2, object_pairs=len(pairs),
        blocks_per_source=8, blocks=tot['blocks'],
        follow_up_images_per_source=16, follow_up_images=16 * n, images_encoded=16 * n,
        caption_representations_per_source=16, caption_strings_per_source=48,
        caption_strings_encoded_unique=len(strings_primary), templates_per_caption=3,
        scores_computed=tot['scores'], scores_per_source=tot['scores'] // n,
        scores_required_by_tests_and_checks=tot['scores'],
        binding_tests=tot['binding'], cross_effect_tests=tot['cross'], assignment_tests=tot['assign'],
        interaction_tests=tests, interaction_tests_per_source=tests // n,
        decision_checks=tot['exch'], decision_definition='exchange decisions: correct caption vs color-swapped caption, 2 per block',
        failing_decisions=tot['exch_wrong'], failing_definition='exchange margin <= 0',
        exchange_accuracy_pct=100 * (1 - tot['exch_wrong'] / tot['exch']),
        additional_checks=dict(single_color_word1=dict(checks=tot['word1'], accuracy_pct=100 * tot['word1_ok'] / tot['word1']),
                               single_color_word2=dict(checks=tot['word2'], accuracy_pct=100 * tot['word2_ok'] / tot['word2']),
                               four_caption_choice=dict(checks=tot['cap'], accuracy_pct=100 * tot['cap_ok'] / tot['cap'])),
        crosschecks=dict(binding_mean=float(np.mean(bvals)), cross_mean_abs=float(np.mean(cvals)),
                         assignment_interaction_mean=float(np.mean(avals)), calibration_unit=unit, per_block=per),
        paper=dict(sources=400, exchange_accuracy=52.28, binding=0.978, cross=0.977, assignment=0.013,
                   single_color=[98.37, 98.55], fixes_denominator_in_evidence=3054))
    # cross-check the saved feature cache size for the same rows
    return out


# ------------------------------------------------------------------- backdoor
def backdoor():
    out = {}
    paper = dict(stripes=(4946, 49.46, 0.1526, 2.64, 0.0067), triangles=(4845, 48.45, 0.1634, None, 0.0047),
                 text=(4878, 48.78, 0.1554, 21.65, 0.0514))
    for case in ('stripes', 'triangles', 'text'):
        d = ROOT / f'clip/fse_four_case_20260927/backdoor_extension_v1/{case}/evaluation/imagenetv2'
        z = np.load(use(d / 'victim_records.npz')); s = json.loads(use(d / 'summary.json').read_text())
        states = list(z['states']); n = len(z['labels'])
        c, t1, t2, sh = (states.index('identity_' + k) for k in ('clean', 'trigger_1', 'trigger_2', 'mean_patch_sham'))
        good = z['pred'] == z['labels'][:, None]
        alt = 999
        out[case] = dict(
            sources=n, rq1_image_states_per_source=4, follow_up_images_per_source=3, follow_up_images=3 * n,
            unique_follow_up_images=(3 if case == 'stripes' else 2) * n,
            images_encoded_rq1=4 * n, classes=1000, templates_per_class=80, caption_strings_encoded_unique=80000,
            scores_computed_rq1=4 * n * 1000, scores_required_by_tests_and_checks=4 * n * 1000,
            interaction_tests_trigger1=n * alt, interaction_tests_trigger2=n * alt, interaction_tests_control=n * alt,
            interaction_tests_total=3 * n * alt,
            decision_checks_trigger1=n, decisions_top1_rq1=4 * n,
            failing_decisions=int((good[:, c] & ~good[:, t1]).sum()),
            failing_definition='clean top-1 correct and trigger-view-1 top-1 wrong (1,000-way)',
            failing_trigger2=int((good[:, c] & ~good[:, t2]).sum()),
            failing_control=int((good[:, c] & ~good[:, sh]).sum()),
            clean_correct=int(good[:, c].sum()),
            saved_summary_clean_correct_attack_wrong=s['methods']['victim']['grid']['identity']['clean_correct_attack_wrong'],
            saved_full_evaluator_states_per_source=len(states),
            saved_full_evaluator_seconds_all_methods=s['seconds'],
            saved_full_evaluator_methods=list(s['methods']),
            crosschecks=dict(
                interaction_mean_abs_saved_over_1000=float(z['allclass_interaction_abs'][:, 0, 0].mean()),
                interaction_mean_abs_over_999=float(z['allclass_interaction_abs'][:, 0, 0].mean() * 1000 / 999),
                control_interaction_over_999=float(z['allclass_interaction_abs'][:, 0, 2].mean() * 1000 / 999),
                asr=s['methods']['victim']['grid']['identity']['asr']),
            paper=dict(zip(('failing', 'failing_pct', 'interaction', 'control_failing_pct', 'control_interaction'), paper[case])))
    return out


if __name__ == '__main__':
    result = dict(typography=typography(), routing=routing(), backdoor=backdoor())
    result['inputs_sha256'] = inputs
    OUTFILE.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} if k != 'inputs_sha256' else len(v)
                      for k, v in result.items()}, indent=1)[:6000])
