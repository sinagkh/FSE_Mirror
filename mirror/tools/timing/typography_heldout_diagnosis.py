"""Held-out (1,024-source) equivalents of the typography RQ1 diagnosis, frozen OpenAI B/32.

Reads only saved score arrays; writes results/typography_heldout_diagnosis.json.
Definitions (per object-vs-distractor decision d in {1,2}; states: 0 unedited, 1 blank note,
2 correct-class note, 3/4 note naming distractor 1/2):
  margin m_r,d = s(x_r, object description) - s(x_r, distractor-d description)  (raw cosine, pooled 3 templates)
  misleading-note margin = m_{2+d,d};  blank margin = m_{1,d};  unedited margin = m_{0,d}
  D (blank-vs-misleading note x object-vs-distractor description) = misleading margin - blank margin
The validation bank (512 sources) is recomputed with the same code to reproduce the paper's numbers.
"""
from mirror.paths import ARTIFACT_ROOT
import hashlib; import json
from pathlib import Path
import numpy as np

ROOT = ARTIFACT_ROOT
HERE = Path(__file__).resolve().parents[1]
TYPO = ROOT / 'mirror/cases/typography_20260926'
HELD = ROOT / 'mirror/cases/typography_strength_20260929/characterization/w4/retest/digital_retest/test_seen_standard'
THRESHOLDS = (0.02, 0.05, 0.1)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def diagnose(scores, rows, vocab):
    idx = {c: i for i, c in enumerate(vocab)}; n = len(rows)
    y = np.array([idx[r['label']] for r in rows]); w = np.array([[idx[s] for s in r['words'][1:]] for r in rows])
    m = scores[np.arange(n)[:, None, None], np.arange(5)[None, :, None], y[:, None, None]] - \
        scores[np.arange(n)[:, None, None], np.arange(5)[None, :, None], w[:, None, :]]
    unedited, blank = m[:, 0], m[:, 1]
    word = np.stack([m[:, 3, 0], m[:, 4, 1]], 1)
    D = (word - blank).astype(np.float64)
    ok_word = word > 0
    latent = {}
    for t in THRESHOLDS:
        big = np.abs(D) > t
        latent[str(t)] = dict(count=int((ok_word & big).sum()),
                              toward_distractor_D_negative=int((ok_word & big & (D < 0)).sum()),
                              toward_object_D_positive=int((ok_word & big & (D > 0)).sum()),
                              pct_of_word_correct=100 * float((ok_word & big).sum() / max(1, ok_word.sum())),
                              pct_of_all_decisions=100 * float((ok_word & big).mean()))
    return dict(
        sources=n, decisions=int(word.size),
        unedited_accuracy_pct=100 * float((unedited > 0).mean()),
        blank_note_accuracy_pct=100 * float((blank > 0).mean()),
        misleading_note_accuracy_pct=100 * float(ok_word.mean()),
        blank_correct=int((blank > 0).sum()), misleading_correct=int(ok_word.sum()),
        reversed_by_word_of_blank_correct=int(((blank > 0) & (word <= 0)).sum()),
        reversed_by_word_and_unedited_also_correct=int(((unedited > 0) & (blank > 0) & (word <= 0)).sum()),
        unedited_correct=int((unedited > 0).sum()),
        reversed_by_blank_note_of_unedited_correct=int(((unedited > 0) & (blank <= 0)).sum()),
        mean_D_blank_vs_word_signed=float(D.mean()), mean_abs_D_blank_vs_word=float(np.abs(D).mean()),
        median_D_blank_vs_word=float(np.median(D)),
        decisions_D_negative=int((D < 0).sum()), max_D=float(D.max()), min_abs_D=float(np.abs(D).min()),
        mean_blank_margin=float(blank.astype(np.float64).mean()),
        mean_unedited_margin=float(unedited.astype(np.float64).mean()),
        mean_misleading_margin=float(word.astype(np.float64).mean()),
        identity_check_word_equals_blank_plus_D=bool(np.allclose(word, blank + D, atol=1e-6)),
        latent_faults=dict(definition='decision correct under the misleading note (margin > 0) with |D_blank-vs-word| > threshold (raw cosine)',
                           word_correct_decisions=int(ok_word.sum()), by_threshold=latent))


def main():
    cfg = json.loads((TYPO / 'protocol.json').read_text()); vocab = cfg['seen_classes'] + cfg['heldout_classes']
    rows = json.loads((HELD / 'rows.json').read_text()); z = np.load(HELD / 'frozen.npz')
    held = diagnose(z['scores'], rows, vocab)
    held['saved_field_check'] = dict(
        blank_matches=bool(np.allclose(z['blank'], diagnose_fields(z['scores'], rows, vocab)[0], atol=0)),
        misleading_matches=bool(np.allclose(z['conflict'], diagnose_fields(z['scores'], rows, vocab)[1], atol=0)),
        saved_bug_count=int(z['bug'].sum()))
    drows = json.loads((TYPO / 'development.json').read_text()); dz = np.load(TYPO / 'frozen_development_scores.npz')
    val = diagnose(dz['scores'], drows, vocab)
    val['paper_v8'] = dict(blank=99.41, misleading=22.66, reversed_by_word=786, reversed_by_blank=4, mean_D=-0.1017, mean_blank_margin=0.0786)
    out = dict(heldout_1024=held, validation_512=val,
               inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in [HELD / 'rows.json', HELD / 'frozen.npz', TYPO / 'protocol.json',
                                                                          TYPO / 'development.json', TYPO / 'frozen_development_scores.npz']})
    (HERE / 'results' / 'typography_heldout_diagnosis.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=1))


def diagnose_fields(scores, rows, vocab):
    idx = {c: i for i, c in enumerate(vocab)}; n = len(rows)
    y = np.array([idx[r['label']] for r in rows]); w = np.array([[idx[s] for s in r['words'][1:]] for r in rows])
    m = scores[np.arange(n)[:, None, None], np.arange(5)[None, :, None], y[:, None, None]] - \
        scores[np.arange(n)[:, None, None], np.arange(5)[None, :, None], w[:, None, :]]
    return m[:, 1], np.stack([m[:, 3, 0], m[:, 4, 1]], 1)


if __name__ == '__main__':
    main()
