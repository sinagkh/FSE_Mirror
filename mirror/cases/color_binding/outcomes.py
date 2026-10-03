"""Behavioral outcomes kept separate from interaction-requirement verdicts.

All functions consume caller-supplied scores/labels. No examples, thresholds,
checkpoints or slices are selected here. A tie is not a correct decision.
"""
import numpy as np


def caption_decisions(scores, correct_captions):
    x = np.asarray(scores, float)
    labels = np.asarray(correct_captions, int)
    if x.ndim != 3 or labels.shape != (x.shape[1],) or x.shape[2] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite [anchor,image,caption] scores and per-image labels")
    if (labels < 0).any() or (labels >= x.shape[2]).any():
        raise ValueError("Invalid correct caption index")
    correct = np.take_along_axis(x, np.broadcast_to(labels[None, :, None], (*x.shape[:2], 1)), axis=2)
    mask = np.arange(x.shape[2])[None, :] != labels[:, None]
    pair_margins = (correct-x)[:, mask].reshape(len(x), x.shape[1], x.shape[2]-1)
    minimum = pair_margins.min(axis=-1)
    return {"pair_margins": pair_margins, "minimum_margins": minimum,
            "correct": minimum > 0, "any_failure": (minimum <= 0).any(axis=1),
            "decision_failure_rate": (minimum <= 0).mean(axis=1)}


def first_order_background(scores):
    """C0/red,C0/blue,C1/red,C1/blue; captions red,blue, no magnitude threshold."""
    x = np.asarray(scores, float)
    if x.ndim != 3 or x.shape[1:] != (4, 2) or not np.isfinite(x).all():
        raise ValueError("Expected [anchor,4,2] background lattice")
    margins = x[:, :, 0] - x[:, :, 1]
    # Tie means this test has not established the expected relation.
    inv = (margins[:, :2] * margins[:, 2:] > 0).all(axis=1)
    direction = (margins[:, [0, 2]] > 0).all(axis=1) & (margins[:, [1, 3]] < 0).all(axis=1)
    return {"invariance_passed": inv, "direction_passed": direction}


def transitions(before_failure, after_failure):
    before, after = np.asarray(before_failure), np.asarray(after_failure)
    if before.dtype != bool or after.dtype != bool or before.shape != after.shape or before.ndim != 1:
        raise ValueError("Paired one-dimensional boolean failure arrays required")
    n = len(before); failed = int(before.sum()); correct = n-failed
    repaired = int((before & ~after).sum()); broken = int((~before & after).sum())
    return {"n": n, "initial_failures": failed, "initial_correct": correct,
            "repaired": repaired, "broken": broken,
            "persistent_failures": int((before & after).sum()),
            "preserved_correct": int((~before & ~after).sum()),
            "repair_rate": repaired/failed if failed else None,
            "break_rate": broken/correct if correct else None,
            "failure_rate_change": float(after.mean()-before.mean()) if n else None}


def paired_seed_source_interval(before, after, source_ids, n_bootstrap=2000, seed=20260922):
    """Paired mean(after-before), sampling seeds then whole source clusters.

    [seed,row] arrays must be aligned by the caller's immutable manifest. A
    frozen one-row baseline may be broadcast over matched seed rows. Repeated
    rows of a source stay together. This estimates source/seed variability;
    three optimization seeds are not three independent model families.
    """
    a, b = np.asarray(before, float), np.asarray(after, float)
    ids = np.asarray(source_ids)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or len(ids) != b.shape[1] or ids.ndim != 1:
        raise ValueError("Expected aligned [seed,row] arrays and row source IDs")
    if a.shape[0] not in (1, b.shape[0]) or min(b.shape) < 1 or n_bootstrap < 2:
        raise ValueError("Invalid seed alignment or bootstrap budget")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite input")
    unique, inverse = np.unique(ids, return_inverse=True)
    if len(unique) < 2: raise ValueError("Need at least two independent sources")
    delta = b-a; rng = np.random.default_rng(seed); draws = np.empty(n_bootstrap)
    for k in range(n_bootstrap):
        seeds = rng.integers(0, len(b), len(b))
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[inverse]
        draws[k] = np.average(delta[seeds].mean(axis=0), weights=weights)
    per_seed = delta.mean(axis=1)
    return {"effect": float(delta.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "per_seed": per_seed.tolist(),
            "sample_sd": float(per_seed.std(ddof=1)) if len(per_seed)>1 else None,
            "n_seeds": len(b), "n_rows": len(ids), "n_sources": len(unique),
            "bootstrap_replicates": n_bootstrap, "bootstrap_seed": seed,
            "estimand": "Row-weighted mean after minus before, paired across arms; hierarchical seeds and source clusters"}


def retrieval_query(scores, row_ids, relevant_ids, incompatible_ids, k_values=(1, 5)):
    """Restricted-gallery label-based retrieval; unknowns are NOT negatives.

    Recall@k here is a query hit (any known relevant image), not the fraction of
    all relevant images retrieved. Ties use lexicographic row IDs fixed before
    scoring; optimistic and pessimistic top-1 tie bounds are retained.
    """
    scores = np.asarray(scores, float); ids = [str(x) for x in row_ids]
    pos, neg = set(map(str, relevant_ids)), set(map(str, incompatible_ids))
    if scores.shape != (len(ids),) or not np.isfinite(scores).all() or len(set(ids)) != len(ids):
        raise ValueError("Unique row IDs and finite scores required")
    if not pos or not neg or pos & neg or not (pos | neg) <= set(ids):
        raise ValueError("Disjoint, nonempty known positive/negative sets required")
    if any(type(k) is not int or k < 1 for k in k_values): raise ValueError("Invalid retrieval cutoff")
    known = [j for j, rid in enumerate(ids) if rid in pos | neg]
    ordered = sorted(known, key=lambda j: (-scores[j], ids[j]))
    top = [ids[j] for j in ordered if scores[j] == scores[ordered[0]]]
    return {"n_known_gallery": len(known), "n_positive": len(pos), "n_incompatible": len(neg),
            "n_unknown_omitted": len(ids)-len(known), "top_id": ids[ordered[0]],
            "top_is_incompatible": ids[ordered[0]] in neg,
            "hits": {str(k): any(ids[j] in pos for j in ordered[:k]) for k in k_values},
            "top1_tie_count": len(top), "top1_pessimistic": all(i in pos for i in top),
            "top1_optimistic": any(i in pos for i in top),
            "ranked_known_ids": [ids[j] for j in ordered]}
