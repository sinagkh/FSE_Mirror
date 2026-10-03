"""Read the exact audit and argmax evidence in compact NumPy records."""
import numpy as np


def recover_prediction(true_score, best_other_score, true_class, best_other_class):
    """Recover NumPy argmax, including first-index tie handling."""
    true_wins = (true_score > best_other_score) | (
        (true_score == best_other_score) & (true_class < best_other_class))
    return np.where(true_wins, true_class, best_other_class)


def typography_views(record, rows, vocabulary):
    """Return object/distractor margins and 70-class correctness by image state."""
    lookup = {name: i for i, name in enumerate(vocabulary)}
    indices = np.array([[lookup[row['label']], *[lookup[w] for w in row['words'][1:]]]
                        for row in rows])
    n = len(rows)
    if 'audit_scores' in record:
        assert np.array_equal(indices, record['audit_class_indices'])
        scores = record['audit_scores'].astype(float)
        assert scores.shape == (n, 5, 3)
        margin = scores[..., :1] - scores[..., 1:]
        predicted = recover_prediction(scores[..., 0], record['recognition_best_other_score'],
                                       indices[:, :1], record['recognition_best_other_class'])
    else:
        scores = record['scores'].astype(float)
        margin = scores[np.arange(n), :, indices[:, 0]][..., None] - np.take_along_axis(
            scores, indices[:, None, 1:], axis=-1)
        predicted = scores.argmax(axis=-1)
    correct = predicted == indices[:, :1]
    assert np.array_equal(correct, record['top1'])
    return margin, correct


def gallery_views(record):
    """Return gallery and object-pair correctness for every method/layout."""
    prefixes = sorted(key.removesuffix('/true_score') for key in record if key.endswith('/true_score'))
    assert prefixes
    result = {}
    for prefix in prefixes:
        true = record[prefix + '/true_class']
        predicted = recover_prediction(record[prefix + '/true_score'], record[prefix + '/best_other_score'],
                                       true, record[prefix + '/best_other_class'])
        result[prefix] = dict(gallery_top1=predicted == true,
                              object_pair_top1=predicted // 4 == true // 4)
    return result
