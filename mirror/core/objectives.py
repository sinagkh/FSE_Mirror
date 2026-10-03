"""Selected repair objectives, independent of data acquisition and run layout.

Inputs are differentiable cosine scores or reduced loss components. These
functions do not load models, select checkpoints, or perform optimization.
"""
import torch
from torch.nn import functional as F


def _typography_margins(scores, labels, distractors):
    correct = scores.gather(2, labels[:, None, None].expand(-1, 5, 1))
    alternatives = scores.gather(2, distractors[:, None, :].expand(-1, 5, -1))
    return correct - alternatives


def typography_objective(scores, frozen_scores, labels, distractors,
                         interaction_coefficient, native_logit_scale=100.0):
    """Five states: clean, blank, congruent, conflict 1, conflict 2.

    Set the interaction coefficient to zero for the matched Ranking arm.
    Labels are source class indices; distractors have shape (sources, 2).
    """
    ce = F.cross_entropy((100 * scores).flatten(0, 1), labels.repeat_interleave(5))
    margins = _typography_margins(scores, labels, distractors)
    reference = _typography_margins(frozen_scores, labels, distractors)
    identity = F.kl_div(
        F.log_softmax(native_logit_scale * scores[:, :2].flatten(0, 1), dim=-1),
        F.softmax(native_logit_scale * frozen_scores[:, :2].flatten(0, 1), dim=-1),
        reduction="batchmean")
    agreement = F.relu(.045 - margins).square().mean()
    retention = F.relu(reference[:, :2].mean((0, 2)) -
                       margins[:, :2].mean((0, 2))).square().mean() / .1**2
    shared = ce + 3 * identity + 1.2 * agreement + retention
    nuisance = torch.stack([margins[:, i] - margins[:, j]
                            for i in range(1, 5) for j in range(i + 1, 5)],
                           1).square().mean() / .1**2
    terms = dict(ce=ce, identity=identity, agreement=agreement,
                 retention=retention, shared=shared, nuisance=nuisance)
    return shared + interaction_coefficient * nuisance, terms


def backdoor_objective(scores, frozen_scores, labels, interaction_coefficient):
    """Four training states: clean, two triggers, and equal-support control.

    Set the interaction coefficient to zero for the matched Ranking arm.
    """
    ce = F.cross_entropy((100 * scores).flatten(0, 1), labels.repeat_interleave(4))
    kl = F.kl_div(F.log_softmax(100 * scores[:, 0], -1),
                  F.softmax(100 * frozen_scores[:, 0], -1), reduction="batchmean")
    margins = scores.gather(-1, labels[:, None, None].expand(-1, 4, 1)) - scores
    reference = frozen_scores.gather(-1, labels[:, None, None].expand(-1, 4, 1)) - frozen_scores
    retain = F.relu(reference[:, 0] - margins[:, 0]).square().mean() / .1**2
    interaction = (margins[:, 1:] - margins[:, :1]).square().mean() / .1**2
    shared = ce + 3 * kl + retain
    terms = dict(ce=ce, kl=kl, retain=retain, interaction=interaction, shared=shared)
    return shared + interaction_coefficient * interaction, terms


def color_binding_objective(terms, cross_entropy, weights, color_weight,
                            method="IS"):
    """Combine the selected color-binding components after their reduction.

    Components must use clause mean, paired-layout maximum, then block mean,
    as specified in Supplement S1. Target/retention definitions are in
    routing_relative_pilot.components and routing_tint_balance_train.parts.
    Zero selected target weights to evaluate the reported term deletions.
    """
    scaled = {key: weights[key] * terms[key] for key in
              ("binding", "cross", "response", "preference", "endpoint",
               "caption_guard", "binding_keep", "response_keep", "object_guard")}
    guard = 4 * (sum(scaled[key] for key in
                     ("caption_guard", "binding_keep", "response_keep", "object_guard"))
                 + terms["natural"] + terms["drift"])
    if method == "Ranking":
        value = cross_entropy + guard
    elif method == "IS":
        # Preserve the original arithmetic ordering for parity checks.
        interaction = sum(scaled[key] for key in ("binding", "cross", "response"))
        value = guard + interaction + 4 * scaled["preference"] + 3 * scaled["cross"]
    else:
        raise ValueError("method must be Ranking or IS")
    return value + 4 * color_weight * terms["color_floor"]
