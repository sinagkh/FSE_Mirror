"""Preference-aware specification and score-only losses; no trainer."""
import copy
import numpy as np
import torch
from mirror.cases.color_binding.routing_diagnostic import full_spec
from mirror.core.specifications import compile_requirement; from mirror.core.specifications import load_requirement


def preference_spec(calibration):
    spec, contexts, _ = full_spec(calibration)
    spec = copy.deepcopy(spec)
    spec['id'] = 'routing/context-and-preference-v3'
    spec['contrasts']['preference'] = [
        dict(image=i, caption=c, weight=w)
        for i in ('rb', 'br') for c, w in [('rb', .5), ('br', -.5)]]
    spec['thresholds']['fractions']['beta'] = calibration['tau_fraction']
    spec['clauses'].append(dict(name='preference_bound', kind='magnitude', contrast='preference', bound='beta'))
    spec['operational_measurement'] += (
        ' Also bounds the mean first-order exchange-caption preference; '
        'K-tau-beta>0 suffices for two strict exchange decisions, not every caption decision.')
    return spec, contexts


def exchange_weights():
    e = np.zeros((4, 4)); b = np.zeros((4, 4))
    e[1, 1] = e[2, 2] = .5; e[1, 2] = e[2, 1] = -.5
    b[1, 1] = b[2, 1] = .5; b[1, 2] = b[2, 2] = -.5
    return e, b


def span_check(rows, target):
    a = np.asarray(rows).reshape(len(rows), -1)
    c = np.linalg.lstsq(a.T, np.asarray(target).reshape(-1), rcond=None)[0]
    residual = np.asarray(target).reshape(-1) - c @ a
    return dict(rank=int(np.linalg.matrix_rank(a)), residual_norm=float(np.linalg.norm(residual)),
                identifiable=bool(np.linalg.norm(residual) < 1e-10), coefficients=c.tolist())


def adequacy(calibration, legacy_path):
    old = compile_requirement(load_requirement(legacy_path), calibration_unit=1., base_model_id='symbolic', calibration_bank_id='symbolic')
    spec, contexts = preference_spec(calibration)
    new = compile_requirement(spec, calibration_unit=1., base_model_id='symbolic', calibration_bank_id='symbolic')
    full = [new.weights[r['contrast']] for r in contexts]
    e, b = exchange_weights()
    result = {}
    for label, rows in [('legacy', list(old.weights.values())), ('complete', full),
                        ('complete_plus_preference', full+[b]), ('two_endpoint_margins', [e+b, e-b])]:
        result[label] = {name: span_check(rows, target) for name, target in [('response', e), ('preference', b), ('margin_rb', e+b), ('margin_br', e-b)]}
    result['guaranteed_margin_in_base_units'] = calibration['kappa'] - 2*calibration['tau_fraction']
    return result


def pilot_losses(scores, compiled, contexts):
    """Differentiable scalar components only. No updates, RNG draws or training."""
    if scores.shape[-2:] != (4, 4) or not bool(torch.isfinite(scores).all()):
        raise ValueError('Finite 4x4 lattices required')
    values = {k: (scores * torch.as_tensor(w, dtype=scores.dtype, device=scores.device)).sum((-2, -1))/compiled.unit
              for k, w in compiled.weights.items()}
    f = compiled.spec['thresholds']['fractions']
    bindings = torch.stack([values[r['contrast']] for r in contexts if r['kind']=='binding'], -1)
    cross = torch.stack([values[r['contrast']] for r in contexts if r['kind']=='unwanted'], -1)
    refs = torch.stack([values[r['relative_reference']] for r in contexts if r['kind']=='unwanted'], -1)
    li = (torch.relu(f['K']-bindings).square().mean()
          + torch.relu(cross.abs()-f['tau']).square().mean()
          + torch.relu(cross.abs()-f['rho']*refs.clamp_min(0)).square().mean())
    lp = torch.relu(values['preference'].abs()-f['beta']).square().mean()
    margins = torch.stack([scores[...,1,1]-scores[...,1,2], scores[...,2,2]-scores[...,2,1]], -1)/compiled.unit
    le = torch.relu(f['K']-f['tau']-f['beta']-margins).square().mean()
    return dict(interaction=li, preference=lp, endpoint=le)
