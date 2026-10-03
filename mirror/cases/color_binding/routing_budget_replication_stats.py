"""Paired crossed seed/anchor bootstrap; no pseudo-replicated corners."""
import numpy as np
from mirror.cases.color_binding.routing_same_class_evaluate import bootstrap_weights


def seed_weights(n_seeds,n_draws=2000,seed=20260923):
    return np.random.default_rng(seed).multinomial(n_seeds,np.full(n_seeds,1/n_seeds),size=n_draws)/n_seeds


def interval(x,weights,seed_draws):
    """x: matched seeds x independent anchors x metrics. Item draw is shared."""
    x=np.asarray(x,float);point,draws=weights
    assert x.ndim==3 and len(seed_draws)==len(draws) and seed_draws.shape[1]==x.shape[0]
    take=point>0;x=x[:,take];point=point[take];draws=draws[:,take]
    per_seed=np.einsum('n,snm->sm',point,x)
    each=np.stack([draws@v for v in x],axis=1)
    item=each.mean(1);hier=np.einsum('bs,bsm->bm',seed_draws,each)
    return dict(mean=per_seed.mean(0),sample_sd=per_seed.std(0,ddof=1) if len(x)>1 else np.full(x.shape[-1],np.nan),
        per_seed=per_seed,ci95_item=np.quantile(item,[.025,.975],axis=0).T,
        ci95_seed_item=np.quantile(hier,[.025,.975],axis=0).T)


def cohort_weights(groups,n=2000):
    groups=np.asarray(groups);out=bootstrap_weights(groups,n=n)
    for group in sorted(set(groups)):
        take=np.flatnonzero(groups==group);p,d=bootstrap_weights(groups[take],n=n)['micro']
        point=np.zeros(len(groups));point[take]=p
        draws=np.zeros((n,len(groups)));draws[:,take]=d;out[group]=(point,draws)
    return out


def assess_gate(current,frozen,reference,criteria):
    """Inputs are aggregate routing-only diagnostics; natural benchmarks absent."""
    checks={
        'cross_reduction':current['all_cross_abs']<=criteria['cross_ratio_max']*frozen['all_cross_abs'],
        'binding_above_frozen':current['all_binding']>frozen['all_binding'],
        'binding_majority':current['fraction_binding_better']>criteria['binding_fraction_better_min'],
        'cross_majority':current['fraction_cross_better']>criteria['cross_fraction_better_min'],
        'exchange_retained':current['exchange_accuracy']>=reference['exchange_accuracy']-criteria['exchange_drop_max'],
        'word1_retained':current['word1_accuracy']>=criteria['word_accuracy_min'],
        'word2_retained':current['word2_accuracy']>=criteria['word_accuracy_min'],
        'object_retained':current['object_guard']>=frozen['object_guard']-criteria['object_accuracy_drop_max'],
    }
    return {k:bool(v) for k,v in checks.items()}
