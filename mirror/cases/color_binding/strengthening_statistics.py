"""Final crossed seed/source intervals, optionally preserving source strata."""
import numpy as np


def intervals(x,cluster,strata=None,draws=10000,seed=20260925):
    x=np.asarray(x,float);cluster=np.asarray(cluster,int);ns,n,m=x.shape
    assert len(cluster)==n and np.isfinite(x).all()
    nc=int(cluster.max())+1;counts=np.bincount(cluster,minlength=nc)
    assert (counts>0).all()
    sums=np.stack([[np.bincount(cluster,weights=row[:,j],minlength=nc) for j in range(m)] for row in x])
    if strata is None:groups=[np.arange(nc)]
    else:
        strata=np.asarray(strata);labels=[]
        for c in range(nc):
            u=np.unique(strata[cluster==c]);assert len(u)==1, 'Shared-source cluster crosses strata'
            labels.append(u[0])
        labels=np.asarray(labels);groups=[np.flatnonzero(labels==s) for s in np.unique(labels)]
    rng=np.random.default_rng(seed);aa=[];bb=[]
    for start in range(0,draws,100):
        size=min(100,draws-start);w=np.zeros((size,nc))
        for ix in groups:w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=size)
        sw=rng.multinomial(ns,np.full(ns,1/ns),size=size)/ns
        per=np.einsum('bc,smc->bsm',w,sums,optimize=True)/(w@counts)[:,None,None]
        aa.extend(per.mean(1));bb.extend((per*sw[:,:,None]).sum(1))
    means=x.mean(1);aa=np.asarray(aa);bb=np.asarray(bb)
    return [dict(mean=float(means[:,j].mean()),per_seed=means[:,j].tolist(),sample_sd=float(means[:,j].std(ddof=1)) if ns>1 else None,
        ci95_source=np.quantile(aa[:,j],[.025,.975]).tolist(),ci95_seed_source=np.quantile(bb[:,j],[.025,.975]).tolist(),
        n_items=n,n_source_clusters=nc,bootstrap_draws=draws,source_strata=len(groups)) for j in range(m)]
