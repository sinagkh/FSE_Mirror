"""Paired source/seed bootstrap with no independent-row inflation."""
import numpy as np

def paired_interval(differences, clusters=None, draws=5000, seed=5342):
    x=np.asarray(differences,dtype=float)
    if x.ndim==1:x=x[None,:]
    if clusters is None:clusters=np.arange(x.shape[1])
    clusters=np.asarray(clusters);unique=np.unique(clusters)
    sums=np.stack([x[:,clusters==g].sum(1) for g in unique],1)
    counts=np.array([(clusters==g).sum() for g in unique])
    rng=np.random.default_rng(seed);result=[]
    for _ in range(draws):
        ss=rng.integers(len(x),size=len(x));cc=rng.integers(len(unique),size=len(unique))
        result.append(sums[ss][:,cc].sum()/len(ss)/counts[cc].sum())
    return dict(mean=float(x.mean()),ci95=np.quantile(result,[.025,.975]).tolist(),
                seeds=len(x),clusters=len(unique),paired=True,draws=draws)
