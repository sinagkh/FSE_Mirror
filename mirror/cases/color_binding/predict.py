"""Source-grouped out-of-fold risk prediction, independent of any test-bank loader."""
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score; from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler; from sklearn.preprocessing import OneHotEncoder


def fold_ids(sources,n_folds=5,seed=20260922):
    """All models, edits and rows of one source go to the same outcome fold."""
    sources=np.asarray(sources)
    unique=np.unique(sources);rng=np.random.default_rng(seed);rng.shuffle(unique)
    mapping={s:i%n_folds for i,s in enumerate(unique)}
    return np.array([mapping[s] for s in sources])


def cross_fitted(frame,numeric,categorical,outcome="failure",source="source_id",n_folds=5):
    if source in numeric+categorical or outcome in numeric+categorical:
        raise ValueError("Source identity/outcomes cannot be predictors")
    if not set(frame[outcome]).issubset({0,1,False,True}):raise ValueError("Binary outcome required")
    folds=fold_ids(frame[source].to_numpy(),n_folds)
    transformer=ColumnTransformer([
        ("numeric",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),numeric),
        ("categorical",OneHotEncoder(handle_unknown="ignore"),categorical)])
    pipeline=make_pipeline(transformer,LogisticRegression(C=1.,max_iter=2000,solver="lbfgs",random_state=20260922))
    predictions=np.full(len(frame),np.nan);audit=[]
    for k in range(n_folds):
        train=folds!=k;test=~train
        if not test.any() or len(np.unique(frame.loc[train,outcome]))<2:
            raise ValueError("Insufficient independent source/outcome support; do not change folds after scores")
        train_sources=set(frame.loc[train,source]);test_sources=set(frame.loc[test,source])
        if train_sources & test_sources:raise AssertionError("Source leakage")
        model=clone(pipeline);model.fit(frame.loc[train],frame.loc[train,outcome])
        predictions[test]=model.predict_proba(frame.loc[test])[:,1]
        audit.append(dict(fold=k,n_train_sources=len(train_sources),n_test_sources=len(test_sources),source_overlap=0))
    if not np.isfinite(predictions).all():raise ValueError("Nonfinite predictions")
    y=frame[outcome].to_numpy(int)
    return dict(predictions=predictions,folds=folds,fold_audit=audit,
                auc=float(roc_auc_score(y,predictions)),brier=float(brier_score_loss(y,predictions)))


def paired_auc_interval(y,base,extended,sources,n=2000,seed=20260922):
    """Paired cluster bootstrap of fixed out-of-fold predictions (conditional CI)."""
    y=np.asarray(y,int);base=np.asarray(base);extended=np.asarray(extended)
    unique,inverse=np.unique(sources,return_inverse=True);rng=np.random.default_rng(seed);draws=[];invalid=0
    if len(np.unique(y))<2:raise ValueError("AUC undefined for a single outcome class")
    for _ in range(n):
        counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique));w=counts[inverse]
        if len(np.unique(y[w>0]))<2:invalid+=1;continue
        draws.append(roc_auc_score(y,extended,sample_weight=w)-roc_auc_score(y,base,sample_weight=w))
    if not draws:raise ValueError("No valid bootstrap replicate")
    return dict(delta_auc=float(roc_auc_score(y,extended)-roc_auc_score(y,base)),
                ci95=np.quantile(draws,[.025,.975]).tolist(),valid_replicates=len(draws),invalid_replicates=invalid,
                scope="Source-cluster sampling conditional on fitted cross-fold models; does not include model-refitting uncertainty")
