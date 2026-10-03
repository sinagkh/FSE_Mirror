"""Freeze future pilot configuration and analyze frozen-regime outcomes; never train."""
from pathlib import Path
import numpy as np
import pandas as pd
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import log
from mirror.cases.color_binding.behavioral_pilot import lines; from mirror.cases.color_binding.behavioral_pilot import CAL
from mirror.cases.color_binding.routing_adapter_reaudit_v2 import OUT as REAUDIT
from mirror.cases.color_binding.routing_diagnostic import REGIMES

OUT=ROOT/'clip/interbind_diagnosis_guided_prep_20260923'
ARMS=('F','R','G','I','P','IP','E')
CONTRASTS=(('I','G'),('P','G'),('IP','I'),('IP','E'),('IP','R'),('IP','F'))
METRICS=('exchange_accuracy','response','absolute_preference','surplus','caption_accuracy','word1_accuracy','word2_accuracy')


def attach_frozen_regimes(frame, assignments):
    if 'regime' in frame:raise ValueError('Outcome-supplied regimes are forbidden')
    if any(r['diagnosis_view']!='canvas' for r in assignments):raise ValueError('Regimes must come from frozen direct audit')
    mapping={r['anchor_id']:r['regime'] for r in assignments}
    if len(mapping)!=len(assignments) or set(mapping.values())-set(REGIMES):raise ValueError('Invalid fixed strata')
    if set(frame.anchor_id)!=set(mapping):raise ValueError('All frozen sources required; no outcome filtering')
    if frame.duplicated(['arm','seed','view','anchor_id']).any():raise ValueError('Duplicate outcome row')
    for _,g in frame.groupby(['arm','seed','view']):
        if set(g.anchor_id)!=set(mapping):raise ValueError('Incomplete arm/view cohort')
    out=frame.copy();out['frozen_regime']=out.anchor_id.map(mapping)
    return out


def regime_contrasts(frame, assignments, n=2000):
    """Paired arm effects and unpaired-stratum effect modification for a one-seed pilot."""
    f=attach_frozen_regimes(frame,assignments)
    if f.seed.nunique()!=1 or set(f.arm)!=set(ARMS):raise ValueError('One-seed, complete seven-arm pilot required')
    if not np.isfinite(f[list(METRICS)].to_numpy()).all():raise ValueError('Nonfinite outcome')
    rng=np.random.default_rng(20260923);results=[];effects={}
    for view,g in f.groupby('view'):
        for regime in REGIMES:
            h=g[g.frozen_regime==regime];ids=sorted(h.anchor_id.unique())
            weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=n)/len(ids) if ids else None
            for a,b in CONTRASTS:
                key=(view,regime,a,b)
                if not ids:
                    results.append(dict(view=view,regime=regime,contrast=f'{a}-{b}',n_anchors=0,mean=None,ci95=None));continue
                x=h[h.arm==a].set_index('anchor_id').loc[ids,list(METRICS)].to_numpy()-h[h.arm==b].set_index('anchor_id').loc[ids,list(METRICS)].to_numpy()
                draws=weights@x;effects[key]=(x.mean(0),draws)
                results.append(dict(view=view,regime=regime,contrast=f'{a}-{b}',n_anchors=len(ids),metrics=METRICS,
                    mean=x.mean(0).tolist(),ci95=np.quantile(draws,[.025,.975],axis=0).T.tolist(),paired=True))
        for a,b,first,second in [('I','G',REGIMES[0],REGIMES[1]),('P','G',REGIMES[1],REGIMES[0])]:
            k1,k2=(view,first,a,b),(view,second,a,b)
            if k1 not in effects or k2 not in effects:
                results.append(dict(view=view,contrast=f'({a}-{b})_{first}-({a}-{b})_{second}',mean=None,ci95=None));continue
            m1,d1=effects[k1];m2,d2=effects[k2]
            results.append(dict(view=view,contrast=f'({a}-{b})_{first}-({a}-{b})_{second}',metrics=METRICS,
                mean=(m1-m2).tolist(),ci95=np.quantile(d1-d2,[.025,.975],axis=0).T.tolist(),
                paired_within_arm=True,paired_between_regimes=False,
                interpretation='Difference of mean paired treatment effects across disjoint frozen-source strata'))
    return results


def freeze():
    log(OUT,'start');assignment=REAUDIT/'frozen_direct_regimes.jsonl';rows=lines(assignment)
    bank=ROOT/'data/color_binding/train/protocol.json'
    cal=read(CAL);beta=cal['tau_fraction'];margin=cal['kappa']-2*beta
    assert margin>0
    paths=[Path(__file__),ROOT/'mirror/cases/color_binding/routing_adequacy.py',ROOT/'mirror/core/repair.py',
        ROOT/'FSE_VLM/plan/19_diagnosis_guided_repair_and_reaudit.md',CAL,assignment,bank,
        REAUDIT/'routing-context-preference-v3.yaml']
    dump(OUT/'pilot_design.json',dict(inputs={str(p):sha(p) for p in paths},
        model='openclip_laion_l14',seed=42,arms={
        'F':[], 'R':['historical_CE','0.2*object_embedding_anchor'],
        'G':['shared_guards'], 'I':['shared_guards','L_I'], 'P':['shared_guards','L_P'],
        'IP':['shared_guards','L_I','L_P'], 'E':['shared_guards','L_E']},
        loss_group_weights={'binding':1,'absolute_cross':1,'relative_cross':1,'preference':1,'exchange_endpoint':1},
        shared_guard_weights={k:1 for k in ('caption_margin','binding_retention','object_caption','natural_consistency','embedding_drift')},
        natural_tolerance=0,drift_tolerance=0,thresholds={'K':cal['kappa'],'tau':beta,'rho':cal['rho'],'beta':beta,'exchange_margin':margin},
        schedule=read(bank)['schedule'],adapter_dimension=768,seed_escalation='author decision after full one-seed report',
        training_scale_rule='Existing derive_scales on frozen TRAINING scores only; store realized values before optimizer initialization',
        frozen_regime_counts={k:sum(r['regime']==k for r in rows) for k in REGIMES},
        outcome_views=['swapped_canvas','in_situ'],metrics=METRICS,contrasts=CONTRASTS,
        bootstrap=2000,statistics_seed=20260923,
        status='configuration and analysis frozen; actual training cache binding/scales and optimizer execution pending',
        gpu=False,training=False,reserve_scored=False))
    log(OUT,'complete',design_sha256=sha(OUT/'pilot_design.json'))


if __name__=='__main__':freeze()
