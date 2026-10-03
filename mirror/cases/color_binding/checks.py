"""Final read-only verification: matched runs, common pixels, main-study replay."""
import argparse
from pathlib import Path
import numpy as np
from mirror.cases.color_binding import evaluate as ev
from mirror.core.io import dump; from mirror.core.io import log; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import verify_files

def replay():
    checks=[]
    for seed,reference in zip((42,43,44),[ev.base.OLD,*[ev.base.OUT/f'seed{s}' for s in (43,44)]]):
        for family in ('primary','transfer','natural'):
            new=ev.OUT/'openclip_laion_l14/evaluation'/f'seed{seed}'/family
            old=reference/'analysis'/family
            verify_files(read(new/'complete.json')['files']);verify_files(read(old/'complete.json')['files'])
            for entry in read(old/'score_index.json'):
                test=entry['test'];aa=ev.data.lines(old/(test+'_per_example.jsonl'));bb=ev.data.lines(new/(test+'_per_example.jsonl'))
                for method in ('Frozen','Ranking','IS'):
                    first=[r for r in aa if r['name']==method];second=[r for r in bb if r['name']==method]
                    assert len(first)==len(second)
                    assert [(r['anchor_id'],r['source_ids']) for r in first]==[(r['anchor_id'],r['source_ids']) for r in second]
                    keys=[k for k in first[0] if k not in ev.stats.METADATA]
                    x=np.array([[r[k] for k in keys] for r in first]);y=np.array([[r[k] for k in keys] for r in second])
                    continuous=[k in ('binding','cross','response','preference','surplus') or k.startswith(('contrast/','absolute/')) for k in keys]
                    np.testing.assert_array_equal(x[:,np.logical_not(continuous)],y[:,np.logical_not(continuous)])
                    # Duplicate text strings were cached in separate encoder batches.
                    # Float32 cosine noise is amplified by the fixed ~.048 unit.
                    np.testing.assert_allclose(x,y,rtol=0,atol=5e-6,err_msg=str((seed,family,test,method)))
                    checks.append(dict(seed=seed,family=family,test=test,method=method,max_error=float(abs(x-y).max()),all_discrete_behavior_identical=True))
    dump(ev.OUT/'main_replay_verification.json',dict(checks=checks,all_existing_main75_values_reproduced=True,
        behavior_tolerance=0,continuous_tolerance=5e-6,units='frozen calibrated units',
        duplicate_text_cache_batch_roundoff='No changed decisions; existing main results remain authoritative'))

def complete():
    ev.data.verify();checks=[];models=[]
    for study in ev.STUDIES:
        root=ev.OUT/study
        for name in ('results','diagnostics'):
            verify_files(read(root/name/'complete.json')['files'])
        verify_files(read(root/'evaluation_complete.json')['inputs'])
        for seed in (42,43,44):
            rows=ev.registry(study,seed)
            trained=[r for r in rows if r['checkpoint'] and not r.get('external_fixed')]
            for r in trained:
                ck=ev.checkpoint(r['checkpoint']);assert ck['tint']==75 and ck['updates']==1944
                assert ck['seed']==seed
                models.append(dict(study=study,name=r['name'],seed=seed,checkpoint=r['checkpoint'],sha256=r['sha256']))
            if study in ev.data.MODELS or study=='joint':
                for key in ('initial_state_hash','schedule_sha256','representation_sha256','final_rng_sha256','first_components','updates'):
                    assert all(r[key]==trained[0][key] for r in trained),(study,seed,key)
            checks.append(dict(study=study,seed=seed,trained_arms=len(trained),all_75_and1944=True))
    for model in ev.data.MODELS:
        for family in ev.data.FAMILIES:
            path=ev.data.bank(model,family);ev.data.verify_cache(path)
            old=ev.data.lines(ev.data.reference(family)/'index.jsonl');new=ev.data.lines(path/'index.jsonl')
            assert [(r['anchor_id'],r['state_names'],r['pixel_sha256']) for r in new]==[(r['anchor_id'],r['state_names'],r['pixel_sha256']) for r in old]
            checks.append(dict(model=model,family=family,n_sources=len(new),all_rendered_pixels_identical=True))
    assert read(ev.OUT/'main_replay_verification.json')['all_existing_main75_values_reproduced']
    new_paths={r['checkpoint'] for r in models if Path(r['checkpoint']).is_relative_to(ev.OUT)}
    assert len(new_paths)==33,len(new_paths)
    dump(ev.OUT/'final_verification.json',dict(checks=checks,models=models,
        seeds=[42,43,44],new_fit_count=len(new_paths),all_rendered_routing_tint=.75,old_results_preserved=True,
        external_labclip=dict(checkpoint=str(ev.LAB),sha256=sha(ev.LAB),independent_models=1),
        verification_code_sha256=sha(__file__),main_replay_sha256=sha(ev.OUT/'main_replay_verification.json')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['replay','complete']);args=ap.parse_args()
    log(ev.OUT,'verification_start',action=args.action);globals()[args.action]();log(ev.OUT,'verification_complete',action=args.action)

if __name__=='__main__':main()
