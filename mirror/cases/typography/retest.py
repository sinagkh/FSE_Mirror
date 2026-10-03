"""Predeclared source-disjoint digital tests; no selection on these results."""
from mirror.cases.typography.diagnose import *
sys.path.insert(0,str(ROOT/'mirror/utils'))
from mirror.utils.two_object_locality_lib import TextLoRAAdapter
sys.path.insert(0,str(ROOT/'clip/fse_color_worker_scale'))
from analyze import interval
import pandas as pd

TESTS=[('test_seen','standard'),('test_heldout','standard'),('test_seen','serif'),('test_seen','top')]

def registry():
    assert sha(OUT/'selection.json')==(OUT/'selection.sha256').read_text().strip()
    selected=json.loads((OUT/'selection.json').read_text())
    reg={'frozen':None,'ranking':selected['ranking'],'IS':selected['IS']}
    reg['same_scale_ablation']=next(r for r in selected['all_candidates'] if r['name']==selected['same_scale_ablation'])
    for r in reg.values():
        if r:assert sha(r['checkpoint'])==r['sha256']
    return reg

def freeze():
    verify();dest=OUT/'digital_retest_protocol.json';cfg=dict(registry=registry(),tests=TESTS,selection_sha256=sha(OUT/'selection.json'),code_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'protocol.json'))
    if dest.exists():assert json.loads(dest.read_text())==json.loads(json.dumps(cfg))
    else:dump(dest,cfg)

def prepare():
    freeze()
    for bank,style in TESTS:encode(bank,style)

def metrics(scores,rr,idx):
    n=len(rr);y=np.array([idx[r['label']] for r in rr]);w=np.array([[idx[s] for s in r['words'][1:]] for r in rr])
    m=scores[np.arange(n)[:,None,None],np.arange(5)[None,:,None],y[:,None,None]]-scores[np.arange(n)[:,None,None],np.arange(5)[None,:,None],w[:,None,:]]
    conflict=np.stack([m[:,3,0],m[:,4,1]],1);inter=m[:,2]-conflict
    return dict(clean=m[:,0],blank=m[:,1],congruent=m[:,2],conflict=conflict,interaction=inter,top1=(scores.argmax(-1)==y[:,None]))

def score():
    freeze();torch.set_num_threads(2);reg=registry();txt=torch.load(OUT/'texts.pt',map_location='cpu');base=txt['features'].cuda();idx={n:i for i,n in enumerate(txt['vocabulary'])};alltext={}
    for name,r in reg.items():
        if r is None:alltext[name]=base.cpu().numpy();continue
        a=TextLoRAAdapter(512,r=64,alpha=64,dropout=.05).cuda();a.load_state_dict(torch.load(r['checkpoint'],map_location='cpu')['state_dict']);a.eval()
        with torch.no_grad():alltext[name]=norm(a(base)).cpu().numpy()
    summaries=[];contrasts=[]
    for bank,style in TESTS:
        rr=json.loads((OUT/f'{bank}.json').read_text());ids=[r['image_id'] for r in rr];meta=json.loads((OUT/f'{bank}_{style}_features.json').read_text());assert sha(meta['path'])==meta['sha256'];v=np.load(meta['path']);dest=OUT/'digital_retest'/f'{bank}_{style}';dest.mkdir(parents=True,exist_ok=True)
        dump(dest/'rows.json',rr);metrics_by_model={};frozen=None
        for name,t in alltext.items():
            scores=np.einsum('bid,cd->bic',v,t);z=metrics(scores,rr,idx)
            if name=='frozen':frozen=z
            bug=(frozen['blank']>0)&(frozen['conflict']<=0);repaired=bug&(z['conflict']>0);retained=repaired&(z['blank']>0)&(z['clean']>0)
            clean_reg=(frozen['clean']>0)&(z['clean']<=0);conflict_reg=(frozen['conflict']>0)&(z['conflict']<=0)
            z.update(bug=bug,repaired=repaired,retained_repaired=retained,clean_regression=clean_reg,conflict_regression=conflict_reg)
            np.savez_compressed(dest/f'{name}.npz',scores=scores,**z);metrics_by_model[name]=z
            summary=dict(bank=bank,style=style,model=name,sources=len(rr),decisions=2*len(rr),**{key:100*float((z[key]>0).mean()) for key in ('clean','blank','congruent','conflict')},clean_top1=100*float(z['top1'][:,0].mean()),conflict_top1=100*float(z['top1'][:,3:].mean()),interaction_abs=float(abs(z['interaction']).mean()),interaction_p90=float(np.quantile(abs(z['interaction']),.9)),bug_count=int(bug.sum()),repaired=int(repaired.sum()),retained_repaired=int(retained.sum()),repair_rate=100*float(repaired.sum()/max(1,bug.sum())),retained_repair_rate=100*float(retained.sum()/max(1,bug.sum())),clean_regression=100*float(clean_reg.sum()/max(1,(frozen['clean']>0).sum())),conflict_regression=100*float(conflict_reg.sum()/max(1,(frozen['conflict']>0).sum())))
            summaries.append(summary);print('RETEST',json.dumps(summary),flush=True)
        x=metrics_by_model['IS']
        for other in ('ranking','same_scale_ablation','frozen'):
            y=metrics_by_model[other]
            for key in ('clean','blank','conflict'):
                delta=100*((x[key]>0).mean(1)-(y[key]>0).mean(1));contrasts.append(dict(bank=bank,style=style,other=other,metric=key,**interval(delta,np.ones(len(ids)),ids)))
            contrasts.append(dict(bank=bank,style=style,other=other,metric='interaction_abs',**interval(abs(x['interaction']).mean(1)-abs(y['interaction']).mean(1),np.ones(len(ids)),ids)))
            for key in ('repaired','retained_repaired'):
                contrasts.append(dict(bank=bank,style=style,other=other,metric=key,**interval(100*(x[key].sum(1)-y[key].sum(1)),x['bug'].sum(1),ids)))
    pd.DataFrame(summaries).to_csv(OUT/'digital_summary.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'digital_intervals.csv',index=False)
    lines=['# Typographic distraction: matched seed42 digital retest','',
           'Development-selected final-epoch models; same data, capacity, updates, guards and search budget. Public SCAM/RTA and natural preservation remain separate required evaluations; this report is not their substitute.','',
           '| Bank / style | Model | Clean % | Blank % | Conflict % | Clean top1 % | Conflict top1 % | Abs interaction | Retained repair % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append('| '+r['bank']+' / '+r['style']+' | '+r['model']+' | '+' | '.join(f'{r[k]:.3f}' for k in ('clean','blank','conflict','clean_top1','conflict_top1','interaction_abs','retained_repair_rate'))+' |')
    lines+=['','Same-scale ranking is the exact interaction-loss ablation, not a separately invented baseline. Full-class top1 uses all70 declared classes; pairwise accuracy compares the annotated object with each of two supplied distractors. Retained repair requires the formerly failed attack decision plus its clean and blank controls; it does not require unrelated corners to pass.','',
           '## IS minus development-selected ranking','', '| Bank / style | Metric | Difference | Paired95% interval |','|---|---|---:|---:|']
    for r in contrasts:
        if r['other']=='ranking':lines.append(f"| {r['bank']} / {r['style']} | {r['metric']} | {r['mean']:+.5f} | [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] |")
    lines+=['','5000 paired source-cluster percentile bootstrap resamples. Both distractors and all five states stay clustered; seed42 only. All registered digital test families are reported.','']
    (OUT/'DIGITAL_REPORT.md').write_text('\n'.join(lines));dump(OUT/'digital_complete.json',dict(protocol_sha256=sha(OUT/'digital_retest_protocol.json'),files={str(p.relative_to(OUT)):sha(p) for p in [OUT/'digital_summary.csv',OUT/'digital_intervals.csv',*list((OUT/'digital_retest').glob('*/*.npz'))]}))

def main():
    log();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','score']);a=ap.parse_args();globals()[a.action]()

if __name__=='__main__':main()
