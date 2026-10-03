"""Full official SugarCrepe preservation with the matching OpenAI backbone."""
from mirror.cases.typography.diagnose import *
from mirror.cases.typography.retest import registry
sys.path.insert(0,str(ROOT/'clip/fse_color_worker_scale'))
from analyze import interval
sys.path.insert(0,str(ROOT/'mirror/utils'))
from mirror.utils.two_object_locality_lib import TextLoRAAdapter
import pandas as pd

REF=ROOT/'clip/fse_constraint_completion_results_20260920/confirmation/sugarcrepe/background/frozen_seed0.csv'

class Natural(Dataset):
    def __init__(self,files,prep):self.files,self.prep=files,prep
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        with Image.open(ROOT/'clip/data/coco/val2017'/self.files[i]) as im:return self.prep(im.convert('RGB'))

def main():
    log();cfg=verify();torch.set_num_threads(4);reg=registry();dest=OUT/'sugarcrepe';dest.mkdir(exist_ok=True)
    df=pd.read_csv(REF);assert len(df)==7511
    files=sorted(set(df.filename));prompts=sorted(set(df.caption)|set(df.negative_caption));fi={x:i for i,x in enumerate(files)};ti={x:i for i,x in enumerate(prompts)}
    plan=dict(registry=reg,source_sha256=sha(REF),code_sha256=sha(Path(__file__)),selection_sha256=sha(OUT/'selection.json'),n=len(df),image_count=len(files),model_weights_sha256=WEIGHT_SHA,all_official_categories=True)
    if (dest/'protocol.json').exists():assert json.loads((dest/'protocol.json').read_text())==plan
    else:dump(dest/'protocol.json',plan)
    cache=Path(cfg['cache'])/'sugarcrepe_features.npz'
    if not cache.exists():
        model,prep,tok=load_model();images=[];texts=[]
        with torch.no_grad():
            for im in DataLoader(Natural(files,prep),batch_size=128,num_workers=8,pin_memory=True,worker_init_fn=worker_init):
                with torch.autocast('cuda',dtype=torch.float16):v=model.encode_image(im.cuda(non_blocking=True))
                images.append(norm(v.float()).cpu().numpy())
            for start in range(0,len(prompts),128):
                with torch.autocast('cuda',dtype=torch.float16):v=model.encode_text(tok(prompts[start:start+128]).cuda())
                texts.append(norm(v.float()).cpu().numpy())
        np.savez_compressed(cache,images=np.concatenate(images),texts=np.concatenate(texts));dump(dest/'cache.json',dict(path=str(cache),sha256=sha(cache),files=files,prompts=prompts));del model
    meta=json.loads((dest/'cache.json').read_text());assert sha(cache)==meta['sha256'] and meta['files']==files and meta['prompts']==prompts
    z=np.load(cache);images=z['images'][[fi[x] for x in df.filename]];pos=[ti[x] for x in df.caption];neg=[ti[x] for x in df.negative_caption];allc={};rows=[]
    for name,r in reg.items():
        t=torch.tensor(z['texts'],device='cuda')
        if r:
            a=TextLoRAAdapter(512,r=64,alpha=64,dropout=.05).cuda();a.load_state_dict(torch.load(r['checkpoint'],map_location='cpu')['state_dict']);a.eval()
            with torch.no_grad():t=norm(a(t))
        t=t.cpu().numpy();ps=np.einsum('nd,nd->n',images,t[pos]);ns=np.einsum('nd,nd->n',images,t[neg]);correct=ps>ns;allc[name]=correct
        result=df[['subset','example_id','filename','caption','negative_caption']].copy();result['positive_score']=ps;result['negative_score']=ns;result['correct']=correct;result.to_csv(dest/f'{name}.csv',index=False)
        for category in ['full']+sorted(set(df.subset)):
            use=np.ones(len(df),bool) if category=='full' else (df.subset==category).to_numpy()
            rows.append(dict(model=name,category=category,n=int(use.sum()),accuracy=100*float(correct[use].mean())))
    comparisons=[]
    for other in ('ranking','same_scale_ablation','frozen'):
        comparisons.append(dict(other=other,**interval(100*(allc['IS'].astype(float)-allc[other].astype(float)),np.ones(len(df)),df.filename)))
    pd.DataFrame(rows).to_csv(dest/'summary.csv',index=False);pd.DataFrame(comparisons).to_csv(dest/'paired_intervals.csv',index=False)
    dump(dest/'complete.json',dict(protocol_sha256=sha(dest/'protocol.json'),summary=rows,contrasts=comparisons,files={str(p.relative_to(dest)):sha(p) for p in dest.glob('*.csv')}))
    print(pd.DataFrame(rows).query('category == "full"').to_string(index=False),flush=True)

if __name__=='__main__':main()
