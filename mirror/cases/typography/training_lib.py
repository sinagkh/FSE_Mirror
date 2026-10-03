"""One bounded, equal-information text-adapter repair comparison."""
from mirror.cases.typography.diagnose import *
sys.path.insert(0,str(ROOT/'mirror/utils'))
from mirror.utils.two_object_locality_lib import TextLoRAAdapter; from mirror.utils.two_object_locality_lib import set_seed
from torch.nn import functional as F

CONFIGS={f'R_s{s}':dict(scale=s,weight=0.) for s in (3,10,30,100)}
CONFIGS.update({f'IS_s{s}_w{w:g}':dict(scale=s,weight=w) for s in (30,100) for w in (.1,1.)})

def model():
    set_seed(42);return TextLoRAAdapter(512,r=64,alpha=64,dropout=.05).cuda()
def statehash(a):
    h=hashlib.sha256()
    for k,v in sorted(a.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def operator(a):return ((a.A.weight@a.A.weight.T)*(a.B.weight.T@a.B.weight)).sum()/512

def data(bank):
    cfg=verify();meta=json.loads((OUT/f'{bank}_standard_features.json').read_text());assert sha(meta['path'])==meta['sha256']
    rr=json.loads((OUT/f'{bank}.json').read_text());txt=torch.load(OUT/'texts.pt',map_location='cpu');vocab=txt['vocabulary'];idx={n:i for i,n in enumerate(vocab)}
    y=torch.tensor([idx[r['label']] for r in rr],device='cuda');wrong=torch.tensor([[idx[w] for w in r['words'][1:]] for r in rr],device='cuda')
    return torch.tensor(np.load(meta['path']),device='cuda'),y,wrong,txt['features'].cuda(),rr

def margins(scores,y,wrong):
    a=scores.gather(2,y[:,None,None].expand(-1,5,1))
    b=scores.gather(2,wrong[:,None,:].expand(-1,5,-1))
    return a-b

def loss(scores,reference,t,base,a,y,wrong,cand,coefficient):
    ce=F.cross_entropy((cand['scale']*scores).flatten(0,1),y.repeat_interleave(5))
    m=margins(scores,y,wrong);fm=margins(reference.detach(),y,wrong)
    agreement=F.relu(.045-m).square().mean();anchor=(1-(t*base).sum(-1)).mean()
    shared=ce+1.2*agreement+.35*anchor+operator(a)
    diffs=torch.stack([m[:,i]-m[:,j] for i in range(1,5) for j in range(i+1,5)],1)
    nuisance=diffs.square().mean()/.1**2
    retention=F.relu(fm[:,:2].mean((0,2))-m[:,:2].mean((0,2))).square().mean()/.1**2
    value=shared+cand['weight']*(coefficient*nuisance+retention)
    return value,dict(ce=ce,nuisance=nuisance,retention=retention,anchor=anchor,shared=shared)

def evaluate(a,v,y,w,base):
    a.eval()
    with torch.no_grad():
        t=norm(a(base));s=torch.einsum('bid,cd->bic',v,t);m=margins(s,y,w).cpu().numpy()
    correct=(m>0).astype(float);conf=np.stack([correct[:,3,0],correct[:,4,1]],1)
    interaction=np.stack([m[:,2,0]-m[:,3,0],m[:,2,1]-m[:,4,1]],1)
    result=dict(clean=float(correct[:,0].mean()),blank=float(correct[:,1].mean()),conflict=float(conf.mean()),congruent=float(correct[:,2].mean()),interaction=float(abs(interaction).mean()),clean_top1=float((s[:,0].argmax(-1)==y).float().mean()))
    result['utility']=(result['clean']+result['blank']+result['conflict'])/3
    return s.cpu().numpy(),result

def prepare():
    cfg=verify();torch.set_num_threads(2);assert not (OUT/'pilot_protocol.json').exists()
    paths=[Path(__file__),CODE/'PILOT_PROTOCOL.md',OUT/'protocol.json',OUT/'protocol.sha256',OUT/'train_standard_features.json',OUT/'development_standard_features.json',OUT/'texts.pt',ROOT/'mirror/utils/two_object_locality_lib.py']
    pp=dict(configurations=CONFIGS,files={str(p):sha(p) for p in paths},epochs=16,updates=1024,seed=42)
    dump(OUT/'pilot_protocol.json',pp);(OUT/'pilot_protocol.sha256').write_text(sha(OUT/'pilot_protocol.json')+'\n')
    v,y,w,base,_=data('train');base=base[:len(cfg['seen_classes'])];a=model();a.eval();gg=[]
    ids=np.random.default_rng(4200).permutation(len(v))
    for start in range(0,512,32):
        ix=ids[start:start+32];t=norm(a(base));s=torch.einsum('bid,cd->bic',v[ix],t);ref=torch.einsum('bid,cd->bic',v[ix],base)
        value,terms=loss(s,ref,t,base,a,y[ix],w[ix],dict(scale=100,weight=0),1.)
        norms={}
        for key in ('ce','nuisance'):
            g=torch.autograd.grad(terms[key],list(a.parameters()),retain_graph=True);norms[key]=float(torch.cat([u.flatten() for u in g]).norm())
        gg.append(norms)
    coefficient=float(np.median([r['ce'] for r in gg])/np.median([r['nuisance'] for r in gg]))
    assert torch.equal(value,terms['shared']);x=torch.autograd.grad(value,list(a.parameters()),retain_graph=True);z=torch.autograd.grad(terms['shared'],list(a.parameters()));assert all(torch.equal(i,j) for i,j in zip(x,z))
    dump(OUT/'pilot_calibration.json',dict(nuisance_coefficient=coefficient,training_gradients=gg,zero_weight_value_gradient_parity=True))
    print('FROZEN PILOT',coefficient,flush=True)

def verify_pilot():
    cfg=verify();assert sha(OUT/'pilot_protocol.json')==(OUT/'pilot_protocol.sha256').read_text().strip()
    pp=json.loads((OUT/'pilot_protocol.json').read_text())
    for p,h in pp['files'].items():assert sha(p)==h,p
    return cfg

def train(name,smoke=False):
    cfg=verify_pilot();torch.set_num_threads(2);n=len(cfg['seen_classes']);cand=CONFIGS[name];v,y,w,base,_=data('train');dv,dy,dw,full,_=data('development');base=base[:n]
    coefficient=json.loads((OUT/'pilot_calibration.json').read_text())['nuisance_coefficient']
    a=model();ih=statehash(a);opt=torch.optim.AdamW(a.parameters(),lr=.0002,weight_decay=.01);dest=OUT/('smoke' if smoke else 'runs')/name;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'last.pt').exists()
    start=time.monotonic();history=[];updates=0
    for epoch in range(1,2 if smoke else 17):
        a.train();torch.manual_seed(420000+epoch);torch.cuda.manual_seed_all(420000+epoch);ids=np.random.default_rng(4200+epoch).permutation(len(v));ids=ids[:64] if smoke else ids;logs=[]
        for j in range(0,len(ids),32):
            ix=ids[j:j+32];t=norm(a(base));s=torch.einsum('bid,cd->bic',v[ix],t);ref=torch.einsum('bid,cd->bic',v[ix],base)
            value,terms=loss(s,ref,t,base,a,y[ix],w[ix],cand,coefficient);assert torch.isfinite(value)
            opt.zero_grad(set_to_none=True);value.backward();gn=torch.nn.utils.clip_grad_norm_(a.parameters(),1.);assert torch.isfinite(gn);opt.step();updates+=1
            logs.append({k:float(x.detach()) for k,x in terms.items()})
        scores,stats=evaluate(a,dv,dy,dw,full);rng=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
        history.append(dict(epoch=epoch,updates=updates,development=stats,rng_hash=rng,train={k:float(np.mean([r[k] for r in logs])) for k in logs[0]}));dump(dest/'history.json',history)
        print('EPOCH',name,epoch,json.dumps(stats),flush=True)
    assert updates==(2 if smoke else 1024)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in a.state_dict().items()},candidate=cand,initial_hash=ih,updates=updates,protocol_sha256=sha(OUT/'pilot_protocol.json'),calibration_sha256=sha(OUT/'pilot_calibration.json')),dest/'last.pt')
    np.savez_compressed(dest/'development.npz',scores=scores)
    dump(dest/'complete.json',dict(updates=updates,initial_hash=ih,rng_hash=rng,sha256=sha(dest/'last.pt'),seconds=time.monotonic()-start))
    print('COMPLETE',name,time.monotonic()-start,flush=True)

def select():
    verify_pilot();rr=[];initial=set();rng=set()
    for name in CONFIGS:
        dest=OUT/'runs'/name;meta=json.loads((dest/'complete.json').read_text());assert meta['updates']==1024 and sha(dest/'last.pt')==meta['sha256']
        initial.add(meta['initial_hash']);rng.add(meta['rng_hash']);r=json.loads((dest/'history.json').read_text())[-1]['development']
        rr.append(dict(name=name,checkpoint=str(dest/'last.pt'),sha256=meta['sha256'],**r))
    assert len(initial)==len(rng)==1
    selected={label:sorted([r for r in rr if r['name'].startswith(prefix)],key=lambda r:(-r['utility'],-r['clean_top1'],r['interaction'],r['name']))[0] for label,prefix in [('ranking','R_'),('IS','IS_')]}
    selected['same_scale_ablation']='R_s'+str(CONFIGS[selected['IS']['name']]['scale']);selected['all_candidates']=rr;selected['identical_init_dropout']=True
    dump(OUT/'selection.json',selected);(OUT/'selection.sha256').write_text(sha(OUT/'selection.json')+'\n');print('SELECTED',json.dumps(selected),flush=True)

def main():
    log();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','smoke','select']);ap.add_argument('candidate',nargs='?');a=ap.parse_args()
    if a.action in ('train','smoke'):train(a.candidate,a.action=='smoke')
    else:globals()[a.action]()

if __name__=='__main__':main()
