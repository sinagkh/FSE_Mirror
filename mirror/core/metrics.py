"""Shared score-only measurements for completion studies; no sample selection."""
from pathlib import Path
import numpy as np
import torch
from mirror.core.io import ROOT; from mirror.core.io import read
from mirror.cases.color_binding.behavioral_pilot import CAL; from mirror.cases.color_binding.behavioral_pilot import CACHE; from mirror.cases.color_binding.behavioral_pilot import EXCLUDE; from mirror.cases.color_binding.behavioral_pilot import lines
from mirror.cases.color_binding.routing_context_coverage import all_contexts


def adapt(text,checkpoint):
    x=torch.as_tensor(np.asarray(text)).float()
    if checkpoint is None:return x.numpy()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
    a,b=state['A.weight'].float(),state['B.weight'].float()
    assert a.shape[1]==x.shape[-1] and a.shape[0]==64 and b.shape==a.T.shape
    with torch.inference_mode():
        y=x+(x@a.T)@b.T
        y=torch.nn.functional.normalize(y,dim=-1)
    return y.numpy()


def routing(x,model='openclip_laion_l14'):
    cal=read(CAL);unit=cal['units'][model]['unit'];x=np.asarray(x,float)
    contexts=all_contexts();vals=np.einsum('nij,kij->nk',x,np.stack([c['weights'] for c in contexts]))/unit
    d=vals[:,[c['kind']=='binding' for c in contexts]];c=vals[:,[c['kind']=='unwanted' for c in contexts]]
    diagonal=x.diagonal(axis1=1,axis2=2)
    q1=(x[:,1,1]-x[:,1,2])/unit;q2=(x[:,2,2]-x[:,2,1])/unit
    e=(q1+q2)/2;b=(q1-q2)/2
    result=dict(binding=d.mean(1),cross=abs(c).mean(1),
        binding_failure=(d<cal['kappa']).mean(1),cross_failure=(abs(c)>cal['tau_fraction']).mean(1),
        caption_accuracy=(diagonal>np.where(np.eye(4,dtype=bool)[None],-np.inf,x).max(2)).mean(1),
        exchange_accuracy=((q1>0).astype(float)+(q2>0))/2,response=e,preference=abs(b),surplus=e-abs(b),
        word1_accuracy=(diagonal>x[:,np.arange(4),np.arange(4)^2]).mean(1),
        word2_accuracy=(diagonal>x[:,np.arange(4),np.arange(4)^1]).mean(1),
        first_order_failure=((diagonal<=x[:,np.arange(4),np.arange(4)^2]).mean(1)+(diagonal<=x[:,np.arange(4),np.arange(4)^1]).mean(1))/2)
    for i,context in enumerate(contexts):result['contrast/'+context['name']]=vals[:,i]
    return result


def background(x,model='openclip_laion_l14'):
    cal=read(CAL);u=cal['units'][model]['unit'];z=np.asarray(x,float)/u
    q=z[:,:,0]-z[:,:,1];d=np.stack((q[:,0]-q[:,1],q[:,2]-q[:,3]),1)
    margins=q*np.array([1,-1,1,-1])[None]
    return dict(binding=d.mean(1),gap=abs(d[:,1]-d[:,0]),
        binding_failure=(d<cal['kappa']).mean(1),gap_failure=(abs(d[:,1]-d[:,0])>cal['tau_fraction']).astype(float),
        caption_accuracy=(margins>0).mean(1),
        context_bias=abs(q[:,2:4].mean(1)-q[:,:2].mean(1)),
        inv_failure=((q[:,:2]>0)!=(q[:,2:]>0)).mean(1),
        dir_failure=((d<=0).mean(1)),
        first_order_failure=(((q[:,:2]>0)!=(q[:,2:]>0)).mean(1)+(d<=0).mean(1))/2)


def bank_arrays(path,family,color='red-blue',view='canvas',prefix='',allowed=None):
    path=Path(path);images=np.load(path/'images.npy',mmap_mode='r');texts=np.load(path/'texts.npy')
    idx=lines(path/'index.jsonl');rows=[];vv=[];tt=[]
    colors=color.split('-');ci=['red-blue','green-yellow','purple-orange'].index(color)
    for r in idx:
        if allowed is not None and r['anchor_id'] not in allowed:continue
        if r['anchor_id'] in EXCLUDE:continue
        if family=='routing':names=[prefix+color+'/'+view+'/'+a+'_'+b for a in colors for b in colors];ncap=4
        else:
            contexts=('gray','blue') if view=='audit' else ('gray',view)
            names=[prefix+color+'/'+ctx+'/'+c for ctx in contexts for c in colors];ncap=2
        ii=[r['state_names'].index(n) for n in names]
        vv.append(np.asarray(images[r['image_offset']+np.asarray(ii)]))
        tt.append(texts[r['text_indices'][ci*ncap:(ci+1)*ncap]]);rows.append(r)
    return np.stack(vv),np.stack(tt),rows


def score_arrays(images,texts,checkpoint):
    adapted=adapt(texts,checkpoint)
    return np.einsum('nid,njd->nij',images,adapted,optimize=True)
