"""CPU-only check of the executed backdoor loss contrast and six final runs.

No new model scores, training, checkpoint selection, or changed artifacts.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from mirror.cases.backdoor.common import OUT; from mirror.cases.backdoor.common import ROOT; from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha

RUN=OUT/'backdoor_par/visual_blocks_v3'
EVAL=OUT/'backdoor_par/imagenetv2_confirmation_v1'
DEST=OUT/'backdoor_par/component_provenance_check'
SOURCE=ROOT/'mirror/cases/backdoor/par_visual_blocks_v3.py'


def main():
    command();torch.set_num_threads(4);assert not (DEST/'complete.json').exists()
    protocol=json.loads((RUN/'protocol.json').read_text())
    assert sha(RUN/'protocol.json')==json.loads((RUN/'protocol_hash.json').read_text())['sha256']
    assert sha(SOURCE)==protocol['source_sha256']
    evaluation=json.loads((EVAL/'protocol.json').read_text())
    assert sha(EVAL/'protocol.json')==json.loads((EVAL/'protocol_hash.json').read_text())['sha256']
    tree=ast.parse(SOURCE.read_text());nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    scope=dict(torch=torch,F=F,np=np,SEED=42,STEPS=1536,BATCH=32)
    exec(compile(ast.Module(body=[nodes['losses'],nodes['sequence']],type_ignores=[]),str(SOURCE),'exec'),scope)
    # Execute the actual trainer's method switch, not a retyped surrogate.
    expressions=[n.value for n in ast.walk(nodes['train']) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='value' for t in n.targets)]
    assert len(expressions)==1
    actual_value=compile(ast.Expression(expressions[0]),str(SOURCE),'eval')
    calibration=json.loads((RUN/'calibration.json').read_text());coefficient=calibration['coefficient']
    ratio=np.median([r['ce'] for r in calibration['gradients']])/np.median([r['interaction'] for r in calibration['gradients']])
    assert ratio==coefficient and len(calibration['gradients'])==8
    torch.manual_seed(646978);scores=(torch.randn(7,4,87,dtype=torch.float64)*.02).requires_grad_()
    reference=torch.randn_like(scores)*.02;labels=torch.tensor([0,4,15,29,43,70,86])
    loss=scope['losses'](scores,reference,labels)
    base=eval(actual_value,dict(ll=loss,coef=coefficient,method='ranking'))
    full=eval(actual_value,dict(ll=loss,coef=coefficient,method='IS'))
    assert torch.equal(base,loss['shared'])
    actual_gradient=torch.autograd.grad(full-base,scores,retain_graph=True)[0]
    target_gradient=torch.autograd.grad(coefficient*loss['interaction'],scores,retain_graph=True)[0]
    gradient_error=float((actual_gradient-target_gradient).abs().max())
    assert gradient_error<1e-12
    margins=scores.gather(-1,labels[:,None,None].expand(-1,4,1))-scores
    four_score=(scores[:,1:].gather(-1,labels[:,None,None].expand(-1,3,1))-
                scores[:,:1].gather(-1,labels[:,None,None].expand(-1,1,1)))-scores[:,1:]+scores[:,:1]
    assert torch.allclose(four_score,margins[:,1:]-margins[:,:1],atol=1e-15,rtol=0)
    assert torch.allclose(loss['interaction'],four_score.square().mean()/.1**2,atol=1e-13,rtol=0)

    data_protocol=json.loads((OUT/'backdoor_par/protocol.json').read_text())
    assert sha(OUT/'backdoor_par/protocol.json')==protocol['parent_sha256']
    vocabulary=data_protocol['vocabulary'];assert vocabulary[-1]=='banana' and len(vocabulary)==87
    train=json.loads((OUT/'backdoor_par/train.json').read_text())
    labels=np.array([vocabulary.index(r['label']) for r in train]);assert len(labels)==3763
    for bank,h in protocol['manifests'].items():assert sha(OUT/'backdoor_par'/(bank+'.json'))==h
    entry=data_protocol['models']['victim'];assert sha(entry['state_path'])==entry['state_sha256']
    state=torch.load(entry['state_path'],map_location='cpu',weights_only=True)
    model=open_clip.create_model('ViT-B-32',pretrained=None,force_quick_gelu=True)
    model.load_state_dict(state,strict=True);model=model.float().eval();del state
    initial={n:p for n,p in model.visual.named_parameters() if n.startswith(('transformer.resblocks.10.','transformer.resblocks.11.'))}
    initial_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in initial.values())).hexdigest()
    count=sum(p.numel() for p in initial.values());assert count==calibration['parameters']==14175744
    observations=[]
    manifest=json.loads((EVAL/'manifest.json').read_text());ids=np.array([r['id'] for r in manifest]);yy=np.array([r['label'] for r in manifest])
    assert len(ids)==10000 and sha(EVAL/'manifest.json')==evaluation['manifest_sha256']
    original=np.load(EVAL/'victim_records.npz');fixed=(original['pred'][:,1]==yy)&(original['pred'][:,2]!=yy)
    assert fixed.sum()==4944
    metrics={}
    for seed in (42,43,44):
        scope['SEED']=seed;sequence=scope['sequence'](labels)
        assert sequence.shape==(1536,32) and np.all((labels[sequence]==86).sum(1)==1)
        stream_hash=hashlib.sha256(sequence.tobytes()).hexdigest()
        for method in ('ranking','IS'):
            name=f'{method}_seed{seed}';directory=RUN/'runs'/name
            receipt=json.loads((directory/'complete.json').read_text());path=directory/'last.pt';digest=sha(path)
            assert receipt['steps']==1536 and receipt['seed']==seed
            assert receipt['initial_sha256']==initial_hash and receipt['sequence_sha256']==stream_hash
            assert digest==receipt['checkpoint_sha256']==evaluation['files'][str(path)]
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert saved['steps']==1536 and saved['method']==method and saved['protocol_sha256']==sha(RUN/'protocol.json')
            assert set(saved['visual_blocks'])==set(initial)
            changed=sum(not torch.equal(value,initial[k]) for k,value in saved['visual_blocks'].items())
            assert changed>0 and all(torch.isfinite(p).all() for p in saved['visual_blocks'].values())
            history=json.loads((directory/'history.json').read_text())
            assert [r['step'] for r in history]==[256,512,768,1024,1280,1536]
            with np.load(EVAL/(name+'_records.npz')) as records:
                assert np.array_equal(ids,records['ids']) and np.array_equal(yy,records['labels'])
                correct=records['pred']==yy[:,None]
                metrics[name]=dict(native_clean=float(correct[:,0].mean()),trigger=float(correct[:,2].mean()),
                    retained_repair=float((correct[fixed,1]&correct[fixed,2]).mean()),
                    target_abs_I=float(abs(records['target_margin'][yy!=954,2]-records['target_margin'][yy!=954,1]).mean()))
            observations.append(dict(method=method,seed=seed,steps=receipt['steps'],overflow_retries=receipt['overflow_retries'],
                initialization_sha256=initial_hash,sequence_sha256=stream_hash,checkpoint_sha256=digest,
                trainable_parameters=count,changed_parameter_tensors=changed,saved_parameter_tensors=len(initial)))
    report=dict(passed=True,source_sha256=sha(__file__),trainer_sha256=sha(SOURCE),
        training_protocol_sha256=sha(RUN/'protocol.json'),test_protocol_sha256=sha(EVAL/'protocol.json'),
        exact_loss_gradient_error=gradient_error,coefficient=coefficient,training_sources=len(labels),
        interaction_coverage='3 edited-versus-clean views times87 class contrasts, including87 fixed zero self-class positions; all86 nonself foils included',
        native_test_sources=len(ids),fixed_original_failure_sources=int(fixed.sum()),runs=observations,metrics=metrics,
        conclusion='The six saved final checkpoints instantiate the same source stream, initialization, capacity and successful update budget; the executed method switch adds only the specified mixed-difference penalty.',
        scope='CPU recheck of existing source/artifacts; no new training or outcomes, no test-based checkpoint selection; published PAR and exact-pattern filter are different information/repair regimes.')
    dump(DEST/'complete.json',report)
    lines=['# Backdoor component and checkpoint provenance check','',report['conclusion'],'',
        f"Executed loss/gradient identity error: {gradient_error:.3g}. The coefficient {coefficient:.12g} reproduces the eight-training-batch gradient calibration.",'',
        '| Method | Seed | Successful updates | AMP overflow retries | Clean | Triggered | Retained repair |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in observations:
        mm=metrics[f"{r['method']}_seed{r['seed']}"]
        lines.append(f"| {r['method']} | {r['seed']} | {r['steps']} | {r['overflow_retries']} | {100*mm['native_clean']:.2f} | {100*mm['trigger']:.2f} | {100*mm['retained_repair']:.2f} |")
    lines+=['','All six initialization hashes match the actual released victim loaded with the training architecture. Source streams are reconstructed from the frozen training manifest and per-seed RNG, not accepted solely from receipts. Every batch contains exactly one genuine banana example. Checkpoint tensors are restricted to the same final two visual blocks, and all six hashes match the pre-test checkpoint registry.','',
        'Native test IDs, labels and the 4,944 original-failure cohort match across methods. Retained repair uses audit-clean correctness, whose geometry matches the trigger; native-clean accuracy is reported separately.','',
        'The component comparison is matched ranking versus IS. Published PAR has different information and cleanup data; the completed exact-pattern input filter is stronger than both parameter repairs. This provenance check does not change that result.']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('BACKDOOR COMPONENT/PROVENANCE CHECK PASSED',gradient_error,observations,flush=True)


if __name__=='__main__':main()
