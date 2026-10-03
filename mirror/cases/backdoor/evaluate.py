"""All-source evaluation of frozen PAR-transfer recipes and input controls."""
import argparse
import copy
import io
import json
import tarfile
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import IterableDataset; from torch.utils.data import DataLoader; from torch.utils.data import get_worker_info
import mirror.cases.backdoor.test_data as data
import mirror.cases.backdoor.par_extension_controls as controls
import mirror.cases.backdoor.train as training
import mirror.cases.backdoor.par_visual_blocks_v3 as base
from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import dump; from mirror.cases.backdoor.common import sha

V2 = data.original.POUT / 'imagenetv2_confirmation_v1'
ARCHIVE = Path('/external-cache/fse_four_case_20260927/downloads/imagenetv2-matched-frequency.tar.gz')
STATES = ['native_clean'] + [g+'_'+s for g in data.GRID for s in data.original.STATES]
MEAN = np.array([.48145466,.4578275,.40821073])
STD = np.array([.26862954,.26130258,.27577711])


def protocol():
    data.verify(); controls.verify()
    path = data.RUN / 'evaluation_protocol.json'
    assert not path.exists()
    files = [Path(__file__), Path(training.__file__), V2/'manifest.json', ARCHIVE,
             data.original.VENDOR/'asset/imagenet/classes.py']
    dump(path, dict(stage='registered before new victim/repair scoring',
        states=STATES, files={str(p):sha(p) for p in files},
        cases=list(data.CASES), methods=['victim','PAR','clean_only','ranking','IS','input_controls'],
        banks=['development','banana87','banana1000','imagenetv2'],
        target_indices={'87class':86,'1000class':954},
        test_selection='all sources, all variants; no recipe changes from independent confirmation',
        precision='FP32 scoring and text ensemble; FP16 visual forwards; identical preprocessing and source pairing',
        per_example='full predicted class/top5, true score, true-minus-banana margin, signed and absolute mean allclass interaction, control metadata; float32 audit summaries',
        activation='poisoned model only on 256 predetermined training sources, before any repaired-model scores',
        uncertainty='4000 paired source percentile bootstrap replicates; crossed seed/source after replication; seed646711',
        released_PAR='operational comparison, not information/budget matched'))
    dump(data.RUN/'evaluation_protocol_hash.json', dict(sha256=sha(path)))


def verify():
    data.verify(); controls.verify()
    path = data.RUN/'evaluation_protocol.json'
    assert sha(path) == json.loads((data.RUN/'evaluation_protocol_hash.json').read_text())['sha256']
    cfg = json.loads(path.read_text())
    for file, digest in cfg['files'].items():
        assert sha(file) == digest, file
    return cfg


def rows_for(bank):
    if bank == 'activation':
        path = data.RUN/'activation_sources.json'
    elif bank.startswith('banana'):
        path = data.RUN/'banana_preservation.json'
    elif bank == 'imagenetv2':
        path = V2/'manifest.json'
    else:
        path = data.original.POUT/'development.json'
    return json.loads(path.read_text())


class Sources(IterableDataset):
    def __init__(self, case, bank, prep, with_controls=True, limit=None):
        self.case, self.bank, self.prep = case, bank, prep
        self.rows = rows_for(bank)
        if limit:
            self.rows = self.rows[:limit]
        self.controls = (['exact_filter','tolerant_filter'] if case == 'stripes' else ['blend_inversion']) if with_controls else []

    def tensor(self, image, row, renderer):
        native = self.prep(image)
        canonical = renderer.images(image, row)
        images = [data.process(im, grid) for grid in data.GRID for im in canonical]
        tensors = [torch.stack([native] + [self.prep(im) for im in images])]
        metadata = []
        if self.controls:
            pixels = np.clip(np.rint((native.numpy().transpose(1,2,0)*STD+MEAN)*255),0,255).astype(np.uint8)
            control_inputs = [Image.fromarray(pixels)] + images
            for method in self.controls:
                changed, m = zip(*(controls.apply(im, self.case, method) for im in control_inputs))
                tensors.append(torch.stack([self.prep(im) for im in changed]))
                metadata.append([z.get('detected_pixels', z.get('clipped_channel_fraction', 0.)) for z in m])
        return torch.stack(tensors), np.asarray(metadata, dtype=np.float32)

    def __iter__(self):
        info = get_worker_info(); wid, nw = (info.id,info.num_workers) if info else (0,1)
        renderer = data.Render(self.case)
        if self.bank == 'imagenetv2':
            index = 0
            with tarfile.open(ARCHIVE, 'r|*') as archive:
                for member in archive:
                    if not member.isfile() or not member.name.lower().endswith(('.jpg','.jpeg','.png')):
                        continue
                    i = index; index += 1
                    if i >= len(self.rows):
                        break
                    assert member.name == self.rows[i]['id']
                    if i % nw != wid:
                        continue
                    image = Image.open(io.BytesIO(archive.extractfile(member).read())).convert('RGB')
                    tensor, meta = self.tensor(image, self.rows[i], renderer)
                    yield i, tensor, meta
        else:
            for i, row in enumerate(self.rows):
                if i % nw == wid:
                    tensor, meta = self.tensor(data.clean_crop(row), row, renderer)
                    yield i, tensor, meta


@torch.no_grad()
def texts(model, tok, case, method, thousand):
    if not thousand:
        return training.texts(model, tok, case, method)
    path = (V2/(method+'_text_prototypes.pt') if case == 'stripes'
            else training.directory(case)/(method+'_1000_texts.pt'))
    if path.exists():
        return torch.load(path, map_location='cuda', weights_only=True)['features']
    cfg = eval((data.original.VENDOR/'asset/imagenet/classes.py').read_text(), {'__builtins__':{}})
    assert len(cfg['classes']) == 1000 and cfg['classes'].index('banana') == 954
    features = []
    for i, name in enumerate(cfg['classes']):
        z = F.normalize(model.encode_text(tok([p(name) for p in cfg['templates']]).cuda()).float(), dim=-1)
        features.append(F.normalize(z.mean(0), dim=-1))
        if i % 200 == 0:
            print('EXTENSION TEXT', case, method, i, flush=True)
    t = torch.stack(features)
    torch.save(dict(features=t.cpu(), classes=cfg['classes'], n_templates=len(cfg['templates'])), path)
    return t


def tail_path(case, method, seed):
    root = training.ORIGINAL_RUN if case == 'stripes' and method in ('ranking','IS') else training.directory(case)
    path = root/'runs'/f'{method}_seed{seed}'/'last.pt'
    receipt = json.loads((path.parent/'complete.json').read_text())
    assert receipt['steps'] == 1536 and receipt['checkpoint_sha256'] == sha(path)
    return path, receipt


def empty_records(n):
    return dict(pred=np.zeros((n,len(STATES)),np.int32), top5=np.zeros((n,len(STATES)),bool),
        true_score=np.zeros((n,len(STATES)),np.float32), target_margin=np.zeros((n,len(STATES)),np.float32),
        allclass_interaction_abs=np.zeros((n,5,3),np.float32),
        allclass_interaction_signed=np.zeros((n,5,3),np.float32))


def record(values, indices, scores, labels, target):
    y = torch.tensor(labels[indices], device='cuda')
    true = scores.gather(-1,y[:,None,None].expand(-1,len(STATES),1)).squeeze(-1)
    margins = true[:,:,None]-scores
    values['pred'][indices] = scores.argmax(-1).cpu().numpy()
    values['top5'][indices] = (scores.topk(5,-1).indices==y[:,None,None]).any(-1).cpu().numpy()
    values['true_score'][indices] = true.cpu().numpy()
    values['target_margin'][indices] = (true-scores[:,:,target]).cpu().numpy()
    for g in range(5):
        c = 1+4*g; d = margins[:,c+1:c+4]-margins[:,c:c+1]
        values['allclass_interaction_abs'][indices,g] = d.abs().mean(-1).cpu().numpy()
        values['allclass_interaction_signed'][indices,g] = d.mean(-1).cpu().numpy()


def summarize(values, labels, target, frozen=None):
    good = values['pred']==labels[:,None]; nt = labels != target
    stats = dict(n=len(labels), nonbanana_n=int(nt.sum()), banana_n=int((~nt).sum()),
        native_clean_top1=float(good[:,0].mean()), native_clean_top5=float(values['top5'][:,0].mean()), grid={})
    for g, grid in enumerate(data.GRID):
        c = 1+g*4; a = c+1
        m = values['target_margin'][:,a]-values['target_margin'][:,c]
        s = dict(clean_top1=float(good[:,c].mean()), attacked_top1=float(good[:,a].mean()),
            second_attack_top1=float(good[:,a+1].mean()), sham_top1=float(good[:,a+2].mean()),
            asr=float((values['pred'][nt,a]==target).mean()) if nt.any() else None,
            banana_clean=float(good[~nt,c].mean()) if (~nt).any() else None,
            banana_attack=float(good[~nt,a].mean()) if (~nt).any() else None,
            target_interaction_signed=float(m[nt].mean()) if nt.any() else None,
            target_interaction_abs=float(abs(m[nt]).mean()) if nt.any() else None,
            allclass_interaction_abs=float(values['allclass_interaction_abs'][:,g,0].mean()),
            allclass_interaction_signed=float(values['allclass_interaction_signed'][:,g,0].mean()),
            clean_correct_attack_wrong=int((good[:,c]&~good[:,a]).sum()))
        if frozen is not None:
            fg = frozen['pred']==labels[:,None]
            bug = fg[:,c]&~fg[:,a]
            retained = good[:,c]&good[:,a]
            s.update(frozen_bug_n=int(bug.sum()), retained_repairs=int((bug&retained).sum()),
                retained_repair_rate=float(retained[bug].mean()) if bug.any() else None,
                original_clean_correct_n=int(fg[:,c].sum()), clean_regressions=int((fg[:,c]&~good[:,c]).sum()))
        stats['grid'][grid] = s
    return stats


@torch.no_grad()
def evaluate(case, bank, seeds, smoke=False):
    verify(); training.CASE = case; training.verify()
    destination = training.directory(case)/('smoke_evaluation' if smoke else 'evaluation')/bank
    marker = destination/'summary.json'
    if marker.exists():
        complete = json.loads(marker.read_text())
        assert complete['seeds'] == seeds
        for name, digest in complete['records_sha256'].items():
            assert sha(destination/(name+'_records.npz')) == digest
        print('VERIFIED EVALUATION', marker, flush=True); return
    activation = bank == 'activation'
    assert not (activation and smoke), 'Activation membership requires all 256 registered sources.'
    if not activation:
        assert (training.directory(case)/'activation_summary.json').exists()
    model, prep, tok = training.model_load(case)
    thousand = bank in ('imagenetv2','banana1000')
    tv = texts(model,tok,case,'victim',thousand); target = 954 if thousand else len(tv)-1
    par, tp, tails, receipts = None, None, {}, {}
    if not activation:
        par, _, ptok = training.model_load(case,'PAR')
        tp = texts(par,ptok,case,'PAR',thousand)
        for seed in seeds:
            matched = []
            for method in ('clean_only','ranking','IS'):
                path, receipt = tail_path(case,method,seed); matched.append(receipt)
                name = method+'_seed'+str(seed); receipts[name] = dict(path=str(path), **receipt)
                state = torch.load(path,map_location='cpu',weights_only=True)['visual_blocks']
                blocks = torch.nn.ModuleList([copy.deepcopy(b) for b in model.visual.transformer.resblocks[10:]])
                for j, block in enumerate(blocks):
                    prefix = f'transformer.resblocks.{10+j}.'
                    block.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
                tails[name] = blocks.eval().requires_grad_(False)
            assert len({r['initial_sha256'] for r in matched}) == 1
            assert len({r['sequence_sha256'] for r in matched}) == 1
    source = Sources(case,bank,prep,with_controls=not activation,limit=8 if smoke else None)
    rows = source.rows; n = len(rows)
    vocab = json.loads((data.original.POUT/'protocol.json').read_text())['vocabulary']
    labels = np.array([target if bank.startswith('banana') else (r['label'] if bank=='imagenetv2' else vocab.index(r['label'])) for r in rows])
    names = ['victim'] + (['PAR',*tails,*source.controls] if not activation else [])
    values = {m:empty_records(n) for m in names}
    metadata = {m:np.zeros((n,len(STATES)),np.float32) for m in source.controls}
    captured = []
    hook = model.visual.transformer.resblocks[10].register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    start = time.monotonic(); seen = []
    loader = DataLoader(source,batch_size=4,num_workers=2,pin_memory=True,prefetch_factor=1,
                        worker_init_fn=data.original.old.typo.worker_init)
    for j, (indices, images, meta) in enumerate(loader):
        ix = indices.numpy(); x = images[:,0].flatten(0,1).cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16):
            features = F.normalize(model.encode_image(x).float(),dim=-1)
            tokens = captured.pop(); assert not captured
            feature_map = {'victim':features}
            if par is not None:
                feature_map['PAR'] = F.normalize(par.encode_image(x).float(),dim=-1)
            for name, blocks in tails.items():
                z = tokens
                for block in blocks:
                    z = block(z)
                pooled, _ = model.visual._pool(z)
                feature_map[name] = F.normalize((pooled@model.visual.proj).float(),dim=-1)
        for name, z in feature_map.items():
            score = (z@(tp if name=='PAR' else tv).T).reshape(len(ix),len(STATES),-1)
            record(values[name],ix,score,labels,target)
        for k, name in enumerate(source.controls):
            with torch.autocast('cuda',dtype=torch.float16):
                z = F.normalize(model.encode_image(images[:,k+1].flatten(0,1).cuda(non_blocking=True)).float(),dim=-1)
            captured.pop(); assert not captured
            record(values[name],ix,(z@tv.T).reshape(len(ix),len(STATES),-1),labels,target)
            metadata[name][ix] = meta[:,k].numpy()
        seen.extend(ix.tolist())
        if j % 50 == 0:
            print('EXTENSION EVALUATE',case,bank,len(seen),'/',n,round(time.monotonic()-start,1),flush=True)
    hook.remove(); assert sorted(seen) == list(range(n))
    destination.mkdir(parents=True,exist_ok=True); digests = {}
    for name, val in values.items():
        path = destination/(name+'_records.npz'); assert not path.exists()
        extra = dict(control_metadata=metadata[name]) if name in metadata else {}
        np.savez_compressed(path,**val,labels=labels,ids=np.array([r['id'] for r in rows]),states=np.array(STATES),**extra)
        digests[name] = sha(path)
    summary = dict(case=case,bank=bank,seeds=seeds,n=n,seconds=time.monotonic()-start,
        methods={name:summarize(v,labels,target,values['victim']) for name,v in values.items()},
        records_sha256=digests,receipts=receipts,evaluation_protocol_sha256=sha(data.RUN/'evaluation_protocol.json'))
    dump(marker,summary)
    if activation:
        grid = summary['methods']['victim']['grid']; canonical = grid['identity']['asr']
        result = dict(case=case,n=n,source_sha256=sha(data.RUN/'activation_sources.json'),
            records_sha256=digests['victim'],rule='ASR>=.50 and >=.80*canonical_ASR',
            variants={name:dict(asr=s['asr'],active=bool(s['asr']>=.50 and s['asr']>=.8*canonical)) for name,s in grid.items()})
        dump(training.directory(case)/'activation_summary.json',result)
    print('EXTENSION COMPLETE',case,bank,{m:s['grid']['identity']['attacked_top1'] for m,s in summary['methods'].items()},flush=True)


if __name__ == '__main__':
    command(); torch.set_num_threads(4)
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','evaluate','smoke'])
    parser.add_argument('--case',choices=data.CASES,default='triangles')
    parser.add_argument('--bank',choices=['activation','development','banana87','banana1000','imagenetv2'],default='development')
    parser.add_argument('--seeds',type=int,nargs='+',default=[42]); args=parser.parse_args()
    if args.action == 'prepare':
        protocol()
    else:
        evaluate(args.case,args.bank,args.seeds,args.action=='smoke')
