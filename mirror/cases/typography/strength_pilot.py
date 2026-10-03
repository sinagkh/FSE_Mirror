"""Isolated strength-only follow-up to manuscript_v6_3 Table 2."""
from mirror.cases.typography.diagnose import *
import itertools
import pandas as pd
import mirror.cases.typography.train as original
import mirror.cases.typography.broad_support_filtered as bank
import mirror.cases.typography.objective as base
import mirror.cases.typography.training_lib as old
import mirror.cases.typography.retest as digital

RUN = ROOT / 'mirror/cases/typography_strength_20260929'
ORIGINAL = original.RUN
MULTIPLIERS = (1, 2, 4)
PAIRS = list(itertools.combinations(range(1, 5), 2))


def command():
    RUN.mkdir(exist_ok=True)
    with (RUN / 'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                               argv=sys.argv)) + '\n')


def load_bank(name):
    if name == 'original':
        v, y, w, text, rows = old.data('development')
        labels = torch.load(OUT / 'texts.pt', map_location='cpu')['vocabulary']
    else:
        v, y, w, text, rows = bank.data('development_new')
        labels = json.loads((bank.DEST / 'filtered_data_protocol.json').read_text())['vocabulary']
    return v, y, w, text, rows, labels


def statistics(scores, frozen, y, w):
    def margins(s):
        return s.gather(2, y[:, None, None].expand(-1, 5, 1)) - s.gather(2, w[:, None, :].expand(-1, 5, -1))
    m, fm = margins(scores), margins(frozen)
    conflict = torch.stack((m[:, 3, 0], m[:, 4, 1]), 1)
    fc = torch.stack((fm[:, 3, 0], fm[:, 4, 1]), 1)
    bug = (fm[:, 1] > 0) & (fc <= 0)
    retained = bug & (conflict > 0) & (m[:, 0] > 0) & (m[:, 1] > 0)
    contrasts = torch.stack([m[:, i] - m[:, j] for i, j in PAIRS], 1)
    return dict(sources=len(y), decisions=int(conflict.numel()),
                all_targets_abs=float(contrasts.abs().mean()),
                attack_accuracy=100 * float((conflict > 0).float().mean()),
                attack_correct=int((conflict > 0).sum()),
                attack_top1=100 * float((scores[:, 3:].argmax(-1) == y[:, None]).float().mean()),
                clean_top1=100 * float((scores[:, 0].argmax(-1) == y).float().mean()),
                blank_accuracy=100 * float((m[:, 1] > 0).float().mean()),
                frozen_bugs=int(bug.sum()), retained=int(retained.sum()),
                retained_repair=100 * float(retained.sum()) / max(1, int(bug.sum())))


def verify():
    original.verify_prompt()
    pp = json.loads((RUN / 'protocol.json').read_text())
    assert sha(RUN / 'protocol.json') == (RUN / 'protocol.sha256').read_text().strip()
    for p, h in pp['files'].items():
        assert sha(p) == h, p
    return pp


def prepare():
    assert not (RUN / 'protocol.json').exists()
    original.verify_prompt()
    selection = json.loads((ORIGINAL / 'selection.json').read_text())
    files = [Path(__file__), CODE / 'STRENGTH_20260929_PROTOCOL.md',
             ORIGINAL / 'protocol.json', ORIGINAL / 'calibration.json',
             ORIGINAL / 'selection.json',
             ROOT / 'FSE_VLM/manuscript_v6_3/generated/typography_primary.tex',
             ROOT / 'FSE_VLM/evidence_completion/target_audit.py']
    for method in ('ranking', 'IS'):
        p = Path(selection[method]['checkpoint'])
        assert sha(p) == selection[method]['sha256']
        files.append(p)
    audit = ROOT / 'clip/fse_two_case_completion_20260927/typography_complete_target_audit/test_seen_standard.npz'
    z = np.load(audit)
    cells = {name:float(np.mean([abs(z[f'{name}_seed{s}']).mean() for s in seeds]))
             for name, seeds in [('frozen', [0]), ('ranking', [42,43,44]), ('IS', [42,43,44])]}
    assert [f'{cells[k]:.5f}' for k in ('frozen','ranking','IS')] == ['0.07704','0.04782','0.03950']
    files.append(audit)
    pp = dict(seed=42, multipliers=MULTIPLIERS, updates=2752, batch_size=32,
              original=selection, table2_reconstruction=cells,
              files={str(p):sha(p) for p in files},
              selection='development only, exact rules in frozen protocol markdown')
    dump(RUN / 'protocol.json', pp)
    (RUN / 'protocol.sha256').write_text(sha(RUN / 'protocol.json') + '\n')
    print('PREPARED', json.dumps(cells), flush=True)


def train(multiplier, smoke=False):
    verify(); torch.set_num_threads(2)
    cfg, model, tok, prefix, v, y, w, t, tt, rows = original.setup()
    coef = json.loads((ORIGINAL / 'calibration.json').read_text())['nuisance_coefficient']
    frozen_scale = float(model.logit_scale.exp())
    ids = bank.stream(y, updates=2752)
    sequence_hash = hashlib.sha256(ids.tobytes()).hexdigest()
    initial_hash = hashlib.sha256(prefix.detach().cpu().numpy().tobytes()).hexdigest()
    reference = json.loads((ORIGINAL / 'runs/IS_s100_w1/complete.json').read_text())
    assert sequence_hash == reference['sequence_hash'] and initial_hash == reference['initial_hash']
    name = f'IS_w{multiplier}'
    dest = RUN / ('smoke' if smoke else 'runs') / name
    dest.mkdir(parents=True, exist_ok=False)
    optim = torch.optim.SGD([prefix], lr=.002)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=2752, eta_min=.00005)
    histories, logs = [], []
    if smoke:
        ix = ids[:32]
        features = base.prototype(model, tt, prefix, len(t))
        scores = torch.einsum('bsd,cd->bsc', v[ix], features)
        ref = torch.einsum('bsd,cd->bsc', v[ix], t)
        zero, terms = base.loss(scores, ref, dict(scale=100, weight=0), y[ix], w[ix], coef, frozen_scale)
        g0 = torch.autograd.grad(zero, prefix, retain_graph=True)[0]
        gs = torch.autograd.grad(terms['shared'], prefix, retain_graph=True)[0]
        assert torch.equal(g0, gs)
        value, _ = base.loss(scores, ref, dict(scale=100, weight=multiplier), y[ix], w[ix], coef, frozen_scale)
        gi = torch.autograd.grad(value, prefix, retain_graph=True)[0]
        gn = torch.autograd.grad(terms['nuisance'], prefix)[0]
        error = float((gi - gs - multiplier * coef * gn).abs().max())
        assert error < 2e-4, error
        dump(dest / 'gradient_checks.json', dict(zero_weight_parity=True, multiplier_identity_max_error=error))
    dev = {}
    if not smoke:
        for key in ('original', 'added'):
            dv, dy, dw, dt, dr, labels = load_bank(key)
            dev[key] = (dv, dy, dw, torch.einsum('bsd,cd->bsc', dv, dt),
                        base.tokens_with_prefix(tok, [s.format(n) for n in labels for s in TEMPLATES]), len(labels))
    start = time.monotonic()
    for step in range(4 if smoke else 2752):
        ix = ids[step*32:(step+1)*32]
        features = base.prototype(model, tt, prefix, len(t))
        scores = torch.einsum('bsd,cd->bsc', v[ix], features)
        ref = torch.einsum('bsd,cd->bsc', v[ix], t)
        value, terms = base.loss(scores, ref, dict(scale=100, weight=multiplier), y[ix], w[ix], coef, frozen_scale)
        assert torch.isfinite(value)
        optim.zero_grad(set_to_none=True); value.backward()
        grad_norm = prefix.grad.detach().norm(); assert torch.isfinite(grad_norm)
        optim.step(); sched.step()
        logs.append({**{k:float(z.detach()) for k,z in terms.items()}, 'gradient_norm':float(grad_norm)})
        if (step+1)%172 == 0 or smoke and step == 3:
            record = dict(updates=step+1, seconds=time.monotonic()-start,
                          train={k:float(np.mean([r[k] for r in logs])) for k in logs[0]})
            with torch.no_grad():
                for key,(dv,dy,dw,df,dtt,n) in dev.items():
                    ds = torch.einsum('bsd,cd->bsc', dv, base.prototype(model,dtt,prefix,n))
                    record[key] = statistics(ds,df,dy,dw)
                    if step == 2751:
                        np.savez_compressed(dest/f'development_{key}.npz', scores=ds.cpu().numpy())
            histories.append(record); logs=[]; dump(dest/'history.json',histories)
            print('UPDATE', name, json.dumps(record), flush=True)
    if not smoke:
        checkpoint = dest/'last.pt'
        torch.save(dict(prefix=prefix.detach().cpu(),candidate=dict(scale=100,weight=multiplier),
                        initial_hash=initial_hash,updates=2752,adapter_location='prefix',
                        protocol_sha256=sha(RUN/'protocol.json')), checkpoint)
        meta=dict(updates=2752,initial_hash=initial_hash,sequence_hash=sequence_hash,
                  sha256=sha(checkpoint),seconds=time.monotonic()-start)
        if multiplier == 1:
            previous=torch.load(ORIGINAL/'runs/IS_s100_w1/last.pt',map_location='cpu')['prefix']
            meta['replay_max_prefix_error']=float((prefix.detach().cpu()-previous).abs().max())
            assert meta['replay_max_prefix_error'] < 1e-4, meta
            previous_scores=np.load(ORIGINAL/'runs/IS_s100_w1/development.npz')['scores']
            current_scores=np.load(dest/'development_original.npz')['scores']
            meta['replay_max_score_error']=float(abs(previous_scores-current_scores).max())
            assert np.array_equal(previous_scores.argmax(-1), current_scores.argmax(-1))
        dump(dest/'complete.json',meta)


def pooled(rows):
    n=sum(r['sources'] for r in rows); bugs=sum(r['frozen_bugs'] for r in rows)
    return dict(all_targets_abs=sum(r['sources']*r['all_targets_abs'] for r in rows)/n,
                retained_repair=100*sum(r['retained'] for r in rows)/max(1,bugs),
                attack_accuracy=100*sum(r['attack_correct'] for r in rows)/sum(r['decisions'] for r in rows))


def select():
    verify(); assert not (RUN/'selection.json').exists()
    results={}
    for m in MULTIPLIERS:
        path=RUN/'runs'/f'IS_w{m}'
        complete=json.loads((path/'complete.json').read_text())
        assert sha(path/'last.pt')==complete['sha256']
        last=json.loads((path/'history.json').read_text())[-1]
        results[m]=dict(original=last['original'],added=last['added'],
                        pooled=pooled([last['original'],last['added']]),
                        checkpoint=str(path/'last.pt'),sha256=complete['sha256'])
    ref=results[1]; eligible=[]
    for m,r in results.items():
        checks=dict(mechanism=r['pooled']['all_targets_abs']<=.9*ref['pooled']['all_targets_abs'],
                    clean_original=r['original']['clean_top1']>=ref['original']['clean_top1']-1,
                    clean_added=r['added']['clean_top1']>=ref['added']['clean_top1']-1,
                    attack=r['original']['attack_accuracy']>=ref['original']['attack_accuracy'],
                    original_retained=r['original']['retained_repair']>=ref['original']['retained_repair'],
                    pooled_retained=r['pooled']['retained_repair']>=ref['pooled']['retained_repair']+1)
        r['eligibility_checks']=checks
        if m!=1 and all(checks.values()):eligible.append(m)
    chosen=sorted(eligible,key=lambda m:(-results[m]['pooled']['retained_repair'],
                  -results[m]['pooled']['attack_accuracy'],results[m]['pooled']['all_targets_abs'],m))[0] if eligible else 1
    dump(RUN/'selection.json',dict(selected_multiplier=chosen,eligible=eligible,candidates=results,
                                  selected_before_test_scoring=True,protocol_sha256=sha(RUN/'protocol.json')))
    (RUN/'selection.sha256').write_text(sha(RUN/'selection.json')+'\n')
    print('SELECTED',chosen,json.dumps(results),flush=True)


def main():
    command(); ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','smoke','train','select']);ap.add_argument('multiplier',type=int,nargs='?',choices=MULTIPLIERS)
    args=ap.parse_args()
    if args.action in ('train','smoke'):train(args.multiplier,args.action=='smoke')
    else:globals()[args.action]()

if __name__=='__main__':main()
