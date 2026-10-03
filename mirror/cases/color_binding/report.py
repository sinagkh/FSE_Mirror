"""Compact completion report and full result index for uniform routing75."""
from pathlib import Path
from mirror.cases.color_binding import evaluate as ev
from mirror.core.io import read; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha; from mirror.core.io import verify_files

def main():
    verify=read(ev.OUT/'final_verification.json');results={s:ev.data.lines(ev.OUT/s/'results/all_metrics.jsonl') for s in ev.STUDIES}
    def one(study,test,metric,method):
        rr=[r for r in results[study] if (r['test'],r['metric'],r['comparison'])==(test,metric,method)]
        assert len(rr)==1;return rr[0]
    def show(r,scale=100,sd=False):
        return f"{scale*r['mean']:.2f}"+(f" ± {scale*r['sample_sd']:.2f}" if sd else '')
    def ci(r,scale=100):
        lo,hi=r['ci95_seed_items'];return f"{scale*r['mean']:+.2f} [{scale*lo:+.2f}, {scale*hi:+.2f}]"
    test='trained_colors_both_orders'
    lines=['# Uniform 75%-tint routing completion','',
        'All routing paper comparisons now use the same75% luminance-modulated tint, two training color pairs, balanced noun order, and complete16-context audit. Old experiments are preserved separately; no old90% numbers are relabeled.',
        '', 'The completed main LAION text-only study is unchanged. This campaign adds33 fixed-last fits:12 backbone fits,3 ordinary-ranking controls, and18 joint fits including four component deletions. No new loss, temperature, tint, subset or checkpoint search.',
        '', 'Every learned arm retains seeds42/43/44. Frozen and released LABCLIP are single fixed models. Bootstrap intervals share source-cluster draws across arms and seeds (5000 draws); seed sample SD uses ddof1.',
        '', '## Main comparisons','', '| Study | Method | Exchange (%) | Binding | Cross-effect | Assignment D=2e |','|---|---|---:|---:|---:|---:|']
    for study in ev.STUDIES:
        methods=list(dict.fromkeys(r['comparison'] for r in results[study] if r['test']==test and ' - ' not in r['comparison']))
        for method in methods:
            values=[show(one(study,test,metric,method),scale,metric=='exchange_accuracy' and method not in ('Frozen','LABCLIP')) for metric,scale in
                    [('exchange_accuracy',100),('binding',1),('cross',1),('response',2)]]
            lines.append('| '+study+' | '+method+' | '+' | '.join(values)+' |')
    lines += ['', '## IS minus matched ranking','', '| Study | Test | Metric | Difference, paired seed/source95% CI |','|---|---|---|---:|']
    tests=[(test,'exchange_accuracy'),('background_shift','exchange_accuracy'),('geometry_shift','exchange_accuracy'),
        ('color_recombination','exchange_accuracy'),('noun_recombination','exchange_accuracy'),('caption_gallery','gallery_top1'),
        ('attribute_clause_red-blue','exchange_accuracy'),('attribute_clause_green-yellow','exchange_accuracy'),
        ('sugarcrepe_full','accuracy'),('aro_full','accuracy'),('coco_t2i','recall1'),('coco_i2t','recall1')]
    for study in ev.STUDIES:
        for target,metric in tests:lines.append('| '+study+' | '+target+' | '+metric+' | '+ci(one(study,target,metric,'IS - Ranking'))+' |')
    lines += ['', '## Integrity and reproduction','',
        '- `final_verification.json`: every new checkpoint, all33 fits, common rendered-pixel identities across backbones, and model/seed mappings.',
        '- `main_replay_verification.json`:243 comparisons of existing main results; exact agreement after retaining their original per-test text caches.',
        '- Each study contains complete per-example raw scores, source IDs, means/sample SD, item-only and crossed seed/source intervals, all categories, component controls where applicable, and diagnosis/fix/break analyses.',
        '- `preflight_evaluation_v1/` preserves preliminary evaluation outputs before correcting duplicate text-cache roundoff. No model was retrained for that correction.',
        '- `FSE_VLM/plan/88_uniform_routing75_completion.md` freezes the fixed protocol. The executable `run_routing75_complete_*.sh` scripts and `routing75_complete_*.py` modules implement the campaign.',
        '- Data and training provenance is logged in `commands.jsonl`, per-stage protocols, checkpoint receipts, and all epoch/update histories.',
        '- Manuscript target: `FSE_VLM/manuscript_v7`; prior versions are not edited.',
        '', '## Interpretation for the paper','',
        'The primary text-only result remains the completed main75% study. Backbone and published-patch claims must use the values above. The joint experiment is an extension with its own measured preservation cost, not a replacement for the main text-only experiment. Ordinary ranking remains a retention control; the primary comparison is ranking with matched guards.',
        '', 'The scope is the fixed 75% rendering and caption protocol. This campaign does not isolate opacity as a causal factor, since the completed main recipe also balances noun order and adds a shared single-word margin floor.']
    with (ev.OUT/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    jsonl(ev.OUT/'all_results.jsonl',[dict(study=study,**r) for study,rows in results.items() for r in rows])
    dump(ev.OUT/'complete.json',dict(report_sha256=sha(ev.OUT/'REPORT.md'),all_results_sha256=sha(ev.OUT/'all_results.jsonl'),
        verification_sha256=sha(ev.OUT/'final_verification.json'),tint=.75,new_fits=33,
        no_new_recipe_search=True,completed=True,
        study_results={s:sha(ev.OUT/s/'results/complete.json') for s in ev.STUDIES}))
    log(ev.OUT,'campaign_complete',new_fits=33,tint=.75)

if __name__=='__main__':main()
