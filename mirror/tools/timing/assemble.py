"""Combine results/counts.json and results/timing_*.json into stats.json and a markdown table."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
R = HERE / 'results'
counts = json.loads((R / 'counts.json').read_text())


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def med(t, k):
    v = t['timing_s'][k]
    return v['median'] if isinstance(v, dict) else v


def timing_block(t):
    if t is None:
        return None
    keys = ('generation', 'exec_image_prep_encode', 'exec_text_encode', 'exec_score_and_checks', 'execution_total', 'end_to_end_pipelined')
    out = {k: t['timing_s'][k] for k in keys}
    meta = {k: t[k] for k in t if k not in ('timing_s', 'work', 'fidelity')}
    return dict(seconds=out, work=t['work'], fidelity=t['fidelity'], settings=meta)


ty, tv = counts['typography'], counts['typography']['validation_bank']
ro = counts['routing']
stats = dict(
    description='RQ1 test-generation statistics, frozen models, held-out suites (paper v8). Counts from saved per-example arrays; '
                'times re-measured 2026-10-01 on 1x NVIDIA A10G (inference only). See PROTOCOL.md and REPORT.md.',
    hardware=dict(gpu='NVIDIA A10G (23 GB)', cpu='AMD EPYC 7R32, 32 vCPU', ram_gb=124, python='/workspace/.venv/bin/python',
                  torch='2.3.1+cu121', open_clip='3.2.0'),
    scenarios=dict(
        typography=dict(model='OpenAI CLIP ViT-B/32 (frozen)', counts={k: v for k, v in ty.items() if k != 'validation_bank'},
                        timing=timing_block(load('timing_typography_test_seen.json')),
                        heldout_diagnosis=load('typography_heldout_diagnosis.json')['heldout_1024'],
                        validation_diagnosis_recomputed=load('typography_heldout_diagnosis.json')['validation_512']),
        typography_validation=dict(model='OpenAI CLIP ViT-B/32 (frozen)', counts=tv,
                                   timing=timing_block(load('timing_typography_development.json'))),
        routing=dict(model='OpenCLIP LAION-2B ViT-L/14 (frozen)', counts=ro, timing=timing_block(load('timing_routing.json'))),
        **{f'backdoor_{c}': dict(model=f'PAR-released poisoned ViT-B/32 ({c})', counts=counts['backdoor'][c],
                                 timing=timing_block(load(f'timing_backdoor_{c}.json')))
           for c in ('stripes', 'triangles', 'text')}),
    inputs_sha256=counts['inputs_sha256'])
prim = ['typography', 'routing', 'backdoor_stripes', 'backdoor_triangles', 'backdoor_text']
def c(name, *keys):
    d = stats['scenarios'][name]['counts']
    for k in keys:
        if k in d: return d[k]
    raise KeyError(keys)
def tsec(name, k):
    v = stats['scenarios'][name]['timing']['seconds'][k]
    return v['median'] if isinstance(v, dict) else v
stats['totals_primary_suites'] = dict(
    note='Sum over the five primary held-out suites (typography, routing, three backdoor victims). The three backdoor suites share the same 10,000 ImageNetV2 sources.',
    source_instances=sum(c(n, 'sources') for n in prim), distinct_source_images_or_anchors=1024 + 400 + 10000,
    follow_up_images=sum(c(n, 'follow_up_images') for n in prim),
    interaction_tests_primary_family=c('typography', 'interaction_tests') + c('routing', 'interaction_tests') + sum(c(n, 'interaction_tests_trigger1') for n in prim[2:]),
    interaction_tests_incl_backdoor_view2_and_control=c('typography', 'interaction_tests') + c('routing', 'interaction_tests') + sum(c(n, 'interaction_tests_total') for n in prim[2:]),
    decision_checks=c('typography', 'decision_checks') + c('routing', 'decision_checks') + sum(c(n, 'decision_checks_trigger1') for n in prim[2:]),
    failing_decisions=sum(c(n, 'failing_decisions') for n in prim),
    generation_s=sum(tsec(n, 'generation') for n in prim), execution_s=sum(tsec(n, 'execution_total') for n in prim),
    end_to_end_pipelined_s=sum(tsec(n, 'end_to_end_pipelined') for n in prim))
stats['caveats'] = [
    "Typography failure definition: saved code and Supplement S1 flag blank-note correct AND misleading-note wrong -> 1,547; additionally requiring the unedited decision correct gives 1,542. The v8 Overview sentence says 'whose unedited and blank-note decisions were correct' with 1,547 (off by 5).",
    "The v8 RQ1 typography paragraph / Table 2A numbers (99.41, 22.66, 786, 4, -0.1017, +0.0786) are from the 512-source validation bank; the Overview's 12,288/2,048/1,547 are the 1,024-source held-out suite. Held-out equivalents are in scenarios.typography.heldout_diagnosis.",
    "Executions: the implementation scores full matrices (typography 5 states x 70 classes = 358,400 scores; only 15,360 are needed by the specified tests/checks). Backdoor decisions are 1,000-way, so all 40M scores are needed.",
    "Routing interaction-test count (54,400) includes the assignment interaction as a 17th test per block; binding + cross-effect alone = 51,200. Routing sources are synthetic two-cutout canvases; all 16 renders are generated inputs.",
    "Triangles/text blends: trigger views 1 and 2 are identical (PAR renderer), so 20,000 distinct follow-up images per victim and trigger-2 tests duplicate trigger-1 tests.",
    "Saved backdoor arrays average |D| over 1,000 classes including the zero true-class term; the paper's 0.1526/0.1634/0.1554 equal saved means x 1000/999 (consistent with '999 alternatives').",
    "Backdoor timing covers only the RQ1 identity-grid states (clean, two triggers, control); the existing evaluator also renders native-clean and four JPEG/resize retest grids (21 states) and scores 7 models (historical full runs 974/637/643 s, not re-measured). Generation is bound by the existing 2-worker tar.gz streaming. The image-encode vs scoring split is unreliable (async CUDA); use their sum. One timed run per victim.",
    "Routing timing mirrors feature_cache.cache_bank pipelining without its file writes and duplicate pixel-hash verification; renderer pixel checks/hashes are included.",
    "Times exclude model load, COCO annotation index (15.4 s, routing) and integrity hashing; OS page cache warm (cannot drop without root). First typography execution repeat slower (4.6 s vs 3.2-3.3 s).",
    "Frozen model only; repaired models reuse the same generated images (only scoring repeats), not timed.",
    "FSE_VLM/overleaf_v9_singlefile (main.tex and LaTeX build outputs) changed ~06:46 UTC by another process during this session; nothing in this folder writes there. No other file outside this folder changed.",
    "REPORT.md was not written: the subagent harness blocks writing report .md files; its content was returned to the caller instead.",
]
(HERE / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
print(json.dumps(stats['totals_primary_suites'], indent=1))


def fmt(x): return f'{x:,}'
def sec(t, k):
    if t is None: return 'n/a'
    return f"{med(t, k):.1f} s"

rows = []
t = load('timing_typography_test_seen.json')
rows.append(['Typography (OpenAI B/32)', fmt(ty['sources']), f"{fmt(ty['follow_up_images'])} images; {fmt(ty['follow_up_caption_representations'])} distractor descriptions ({ty['caption_strings_encoded_unique']} strings)",
             f"{fmt(ty['images_encoded'])} img + {ty['caption_strings_encoded_unique']} txt encodings; {fmt(ty['scores_computed'])} scores ({fmt(ty['scores_required_by_tests_and_checks'])} needed)",
             fmt(ty['interaction_tests']), fmt(ty['decision_checks']), f"{fmt(ty['failing_decisions'])} ({100*ty['failing_decisions']/ty['decision_checks']:.2f}%)",
             sec(t, 'generation'), sec(t, 'execution_total'), sec(t, 'end_to_end_pipelined')])
t = load('timing_routing.json')
rows.append(['Color binding (LAION L/14)', fmt(ro['sources']), f"{fmt(ro['follow_up_images'])} images; {ro['caption_representations_per_source']} caption assignments/source ({ro['caption_strings_encoded_unique']} strings)",
             f"{fmt(ro['images_encoded'])} img + {ro['caption_strings_encoded_unique']} txt encodings; {fmt(ro['scores_computed'])} scores",
             f"{fmt(ro['interaction_tests'])} ({fmt(ro['binding_tests'])} binding + {fmt(ro['cross_effect_tests'])} cross + {fmt(ro['assignment_tests'])} assignment)",
             fmt(ro['decision_checks']), f"{fmt(ro['failing_decisions'])} ({100*ro['failing_decisions']/ro['decision_checks']:.2f}%)",
             sec(t, 'generation'), sec(t, 'execution_total'), sec(t, 'end_to_end_pipelined')])
for c in ('stripes', 'triangles', 'text'):
    b = counts['backdoor'][c]; t = load(f'timing_backdoor_{c}.json')
    rows.append([f'Backdoor {c} (poisoned B/32)', fmt(b['sources']), f"{fmt(b['follow_up_images'])} images ({fmt(b['unique_follow_up_images'])} distinct); 1,000 class descriptions (80,000 strings)",
                 f"{fmt(b['images_encoded_rq1'])} img + 80,000 txt encodings; {fmt(b['scores_computed_rq1'])} scores",
                 f"{fmt(b['interaction_tests_trigger1'])} trigger ({fmt(b['interaction_tests_total'])} incl. view 2 + control)",
                 fmt(b['decision_checks_trigger1']), f"{fmt(b['failing_decisions'])} ({100*b['failing_decisions']/b['decision_checks_trigger1']:.2f}%)",
                 sec(t, 'generation'), sec(t, 'execution_total'), sec(t, 'end_to_end_pipelined')])
head = ['Scenario', 'Sources', 'Follow-up inputs', 'Executions', 'Interaction tests', 'Decisions', 'Failing decisions',
        'Generation time', 'Execution time', 'End-to-end (pipelined)']
md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
(R / 'summary_table.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md))
