"""CPU value/gradient parity of standalone objectives and selected source."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from mirror.core.objectives import typography_objective, backdoor_objective, color_binding_objective

ROOT = Path(__file__).resolve().parents[1]


def functions(relative, names, scope):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def compare(old, new, variable):
    torch.testing.assert_close(old, new, rtol=1e-6, atol=1e-6)
    a = torch.autograd.grad(old, variable, retain_graph=True)[0]
    b = torch.autograd.grad(new, variable, retain_graph=True)[0]
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


torch.set_num_threads(2)
torch.manual_seed(42)
base = dict(torch=torch, F=F)
margins = functions("mirror/cases/typography/training_lib.py", {"margins"}, base.copy())
typo = functions("mirror/cases/typography/objective.py", {"loss"}, dict(base, old=margins))
scores = (torch.randn(7, 5, 6) * .05).requires_grad_()
frozen = torch.randn_like(scores) * .05
labels = torch.arange(7) % 6
distractors = torch.stack(((labels + 1) % 6, (labels + 2) % 6), 1)
for coefficient in (0., 1.5199369581165407, 6.079747832466163):
    old, _ = typo.loss(scores, frozen, dict(scale=100, weight=1), labels, distractors,
                       coefficient, 100.)
    new, _ = typography_objective(scores, frozen, labels, distractors, coefficient)
    compare(old, new, scores)
back = functions("mirror/cases/backdoor/par_visual_blocks_v3.py", {"losses"}, base.copy())
scores = (torch.randn(7, 4, 6) * .05).requires_grad_()
frozen = torch.randn_like(scores) * .05
for coefficient in (0., 10.171653221890306, 10.54683470363096, 12.357516534604287):
    old = back.losses(scores, frozen, labels)
    new, _ = backdoor_objective(scores, frozen, labels, coefficient)
    compare(old["shared"] + coefficient * old["interaction"], new, scores)
parts = ("binding", "cross", "response", "preference", "endpoint",
         "caption_guard", "binding_keep", "response_keep", "object_guard")
prior = functions("mirror/cases/color_binding/routing_relative_pilot.py", {"objective"},
                  dict(SCORE_PARTS=parts, PRIORITY=4.))
common = functions("mirror/cases/color_binding/routing_common_noise_pilot.py", {"objective"},
                   dict(prior=prior, CROSS_PRIORITY=4.))
selected = functions("mirror/cases/color_binding/routing_tint_balance_train.py", {"objective"},
                     dict(p=SimpleNamespace(prior=SimpleNamespace(objective=prior.objective,
                                                                    PRIORITY=4.), cn=common)))
variable = torch.rand(13, requires_grad=True)
terms = dict(zip((*parts, "natural", "drift", "color_floor", "ranking"), variable))
weights = {name: .2 + index * .17 for index, name in enumerate(parts)}
for method in ("Ranking", "IS"):
    old = selected.objective(terms, variable[-1], weights, "color_order", method, .45021634)
    new = color_binding_objective(terms, variable[-1], weights, .45021634, method)
    compare(old, new, variable)
print("Standalone selected objectives: 9 CPU loss-value and gradient parity checks passed; no optimization or model execution.")
