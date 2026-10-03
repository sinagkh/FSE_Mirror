"""CPU checks of the bundled metamorphic-requirement compiler and sources."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from score_views import recover_prediction

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'mirror/core'
name='artifact_requirement_compiler'
spec=importlib.util.spec_from_file_location(name,CODE/'specifications.py')
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
declared=module.load_requirement(CODE/'templates/routing-v1.yaml')
compiled=module.compile_requirement(declared,calibration_unit=.05,
    base_model_id='example',calibration_bank_id='example')
rng=np.random.default_rng(42);scores=rng.normal(size=(9,*compiled.shape))
result=compiled.evaluate(scores)
for contrast,weights in compiled.weights.items():
    np.testing.assert_allclose(result['contrasts'][contrast],(scores*weights).sum((-2,-1))/.05)
    # Interactions vanish for image-only and description-only score offsets.
    additive=rng.normal(size=(9,compiled.shape[0],1))+rng.normal(size=(9,1,compiled.shape[1]))
    np.testing.assert_allclose(compiled.evaluate(scores+additive)['contrasts'][contrast],
                               result['contrasts'][contrast],atol=1e-12)
assert len(compiled.score_requests)==16 and np.isfinite(result['passed']).all()
complete=module.compile_requirement(module.load_requirement(
    ROOT/'data/color_binding/specifications/routing-context-preference-v3.yaml'),
    calibration_unit=.05,base_model_id='example',calibration_bank_id='example')
full=complete.evaluate(scores)
for contrast,weights in complete.weights.items():
    np.testing.assert_allclose(full['contrasts'][contrast],(scores*weights).sum((-2,-1))/.05)
    if np.allclose(weights.sum(0),0) and np.allclose(weights.sum(1),0):
        np.testing.assert_allclose(complete.evaluate(scores+additive)['contrasts'][contrast],
                                   full['contrasts'][contrast],atol=1e-12)
assert len(complete.weights)==17
try:
    compiled.evaluate(np.full((4,4),np.nan))
except ValueError:pass
else:raise AssertionError('Non-finite scores must be rejected.')
# The compact classification evidence preserves argmax even with score ties.
toy=rng.integers(-2,3,size=(200,5,70)).astype(np.float32)
true=rng.integers(0,70,size=(200,5));other=toy.copy()
np.put_along_axis(other,true[...,None],-np.inf,axis=-1)
other_index=other.argmax(-1)
true_score=np.take_along_axis(toy,true[...,None],axis=-1)[...,0]
other_score=np.take_along_axis(toy,other_index[...,None],axis=-1)[...,0]
np.testing.assert_array_equal(recover_prediction(true_score,other_score,true,other_index),toy.argmax(-1))
sources=list((ROOT/'mirror').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))
for p in sources:compile(p.read_text(),str(p),'exec')
print(json.dumps(dict(python_sources_checked=len(sources),
    batched_compiler_contrasts_checked=len(compiled.weights)+len(complete.weights),
    additive_offsets_cancel=True,nonfinite_scores_rejected=True,
    compact_classification_ties_checked=1000,new_model_execution=False)))
