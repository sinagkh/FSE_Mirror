"""Fixed 90% RGB blend, original-mask context scope, versioned separately."""
import itertools
import numpy as np
from mirror.cases.color_binding.banks_v2 import load
from mirror.cases.color_binding.generators import context; from mirror.cases.color_binding.generators import tint; from mirror.cases.color_binding.generators import hsv_recolor; from mirror.cases.color_binding.generators import pixel_hash; from mirror.cases.color_binding.generators import edit_checks
from mirror.cases.color_binding.rendering import CONTEXTS; from mirror.cases.color_binding.rendering import LAYOUTS; from mirror.cases.color_binding.rendering import canvas_pair; from mirror.cases.color_binding.rendering import captions
from mirror.cases.color_binding.transparency_development import recolor as blend

COLOR_PAIRS=(("red","blue"),("green","yellow"),("purple","orange"))


def recolor(rgb,mask,color):
    return blend(rgb,mask,color,strength=.9)


METHODS={"legacy_tint":tint,"native_hsv":hsv_recolor,"blend90_luminance":recolor}


def render_from_arrays(family,rgb,masks,donor=None,colors=("red","blue"),method="blend90_luminance",with_checks=False):
    if family not in ("background","routing") or method not in METHODS:raise ValueError("Unknown family/renderer")
    if len(colors)!=2 or colors[0]==colors[1]:raise ValueError("Two distinct colors required")
    fn=METHODS[method];images=[];names=[];checks=[]
    def apply(base,mm,cc,name):
        after=base.copy()
        for slot,(mask,color) in enumerate(zip(mm,cc)):
            before=after;after=fn(before,mask,color)
            if not np.array_equal(after[~mask],before[~mask]):raise AssertionError("Object edit spilled")
            if with_checks:
                checks.append(dict(state=name,slot=slot,color=color,
                    edit=edit_checks(before,after,mask,color,"hsv" if method=="native_hsv" else "rgb_blend")))
        images.append(after);names.append(name)
    if family=="background":
        if len(masks)!=1:raise ValueError("One foreground mask required")
        for ctx in CONTEXTS:
            base=context(rgb,masks[0],ctx,donor)
            if not np.array_equal(base[masks[0]],rgb[masks[0]]):raise AssertionError("Context altered foreground")
            for color in colors:apply(base,masks,[color],f"{ctx}/{color}")
    else:
        if len(masks)!=2 or np.any(masks[0]&masks[1]):raise ValueError("Two nonoverlapping slots required")
        for layout in LAYOUTS:
            base,mm=(rgb,masks) if layout=="in_situ" else canvas_pair(rgb,masks,layout=="swapped_canvas")
            if np.any(mm[0]&mm[1]):raise AssertionError("Composite slots overlap")
            for combo in itertools.product(colors,repeat=2):apply(base,mm,combo,f"{layout}/{combo[0]}_{combo[1]}")
    result=(images,names,[pixel_hash(im) for im in images])
    return (*result,checks) if with_checks else result


def render_anchor(row,colors=("red","blue"),method="blend90_luminance",with_checks=False):
    rgb,masks,donor=load(row)
    return render_from_arrays(row["family"],rgb,masks,donor,colors,method,with_checks)
