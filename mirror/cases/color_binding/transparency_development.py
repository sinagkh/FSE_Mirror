"""Score-free opacity comparison; no frozen production renderer is changed."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mirror.cases.color_binding.banks_v2 import load
from mirror.cases.color_binding.generators import COLORS; from mirror.cases.color_binding.generators import edit_checks; from mirror.cases.color_binding.generators import pixel_hash
from mirror.core.io import ROOT; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha
from mirror.cases.color_binding.rendering_v2 import recolor as opaque_recolor
from mirror.cases.color_binding.visual_panels import panel

STRENGTHS = (1.0, .9, .8)


def recolor(rgb, mask, color, strength=.8):
    """Blend the current opaque target-color rendering with source RGB.

    This retains some original texture AND chroma: intended hue is no longer
    guaranteed. Alpha is fixed for every pixel, source, color and model. Geometry
    and pixels outside the annotation mask are exactly unchanged.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Strength must be finite and in [0, 1]")
    rgb = np.asarray(rgb, np.uint8)
    mask = np.asarray(mask, bool)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("RGB image and mask shapes must agree")
    target = opaque_recolor(rgb, mask, color)
    out = rgb.copy()
    out[mask] = np.rint(strength * target[mask].astype(np.float32)
                       + (1-strength) * rgb[mask].astype(np.float32)).clip(0,255).astype(np.uint8)
    return out


def prepare(out):
    source = ROOT / "clip/interbind_source_quality_20260922/development/accepted_rows.jsonl"
    rows = sorted([json.loads(s) for s in source.read_text().splitlines()], key=lambda r:r["anchor_id"])
    selected = [r for family in ("background", "routing")
                for r in [x for x in rows if x["family"]==family][:4]]
    dump(out / "protocol.json", {
        "purpose":"User requested modestly less opaque colors; development-only preview",
        "no_vlm_scores":True, "no_new_training":True, "production_renderer_changed":False,
        "partition":"development", "source_sha256":sha(source),
        "code_sha256":sha(__file__), "opaque_renderer_sha256":sha(Path(__file__).with_name("rendering_v2.py")),
        "rule":"alpha * existing opaque target-color rendering + (1-alpha) * original RGB, within same mask only",
        "strengths":list(STRENGTHS), "checks":"All qualified development rows, all six colors for each mask",
        "preview_rule":"First four anchor IDs per family, lexical order; colors cycled red/blue, green/yellow, purple/orange",
        "preview_ids":[r["anchor_id"] for r in selected],
        "context":"Original context; no mask, crop, donor or geometry changes",
        "selection_basis":"Rendering quality only; model outcomes never loaded",
        "caution":"Blending source chroma can shift target hue; retain every diagnostic failure",
    })
    checks=[]; previews=[]; selected_ids={r["anchor_id"] for r in selected}
    pair_options=(("red","blue"),("green","yellow"),("purple","orange"))
    family_index={"background":0,"routing":0}
    for row in rows:
        rgb,masks,_ = load(row)
        for slot,mask in enumerate(masks):
            for color in COLORS:
                for strength in STRENGTHS:
                    after = recolor(rgb,mask,color,strength)
                    checks.append({"anchor_id":row["anchor_id"],"slot":slot,"color":color,"strength":strength,
                        "edit":edit_checks(rgb,after,mask,color,"rgb_blend"),"pixel_sha256":pixel_hash(after)})
        if row["anchor_id"] not in selected_ids: continue
        idx=family_index[row["family"]]; family_index[row["family"]]+=1
        colors=pair_options[idx%len(pair_options)]
        if row["family"]=="background": colors=(list(COLORS)[idx],)
        images=[rgb];labels=["Original"]
        for strength in STRENGTHS:
            after=rgb.copy()
            for mask,color in zip(masks,colors): after=recolor(after,mask,color,strength)
            images.append(after);labels.append(f"{strength:.0%} recoloring"+(" (current)" if strength==1 else ""))
        dest=out/"panels"/f"{row['anchor_id']}.png";dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("xb") as stream: panel(images,labels,cell=240).save(stream,format="PNG")
        previews.append({"anchor_id":row["anchor_id"],"objects":row["objects"],"colors":list(colors),
            "panel":str(dest),"panel_sha256":sha(dest),"source_ids":row["source_ids"]})
    jsonl(out/"checks.jsonl",checks);jsonl(out/"previews.jsonl",previews)
    summaries=[]
    for strength in STRENGTHS:
        subset=[r for r in checks if r["strength"]==strength]
        summaries.append({"strength":strength,"checks":len(subset),"passed":sum(r["edit"]["passes"] for r in subset),
            "outside_unchanged":all(r["edit"]["outside_unchanged"] for r in subset),
            "maximum_hue_error_p90_degrees":max(r["edit"]["hue_error_p90_degrees"] for r in subset)})
    dump(out/"complete.json",{"protocol_sha256":sha(out/"protocol.json"),"checks_sha256":sha(out/"checks.jsonl"),
        "previews_sha256":sha(out/"previews.jsonl"),"summary":summaries,"selected_strength":None,
        "semantic_validation":"Not established by deterministic pixel checks"})
    print(json.dumps(summaries),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT/"clip/interbind_transparency_development_20260922")
    a=p.parse_args();cv2.setNumThreads(1);log(a.out,"start")
    try:prepare(a.out)
    except BaseException as exc:log(a.out,"failed",error=repr(exc));raise
    log(a.out,"complete")


if __name__=="__main__":main()
