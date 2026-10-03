"""Consistently scaled review panels; originals are never modified.

PIL.thumbnail only shrinks. Small square crops consequently appeared much
smaller to the judges than to the 224px model preprocessors. Resize explicitly
in both directions and record that interpolation adds no source information.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image; from PIL import ImageDraw
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import log


def fit_view(image, cell):
    im = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, np.uint8)).convert("RGB")
    if type(cell) is not int or cell < 1: raise ValueError("Invalid review cell")
    scale = min(cell/im.width, cell/im.height)
    size = (max(1, round(im.width*scale)), max(1, round(im.height*scale)))
    return im.resize(size, Image.Resampling.BICUBIC)


def panel(images, labels, cell=224):
    if not images or len(images) != len(labels): raise ValueError("Images and labels must match")
    canvas = Image.new("RGB", (cell*len(images), cell+34), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        im = fit_view(image, cell)
        canvas.paste(im, (i*cell+(cell-im.width)//2, (cell-im.height)//2))
        draw.text((i*cell+4, cell+5), label, fill="black")
    return canvas


def correct_natural_display(out):
    from mirror.cases.color_binding.natural_data import load_gallery_image
    prior = ROOT / "clip/interbind_validity_v2_20260922"
    quality = ROOT / "clip/interbind_natural_quality_20260922"
    inputs = [json.loads(s) for s in (prior/"judge_input.jsonl").read_text().splitlines()]
    rows = {"natural_"+str(r["image_id"]): r for r in map(json.loads, (quality/"development.jsonl").read_text().splitlines())}
    dump(out/"protocol.json", {"purpose":"Display correction only; unchanged sources, IDs, crops and labels",
        "prior_input_sha256":sha(prior/"judge_input.jsonl"),
        "quality_manifest_sha256":sha(quality/"development.jsonl"),
        "code_sha256":sha(__file__),"view_side":224,"interpolation":"BICUBIC, including enlargement",
        "new_pixel_information":False,"existing_verdicts_amended":False,
        "new_review_claimed":False,"no_model_scores":True})
    records=[]
    for item in inputs:
        if item["views_to_review"] != ["natural"]: continue
        row=rows[item["item_id"]]; image=load_gallery_image(row, foreground_context=True)
        path=out/f"{item['item_id']}.png"
        panel([image],[f"Label: {row['color']} {row['noun']}"]).save(path)
        records.append({"item_id":item["item_id"],"source_crop_size":list(image.size),"view_side":224,
            "panel":str(path),"sha256":sha(path),"source_image_sha256":row["image_sha256"]})
    dump(out/"freeze.json",{"protocol_sha256":sha(out/"protocol.json"),"panels":records,
        "no_new_review_or_model_outcomes":True})
    print(f"Corrected display for {len(records)} unchanged natural rows; not re-reviewed", flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT/"clip/interbind_validity_v2_20260922/display_correction")
    a=p.parse_args();log(a.out,"start")
    try:correct_natural_display(a.out)
    except BaseException as exc:log(a.out,"failed",error=repr(exc));raise
    log(a.out,"complete")


if __name__=="__main__":main()
