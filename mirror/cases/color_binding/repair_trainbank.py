"""Fixed, source-disjoint four-color repair bank; no model loading or scoring."""
import argparse
from collections import Counter; from collections import defaultdict
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.banks import decode; from mirror.cases.color_binding.banks import order
from mirror.cases.color_binding.banks_v2 import annotations_for; from mirror.cases.color_binding.banks_v2 import square_crop; from mirror.cases.color_binding.banks_v2 import crop_array
from mirror.cases.color_binding.rendering import canvas_pair
from mirror.cases.color_binding.rendering_v3 import recolor
from mirror.cases.color_binding.generators import context; from mirror.cases.color_binding.generators import edit_checks; from mirror.cases.color_binding.generators import pixel_hash

OUT = ROOT / "clip/interbind_repair_trainbank_20260922"
OLD = ROOT / "clip/fse_constraint_completion_results_20260920/confirmation"
COCO = ROOT / "clip/data/coco"
NATURAL = ROOT / "clip/fse_background_preservation_results_20260920/manifests/training.jsonl"


def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def exclusion_inputs():
    paths = []
    for part in ("pilot", "reserve"):
        paths.append(ROOT / f"clip/interbind_source_quality_20260922/{part}/accepted_rows.jsonl")
    for part in ("calibration", "pilot", "reserve"):
        paths += [ROOT / f"clip/interbind_natural_20260922/v3/{part}.jsonl",
                  ROOT / f"clip/interbind_paco_natural_20260922/{part}/rows.jsonl"]
    return paths


def blocked_ids(paths):
    ids = set()
    for path in paths:
        for row in lines(path):
            ids.update(row.get("source_ids", []))
            for key in ("coco_id",):
                if row.get(key) is not None: ids.add(int(row[key]))
            if row.get("donor"): ids.add(int(row["donor"]["image_id"]))
    return ids


def freeze(out):
    files = [OLD / "training_streams_complete.json", NATURAL, *exclusion_inputs(),
             COCO / "annotations/instances_train2017.json"]
    files += [OLD / f"training_manifests/{family}_stream_{seed}.jsonl"
              for family in ("background", "routing") for seed in range(43, 49)]
    files += [Path(__file__).with_name(m + ".py") for m in ("repair_trainbank", "banks", "banks_v2", "rendering", "rendering_v3", "transparency_development", "generators")]
    dump(out / "protocol.json", {"version": "interbind-four-color-shared-repair-bank-v1",
        "inputs": {str(p): sha(p) for p in files}, "sources": "Source/annotation rows from six existing final-confirmation streams (43..48), deduplicated before fixed hash sampling",
        "per_class_or_pair": 128, "selection": "Fixed namespace SHA256 order within each existing six-background-class / five-routing-pair group; first 128 geometrically eligible unique source tuples",
        "geometry": {"mask_area_fraction": [.02, .60], "minimum_bbox_side": 24, "border_pixels": 1,
                     "background_crop": "same 1.25x square crop as new audit; no padding or truncation",
                     "routing_layout": "same 256px canvas and 112px cutout size as new audit; two historical source images"},
        "colors": [["red", "blue"], ["green", "yellow"]], "held_out_colors": ["purple", "orange"],
        "background_contexts": ["gray", "blue"], "routing_layouts": ["canvas"],
        "renderer": "fixed 90% RGB blend, identical to held-out audit",
        "source_firewall": "Reject actual new calibration/pilot/reserve source and donor IDs and known natural COCO mappings; historical exclusion list protected evaluation construction and is not misused to reject intended training reuse",
        "natural_guard": "Existing natural-caption TRAINING manifest only; retain all five captions, matched by fixed source-row hash; no held-out captions",
        "seeds": [42, 43, 44], "schedule": {"epochs": 6, "batch": 24, "rank": 64, "alpha": 64, "dropout": .05, "lr": .0002, "weight_decay": .01, "selection": "fixed_last"},
        "scientific_role": "New common-default training bank, not exact replication of historical epoch draws or historical sample count",
        "gpu_work": "Not launched by this constructor; wait for GPU availability before feature encoding",
        "no_model_outcomes": True, "no_training": True, "no_new_judge_round": True})
    print("FROZEN repair bank", sha(out / "protocol.json"), flush=True)


def source(row):
    im = Image.open(COCO / row["coco_split"] / row["file"]).convert("RGB")
    ann = annotations_for(row["coco_split"])[1][row["ann_id"]]
    if ann["image_id"] != row["image_id"]: raise ValueError("Wrong source annotation")
    if im.size != (annotations_for(row["coco_split"])[0][row["image_id"]]["width"], annotations_for(row["coco_split"])[0][row["image_id"]]["height"]):
        raise ValueError("Source dimensions mismatch")
    rgb = np.asarray(im); mask = decode(ann, im.height, im.width)
    return rgb, mask, ann


def render(row, colors=("red", "blue")):
    sources = [source(s) for s in row["sources"]]
    if row["family"] == "background":
        rgb, mask, ann = sources[0]; crop = row["crop_xyxy"]
        rgb, mask = crop_array(rgb, crop), crop_array(mask, crop)
        bases = [(context(rgb, mask, ctx), [mask], [color], f"{ctx}/{color}") for ctx in ("gray", "blue") for color in colors]
    else:
        height = max(s[0].shape[0] for s in sources); width = sum(s[0].shape[1] for s in sources)
        packed = np.zeros((height, width, 3), np.uint8); masks = []; left = 0
        for rgb, mask, _ in sources:
            h, w = mask.shape; packed[:h,left:left+w] = rgb
            full = np.zeros((height, width), bool); full[:h,left:left+w] = mask; masks.append(full); left += w
        base, masks = canvas_pair(packed, masks)
        bases = [(base, masks, combo, "canvas/" + "_".join(combo)) for combo in itertools.product(colors, repeat=2)]
    images = []; checks = []; names = []
    for base, masks, combo, name in bases:
        after = base.copy()
        for slot, (mask, color) in enumerate(zip(masks, combo)):
            before = after; after = recolor(before, mask, color)
            checks.append({"state": name, "slot": slot, "color": color, "edit": edit_checks(before, after, mask, color, "rgb_blend")})
        images.append(after); names.append(name)
    return images, names, [pixel_hash(x) for x in images], checks


def build(out):
    p = read(out / "protocol.json"); verify_files(p["inputs"]); blocked = blocked_ids(exclusion_inputs())
    natural = [r for r in lines(NATURAL) if r["image_id"] not in blocked]
    if not natural: raise ValueError("No disjoint natural training captions")
    streams = read(OLD / "training_streams_complete.json"); selected = []; rejects = Counter(); counts = {}
    for family in ("background", "routing"):
        unique = {}; groups = defaultdict(list)
        for seed in range(43, 49):
            path = OLD / f"training_manifests/{family}_stream_{seed}.jsonl"
            if sha(path) != streams["records"][f"{family}_{seed}"]["sha256"]: raise ValueError("Historical stream mismatch")
            for r in lines(path): unique.setdefault(tuple((s["image_id"], s["ann_id"]) for s in r["sources"]), r)
        for key, row in unique.items(): groups[tuple(row["objects"])].append((key, row))
        for group, candidates in sorted(groups.items()):
            n = 0
            for key, row in sorted(candidates, key=lambda x: order("repair-bank-four-color-v1", family, x[0])):
                if blocked.intersection(row["source_ids"]): rejects["heldout_source"] += 1; continue
                records = [source(s) for s in row["sources"]]; valid = True
                for rgb, mask, ann in records:
                    h, w = mask.shape; x, y, bw, bh = ann["bbox"]
                    if not .02 <= mask.mean() <= .60 or min(bw, bh) < 24 or min(x, y, w-x-bw, h-y-bh) < 1:
                        valid = False; break
                if not valid: rejects["source_geometry"] += 1; continue
                record = {**row, "anchor_id": "repair_" + family + "_" + order("repair-anchor-v1", key)[:18]}
                if family == "background":
                    rgb, mask, ann = records[0]; crop = square_crop([ann["bbox"]], rgb.shape[1], rgb.shape[0])
                    if crop is None or crop_array(mask, crop).sum()/mask.sum() < .99:
                        rejects["crop_visibility"] += 1; continue
                    record["crop_xyxy"] = crop
                record["source_image_sha256"] = {str(COCO / s["coco_split"] / s["file"]): sha(COCO / s["coco_split"] / s["file"]) for s in row["sources"]}
                guard = natural[int(order("repair-natural-guard-v1", key), 16) % len(natural)]
                record["natural_guard"] = guard
                selected.append(record); n += 1
                if n == p["per_class_or_pair"]: break
            counts["+".join(group)] = n
            if n != p["per_class_or_pair"]: raise ValueError(f"Insufficient fixed-rule sources for {group}: {n}; do not relax")
        print("BANK", family, {k: v for k, v in counts.items()}, flush=True)
    jsonl(out / "rows.jsonl", selected); dump(out / "excluded_source_ids.json", sorted(blocked))
    dump(out / "complete.json", {"protocol_sha256": sha(out / "protocol.json"), "rows_sha256": sha(out / "rows.jsonl"),
        "excluded_sha256": sha(out / "excluded_source_ids.json"), "n_anchors": len(selected), "n_pair_specific_training_rows": 2*len(selected),
        "source_overlap": len(blocked.intersection(i for r in selected for i in r["source_ids"])),
        "by_class_or_pair": counts, "rejected_before_quota": dict(rejects), "gpu_used": False, "training_launched": False})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["freeze", "build"])
    parser.add_argument("--out", type=Path, default=OUT); args = parser.parse_args(); cv2.setNumThreads(1); log(args.out, "start")
    try: {"freeze": freeze, "build": build}[args.action](args.out)
    except BaseException as exc: log(args.out, "failed", error=repr(exc)); raise
    log(args.out, "complete")


if __name__ == "__main__": main()
