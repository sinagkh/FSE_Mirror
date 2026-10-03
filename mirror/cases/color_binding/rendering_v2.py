"""Development renderer with explicitly protected local context.

COCO masks label visible pixels, not complete amodal objects. Transplanting just
the mask can delete occluders and expose an incomplete silhouette. This variant
retains a target bounding rectangle and its local context, and changes only the
context outside it. It does not claim to replace the entire physical background.
The earlier renderer and its failed development review remain unchanged.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mirror.cases.color_binding.banks import order; from mirror.cases.color_binding.banks import _panel
from mirror.cases.color_binding.banks_v2 import load; from mirror.cases.color_binding.banks_v2 import records
from mirror.cases.color_binding.generators import COLORS; from mirror.cases.color_binding.generators import context; from mirror.cases.color_binding.generators import edit_checks; from mirror.cases.color_binding.generators import pixel_hash
from mirror.core.io import ROOT; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import sha

VALUE_FLOOR = .65
PROTECTED_PADDING_FRACTION = .04


def recolor(rgb, mask, color):
    """Fixed color ray with monotone shading in [.65, 1] of maximum value.

    The brighter interval addresses dark yellow being perceived as olive.
    Every color uses the same rule, chosen on development images only. This
    deliberately compresses contrast; no absolute-luminance preservation claim.
    """
    rgb = np.asarray(rgb, np.uint8); mask = np.asarray(mask, bool)
    gray = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY)
    ray = np.array(COLORS[color], np.float32); ray /= ray.max()
    value = VALUE_FLOOR + (1 - VALUE_FLOOR) * gray
    out = rgb.copy()
    out[mask] = np.rint(255 * value[mask, None] * ray).clip(0, 255).astype(np.uint8)
    return out


def protected_region(mask):
    mask = np.asarray(mask, bool)
    ys, xs = np.where(mask)
    if not len(ys): raise ValueError("Cannot protect an empty target")
    h, w = mask.shape
    pad = int(np.ceil(PROTECTED_PADDING_FRACTION * min(h, w)))
    region = np.zeros_like(mask)
    region[max(0, ys.min()-pad):min(h, ys.max()+1+pad),
           max(0, xs.min()-pad):min(w, xs.max()+1+pad)] = True
    if not region[mask].all(): raise AssertionError("Target not protected")
    return region


def protected_context(rgb, mask, kind, donor=None):
    protected = protected_region(mask)
    out = context(rgb, protected, kind, donor)
    if not np.array_equal(out[protected], rgb[protected]):
        raise AssertionError("Context edit changed the protected rectangle")
    return out, protected


def prepare(out):
    # Freeze the actual procedural change before rendering any development item.
    dump(out / "protocol.json", {
        "version": "protected-context-bright-ray-v2-development",
        "purpose": "Correct observed construction/label-scope problems before model outcomes",
        "source_banks_unchanged": True, "source_partitions": ["development"],
        "no_model_scores": True, "outcome_status": "No pilot or reserve scorer results inspected",
        "color_rule": "RGB target ray * (.65 + .35 * source grayscale intensity)",
        "absolute_brightness_preserved": False, "relative_shading_preserved": True,
        "context_scope": "All pixels outside the target bounding rectangle padded by .04 image side; includes other objects outside that region. Protected rectangle retains target, holes, occluders and local context.",
        "scene_claim": "Partial outer-context substitution, NOT a complete object-background separation",
        "sampling": "Exact original development panel source IDs, not selected for model or judge success",
        "frozen_prior_packet_sha256": sha(ROOT / "clip/interbind_validity_20260922/square_views/packet_frozen.json"),
        "renderer_sha256": sha(__file__),
        "old_renderer_sha256": sha(ROOT / "mirror/cases/color_binding/generators.py"),
        "diagnostic_gates": {"minimum_edited_context_area_fraction": .10, "semantic_fidelity": .90},
        "validity_status": "Development candidate, no approval inferred from deterministic checks"})
    val = ROOT / "clip/interbind_banks_v2_20260922"
    transfer = ROOT / "clip/interbind_transfer_bank_20260922"
    bg = sorted(records(val, "development", "background"), key=lambda r: order("v2-judge-bg", r["anchor_id"]))[:12]
    routing = sorted(records(transfer, "development", "routing"), key=lambda r: order("v2-judge-route", r["anchor_id"]))[:12]
    checks = []; packet = []; hashes = {}
    panels = out / "panels"; panels.mkdir(parents=True, exist_ok=True)
    for j, row in enumerate(bg):
        rgb, masks, donor = load(row); mask = masks[0]
        color = list(COLORS)[j % 6]
        images = [rgb]; labels = ["Original"]
        for kind in ["original", "hue_cast", "scene_swap"]:
            base, protected = protected_context(rgb, mask, kind, donor)
            after = recolor(base, mask, color)
            q = edit_checks(base, after, mask, color, "luminance_affine")
            # Check actual changed pixels as well as the allowed region.
            eligible = float((~protected).mean())
            changed = float(np.any(base != rgb, axis=-1).mean())
            checks.append(dict(anchor_id=row["anchor_id"], context=kind, color=color,
                edit=q, protected_fraction=float(protected.mean()), eligible_context_fraction=eligible,
                actual_context_changed_fraction=changed,
                context_target_pixels_identical=bool(np.array_equal(base[mask], rgb[mask])),
                pixel_sha256=pixel_hash(after)))
            images.append(after); labels.append(f"{color}; {kind}; protected rectangle")
        path = panels / f"{row['anchor_id']}.png"
        _panel(images, labels, cell=280).save(path); hashes[str(path)] = sha(path)
        packet.append(dict(anchor_id=row["anchor_id"], family="background", objects=row["objects"], panel=str(path)))
    for j, row in enumerate(routing):
        rgb, masks, _ = load(row)
        colors = [("red", "blue"), ("green", "yellow"), ("purple", "orange")][j % 3]
        after = rgb.copy()
        for mask, color in zip(masks, colors):
            before = after.copy(); after = recolor(after, mask, color)
            checks.append(dict(anchor_id=row["anchor_id"], context="in_situ", color=color,
                edit=edit_checks(before, after, mask, color, "luminance_affine")))
        path = panels / f"{row['anchor_id']}.png"
        _panel([rgb, after], ["Original", f"{row['objects']}; {colors}"], cell=360).save(path)
        hashes[str(path)] = sha(path)
        packet.append(dict(anchor_id=row["anchor_id"], family="routing", objects=row["objects"], panel=str(path)))
    jsonl(out / "deterministic_checks.jsonl", checks)
    jsonl(out / "development_panels.jsonl", packet)
    dump(out / "freeze.json", {"protocol_sha256": sha(out / "protocol.json"), "panels": hashes,
        "checks_sha256": sha(out / "deterministic_checks.jsonl"),
        "n_color_checks": len(checks), "color_checks_passed": sum(x["edit"]["passes"] for x in checks),
        "no_model_scores": True, "semantic_gate": "not established"})
    print(f"Development candidate: {len(packet)} panels; {sum(x['edit']['passes'] for x in checks)}/{len(checks)} edit checks", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=ROOT / "clip/interbind_renderer_v2_20260922")
    a = p.parse_args(); log(a.out, "start")
    try: prepare(a.out)
    except BaseException as exc: log(a.out, "failed", error=repr(exc)); raise
    log(a.out, "complete")


if __name__ == "__main__": main()
