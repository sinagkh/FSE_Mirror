"""Score-blind Visual Genome natural-color manifests for InterBind.

Run ``python -m mirror.cases.color_binding.natural_data freeze`` before ``build``.
No model, checkpoint, score, or outcome artifact is read here.
"""
from __future__ import annotations
from mirror.paths import ARTIFACT_ROOT

import argparse
from collections import Counter; from collections import defaultdict
from datetime import datetime; from datetime import timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from urllib.request import urlopen
from PIL import Image

ROOT = ARTIFACT_ROOT
OUT = ROOT / "clip/interbind_natural_20260922"
ATTR = OUT / "source/attributes.json.zip"
META = OUT / "source/image_data.json.zip"
IMAGES = ROOT / "clip/aro_bows/aro_data/images"
BANKS = ROOT / "clip/interbind_foundations_20260922/banks"
COLORS = ("red", "blue", "green", "yellow", "purple", "orange")
OTHER_COLOR_WORDS = {"black", "white", "gray", "grey", "pink", "brown", "tan", "beige",
                     "silver", "gold", "golden", "cream", "teal", "turquoise", "maroon",
                     "blond", "blonde", "multicolor", "multicolored"}
ALL_COLOR_WORDS = set(COLORS) | OTHER_COLOR_WORDS
NOUN_ALIASES = {
    "airplane": ["airplane", "airplanes", "aeroplane", "aeroplanes", "aircraft", "plane", "planes", "jet", "jets"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
    "boat": ["boat", "boats"], "bus": ["bus", "buses"],
    "car": ["car", "cars", "automobile", "automobiles"], "cat": ["cat", "cats"],
    "chair": ["chair", "chairs"], "couch": ["couch", "couches", "sofa", "sofas"],
    "dog": ["dog", "dogs"], "horse": ["horse", "horses"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children"],
    "sports ball": ["sports ball", "sports balls", "ball", "balls"],
    "train": ["train", "trains"], "truck": ["truck", "trucks"],
    "traffic light": ["traffic light", "traffic lights", "traffic signal", "traffic signals", "stoplight", "stoplights"],
    "tv": ["tv", "tvs", "television", "televisions", "television set", "television sets"],
    "remote": ["remote", "remotes", "remote control", "remote controls"],
}
WORD = re.compile(r"[a-z]+")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(value: str) -> str:
    return " ".join(WORD.findall(str(value).lower()))


ALIASES = {norm(alias): noun for noun, variants in NOUN_ALIASES.items() for alias in variants}


def partition(image_id: int, coco_id: int | None) -> str:
    """Mapped COCO IDs use precisely the foundations bank partition."""
    if coco_id is not None:
        digest = hashlib.sha256(json.dumps(["interbind_v1_20260922", "partition", int(coco_id)], sort_keys=True).encode()).hexdigest()
    else:
        digest = hashlib.sha256(json.dumps(["interbind_natural_v1_20260922", "vg_partition", int(image_id)], sort_keys=True).encode()).hexdigest()
    bucket = int(digest, 16) % 100
    return "development" if bucket < 10 else "calibration" if bucket < 25 else "pilot" if bucket < 50 else "reserve" if bucket < 80 else "donor"


def object_color(attrs: list[str]) -> str | None:
    """Exact singleton six-color attribute with no contradictory color tokens."""
    exact = {norm(a) for a in attrs if norm(a) in COLORS}
    seen = {token for a in attrs for token in WORD.findall(str(a).lower()) if token in ALL_COLOR_WORDS}
    return next(iter(exact)) if len(exact) == 1 and seen == exact else None


def box_ok(obj: dict, width: int, height: int, rules: dict) -> bool:
    x, y, w, h = (obj[k] for k in ("x", "y", "w", "h"))
    if not all(isinstance(v, (int, float)) for v in (x, y, w, h)):
        return False
    if min(w, h) <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        return False
    return (w * h / (width * height) >= rules["minimum_box_area_fraction"]
            and w / width >= rules["minimum_box_side_fraction"]
            and h / height >= rules["minimum_box_side_fraction"])


def rules() -> dict:
    return {
        "version": "interbind_vg_natural_v1_20260922", "created_before_enumeration": True,
        "source": {"attributes_url": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip",
                   "image_data_url": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
                   "attributes_zip_sha256": sha(ATTR), "image_data_zip_sha256": sha(META),
                   "local_images_root": str(IMAGES), "official_field_reference": "https://visualgenome.org/api/v0/api_readme"},
        "colors": list(COLORS), "noun_aliases": NOUN_ALIASES,
        "partition": "VG image ID first; if mapped COCO ID, exact foundations banks.partition(coco_id); otherwise independent VG namespace; 10/15/25/30/20 development/calibration/pilot/reserve/donor",
        "bank_exclusion": "Exclude mapped COCO IDs in every foundations source and donor bank; also exact VG ID cannot appear twice in one gallery",
        "minimum_box_area_fraction": .02, "minimum_box_side_fraction": .06,
        "object_rule": "one mapped instance of the noun per image among attribute objects, one exact six-color label and no other basic-color token; at most one target noun/image, chosen by hash",
        "image_rule": "local original JPEG required; metadata dimensions and valid box required; no model-dependent selection",
        "calibration": "all eligible calibration rows; canonical original-context red/blue rows for A3 unit, no crop required by data manifest; unit itself is computed only by downstream scorer",
        "retrieval": "focused per-noun gallery of original images and optional 1.5x boxed-object context crop; explicit same-noun other-color labels are incompatible only within this restricted gallery; unannotated images never negative",
        "relevance": "positive=exact object-color annotation; incompatible=unique mapped object explicitly labeled another singleton color; otherwise UNKNOWN; no closed-world full-image claim",
        "context": "other separately boxed object with exact query-color label is an annotation-supported decoy; background hue is not asserted from a box or measured here",
        "selection": "all eligible local images; at most one target noun/image; deterministic SHA256 tie-break; no outcome inspection or quality cherry-picking",
        "scope_limitations": ["VG attributes are sparse and noisy", "an annotated object's color can describe a part rather than the whole",
                              "a full image may contain an unannotated target, so restricted-gallery incompatibility is not exhaustive full-image absence",
                              "a box is not a mask and a 1.5x context crop is not a full-scene test", "calibration candidates require later visual label-noise assessment"],
        "no_models_or_outcomes_read": True,
    }


def rules_v2(exclusion_path: Path) -> dict:
    p = rules()
    p["version"] = "interbind_vg_natural_v2_20260922"
    p["image_rule"] = "official metadata URL JPEG or identical locally retained ARO JPEG; frozen candidates downloaded only after annotation-based selection"
    p["remote_source"] = "Only cs.stanford.edu Visual Genome VG_100K/VG_100K_2 JPEG URLs from official image_data.json; no search or alternative images"
    p["quotas_per_partition_noun_color"] = {"development": 3, "calibration": 5, "pilot": 8, "reserve": 8}
    p["total_cap_per_partition"] = {"development": 90, "calibration": 150, "pilot": 240, "reserve": 240}
    p["selection"] = "before downloading, SHA256 order within partition/noun/color then global SHA256 order to cap; no replacement for unavailable images"
    p["download_limit"] = "at most 720 JPEGs, expected well below 3 GB; preserve archive and manifest hashes"
    p["historical_coco_exclusion"] = {"path": str(exclusion_path.resolve()), "sha256": sha(exclusion_path)}
    return p


def dump(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def log(command: str) -> None:
    with (OUT / "commands.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "command": command,
                            "module_sha256": sha(Path(__file__))}, sort_keys=True) + "\n")


def load_zip_json(path: Path, member: str):
    with zipfile.ZipFile(path) as z:
        with z.open(member) as f:
            return json.load(f)


def frozen_bank_ids() -> tuple[set[int], dict]:
    freeze = json.loads((BANKS / "freeze.json").read_text())
    ids = set()
    for filename, expected in freeze["files"].items():
        path = Path(filename)
        if sha(path) != expected:
            raise ValueError(f"Foundations bank hash changed: {path}")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            ids.update(map(int, row["source_ids"]))
            if "donor" in row:
                ids.add(int(row["donor"]["image_id"]))
    return ids, {"freeze_sha256": sha(BANKS / "freeze.json"), "excluded_coco_ids": len(ids)}


def crop_box(box: list[int], width: int, height: int, scale: float = 1.5) -> list[int]:
    x, y, w, h = box
    side_w, side_h = min(width, max(w, round(scale * w))), min(height, max(h, round(scale * h)))
    left = max(0, min(width - side_w, round(x + w / 2 - side_w / 2)))
    top = max(0, min(height - side_h, round(y + h / 2 - side_h / 2)))
    return [left, top, side_w, side_h]


def square_context_crop(box: list[int], width: int, height: int) -> list[int] | None:
    """Canonical scorer view: square context, full boxed target or reject."""
    x, y, w, h = box
    side = min(math.ceil(1.5 * max(w, h)), min(width, height))
    if side < max(w, h):
        return None
    left = max(0, min(width - side, round(x + w / 2 - side / 2)))
    top = max(0, min(height - side, round(y + h / 2 - side / 2)))
    if not (left <= x and top <= y and x + w <= left + side and y + h <= top + side):
        return None
    return [left, top, side, side]


def relevance(query: dict, row: dict) -> str:
    """Three-valued relevance; incompatible is valid only in the frozen focused gallery."""
    if row.get("partition") != query.get("partition") or row.get("noun") != query.get("noun"):
        return "UNKNOWN"
    if row.get("id") not in query.get("gallery_ids", ()):
        return "UNKNOWN"
    if row.get("color") not in COLORS or query.get("color") not in COLORS:
        return "UNKNOWN"
    return "positive" if row["color"] == query["color"] else "incompatible"


def _intersection(a: list[int], b: list[int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))


def context_color_decoy(query: dict, row: dict, view: str = "foreground_context") -> bool:
    """Visible other-object label in the specified view; never a background label."""
    if view not in ("foreground_context", "full_image"):
        raise ValueError("view must be foreground_context or full_image")
    if relevance(query, row) != "incompatible":
        return False
    target = row["box"]
    crop = row.get("context_crop_box") if view == "foreground_context" else None
    if view == "foreground_context" and crop is None:
        raise ValueError("context_crop_box required for foreground_context view")
    for other in row.get("other_labeled_color_objects", ()):
        if other["color"] != query["color"]:
            continue
        other_box = other["box"]
        target_area = target[2] * target[3]
        other_area = other_box[2] * other_box[3]
        if min(target_area, other_area) <= 0:
            continue
        overlap = _intersection(target, other_box)
        if overlap / min(target_area, other_area) >= .1:
            continue
        visible_area = other_area if crop is None else _intersection(other_box, crop)
        # An overlapping portion is not visibly a distinct other-object cue.
        visible_outside_target = max(0, visible_area - overlap)
        if visible_outside_target / other_area >= .5:
            return True
    return False


def load_gallery_image(row: dict, foreground_context: bool = True) -> Image.Image:
    """Load untouched VG pixels, optionally restricting to the frozen 1.5x box crop."""
    path = Path(row["image_path"])
    if "image_sha256" in row and sha(path) != row["image_sha256"]:
        raise ValueError(f"Gallery image hash changed: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    if foreground_context:
        x, y, w, h = row["context_crop_box"]
        image = image.crop((x, y, x + w, y + h))
    return image


def _fetch(row: dict) -> dict:
    path = Path(row["image_path"])
    if path.is_file():
        with Image.open(path) as im:
            if im.size != (row["width"], row["height"]):
                return {"id": row["id"], "ok": False, "reason": "reused_image_dimension_mismatch"}
        return {"id": row["id"], "ok": True, "sha256": sha(path), "bytes": path.stat().st_size, "reused": True}
    url = row["image_url"]
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http") or parsed.hostname != "cs.stanford.edu" or "/VG_100K" not in parsed.path or not parsed.path.endswith(f"/{row['image_id']}.jpg"):
        return {"id": row["id"], "ok": False, "reason": "unapproved_official_metadata_url"}
    url = "https" + url[4:] if url.startswith("http:") else url
    try:
        with urlopen(url, timeout=30) as response:
            content = response.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024:
            return {"id": row["id"], "ok": False, "reason": "image_over_8mb"}
        path.parent.mkdir(parents=True, exist_ok=True)
        # This path is create-only; no replacement of a pre-existing file.
        with path.open("xb") as f:
            f.write(content)
        with Image.open(path) as im:
            if im.size != (row["width"], row["height"]):
                return {"id": row["id"], "ok": False, "reason": "metadata_dimension_mismatch"}
        return {"id": row["id"], "ok": True, "sha256": sha(path), "bytes": len(content), "reused": False}
    except Exception as exc:
        return {"id": row["id"], "ok": False, "reason": type(exc).__name__}


def build(v2: bool = False) -> None:
    dest = OUT / "v2" if v2 else OUT
    protocol_path = dest / "protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError("Freeze protocol before enumeration")
    p = json.loads(protocol_path.read_text())
    if sha(ATTR) != p["source"]["attributes_zip_sha256"] or sha(META) != p["source"]["image_data_zip_sha256"]:
        raise ValueError("Annotation archive changed since freeze")
    if (dest / "freeze.json").exists():
        raise FileExistsError("Natural manifests already frozen")
    used_coco, bank_info = frozen_bank_ids()
    if v2:
        protected = p["historical_coco_exclusion"]
        exclusion_path = Path(protected["path"])
        if sha(exclusion_path) != protected["sha256"]:
            raise ValueError("Historical source exclusion changed since protocol freeze")
        exclusion_data = json.loads(exclusion_path.read_text())
        if isinstance(exclusion_data, list):
            prior_ids = set(map(int, exclusion_data))
        else:
            field = "source_ids" if "source_ids" in exclusion_data else "excluded_coco_ids"
            prior_ids = set(map(int, exclusion_data[field]))
        used_coco |= prior_ids
        bank_info["historical_exclusion_sha256"] = protected["sha256"]
        bank_info["historical_excluded_coco_ids"] = len(prior_ids)
    metadata = {int(r["image_id"]): r for r in load_zip_json(META, "image_data.json")}
    raw = load_zip_json(ATTR, "attributes.json")
    rows = defaultdict(list)
    rejected = Counter()
    for image in raw:
        iid = int(image["image_id"])
        meta = metadata.get(iid)
        if meta is None:
            rejected["no_metadata"] += 1; continue
        coco = meta.get("coco_id")
        coco = int(coco) if coco is not None else None
        part = partition(iid, coco)
        if part == "donor":
            rejected["donor_partition"] += 1; continue
        if coco in used_coco:
            rejected["protected_coco_overlap"] += 1; continue
        local = IMAGES / f"{iid}.jpg"
        if not local.is_file() and not v2:
            rejected["image_not_local"] += 1; continue
        width, height = int(meta["width"]), int(meta["height"])
        if width <= 0 or height <= 0:
            rejected["invalid_metadata_size"] += 1; continue
        byid = {}
        conflict = set()
        for obj in image.get("attributes", []):
            oid = int(obj["object_id"])
            if oid in byid and byid[oid] != obj:
                conflict.add(oid)
            else:
                byid[oid] = obj
        mapped = defaultdict(list)
        for oid, obj in byid.items():
            if oid in conflict:
                continue
            names = obj.get("names", [obj.get("name", "")])
            nouns = {ALIASES[norm(n)] for n in names if norm(n) in ALIASES}
            if len(nouns) == 1:
                mapped[next(iter(nouns))].append(obj)
        target_candidates = []
        for noun, objects in mapped.items():
            if len(objects) != 1:
                rejected["nonunique_noun_instance"] += 1; continue
            obj = objects[0]
            color = object_color(obj.get("attributes", []))
            if color is None:
                continue
            if not box_ok(obj, width, height, p):
                rejected["box_rejected"] += 1; continue
            target_candidates.append((noun, obj, color))
        if not target_candidates:
            rejected["no_target"] += 1; continue
        target_candidates.sort(key=lambda t: hashlib.sha256(f"natural-target-v1:{iid}:{t[0]}:{t[1]['object_id']}".encode()).hexdigest())
        noun, obj, color = target_candidates[0]
        box = [int(obj[k]) for k in ("x", "y", "w", "h")]
        decoys = []
        for other_noun, objects in mapped.items():
            if other_noun == noun:
                continue
            for other in objects:
                other_color = object_color(other.get("attributes", []))
                if other_color and box_ok(other, width, height, p):
                    decoys.append({"noun": other_noun, "color": other_color,
                                   "object_id": int(other["object_id"]),
                                   "box": [int(other[k]) for k in ("x", "y", "w", "h")]})
        selected_path = local if local.is_file() else dest / "images" / f"{iid}.jpg"
        rows[part].append({"id": f"vg_{iid}_{noun.replace(' ', '_')}", "image_id": iid, "coco_id": coco,
                           "partition": part, "image_path": str(selected_path.resolve()), "image_url": meta.get("url"),
                           "width": width, "height": height,
                           "noun": noun, "color": color, "object_id": int(obj["object_id"]), "box": box,
                           "context_crop_box": crop_box(box, width, height),
                           "other_labeled_color_objects": sorted(decoys, key=lambda x: (x["object_id"], x["noun"])),
                           "annotation_status": "explicit singleton target color; other/missing colors UNKNOWN",
                           "relevance_scope": "restricted per-noun, explicit-label gallery"})
    if v2:
        selected = []
        reduced = defaultdict(list)
        for part in ("development", "calibration", "pilot", "reserve"):
            buckets = defaultdict(list)
            for row in rows[part]:
                buckets[(row["noun"], row["color"])].append(row)
            pool = []
            for key, bucket in buckets.items():
                bucket.sort(key=lambda r: hashlib.sha256(f"natural-v2-in-stratum:{r['image_id']}:{key}".encode()).hexdigest())
                pool.extend(bucket[:p["quotas_per_partition_noun_color"][part]])
            pool.sort(key=lambda r: hashlib.sha256(f"natural-v2-global:{r['image_id']}".encode()).hexdigest())
            reduced[part] = pool[:p["total_cap_per_partition"][part]]
            selected.extend(reduced[part])
        rows = reduced
        selection_path = dest / "selection_before_download.jsonl"
        jsonl(selection_path, sorted(selected, key=lambda r: r["id"]))
        with ThreadPoolExecutor(max_workers=8) as executor:
            availability = list(executor.map(_fetch, selected))
        jsonl(dest / "image_availability.jsonl", availability)
        good = {r["id"]: r for r in availability if r["ok"]}
        if len(good) != len(selected):
            raise RuntimeError(f"{len(selected)-len(good)} selected official images unavailable; no replacement performed")
        for part in rows:
            for row in rows[part]:
                row["image_sha256"] = good[row["id"]]["sha256"]
    files = {}
    counts = {}
    for part in ("development", "calibration", "pilot", "reserve"):
        part_rows = sorted(rows[part], key=lambda r: (r["image_id"], r["noun"]))
        path = dest / f"{part}.jsonl"
        jsonl(path, part_rows)
        files[path.name] = sha(path)
        counts[part] = {"images": len(part_rows), "by_color": dict(Counter(r["color"] for r in part_rows)),
                        "by_noun": dict(Counter(r["noun"] for r in part_rows)),
                        "red_blue_calibration_candidates": sum(r["color"] in ("red", "blue") for r in part_rows)}
    # Query definitions are score-blind and can be derived without treating unlabeled images as negatives.
    queries = []
    for part in ("pilot", "reserve"):
        grouped = defaultdict(list)
        for row in rows[part]:
            grouped[row["noun"]].append(row)
        for noun, gallery in grouped.items():
            bycolor = Counter(r["color"] for r in gallery)
            if len(bycolor) < 2:
                continue
            for color, n in sorted(bycolor.items()):
                queries.append({"partition": part, "query_id": f"{part}_{color}_{noun.replace(' ', '_')}",
                                "caption": f"a {color} {noun}", "noun": noun, "color": color,
                                "gallery_ids": sorted(r["id"] for r in gallery), "positive_count": n,
                                "incompatible_count": len(gallery) - n,
                                "scope": "unique explicitly colored target noun only; not open-world full-image relevance"})
    qpath = dest / "queries.jsonl"
    jsonl(qpath, sorted(queries, key=lambda r: r["query_id"]))
    files[qpath.name] = sha(qpath)
    if v2:
        files["selection_before_download.jsonl"] = sha(dest / "selection_before_download.jsonl")
        files["image_availability.jsonl"] = sha(dest / "image_availability.jsonl")
    dump(dest / "freeze.json", {"protocol_sha256": sha(protocol_path), "source_hashes": p["source"],
                               "bank_coordination": bank_info, "files": files, "counts": counts,
                               "queries": dict(Counter(q["partition"] for q in queries)),
                               "rejections": dict(rejected), "no_models_or_outcomes_read": True,
                               "scope": "natural original images with annotated object boxes; optional foreground-with-context crop coordinates; restricted explicit-label retrieval"})
    print(json.dumps({"counts": counts, "queries": len(queries)}, sort_keys=True))


def build_square_v3() -> None:
    dest = OUT / "v3"
    protocol_path = dest / "view_protocol.json"
    p = json.loads(protocol_path.read_text())
    if p["version"] != "interbind_vg_natural_square_view_v3_20260922" or p["score_information_used"] is not False:
        raise ValueError("Invalid score-blind v3 view protocol")
    source = OUT / "v2"
    if sha(source / "freeze.json") != p["input_freeze_sha256"] or sha(source / "queries.jsonl") != p["input_queries_sha256"]:
        raise ValueError("Frozen v2 source changed")
    if (dest / "freeze.json").exists():
        raise FileExistsError("Square v3 already frozen")
    v2_freeze = json.loads((source / "freeze.json").read_text())
    output_files = {}
    counts = {}
    kept_by_id = {}
    rejected = []
    for part in ("development", "calibration", "pilot", "reserve"):
        source_file = source / f"{part}.jsonl"
        if sha(source_file) != v2_freeze["files"][source_file.name]:
            raise ValueError(f"v2 manifest hash changed: {source_file}")
        rows = []
        for line in source_file.read_text().splitlines():
            row = json.loads(line)
            crop = square_context_crop(row["box"], row["width"], row["height"])
            if crop is None:
                rejected.append({"id": row["id"], "partition": part, "reason": "full_target_box_cannot_fit_square"})
                continue
            row["v2_context_crop_box"] = row["context_crop_box"]
            row["context_crop_box"] = crop
            row["view"] = "canonical_square"
            row["full_target_box_preserved"] = True
            rows.append(row)
            kept_by_id[row["id"]] = row
        path = dest / f"{part}.jsonl"
        jsonl(path, rows)
        output_files[path.name] = sha(path)
        counts[part] = {"input": v2_freeze["counts"][part]["images"], "kept": len(rows),
                        "rejected": v2_freeze["counts"][part]["images"] - len(rows),
                        "by_color": dict(Counter(r["color"] for r in rows)),
                        "red_blue_calibration_candidates": sum(r["color"] in ("red", "blue") for r in rows)}
    qrows = []
    rejected_queries = []
    for line in (source / "queries.jsonl").read_text().splitlines():
        q = json.loads(line)
        ids = [rid for rid in q["gallery_ids"] if rid in kept_by_id]
        positives = sum(kept_by_id[rid]["color"] == q["color"] for rid in ids)
        incompatible = len(ids) - positives
        if positives == 0 or incompatible == 0:
            rejected_queries.append({"query_id": q["query_id"], "reason": "no_positive_or_incompatible_after_view_gate",
                                     "positive_count": positives, "incompatible_count": incompatible})
            continue
        q.update(gallery_ids=ids, positive_count=positives, incompatible_count=incompatible,
                 view="canonical_square")
        qrows.append(q)
    qpath = dest / "queries.jsonl"
    jsonl(qpath, qrows)
    output_files[qpath.name] = sha(qpath)
    jsonl(dest / "rejected_rows.jsonl", rejected)
    output_files["rejected_rows.jsonl"] = sha(dest / "rejected_rows.jsonl")
    jsonl(dest / "rejected_queries.jsonl", rejected_queries)
    output_files["rejected_queries.jsonl"] = sha(dest / "rejected_queries.jsonl")
    dump(dest / "freeze.json", {"view_protocol_sha256": sha(protocol_path), "input_v2_freeze_sha256": sha(source / "freeze.json"),
                               "files": output_files, "counts": counts,
                               "queries": dict(Counter(q["partition"] for q in qrows)),
                               "rejected_rows": len(rejected), "rejected_queries": len(rejected_queries),
                               "model_scores_read": False, "no_new_downloads_or_labels": True})
    print(json.dumps({"counts": counts, "queries": len(qrows), "rejected_rows": len(rejected)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "build", "freeze-v2", "build-v2", "build-v3"))
    parser.add_argument("--exclude-json", type=Path, help="consolidated protected COCO source IDs; required for freeze-v2")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log(shlex.join([sys.executable, "-m", "mirror.cases.color_binding.natural_data", *sys.argv[1:]]))
    if args.action == "freeze":
        dump(OUT / "protocol.json", rules())
    elif args.action == "freeze-v2":
        if args.exclude_json is None or not args.exclude_json.is_file():
            parser.error("freeze-v2 requires --exclude-json")
        dump(OUT / "v2/protocol.json", rules_v2(args.exclude_json))
    elif args.action == "build-v2":
        build(v2=True)
    elif args.action == "build-v3":
        build_square_v3()
    else:
        build()


if __name__ == "__main__":
    main()
