"""Bounded, outcome-free feature caches for frozen InterBind inputs.

The caller owns bank selection, renderer, and captions. This module never forms
image/text scores, selects examples, trains, or loads a model. Each destination
is create-only; ``complete.json`` is written last and verifies every artifact.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from PIL import Image

from mirror.core.encoders import file_sha256; from mirror.core.encoders import read_registry


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical(row) + "\n")


def _write_array(path, value):
    with Path(path).open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def _checked_features(value, count, width=None):
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] != count or tensor.shape[1] < 1:
        raise ValueError("Encoder returned wrong feature shape")
    if width is not None and tensor.shape[1] != width:
        raise ValueError("Image/text feature widths differ")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Encoder returned nonfinite features")
    return tensor.numpy().copy()


def _verify_hashes(paths):
    if not isinstance(paths, dict) or not paths:
        raise ValueError("Nonempty input path-to-SHA256 mapping is required")
    for name, expected in paths.items():
        if not isinstance(expected, str) or len(expected) != 64 or file_sha256(name) != expected:
            raise ValueError(f"Missing or changed input: {name}")


def model_binding(scorer, registry_path, activation_addendum_path=None):
    """Require exact registry ID, activation addendum, and local checkpoint bytes."""
    registry_path = Path(registry_path)
    addendum = Path(activation_addendum_path or registry_path.with_name("activation_addendum.json"))
    if addendum != registry_path.with_name("activation_addendum.json") or not addendum.is_file():
        raise ValueError("Matching activation addendum is mandatory")
    registry_sha = file_sha256(registry_path)
    activation_sha = file_sha256(addendum)
    registry = read_registry(registry_path)
    subject = scorer.subject
    row = next((r for r in registry["subjects"] if r["id"] == subject.get("id")), None)
    if row is None or row != subject or row.get("status") != "ready" or not row.get("activation"):
        raise ValueError("Scorer metadata does not match a ready, activation-pinned registry subject")
    _verify_hashes({entry["path"]: entry["sha256"] for entry in row["files"]})
    return {"subject_id": row["id"], "registry_path": str(registry_path.resolve()),
            "registry_sha256": registry_sha, "activation_addendum_sha256": activation_sha,
            "activation": row["activation"], "subject": row,
            "checkpoint_files": row["files"]}


def verify_cache(path, *, binding=None, input_hashes=None):
    """Read-only integrity check; optional expected model and source binding."""
    path = Path(path)
    complete = json.loads((path / "complete.json").read_text())
    protocol = json.loads((path / "protocol.json").read_text())
    if binding is not None and protocol["model_binding"] != binding:
        raise ValueError("Cache model binding changed")
    if input_hashes is not None and protocol["input_hashes"] != input_hashes:
        raise ValueError("Cache input binding changed")
    for name, expected in complete["files"].items():
        if file_sha256(path / name) != expected:
            raise ValueError(f"Cache artifact changed: {name}")
    if protocol["kind"] == "natural_gallery":
        for line in (path / "gallery_index.jsonl").read_text().splitlines():
            row = json.loads(line)
            if file_sha256(row["image_path"]) != row["image_sha256"]:
                raise ValueError(f"Natural source image changed: {row['image_path']}")
    _verify_hashes(protocol["input_hashes"])
    _verify_hashes({entry["path"]: entry["sha256"]
                    for entry in protocol["model_binding"]["checkpoint_files"]})
    if file_sha256(protocol["model_binding"]["registry_path"]) != protocol["model_binding"]["registry_sha256"]:
        raise ValueError("Registry changed")
    addendum = Path(protocol["model_binding"]["registry_path"]).with_name("activation_addendum.json")
    if file_sha256(addendum) != protocol["model_binding"]["activation_addendum_sha256"]:
        raise ValueError("Activation addendum changed")
    return protocol, complete


def _pixel_sha(image):
    array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("Renderer must return RGB uint8 images")
    return sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def _image_batches(scorer, images, batch_size):
    return _checked_features(scorer.encode_images(images, batch_size=batch_size), len(images))


def _finish(out, *, kind, scorer, binding, input_hashes, index, image_features,
            text_groups, image_forward_count, text_batch_size, started, details,
            extra_files=()):
    unique = []
    indices = {}
    for group in text_groups:
        key = _canonical(group)
        if key not in indices:
            indices[key] = len(unique)
            unique.append(group)
    texts = _checked_features(scorer.encode_texts(unique, batch_size=text_batch_size), len(unique),
                              width=image_features.shape[1])
    for row in index:
        row["text_indices"] = [indices[_canonical(group)] for group in row.pop("_text_groups")]
    protocol = {"schema_version": 1, "kind": kind, "model_binding": binding,
                "input_hashes": input_hashes, "details": details,
                "feature_rule": "scorer.encode_images/encode_texts; CPU float32; no score matrix",
                "template_group_key": "canonical JSON array; exact order and text preserved",
                "no_scores_or_training": True}
    _write_json(out / "protocol.json", protocol)
    _write_jsonl(out / "index.jsonl", index)
    _write_jsonl(out / "text_groups.jsonl", [{"index": i, "templates": group} for i, group in enumerate(unique)])
    _write_array(out / "images.npy", image_features)
    _write_array(out / "texts.npy", texts)
    elapsed = time.monotonic() - started
    cuda_peak = None
    if torch.cuda.is_available() and getattr(scorer.device, "type", None) == "cuda":
        cuda_peak = int(torch.cuda.max_memory_allocated(scorer.device))
    files = {name: file_sha256(out / name) for name in
             ("protocol.json", "index.jsonl", "text_groups.jsonl", "images.npy", "texts.npy", *extra_files)}
    _write_json(out / "complete.json", {"files": files, "n_images": len(image_features),
                "n_unique_text_groups": len(unique), "elapsed_seconds": elapsed,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_cuda_bytes": cuda_peak, "image_forward_batch_count": image_forward_count,
                "text_forward_batch_count": (sum(map(len, unique)) + text_batch_size - 1) // text_batch_size,
                "score_forward_count": 0, "no_scores_or_training": True})
    return out


def cache_bank(out, rows, scorer, renderer, captions, *, registry_path,
               input_hashes, image_batch_size=32, text_batch_size=64, render_workers=0,
               details=None):
    """Cache all ordered bank states. Renderer returns (images, names, pixel_hashes).

    Captions receives ``(family, objects)`` and returns ordered template groups.
    Render workers are bounded to two anchors per worker; GPU encoding remains
    on the calling thread. No scorer ``scores`` method is called.
    """
    started = time.monotonic()
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if image_batch_size < 1 or text_batch_size < 1 or render_workers < 0:
        raise ValueError("Invalid batch or worker size")
    _verify_hashes(input_hashes)
    binding = model_binding(scorer, registry_path)
    rows = list(rows)
    if not rows:
        raise ValueError("Empty bank")
    ids = [row["anchor_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate anchor ID")
    if torch.cuda.is_available() and getattr(scorer.device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats(scorer.device)
    index, features, groups, pending_images = [], [], [], []
    count = forward_batches = 0
    def flush_images():
        nonlocal forward_batches
        if pending_images:
            features.append(_image_batches(scorer, pending_images, image_batch_size))
            forward_batches += (len(pending_images) + image_batch_size - 1) // image_batch_size
            pending_images.clear()
    def consume(row, rendered):
        nonlocal count
        images, names, hashes = rendered
        images, names, hashes = list(images), list(names), list(hashes)
        if not images or len(images) != len(names) or len(images) != len(hashes) or len(set(names)) != len(names):
            raise ValueError("Renderer state count or names invalid")
        if [_pixel_sha(im) for im in images] != hashes:
            raise ValueError("Renderer pixel hash mismatch")
        prompts = [list(g) for g in captions(row["family"], row["objects"])]
        if not prompts or any(not g or any(not isinstance(t, str) or not t for t in g) for g in prompts):
            raise ValueError("Invalid template groups")
        pending_images.extend(images)
        if len(pending_images) >= image_batch_size:
            flush_images()
        index.append({"anchor_id": row["anchor_id"], "family": row["family"],
                      "image_offset": count, "image_count": len(images), "state_names": names,
                      "pixel_sha256": hashes, "_text_groups": prompts})
        groups.extend(prompts)
        count += len(images)
    if render_workers:
        with ThreadPoolExecutor(max_workers=render_workers) as pool:
            for start in range(0, len(rows), 2 * render_workers):
                chunk = rows[start:start + 2 * render_workers]
                for row, rendered in zip(chunk, pool.map(renderer, chunk)):
                    consume(row, rendered)
    else:
        for row in rows:
            consume(row, renderer(row))
    flush_images()
    image_features = np.concatenate(features, axis=0)
    out.mkdir(parents=True, exist_ok=False)
    return _finish(out, kind="bank", scorer=scorer, binding=binding, input_hashes=input_hashes,
                   index=index, image_features=image_features, text_groups=groups,
                   image_forward_count=forward_batches, text_batch_size=text_batch_size,
                   started=started, details=details or {})


def cache_natural_gallery(out, rows, queries, scorer, *, registry_path, input_hashes,
                          view, loader=None, image_batch_size=32, text_batch_size=64,
                          details=None):
    """Cache one natural-gallery view and every supplied query in supplied order.

    ``view`` is ``full_image`` or ``foreground_context``; call separately for
    both views. Rows retain IDs, frozen crop coordinates, and source hashes.
    """
    from mirror.cases.color_binding.natural_data import load_gallery_image
    started = time.monotonic()
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if view not in ("full_image", "foreground_context") or image_batch_size < 1 or text_batch_size < 1:
        raise ValueError("Invalid view or batch size")
    _verify_hashes(input_hashes)
    binding = model_binding(scorer, registry_path)
    rows, queries = list(rows), list(queries)
    if not rows or not queries:
        raise ValueError("Empty natural gallery or query set")
    ids = [row["id"] for row in rows]
    qids = [query["query_id"] for query in queries]
    if len(set(ids)) != len(ids) or len(set(qids)) != len(qids):
        raise ValueError("Duplicate gallery or query ID")
    known = set(ids)
    for row in rows:
        if file_sha256(row["image_path"]) != row["image_sha256"]:
            raise ValueError("Natural source image changed")
        if view == "foreground_context" and not row.get("context_crop_box"):
            raise ValueError("Missing frozen context crop")
    for query in queries:
        if not set(query["gallery_ids"]) <= known:
            raise ValueError("Query refers to missing gallery row")
    load = loader or load_gallery_image
    index, chunks, forward_batches = [], [], 0
    for offset in range(0, len(rows), image_batch_size):
        block = rows[offset:offset + image_batch_size]
        images = [load(row, foreground_context=(view == "foreground_context")) for row in block]
        chunks.append(_image_batches(scorer, images, image_batch_size))
        forward_batches += 1
        for row, image in zip(block, images):
            index.append({"row_id": row["id"], "image_offset": len(index), "view": view,
                          "image_path": row["image_path"], "image_sha256": row["image_sha256"],
                          "crop_box": row.get("context_crop_box") if view == "foreground_context" else None,
                          "view_pixel_sha256": _pixel_sha(image)})
    image_features = np.concatenate(chunks, axis=0)
    groups = [[query["caption"]] for query in queries]
    query_index = [{"query_id": q["query_id"], "gallery_ids": q["gallery_ids"],
                    "partition": q["partition"], "noun": q["noun"], "color": q["color"],
                    "_text_groups": [group]} for q, group in zip(queries, groups)]
    out.mkdir(parents=True, exist_ok=False)
    _write_jsonl(out / "gallery_index.jsonl", index)
    _finish(out, kind="natural_gallery", scorer=scorer, binding=binding,
            input_hashes=input_hashes, index=query_index, image_features=image_features,
            text_groups=groups, image_forward_count=forward_batches,
            text_batch_size=text_batch_size, started=started,
            details={**(details or {}), "view": view}, extra_files=("gallery_index.jsonl",))
    return out
