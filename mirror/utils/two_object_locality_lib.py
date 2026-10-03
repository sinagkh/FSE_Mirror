#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader; from torch.utils.data import Dataset

import open_clip
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


COLOR_RGB = {
    "red": (220, 30, 30),
    "blue": (30, 90, 220),
    "green": (35, 170, 80),
    "yellow": (235, 205, 35),
    "purple": (135, 70, 190),
    "orange": (235, 130, 35),
}

STATE_KEYS = ["rr", "rb", "br", "bb"]

COCO_NAME_MAP = {
    "airplane": "airplane",
    "ball": "sports ball",
    "bike": "bicycle",
    "couch": "couch",
    "fire_hydrant": "fire hydrant",
    "sofa": "couch",
    "stop_sign": "stop sign",
    "traffic_light": "traffic light",
    "tv": "tv",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unit_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)


def canonical_class(name: str) -> str:
    key = name.strip().replace("_", " ")
    return COCO_NAME_MAP.get(name.strip(), COCO_NAME_MAP.get(key, key))


def parse_object_pairs(value: str) -> list[tuple[str, str]]:
    if not value:
        value = "car:boat,dog:ball,person:bicycle,cat:couch,bus:airplane"
    chunks = [c.strip() for c in value.replace("|", ",").split(",") if c.strip()]
    pairs: list[tuple[str, str]] = []
    for chunk in chunks:
        if ":" in chunk:
            a, b = chunk.split(":", 1)
        elif "/" in chunk:
            a, b = chunk.split("/", 1)
        else:
            parts = chunk.split()
            if len(parts) != 2:
                continue
            a, b = parts
        pairs.append((canonical_class(a), canonical_class(b)))
    if not pairs:
        raise ValueError(f"No valid object pairs parsed from: {value!r}")
    return pairs


def parse_colors(value: str) -> tuple[str, str]:
    colors = [c.strip().lower() for c in value.replace("|", ",").split(",") if c.strip()]
    if len(colors) != 2:
        raise ValueError(f"Expected exactly two colors, got: {value!r}")
    unknown = [c for c in colors if c not in COLOR_RGB]
    if unknown:
        raise ValueError(f"Unknown colors {unknown}; available colors: {sorted(COLOR_RGB)}")
    return colors[0], colors[1]


def state_colors(colors: tuple[str, str]) -> dict[str, tuple[str, str]]:
    c0, c1 = colors
    return {
        "rr": (c0, c0),
        "rb": (c0, c1),
        "br": (c1, c0),
        "bb": (c1, c1),
    }


def ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = maskUtils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return maskUtils.decode(rle).astype(bool)


class LRU:
    def __init__(self, cap: int = 256):
        self.cap = cap
        self.data: OrderedDict[str, Image.Image | np.ndarray] = OrderedDict()

    def get(self, key: str):
        value = self.data.get(key)
        if value is not None:
            self.data.move_to_end(key)
        return value

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.cap:
            self.data.popitem(last=False)


def crop_rgba(img: Image.Image, mask: np.ndarray, pad: int = 4) -> Image.Image | None:
    yx = np.argwhere(mask)
    if yx.size == 0:
        return None
    y0, x0 = yx.min(axis=0)
    y1, x1 = yx.max(axis=0) + 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0], y1 + pad)
    x1 = min(mask.shape[1], x1 + pad)
    crop = img.crop((x0, y0, x1, y1))
    mask_crop = mask[y0:y1, x0:x1]
    rgba = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    rgba.paste(crop, (0, 0), Image.fromarray((mask_crop * 255).astype(np.uint8)))
    return rgba


def fit_rgba(img: Image.Image, target: int = 112) -> Image.Image:
    w, h = img.size
    scale = target / max(w, h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def paste_pair_canvas(
    o1_rgba: Image.Image,
    o2_rgba: Image.Image,
    *,
    canvas: int = 256,
    object_size: int = 112,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    o1 = fit_rgba(o1_rgba, target=object_size)
    o2 = fit_rgba(o2_rgba, target=object_size)
    base = Image.new("RGB", (canvas, canvas), (235, 235, 235))
    mask1 = np.zeros((canvas, canvas), dtype=bool)
    mask2 = np.zeros((canvas, canvas), dtype=bool)
    pad = 16
    x1 = pad
    y1 = (canvas - o1.size[1]) // 2
    x2 = canvas - pad - o2.size[0]
    y2 = (canvas - o2.size[1]) // 2
    base.paste(o1, (x1, y1), o1)
    base.paste(o2, (x2, y2), o2)
    a1 = np.array(o1.getchannel("A")) > 0
    a2 = np.array(o2.getchannel("A")) > 0
    mask1[y1 : y1 + o1.size[1], x1 : x1 + o1.size[0]] = a1
    mask2[y2 : y2 + o2.size[1], x2 : x2 + o2.size[0]] = a2
    return base, mask1, mask2


def tint_two_objects(
    base: Image.Image,
    mask1: np.ndarray,
    mask2: np.ndarray,
    color1: str,
    color2: str,
    *,
    alpha: float = 0.55,
) -> Image.Image:
    arr = np.array(base.convert("RGB")).astype(np.float32)
    for mask, color_name in [(mask1, color1), (mask2, color2)]:
        rgb = np.array(COLOR_RGB[color_name], dtype=np.float32).reshape(1, 1, 3)
        m = mask[..., None].astype(np.float32)
        arr = (1.0 - m) * arr + m * ((1.0 - alpha) * arr + alpha * rgb)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def resolve_cid(coco: COCO, name: str) -> int | None:
    cname = canonical_class(name)
    ids = coco.getCatIds(catNms=[cname])
    if ids:
        return ids[0]
    compact = cname.replace(" ", "")
    for cat in coco.loadCats(coco.getCatIds()):
        if cat["name"].replace(" ", "") == compact:
            return cat["id"]
    return None


class TwoObjectLocalityStream(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_json: str,
        preprocess,
        object_pairs: list[tuple[str, str]],
        *,
        n_samples: int,
        min_area_ratio: float = 0.02,
        seed: int = 7,
        canvas: int = 256,
        object_size: int = 112,
        tint_alpha: float = 0.55,
        colors: tuple[str, str] = ("red", "blue"),
        position_swap_prob: float = 0.0,
    ):
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_json)
        self.preprocess = preprocess
        self.object_pairs = object_pairs
        self.n_samples = n_samples
        self.min_area_ratio = min_area_ratio
        self.rng = random.Random(seed)
        self.canvas = canvas
        self.object_size = object_size
        self.tint_alpha = tint_alpha
        self.colors = colors
        if not 0.0 <= position_swap_prob <= 1.0:
            raise ValueError("position_swap_prob must be in [0, 1]")
        self.position_swap_prob = position_swap_prob
        self.state_colors = state_colors(colors)
        self.img_cache = LRU(256)
        self.mask_cache = LRU(512)
        self.pool = self._build_pool()

    def _build_pool(self) -> dict[str, list[tuple[dict, dict]]]:
        classes = sorted({c for pair in self.object_pairs for c in pair})
        pool: dict[str, list[tuple[dict, dict]]] = {}
        for cname in classes:
            cid = resolve_cid(self.coco, cname)
            if cid is None:
                pool[cname] = []
                continue
            rows: list[tuple[dict, dict]] = []
            for img_id in self.coco.getImgIds(catIds=[cid]):
                img = self.coco.loadImgs([img_id])[0]
                anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id], catIds=[cid]))
                if not anns:
                    continue
                best = max(anns, key=lambda ann: ann.get("area", 0))
                area_ratio = best.get("area", 0) / max(1, img["width"] * img["height"])
                if area_ratio >= self.min_area_ratio:
                    rows.append((img, best))
            pool[cname] = rows
            print(f"[pool] {cname}: {len(rows)} instances")
        valid = [p for p in self.object_pairs if pool.get(p[0]) and pool.get(p[1])]
        if not valid:
            raise ValueError("No valid object pairs have non-empty COCO pools.")
        self.object_pairs = valid
        return pool

    def __len__(self) -> int:
        return self.n_samples

    def _pil(self, img: dict) -> Image.Image:
        key = img["file_name"]
        cached = self.img_cache.get(key)
        if cached is None:
            cached = Image.open(self.img_dir / key).convert("RGB")
            self.img_cache.put(key, cached)
        return cached.copy()

    def _mask(self, ann: dict, h: int, w: int) -> np.ndarray:
        key = str(ann["id"])
        cached = self.mask_cache.get(key)
        if cached is None:
            cached = ann_to_mask(ann, h, w)
            self.mask_cache.put(key, cached)
        return np.array(cached, copy=True)

    def _sample_rgba(self, cname: str) -> Image.Image | None:
        for _ in range(20):
            img, ann = self.rng.choice(self.pool[cname])
            pil = self._pil(img)
            mask = self._mask(ann, img["height"], img["width"])
            rgba = crop_rgba(pil, mask)
            if rgba is not None and min(rgba.size) >= 6:
                return rgba
        return None

    def __getitem__(self, idx: int):
        for _ in range(50):
            o1, o2 = self.rng.choice(self.object_pairs)
            r1 = self._sample_rgba(o1)
            r2 = self._sample_rgba(o2)
            if r1 is None or r2 is None:
                continue
            # Preserve the submitted RNG stream exactly when augmentation is off.
            if self.position_swap_prob > 0.0 and self.rng.random() < self.position_swap_prob:
                # Put semantic object 2 on the left and object 1 on the right,
                # while retaining mask1/object1 and mask2/object2 identities.
                base, mask2, mask1 = paste_pair_canvas(
                    r2,
                    r1,
                    canvas=self.canvas,
                    object_size=self.object_size,
                )
            else:
                base, mask1, mask2 = paste_pair_canvas(
                    r1,
                    r2,
                    canvas=self.canvas,
                    object_size=self.object_size,
                )
            imgs = []
            for key in STATE_KEYS:
                c1, c2 = self.state_colors[key]
                imgs.append(
                    self.preprocess(
                        tint_two_objects(base, mask1, mask2, c1, c2, alpha=self.tint_alpha)
                    )
                )
            return torch.stack(imgs, dim=0), o1, o2
        raise RuntimeError("Failed to sample a valid two-object anchor.")


class TextLoRAAdapter(nn.Module):
    def __init__(self, d: int, r: int = 64, alpha: int = 64, dropout: float = 0.0):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(dropout)
        self.scale = alpha / r

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return unit_norm(t + self.scale * self.B(self.drop(self.A(t))))


def caption_templates(o1: str, o2: str, c1: str, c2: str) -> list[str]:
    return [
        f"a {c1} {o1} and a {c2} {o2}",
        f"a {c1} {o1} next to a {c2} {o2}",
        f"the {o1} is {c1} and the {o2} is {c2}",
    ]


def object_templates(o1: str, o2: str) -> list[str]:
    return [
        f"a photo of a {o1} and a {o2}",
        f"a {o1} next to a {o2}",
    ]


class TextBank:
    def __init__(self, model, tokenizer, adapter: TextLoRAAdapter | None, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.device = device
        self.cache: dict[tuple, torch.Tensor] = {}

    def base_mean(self, key: tuple, prompts: list[str]) -> torch.Tensor:
        if key not in self.cache:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                toks = self.tokenizer(prompts).to(self.device)
                emb = unit_norm(self.model.encode_text(toks)).mean(dim=0, keepdim=True)
                self.cache[key] = unit_norm(emb.detach())
        return self.cache[key]

    def adapted(self, key: tuple, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_mean(key, prompts)
        if self.adapter is None:
            return base, base
        return base, unit_norm(self.adapter(base))


def load_model(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.arch,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.arch)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    dim = model.text_projection.shape[1] if hasattr(model, "text_projection") else model.text_output_dim
    return model, tokenizer, preprocess, device, dim


def load_adapter(path: str, dim: int, args: argparse.Namespace, device: torch.device) -> TextLoRAAdapter | None:
    if not path:
        return None
    raw = torch.load(path, map_location=device)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    meta_args = raw.get("args", {}) if isinstance(raw, dict) else {}
    rank = int(meta_args.get("rank", args.rank))
    alpha = int(meta_args.get("alpha", args.alpha))
    adapter = TextLoRAAdapter(dim, r=rank, alpha=alpha, dropout=0.0).to(device)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    print(f"[adapter] loaded {path} rank={rank} alpha={alpha}")
    return adapter


def make_texts(
    text_bank: TextBank,
    o1s: Iterable[str],
    o2s: Iterable[str],
    colors: tuple[str, str] = ("red", "blue"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    color_states = state_colors(colors)
    base_caps = []
    adapted_caps = []
    base_obj = []
    adapted_obj = []
    for o1, o2 in zip(o1s, o2s):
        caps_base = []
        caps_adapted = []
        for key in STATE_KEYS:
            c1, c2 = color_states[key]
            base, adapted = text_bank.adapted(("cap", o1, o2, c1, c2), caption_templates(o1, o2, c1, c2))
            caps_base.append(base.squeeze(0))
            caps_adapted.append(adapted.squeeze(0))
        obj_base, obj_adapted = text_bank.adapted(("obj", o1, o2), object_templates(o1, o2))
        base_caps.append(torch.stack(caps_base, dim=0))
        adapted_caps.append(torch.stack(caps_adapted, dim=0))
        base_obj.append(obj_base.squeeze(0))
        adapted_obj.append(obj_adapted.squeeze(0))
    return (
        torch.stack(base_caps, dim=0),
        torch.stack(adapted_caps, dim=0),
        torch.stack(base_obj, dim=0),
        torch.stack(adapted_obj, dim=0),
    )


def locality_metrics(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    # scores: [B, 4 image states, 4 text states], order rr, rb, br, bb.
    rr, rb, br, bb = 0, 1, 2, 3
    d_o1_diag = scores[:, bb, bb] - scores[:, bb, rb] - scores[:, rb, bb] + scores[:, rb, rb]
    d_o2_diag = scores[:, rb, rb] - scores[:, rb, rr] - scores[:, rr, rb] + scores[:, rr, rr]
    d_o1_to_t2 = scores[:, bb, rb] - scores[:, bb, rr] - scores[:, rb, rb] + scores[:, rb, rr]
    d_o2_to_t1 = scores[:, rb, bb] - scores[:, rb, rb] - scores[:, rr, bb] + scores[:, rr, rb]

    m_o1_red = scores[:, rb, rb] - scores[:, rb, bb]
    m_o1_blue = scores[:, bb, bb] - scores[:, bb, rb]
    m_o2_red = scores[:, rr, rr] - scores[:, rr, rb]
    m_o2_blue = scores[:, rb, rb] - scores[:, rb, rr]

    correct = torch.arange(4, device=scores.device).view(1, 4)
    pred = scores.argmax(dim=2)
    caption_state_acc = (pred == correct).float()
    caption_all = caption_state_acc.prod(dim=1)
    bli = (d_o1_diag.abs() + d_o2_diag.abs()) / (1e-6 + d_o1_to_t2.abs() + d_o2_to_t1.abs())
    return {
        "d_o1_diag": d_o1_diag,
        "d_o2_diag": d_o2_diag,
        "d_o1_to_t2": d_o1_to_t2,
        "d_o2_to_t1": d_o2_to_t1,
        "m_o1_red": m_o1_red,
        "m_o1_blue": m_o1_blue,
        "m_o2_red": m_o2_red,
        "m_o2_blue": m_o2_blue,
        "caption_acc": caption_state_acc.mean(dim=1),
        "caption_all_acc": caption_all,
        "bli": bli,
    }


def summarize_rows(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {"n": 0}
    out: dict[str, float | int] = {"n": len(rows)}
    numeric_keys = [
        "d_o1_diag",
        "d_o2_diag",
        "d_o1_to_t2",
        "d_o2_to_t1",
        "abs_off_mean",
        "bli",
        "caption_acc",
        "caption_all_acc",
        "object_guard",
    ]
    for key in numeric_keys:
        vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_median"] = float(np.median(vals))
        out[f"{key}_p90"] = float(np.quantile(vals, 0.9))
    out["diag_signed_median"] = float(
        np.median(
            np.concatenate(
                [
                    np.array([float(r["d_o1_diag"]) for r in rows]),
                    np.array([float(r["d_o2_diag"]) for r in rows]),
                ]
            )
        )
    )
    out["off_abs_median"] = float(np.median(np.array([float(r["abs_off_mean"]) for r in rows])))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_rows(args: argparse.Namespace, adapter_path: str = "") -> tuple[list[dict], dict]:
    set_seed(args.seed)
    pairs = parse_object_pairs(args.object_pairs)
    colors = parse_colors(args.colors)
    model, tokenizer, preprocess, device, dim = load_model(args)
    adapter = load_adapter(adapter_path, dim, args, device)
    text_bank = TextBank(model, tokenizer, adapter, device)
    ds = TwoObjectLocalityStream(
        args.img_dir,
        args.ann_json,
        preprocess,
        pairs,
        n_samples=args.samples,
        min_area_ratio=args.min_area_ratio,
        seed=args.seed,
        canvas=args.canvas,
        object_size=args.object_size,
        tint_alpha=args.tint_alpha,
        colors=colors,
        position_swap_prob=args.position_swap_prob,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    rows: list[dict] = []
    pair_cycle = pairs
    with torch.no_grad():
        for batch_idx, (imgs, o1s, o2s) in enumerate(loader):
            bsz = imgs.shape[0]
            flat = imgs.view(bsz * 4, *imgs.shape[2:]).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                image_emb = unit_norm(model.encode_image(flat)).view(bsz, 4, -1)
            _, text_emb, _, obj_emb = make_texts(text_bank, o1s, o2s, colors)
            text_emb = text_emb.to(device)
            obj_emb = obj_emb.to(device)
            scores = torch.einsum("bid,bjd->bij", image_emb, text_emb)
            metrics = locality_metrics(scores)

            wrong_embs = []
            for i, (o1, o2) in enumerate(zip(o1s, o2s)):
                wrong = pair_cycle[(batch_idx * args.batch + i + 1) % len(pair_cycle)]
                if wrong == (o1, o2) and len(pair_cycle) > 1:
                    wrong = pair_cycle[(batch_idx * args.batch + i + 2) % len(pair_cycle)]
                _, wrong_obj = text_bank.adapted(("obj", wrong[0], wrong[1]), object_templates(wrong[0], wrong[1]))
                wrong_embs.append(wrong_obj.squeeze(0))
            wrong_obj_emb = torch.stack(wrong_embs, dim=0).to(device)
            obj_score = cosine(image_emb.mean(dim=1), obj_emb)
            wrong_score = cosine(image_emb.mean(dim=1), wrong_obj_emb)
            object_guard = (obj_score > wrong_score).float()

            for i in range(bsz):
                row = {
                    "o1": o1s[i],
                    "o2": o2s[i],
                    "color0": colors[0],
                    "color1": colors[1],
                    "d_o1_diag": float(metrics["d_o1_diag"][i].cpu()),
                    "d_o2_diag": float(metrics["d_o2_diag"][i].cpu()),
                    "d_o1_to_t2": float(metrics["d_o1_to_t2"][i].cpu()),
                    "d_o2_to_t1": float(metrics["d_o2_to_t1"][i].cpu()),
                    "abs_off_mean": float(
                        0.5
                        * (
                            metrics["d_o1_to_t2"][i].abs()
                            + metrics["d_o2_to_t1"][i].abs()
                        ).cpu()
                    ),
                    "bli": float(metrics["bli"][i].cpu()),
                    "caption_acc": float(metrics["caption_acc"][i].cpu()),
                    "caption_all_acc": float(metrics["caption_all_acc"][i].cpu()),
                    "object_guard": float(object_guard[i].cpu()),
                    "obj_score": float(obj_score[i].cpu()),
                    "wrong_obj_score": float(wrong_score[i].cpu()),
                }
                for img_key_idx, img_key in enumerate(STATE_KEYS):
                    for text_key_idx, text_key in enumerate(STATE_KEYS):
                        row[f"s_{img_key}_{text_key}"] = float(scores[i, img_key_idx, text_key_idx].cpu())
                rows.append(row)
    return rows, summarize_rows(rows)


def train_two_object_locality(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    pairs = parse_object_pairs(args.object_pairs)
    colors = parse_colors(args.colors)
    model, tokenizer, preprocess, device, dim = load_model(args)
    adapter = TextLoRAAdapter(dim, r=args.rank, alpha=args.alpha, dropout=args.dropout).to(device)
    text_bank = TextBank(model, tokenizer, adapter, device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    summary_dir = out_dir / "summaries"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    history = []
    for epoch in range(1, args.epochs + 1):
        ds = TwoObjectLocalityStream(
            args.img_dir,
            args.ann_json,
            preprocess,
            pairs,
            n_samples=args.samples,
            min_area_ratio=args.min_area_ratio,
            seed=args.seed + epoch,
            canvas=args.canvas,
            object_size=args.object_size,
            tint_alpha=args.tint_alpha,
            colors=colors,
            position_swap_prob=args.position_swap_prob,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
        )
        adapter.train()
        logs: dict[str, list[float]] = {
            "loss": [],
            "diag": [],
            "diag_high": [],
            "off": [],
            "agree": [],
            "anchor": [],
            "caption": [],
            "caption_loss": [],
        }
        for step, (imgs, o1s, o2s) in enumerate(loader, start=1):
            bsz = imgs.shape[0]
            flat = imgs.view(bsz * 4, *imgs.shape[2:]).to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                image_emb = unit_norm(model.encode_image(flat)).view(bsz, 4, -1)
            base_text, text_emb, base_obj, obj_emb = make_texts(text_bank, o1s, o2s, colors)
            base_text = base_text.to(device)
            text_emb = text_emb.to(device)
            base_obj = base_obj.to(device)
            obj_emb = obj_emb.to(device)
            scores = torch.einsum("bid,bjd->bij", image_emb, text_emb)
            metrics = locality_metrics(scores)

            diag = 0.5 * (metrics["d_o1_diag"] + metrics["d_o2_diag"])
            off_abs = 0.5 * (metrics["d_o1_to_t2"].abs() + metrics["d_o2_to_t1"].abs())
            loss_diag = (
                F.relu(args.K_diag - metrics["d_o1_diag"]).pow(2).mean()
                + F.relu(args.K_diag - metrics["d_o2_diag"]).pow(2).mean()
            )
            if args.K_diag_high > 0:
                loss_diag_high = (
                    F.relu(metrics["d_o1_diag"] - args.K_diag_high).pow(2).mean()
                    + F.relu(metrics["d_o2_diag"] - args.K_diag_high).pow(2).mean()
                )
            else:
                loss_diag_high = torch.zeros((), device=device)
            loss_off = (
                F.relu(metrics["d_o1_to_t2"].abs() - args.K_off).pow(2).mean()
                + F.relu(metrics["d_o2_to_t1"].abs() - args.K_off).pow(2).mean()
            )
            margins = torch.stack(
                [
                    metrics["m_o1_red"],
                    metrics["m_o1_blue"],
                    metrics["m_o2_red"],
                    metrics["m_o2_blue"],
                ],
                dim=0,
            )
            loss_agree = F.relu(args.m_agree - margins).pow(2).mean()
            loss_anchor = (1.0 - cosine(text_emb.reshape(-1, text_emb.shape[-1]), base_text.reshape(-1, base_text.shape[-1]))).mean()
            loss_anchor = loss_anchor + (1.0 - cosine(obj_emb, base_obj)).mean()
            loss_caption = F.cross_entropy(scores.reshape(bsz * 4, 4), torch.arange(4, device=device).repeat(bsz))
            if args.objective == "caption_ranking":
                loss = loss_caption + args.lam_anchor * loss_anchor
            else:
                loss = (
                    loss_diag
                    + args.lam_diag_high * loss_diag_high
                    + args.lam_off * loss_off
                    + args.lam_guard * loss_agree
                    + args.lam_anchor * loss_anchor
                    + args.lam_caption * loss_caption
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            logs["loss"].append(float(loss.item()))
            logs["diag"].append(float(diag.detach().mean().cpu()))
            logs["diag_high"].append(float(loss_diag_high.detach().cpu()))
            logs["off"].append(float(off_abs.detach().mean().cpu()))
            logs["agree"].append(float((margins > 0).float().mean().detach().cpu()))
            logs["anchor"].append(float(loss_anchor.detach().cpu()))
            logs["caption"].append(float(metrics["caption_acc"].mean().detach().cpu()))
            logs["caption_loss"].append(float(loss_caption.detach().cpu()))
            if args.log_every > 0 and step % args.log_every == 0:
                print(
                    f"[epoch {epoch:02d} step {step:04d}/{len(loader):04d}] "
                    f"loss={np.mean(logs['loss']):.4f} diag={np.mean(logs['diag']):.4f} "
                    f"off={np.mean(logs['off']):.4f} caption={np.mean(logs['caption']):.3f} "
                    f"ce={np.mean(logs['caption_loss']):.4f}"
                )

        adapter.eval()
        val_args = argparse.Namespace(**vars(args))
        val_args.img_dir = args.val_img_dir or args.img_dir
        val_args.ann_json = args.val_ann_json or args.ann_json
        val_args.samples = max(1, args.val_samples)
        tmp_path = ckpt_dir / "_tmp_epoch.pt"
        torch.save({"state_dict": adapter.state_dict(), "args": vars(args)}, tmp_path)
        rows, summary = audit_rows(val_args, str(tmp_path))
        tmp_path.unlink(missing_ok=True)
        routing_pass = float(
            np.mean(
                [
                    (
                        float(r["d_o1_diag"]) > args.K_diag
                        and float(r["d_o2_diag"]) > args.K_diag
                        and abs(float(r["d_o1_to_t2"])) < args.K_off
                        and abs(float(r["d_o2_to_t1"])) < args.K_off
                    )
                    for r in rows
                ]
            )
            if rows
            else 0.0
        )
        summary["routing_pass_mean"] = routing_pass
        if args.objective == "caption_ranking":
            score = float(summary["caption_acc_mean"]) + 0.25 * float(summary["caption_all_acc_mean"])
        elif args.selection_mode == "behavior_routing":
            score = (
                args.selection_caption_weight * float(summary["caption_acc_mean"])
                + args.selection_strict_weight * float(summary["caption_all_acc_mean"])
                + args.selection_routing_weight * routing_pass
                + args.selection_diag_weight * float(summary["diag_signed_median"])
                + args.selection_object_weight * float(summary["object_guard_mean"])
                - args.selection_off_weight * float(summary["off_abs_median"])
            )
        else:
            score = float(summary["diag_signed_median"]) + 0.25 * float(summary["caption_acc_mean"]) - float(summary["off_abs_median"])
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(logs["loss"])),
            "train_diag": float(np.mean(logs["diag"])),
            "train_diag_high_loss": float(np.mean(logs["diag_high"])),
            "train_off": float(np.mean(logs["off"])),
            "train_caption_loss": float(np.mean(logs["caption_loss"])),
            "val_score": score,
            **summary,
        }
        history.append(epoch_row)
        print(
            f"[epoch {epoch:02d}] val diag_med={summary['diag_signed_median']:.4f} "
            f"off_abs_med={summary['off_abs_median']:.4f} BLI={summary['bli_median']:.2f} "
            f"caption={summary['caption_acc_mean']:.3f} strict={summary['caption_all_acc_mean']:.3f} "
            f"routing={routing_pass:.3f} score={score:.4f}"
        )
        ckpt_payload = {
            "epoch": epoch,
            "state_dict": adapter.state_dict(),
            "args": vars(args),
            "summary": summary,
            "score": score,
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
            torch.save(ckpt_payload, out_dir / "two_object_locality_best.pt")
            print(f"[checkpoint] saved best score={best_score:.4f}")

    write_json(summary_dir / "train_history.json", {"history": history, "best_score": best_score, "args": vars(args)})
    write_csv(summary_dir / "train_history.csv", history)


def audit_two_object_locality(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    audit_dir = out_dir / "audits"
    summary_dir = out_dir / "summaries"
    figure_dir = out_dir / "figures"
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = args.adapter_ckpt or args.checkpoint
    rows, summary = audit_rows(args, adapter_path)
    name = "steered" if adapter_path else "baseline"
    write_csv(audit_dir / f"{name}_audit.csv", rows)
    write_json(summary_dir / f"{name}_summary.json", summary)
    try:
        import matplotlib.pyplot as plt

        xs_diag = np.sort(np.array([0.5 * (abs(float(r["d_o1_diag"])) + abs(float(r["d_o2_diag"]))) for r in rows]))
        xs_off = np.sort(np.array([float(r["abs_off_mean"]) for r in rows]))
        ys_diag = np.arange(1, len(xs_diag) + 1) / max(1, len(xs_diag))
        ys_off = np.arange(1, len(xs_off) + 1) / max(1, len(xs_off))
        plt.figure(figsize=(4.2, 3.0))
        plt.plot(xs_diag, ys_diag, label="diag |Delta2|")
        plt.plot(xs_off, ys_off, label="offdiag |Delta2|")
        plt.xlabel("|Delta2|")
        plt.ylabel("ECDF")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(figure_dir / f"{name}_ecdf_delta.png", dpi=170)
        plt.close()
    except Exception as exc:  # pragma: no cover - plotting should not fail audits
        print(f"[warn] plotting failed: {exc}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def eval_two_object_locality(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.checkpoint = ""
    baseline_args.adapter_ckpt = ""
    steered_args = argparse.Namespace(**vars(args))
    steered_args.checkpoint = args.adapter_ckpt or args.checkpoint
    steered_args.adapter_ckpt = steered_args.checkpoint
    rows_base, summary_base = audit_rows(baseline_args, "")
    rows_steer, summary_steer = audit_rows(steered_args, steered_args.checkpoint) if steered_args.checkpoint else ([], {})
    audit_dir = out_dir / "audits"
    summary_dir = out_dir / "summaries"
    write_csv(audit_dir / "baseline_audit.csv", rows_base)
    write_json(summary_dir / "baseline_summary.json", summary_base)
    if rows_steer:
        write_csv(audit_dir / "steered_audit.csv", rows_steer)
        write_json(summary_dir / "steered_summary.json", summary_steer)
    delta = {}
    if summary_steer:
        for key, val in summary_steer.items():
            if isinstance(val, (int, float)) and key in summary_base:
                delta[key] = float(val) - float(summary_base[key])
    write_json(summary_dir / "eval_summary.json", {"baseline": summary_base, "steered": summary_steer, "delta": delta})
    print(json.dumps({"baseline": summary_base, "steered": summary_steer, "delta": delta}, indent=2, sort_keys=True))


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--ann_json", required=True)
    ap.add_argument("--val_img_dir", default="")
    ap.add_argument("--val_ann_json", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--adapter_ckpt", default="")
    ap.add_argument("--baseline_checkpoint", default="")
    ap.add_argument("--arch", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--val_samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--classes", default="")
    ap.add_argument("--target_classes", default="")
    ap.add_argument("--distractor_classes", default="")
    ap.add_argument("--colors", default="red,blue")
    ap.add_argument("--object_pairs", default="car:boat,dog:ball,person:bicycle,cat:couch,bus:airplane")
    ap.add_argument("--template_groups", default="")
    ap.add_argument("--objective", choices=["interaction", "caption_ranking"], default="interaction")
    ap.add_argument("--K", type=float, default=0.20)
    ap.add_argument("--K_diag", type=float, default=0.10)
    ap.add_argument("--K_diag_high", type=float, default=0.0)
    ap.add_argument("--K_off", type=float, default=0.03)
    ap.add_argument("--K3", type=float, default=0.05)
    ap.add_argument("--K_low", type=float, default=0.10)
    ap.add_argument("--K_high", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--m_agree", type=float, default=0.03)
    ap.add_argument("--lam_anchor", type=float, default=0.2)
    ap.add_argument("--lam_diag_high", type=float, default=0.0)
    ap.add_argument("--lam_off", type=float, default=1.0)
    ap.add_argument("--lam_invariant", type=float, default=1.0)
    ap.add_argument("--lam_guard", type=float, default=0.8)
    ap.add_argument("--lam_caption", type=float, default=0.05)
    ap.add_argument("--selection_mode", choices=["legacy", "behavior_routing"], default="legacy")
    ap.add_argument("--selection_caption_weight", type=float, default=1.0)
    ap.add_argument("--selection_strict_weight", type=float, default=0.5)
    ap.add_argument("--selection_routing_weight", type=float, default=0.5)
    ap.add_argument("--selection_diag_weight", type=float, default=0.25)
    ap.add_argument("--selection_off_weight", type=float, default=1.0)
    ap.add_argument("--selection_object_weight", type=float, default=0.1)
    ap.add_argument("--min_area_ratio", type=float, default=0.02)
    ap.add_argument("--canvas", type=int, default=256)
    ap.add_argument("--object_size", type=int, default=112)
    ap.add_argument("--tint_alpha", type=float, default=0.55)
    ap.add_argument(
        "--position_swap_prob",
        type=float,
        default=0.0,
        help="Probability of placing semantic object 1 on the right; default 0 preserves submitted runs.",
    )
    ap.add_argument("--notes", default="")
    ap.add_argument("--scaffold_only", action="store_true")
