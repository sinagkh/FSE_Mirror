"""Frozen, provenance-checked image/text scorers for the bounded B2 subjects.

This module only encodes supplied inputs and forms score matrices. It does not
construct banks, select checkpoints, calibrate thresholds, or report outcomes.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import torch
from PIL import Image


DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "data/models/registry.json"
DEFAULT_ACTIVATION_ADDENDUM = DEFAULT_REGISTRY.with_name("activation_addendum.json")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_registry(path=DEFAULT_REGISTRY):
    path = Path(path)
    registry = json.loads(path.read_text())
    if registry.get("schema_version") != 1 or len(registry.get("subjects", [])) != 7:
        raise ValueError("Expected the frozen seven-subject registry")
    ids = [row["id"] for row in registry["subjects"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate subject IDs")
    addendum_path = path.with_name("activation_addendum.json")
    if path == DEFAULT_REGISTRY and not addendum_path.is_file():
        raise ValueError("Activation addendum is required for frozen registry")
    if addendum_path.is_file():
        addendum = json.loads(addendum_path.read_text())
        if addendum.get("registry_sha256") != file_sha256(path):
            raise ValueError("Activation addendum does not match original registry")
        activations = addendum["activations"]
        if set(activations) != set(ids) or set(activations.values()) - {"gelu", "quick_gelu", "siglip"}:
            raise ValueError("Invalid activation addendum")
        for row in registry["subjects"]:
            row["activation"] = activations[row["id"]]
    return registry


def _verify_files(row):
    for item in row["files"]:
        path = Path(item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError(f"Missing or changed subject input: {path}")


def _groups(prompts):
    if not prompts:
        raise ValueError("Need at least one prompt group")
    if all(isinstance(p, str) and p for p in prompts):
        return [[p] for p in prompts]
    if not all(isinstance(p, (list, tuple)) and p and
               all(isinstance(v, str) and v for v in p) for p in prompts):
        raise ValueError("Prompts must be nonempty strings or nonempty groups of strings")
    return [list(p) for p in prompts]


def legacy_unit(features):
    """Exactly the embedding normalization in interbind/encode.py."""
    features = features.float()
    return features / (features.norm(dim=-1, keepdim=True) + 1e-8)


def pool_templates(features, lengths):
    """Legacy pooling: unit per template, arithmetic mean, then unit again."""
    features = legacy_unit(features)
    if sum(lengths) != len(features) or any(n < 1 for n in lengths):
        raise ValueError("Template lengths do not match features")
    return torch.stack([legacy_unit(chunk.mean(0))
                        for chunk in features.split(lengths)])


@dataclass
class FrozenScorer:
    subject: dict
    model: object
    preprocess: object
    tokenizer: object
    device: torch.device

    @property
    def score_unit(self):
        return self.subject["score_unit"]

    def _run_image(self, images):
        if self.subject["backend"] == "siglip_transformers":
            pixels = self.preprocess(images=[_as_pil(im) for im in images],
                                     return_tensors="pt")["pixel_values"].to(self.device)
            return self.model.get_image_features(pixel_values=pixels)
        pixels = torch.stack([self.preprocess(_as_pil(im)) for im in images]).to(self.device)
        return self.model.encode_image(pixels)

    def _run_text(self, texts):
        if self.subject["backend"] == "siglip_transformers":
            inputs = self.tokenizer(texts, padding="max_length", truncation=True,
                                    return_tensors="pt")
            return self.model.get_text_features(**{k: v.to(self.device) for k, v in inputs.items()})
        return self.model.encode_text(self.tokenizer(texts).to(self.device))

    @torch.inference_mode()
    def encode_images(self, images, batch_size=32):
        images = list(images)
        if not images or batch_size < 1:
            raise ValueError("Need images and positive batch size")
        chunks = [legacy_unit(self._run_image(images[i:i + batch_size])).cpu()
                  for i in range(0, len(images), batch_size)]
        return torch.cat(chunks)

    @torch.inference_mode()
    def encode_texts(self, prompts, batch_size=64):
        groups = _groups(prompts)
        if batch_size < 1:
            raise ValueError("Need positive batch size")
        flat = [text for group in groups for text in group]
        chunks = [self._run_text(flat[i:i + batch_size]).float().cpu()
                  for i in range(0, len(flat), batch_size)]
        return pool_templates(torch.cat(chunks), [len(group) for group in groups])

    def scores(self, image_features, text_features):
        """[images, texts]; CLIP raw cosine, SigLIP pre-sigmoid logits."""
        images = torch.as_tensor(image_features).float()
        texts = torch.as_tensor(text_features).float()
        if (images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]
                or not torch.isfinite(images).all() or not torch.isfinite(texts).all()):
            raise ValueError("Expected finite, matched two-dimensional features")
        score = images @ texts.T
        if self.subject["backend"] == "siglip_transformers":
            score = score * self.model.logit_scale.detach().float().exp().to(score.device)
            score = score + self.model.logit_bias.detach().float().to(score.device)
        return score


def _as_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def load_subject(subject_id, registry_path=DEFAULT_REGISTRY, device="cpu"):
    """Load only a verified local checkpoint; never fetch or silently substitute."""
    registry = read_registry(registry_path)
    row = next((r for r in registry["subjects"] if r["id"] == subject_id), None)
    if row is None:
        raise KeyError(subject_id)
    if row["status"] != "ready":
        raise RuntimeError(f"Subject {subject_id} is not ready: {row['status']}")
    _verify_files(row)
    device = torch.device(device)
    if row["backend"] == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            row["architecture"], pretrained=row["files"][0]["path"], device=str(device),
            force_quick_gelu=row["activation"] == "quick_gelu",
            **({"weights_only": False} if row.get("trusted_original_pickle") else {}))
        tokenizer = open_clip.get_tokenizer(row["architecture"])
    elif row["backend"] == "siglip_transformers":
        from transformers import SiglipModel; from transformers import SiglipProcessor
        directory = str(Path(row["files"][0]["path"]).parent)
        model = SiglipModel.from_pretrained(directory, local_files_only=True).to(device)
        processor = SiglipProcessor.from_pretrained(directory, local_files_only=True,
                                                   use_fast=False)
        preprocess, tokenizer = processor.image_processor, processor.tokenizer
    else:
        raise ValueError(f"Unsupported backend: {row['backend']}")
    model.eval().requires_grad_(False)
    return FrozenScorer(row, model, preprocess, tokenizer, device)
