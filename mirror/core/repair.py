"""Create-only, cached-feature repair engine. No encoder, renderer, or evaluator lives here.

All arms use the same text forward, batches, initialization, dropout stream,
optimizer and fixed-last selection. The registered objective terms and weights
select the arm; the main ranking arm retains its historical ancillary terms. A cache
must be constructed from frozen encoders before calling :func:`train_cached`.
"""
from __future__ import annotations

from dataclasses import asdict; from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mirror.core.specifications import CompiledRequirement


@dataclass(frozen=True)
class RepairCache:
    images: torch.Tensor                 # [anchor, image state, dimension]
    texts: torch.Tensor                  # [anchor, caption state + object captions, dimension]
    natural_texts: torch.Tensor          # [anchor, natural caption, dimension]
    source_ids: tuple[str | int, ...]   # source, not row, identity; repetitions permitted
    base_model_id: str
    encoder_sha256: str
    bank_id: str
    score_scale: float = 1.             # SigLIP exp(logit_scale); CLIP 1
    score_bias: float = 0.               # SigLIP logit_bias; CLIP 0


@dataclass(frozen=True)
class LossScales:
    interaction: float
    margin: float
    rule: str = "max(.01, median absolute frozen training contrast/margin)"


@dataclass(frozen=True)
class LegacyRanking:
    """Frozen confirmation rank_loss recipe, with row-aligned object-caption indices.

    Background coefficients: 1*CE + 1.2*agreement(m=.045) + .35*anchor.
    Routing coefficients: 1*CE + .2*anchor. The original rank_loss reads a
    different object caption per background row (2+nouns); record that mapping
    explicitly, rather than silently using an arbitrary object guard caption.
    """
    family: str
    object_text_indices: tuple[int, ...]
    endpoint_weight: float = 1.
    agreement_weight: float = 0.
    agreement_margin: float = 0.
    anchor_weight: float = 0.

    def __post_init__(self):
        if type(self.object_text_indices) is not tuple or not all(type(i) is int for i in self.object_text_indices):
            raise TypeError("Legacy object-caption indices must be an immutable tuple of ints")


def historical_ranking(family: str, object_text_indices: tuple[int, ...]) -> LegacyRanking:
    """Values transcribed from the frozen 2026-09-20 training protocol."""
    if family == "background":
        return LegacyRanking(family, object_text_indices, 1., 1.2, .045, .35)
    if family == "routing":
        return LegacyRanking(family, object_text_indices, 1., 0., 0., .2)
    raise ValueError("No historical ranking recipe for this requirement family")


@dataclass(frozen=True)
class RepairConfig:
    arm: str                             # ranking (historical), ce_only, guard_only, full_is, or ablation
    seed: int
    expected_cache_sha256: str
    exclusion_sha256: str
    encoder_sha256: str
    base_model_id: str
    bank_id: str
    excluded_source_ids: tuple[str | int, ...]
    correct_captions: tuple[int, ...]    # one correct caption per image state
    object_pairs: tuple[tuple[int, int, int], ...]  # (image, correct, distractor)
    binding_contrasts: tuple[str, ...]
    clause_weights: dict[str, float]
    guard_weights: dict[str, float]
    scales: LossScales
    legacy_ranking: LegacyRanking | None = None
    epochs: int = 6
    batch_size: int = 24
    lr: float = 2e-4
    weight_decay: float = .01
    rank: int = 64
    alpha: float = 64.
    dropout: float = .05
    grad_clip: float = 1.
    natural_tolerance: float = .01
    drift_tolerance: float = .01
    checkpoint_selection: str = "fixed_last"


GUARDS = ("caption_margin", "binding_retention", "object_caption",
          "natural_consistency", "embedding_drift")


class TextLowRankAdapter(nn.Module):
    """Output-embedding adapter; zero B gives an exact identity at initialization."""

    def __init__(self, dimension: int, rank: int = 64, alpha: float = 64., dropout: float = .05):
        super().__init__()
        self.A = nn.Linear(dimension, rank, bias=False)
        self.B = nn.Linear(rank, dimension, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, frozen_text: torch.Tensor) -> torch.Tensor:
        return F.normalize(frozen_text + self.scale * self.B(self.drop(self.A(frozen_text))), dim=-1)


def _tensor_bytes(t: torch.Tensor) -> bytes:
    a = t.detach().cpu().contiguous().numpy()
    return (str(a.dtype) + repr(a.shape)).encode() + a.tobytes()


def cache_sha256(cache: RepairCache) -> str:
    h = sha256()
    for x in (cache.images, cache.texts, cache.natural_texts):
        h.update(_tensor_bytes(x))
    for x in (cache.source_ids, cache.base_model_id, cache.encoder_sha256,
              cache.bank_id, cache.score_scale, cache.score_bias):
        h.update(json.dumps(x, sort_keys=True).encode())
    return h.hexdigest()


def exclusion_sha256(ids: tuple[str | int, ...]) -> str:
    return sha256(json.dumps(sorted(set(ids), key=str), separators=(",", ":")).encode()).hexdigest()


def _contrast_values(spec: CompiledRequirement, scores: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = scores.flatten(-2) / spec.unit
    return {k: flat @ scores.new_tensor(w.reshape(-1)) for k, w in spec.weights.items()}


def derive_scales(cache: RepairCache, spec: CompiledRequirement,
                  correct_captions: tuple[int, ...]) -> LossScales:
    """Frozen training rows only; never derive scales from validation outcomes."""
    with torch.no_grad():
        scores = (cache.images.float() @ cache.texts[:, :spec.shape[1]].float().transpose(1, 2))*cache.score_scale+cache.score_bias
        vals = torch.stack(list(_contrast_values(spec, scores).values()), -1).abs().flatten()
        margin = _correct_margins(scores, correct_captions).abs().flatten()
        return LossScales(max(.01, float(vals.median())), max(.01, float(margin.median())))


def _correct_margins(scores: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    # [B,I,C] -> all correct-v-incorrect comparisons; correct positions excluded.
    ix = torch.tensor(labels, device=scores.device)
    correct = scores.gather(2, ix[None, :, None].expand(len(scores), -1, 1))
    margins = correct - scores
    mask = torch.arange(scores.shape[-1], device=scores.device)[None, None, :] != ix[None, :, None]
    return margins[:, mask[0]].reshape(len(scores), -1)


def _check(cfg: RepairConfig, cache: RepairCache, spec: CompiledRequirement) -> None:
    if cfg.checkpoint_selection != "fixed_last":
        raise ValueError("Only fixed-last checkpoint selection is supported")
    if cfg.arm not in {"ranking", "ce_only", "guard_only", "full_is", "no_gap", "no_cross", "no_delta"} and not cfg.arm.startswith("without:"):
        raise ValueError("Unknown arm")
    if cfg.arm.startswith("without:") and cfg.arm.split(":", 1)[1] not in cfg.clause_weights:
        raise ValueError("Unknown ablated clause")
    if cfg.seed < 0 or cfg.epochs < 1 or cfg.batch_size < 1 or cfg.rank < 1 or cfg.alpha <= 0:
        raise ValueError("Invalid schedule or adapter")
    if not (0 <= cfg.dropout < 1) or min(cfg.lr, cfg.grad_clip) <= 0 or cfg.weight_decay < 0:
        raise ValueError("Invalid optimizer")
    if cfg.base_model_id != spec.base_model_id or cache.base_model_id != cfg.base_model_id:
        raise ValueError("Base model mismatch")
    if spec.spec["scorer_output"] not in {"cosine", "logit"}:
        raise ValueError("Repair requires cosine or pre-sigmoid logit requirements")
    if cfg.bank_id != cache.bank_id or cfg.encoder_sha256 != cache.encoder_sha256:
        raise ValueError("Cache provenance mismatch")
    if not math.isfinite(cache.score_scale) or cache.score_scale <= 0 or not math.isfinite(cache.score_bias):
        raise ValueError("Invalid frozen scorer scale/bias")
    if (cache.score_scale != 1. or cache.score_bias != 0.) and spec.spec["scorer_output"] != "logit":
        raise ValueError("Scaled scorer requires a logit requirement")
    if cfg.expected_cache_sha256 != cache_sha256(cache):
        raise ValueError("Cache hash mismatch")
    if cfg.exclusion_sha256 != exclusion_sha256(cfg.excluded_source_ids):
        raise ValueError("Exclusion hash mismatch")
    if not cache.source_ids or not cfg.excluded_source_ids or set(cache.source_ids) & set(cfg.excluded_source_ids):
        raise ValueError("Missing or excluded training source")
    n, ni, d = cache.images.shape
    if cache.texts.ndim != 3 or cache.natural_texts.ndim != 3 or cache.texts.shape[0] != n or cache.natural_texts.shape[0] != n or len(cache.source_ids) != n:
        raise ValueError("Invalid cache row dimensions")
    if ni != spec.shape[0] or cache.texts.shape[1] < spec.shape[1] or cache.texts.shape[2] != d or cache.natural_texts.shape[2] != d:
        raise ValueError("Cache does not match requirement lattice")
    if cache.natural_texts.shape[1] < 1 or cache.texts.shape[1] <= spec.shape[1]:
        raise ValueError("Natural and object-caption guards require cached texts")
    if any(not torch.isfinite(x).all() for x in (cache.images, cache.texts, cache.natural_texts)):
        raise ValueError("Nonfinite frozen cache")
    if any(bool(((torch.linalg.vector_norm(x.float(), dim=-1)-1).abs() > .005).any())
           for x in (cache.images, cache.texts, cache.natural_texts)):
        raise ValueError("Cached encoder embeddings must be unit-normalized")
    if len(cfg.correct_captions) != ni or any(c < 0 or c >= spec.shape[1] for c in cfg.correct_captions):
        raise ValueError("Correct caption labels do not cover image states")
    if cfg.legacy_ranking is not None:
        rank = cfg.legacy_ranking
        family = spec.spec["id"].split("/", 1)[0]
        if rank.family != family or family not in {"background", "routing"}:
            raise ValueError("Legacy ranking family mismatch")
        if rank != historical_ranking(family, rank.object_text_indices):
            raise ValueError("Legacy ranking coefficients differ from frozen protocol")
        if len(rank.object_text_indices) != n or any(i < spec.shape[1] or i >= cache.texts.shape[1] for i in rank.object_text_indices):
            raise ValueError("Legacy ranking object-caption indices must match cache rows")
        expected_shape = (4, 2) if family == "background" else (4, 4)
        expected_labels = (0, 1, 0, 1) if family == "background" else (0, 1, 2, 3)
        lattice = spec.spec["score_lattice"]
        image_ids = tuple(x["id"] for x in lattice["image_states"])
        caption_ids = tuple(x["id"] for x in lattice["caption_states"])
        expected_images = ("orig_red", "orig_blue", "swap_red", "swap_blue") if family == "background" else ("rr", "rb", "br", "bb")
        expected_captions = ("red", "blue") if family == "background" else ("rr", "rb", "br", "bb")
        if spec.shape != expected_shape or cfg.correct_captions != expected_labels or image_ids != expected_images or caption_ids != expected_captions:
            raise ValueError("Legacy ranking lattice/order differs from frozen objective")
        if family == "routing" and any(i != 4 for i in rank.object_text_indices):
            raise ValueError("Legacy routing anchor requires object caption index 4")
    if cfg.arm == "ranking" and cfg.legacy_ranking is None:
        raise ValueError("Historical ranking requires explicit immutable legacy recipe")
    if not cfg.object_pairs or any(i < 0 or i >= ni or c < spec.shape[1] or c >= cache.texts.shape[1] or w < spec.shape[1] or w >= cache.texts.shape[1] or c == w for i, c, w in cfg.object_pairs):
        raise ValueError("Invalid object-caption guard pairs")
    if not cfg.binding_contrasts or set(cfg.binding_contrasts) - set(spec.weights):
        raise ValueError("Unknown binding retention contrasts")
    if set(cfg.clause_weights) != {c["name"] for c in spec.spec["clauses"]} or set(cfg.guard_weights) != set(GUARDS):
        raise ValueError("Weights must explicitly cover every penalty and guard")
    numbers = list(cfg.clause_weights.values()) + list(cfg.guard_weights.values()) + [cfg.scales.interaction, cfg.scales.margin, cfg.natural_tolerance, cfg.drift_tolerance]
    if any(not math.isfinite(v) or v < 0 for v in numbers) or min(cfg.scales.interaction, cfg.scales.margin) <= 0:
        raise ValueError("Nonfinite or negative weights/scales")
    if cfg.scales.rule != LossScales(1, 1).rule:
        raise ValueError("Unknown normalization rule")
    arm_weights(cfg)  # reject a term ablation incompatible with this requirement
    derived = derive_scales(cache, spec, cfg.correct_captions)
    if not math.isclose(cfg.scales.interaction, derived.interaction, rel_tol=1e-6, abs_tol=1e-7) or not math.isclose(cfg.scales.margin, derived.margin, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("Loss scales do not match frozen training cache")


def arm_weights(cfg: RepairConfig) -> tuple[dict[str, float], dict[str, float]]:
    clauses = dict(cfg.clause_weights)
    guards = dict(cfg.guard_weights)
    if cfg.arm in {"ranking", "ce_only"}:
        clauses = {k: 0. for k in clauses}
        guards = {k: 0. for k in guards}
    elif cfg.arm == "guard_only":
        clauses = {k: 0. for k in clauses}
    elif cfg.arm.startswith("without:"):
        clauses[cfg.arm.split(":", 1)[1]] = 0.
    elif cfg.arm == "no_gap":
        if not any(k.startswith("gap_") for k in clauses):
            raise ValueError("No background gap clauses to ablate")
        clauses = {k: (0. if k.startswith("gap_") else v) for k, v in clauses.items()}
    elif cfg.arm == "no_cross":
        if not any(k.startswith(("leak_", "relative_")) for k in clauses):
            raise ValueError("No routing cross clauses to ablate")
        clauses = {k: (0. if k.startswith(("leak_", "relative_")) else v) for k, v in clauses.items()}
    elif cfg.arm == "no_delta":
        if not any(k.startswith("positive_synergy") for k in clauses):
            raise ValueError("No conjunction delta clause to ablate")
        clauses = {k: (0. if k.startswith("positive_synergy") else v) for k, v in clauses.items()}
    return clauses, guards


def legacy_ranking_loss(scores: torch.Tensor, adapted: torch.Tensor,
                        base: torch.Tensor, recipe: LegacyRanking,
                        rows: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The frozen rank_loss arithmetic, on an already matched cached forward."""
    b = len(scores)
    selected = torch.tensor(recipe.object_text_indices, device=scores.device)[rows]
    ix = torch.arange(b, device=scores.device)
    if recipe.family == "background":
        labels = torch.tensor([0, 1, 0, 1], device=scores.device).repeat(b)
        ce = F.cross_entropy(scores[:, :, :2].reshape(-1, 2), labels)
        margin = (scores[:, :, 0]-scores[:, :, 1])*scores.new_tensor([1, -1, 1, -1])
        agreement = F.relu(recipe.agreement_margin-margin).square().mean()
        anchor = (1-F.cosine_similarity(adapted[:, :2], base[:, :2], dim=-1)).mean()
        anchor = anchor+(1-F.cosine_similarity(adapted[ix, selected], base[ix, selected], dim=-1)).mean()
        return (recipe.endpoint_weight*ce + recipe.agreement_weight*agreement
                + recipe.anchor_weight*anchor), {"ranking_endpoint": ce,
                                                  "ranking_agreement": agreement,
                                                  "ranking_anchor": anchor}
    if recipe.family == "routing":
        ce = F.cross_entropy(scores[:, :, :4].reshape(-1, 4),
                             torch.arange(4, device=scores.device).repeat(b))
        anchor = (1-F.cosine_similarity(adapted[:, :4], base[:, :4], dim=-1)).mean()
        anchor = anchor+(1-F.cosine_similarity(adapted[ix, selected], base[ix, selected], dim=-1)).mean()
        return ce+recipe.anchor_weight*anchor, {"ranking_endpoint": ce,
                                                "ranking_agreement": ce.new_zeros(()),
                                                "ranking_anchor": anchor}
    raise ValueError("Unknown historical ranking family")


def validate_matched_configs(configs: tuple[RepairConfig, ...]) -> None:
    """Fail before a grid if anything other than the arm was changed."""
    if not configs or len({c.arm for c in configs}) != len(configs):
        raise ValueError("Need distinct arms")
    reference = asdict(configs[0])
    reference.pop("arm")
    for config in configs[1:]:
        candidate = asdict(config)
        candidate.pop("arm")
        if candidate != reference:
            raise ValueError("Matched arms must differ only by arm")


def loss_components(adapter: TextLowRankAdapter, cache: RepairCache,
                    spec: CompiledRequirement, cfg: RepairConfig,
                    rows: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute *all* auxiliary forwards/components even for zero-weight arms."""
    images = cache.images[rows].detach().float()
    base_text = cache.texts[rows].detach().float()
    base_natural = cache.natural_texts[rows].detach().float()
    all_base = torch.cat((base_text, base_natural), dim=1)
    all_adapted = adapter(all_base)  # identical shape and dropout consumption in every arm
    adapted_text = all_adapted[:, :base_text.shape[1]]
    adapted_natural = all_adapted[:, base_text.shape[1]:]
    scores = (images @ adapted_text.transpose(1, 2))*cache.score_scale+cache.score_bias
    with torch.no_grad():
        frozen = (images @ base_text.transpose(1, 2))*cache.score_scale+cache.score_bias
    ncap = spec.shape[1]
    lattice = scores[:, :, :ncap]
    reference = frozen[:, :, :ncap]
    labels = torch.tensor(cfg.correct_captions, device=scores.device)
    endpoint = F.cross_entropy(lattice.reshape(-1, ncap), labels.repeat(len(rows)))
    current_margin = _correct_margins(lattice, cfg.correct_captions)
    frozen_margin = _correct_margins(reference, cfg.correct_captions)
    good = frozen_margin > 0
    caption_margin = ((F.relu((frozen_margin-current_margin)/cfg.scales.margin).square())*good).sum()/good.sum().clamp_min(1)
    current_binding = _contrast_values(spec, lattice)
    frozen_binding = _contrast_values(spec, reference)
    binding_retention = torch.stack([
        (F.relu((frozen_binding[k]-current_binding[k])/cfg.scales.interaction).square()*(frozen_binding[k] > 0)).mean()
        for k in cfg.binding_contrasts]).mean()
    object_losses = []
    for i, correct, distractor in cfg.object_pairs:
        old = frozen[:, i, correct] - frozen[:, i, distractor]
        new = scores[:, i, correct] - scores[:, i, distractor]
        object_losses.append((F.relu((old-new)/cfg.scales.margin).square()*(old > 0)).mean())
    object_caption = torch.stack(object_losses).mean()
    natural_consistency = F.relu(1-F.cosine_similarity(adapted_natural, base_natural, dim=-1)-cfg.natural_tolerance).square().mean()
    embedding_drift = F.relu(1-F.cosine_similarity(adapted_text, base_text, dim=-1)-cfg.drift_tolerance).square().mean()
    penalties = {k: v.mean() for k, v in spec.penalty(lattice).items()}
    if cfg.legacy_ranking is not None:
        ranking_total, ranking_terms = legacy_ranking_loss(scores, adapted_text, base_text,
                                                           cfg.legacy_ranking, rows)
    else:
        ranking_total = endpoint
        ranking_terms = {"ranking_endpoint": endpoint,
                         "ranking_agreement": endpoint.new_zeros(()),
                         "ranking_anchor": endpoint.new_zeros(())}
    components = {"endpoint": endpoint, "caption_margin": caption_margin,
                  "binding_retention": binding_retention, "object_caption": object_caption,
                  "natural_consistency": natural_consistency, "embedding_drift": embedding_drift,
                  **ranking_terms,
                  **{f"clause:{k}": v for k, v in penalties.items()}}
    cw, gw = arm_weights(cfg)
    total = (ranking_total if cfg.arm == "ranking" else endpoint if cfg.arm == "ce_only" else
             sum(gw[k]*components[k] for k in GUARDS) + sum(cw[k]*components[f"clause:{k}"] for k in cw))
    return total, components


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    h = sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(_tensor_bytes(v))
    return h.hexdigest()


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def train_cached(config: RepairConfig, cache: RepairCache,
                 spec: CompiledRequirement, out: str | Path) -> dict:
    """Train one create-only arm; interrupted directories are errors, not resumes.

    Outputs initial.pt, last.pt, history.json and complete.json. No best checkpoint
    is calculated; downstream evaluation must use last.pt.
    """
    _check(config, cache, spec)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Run exists (resume is not supported): {out}")
    out.mkdir(parents=True, exist_ok=False)
    device = cache.images.device
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    adapter = TextLowRankAdapter(cache.images.shape[-1], config.rank, config.alpha, config.dropout).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    torch.save({"state_dict": initial, "state_hash": _state_hash(initial)}, out / "initial.pt")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    schedule = [rng.permutation(len(cache.source_ids)).tolist() for _ in range(config.epochs)]
    history = []
    for epoch, order in enumerate(schedule, 1):
        adapter.train()
        torch.manual_seed(config.seed*10000+epoch)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed*10000+epoch)
        for start in range(0, len(order), config.batch_size):
            rows = torch.tensor(order[start:start+config.batch_size], device=device)
            total, components = loss_components(adapter, cache, spec, config, rows)
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite repair loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite repair gradient")
            optimizer.step()
            history.append({"epoch": epoch, "rows": rows.cpu().tolist(), "total": float(total.detach()),
                            "components": {k: float(v.detach()) for k, v in components.items()}})
    state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    payload = {"state_dict": state, "optimizer": optimizer.state_dict(), "epoch": config.epochs,
               "updates": len(history), "selection": "fixed_last", "arm": config.arm,
               "seed": config.seed, "cache_sha256": config.expected_cache_sha256,
               "requirement_sha256": spec.spec_hash, "base_model_id": config.base_model_id,
               "encoder_sha256": config.encoder_sha256, "bank_id": config.bank_id,
               "initial_state_hash": _state_hash(initial), "state_hash": _state_hash(state),
               "config": asdict(config)}
    torch.save(payload, out / "last.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    result = {"arm": config.arm, "selection": "fixed_last", "epochs": config.epochs,
              "updates": len(history), "initial_state_hash": _state_hash(initial),
              "last_state_hash": _state_hash(state), "last_sha256": sha256((out / "last.pt").read_bytes()).hexdigest(),
              "cache_sha256": config.expected_cache_sha256, "requirement_sha256": spec.spec_hash,
              "clause_weights": arm_weights(config)[0], "guard_weights": arm_weights(config)[1],
              "normalization": asdict(config.scales)}
    (out / "complete.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
