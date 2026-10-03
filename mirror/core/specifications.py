"""Safe, declarative interaction requirements and their score-only compiler.

No expression language, Python evaluation, model calibration, or threshold search
lives here. A caller supplies the frozen *base* model's calibration unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import copy

import numpy as np
import yaml


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*/[A-Za-z][A-Za-z0-9_.-]*$")
_KINDS = {"lower", "magnitude", "band", "relative_leakage", "relative_context"}


def _finite_number(x, label, *, positive=False, nonnegative=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{label} must be a finite number")
    if positive and x <= 0 or nonnegative and x < 0:
        raise ValueError(f"{label} is outside its allowed range")
    return float(x)


def _name(value, label):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _mapping(x, label):
    if not isinstance(x, dict):
        raise ValueError(f"{label} must be a mapping")
    return x


def _keys(x, required, optional, label):
    missing = set(required) - set(x)
    extra = set(x) - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{label}: missing={sorted(missing)}, unknown={sorted(extra)}")


def load_requirement(path):
    """Load one UTF-8 YAML requirement, rejecting tags and duplicate keys."""
    class UniqueSafeLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if not isinstance(key, (str, int, float, bool, tuple)) or key in result:
                raise ValueError("YAML mapping has a duplicate or unsupported key")
            result[key] = loader.construct_object(value_node, deep=True)
        return result

    UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    with Path(path).open("r", encoding="utf-8") as fh:
        spec = yaml.load(fh, Loader=UniqueSafeLoader)
    return validate_requirement(spec)


def requirement_hash(spec):
    """Canonical content hash for result provenance, independent of YAML layout."""
    validate_requirement(spec)
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def validate_requirement(spec):
    """Validate the bounded v1 grammar; return the original mapping."""
    s = _mapping(spec, "requirement")
    _keys(s, {"schema_version", "id", "bug_type", "semantic_intent", "operational_measurement",
              "scorer_output", "factors", "score_lattice", "contrasts", "clauses", "thresholds"},
          {"anchors", "guards", "evidence_status"}, "requirement")
    if type(s["schema_version"]) is not int or s["schema_version"] != 1:
        raise ValueError("Only requirement schema_version 1 is supported")
    if not isinstance(s["id"], str) or not _ID.fullmatch(s["id"]):
        raise ValueError("Requirement id must be family/name")
    if s["bug_type"] not in {"under_binding", "leakage", "context_dependence", "conjunction", "relational"}:
        raise ValueError("Unsupported bug type")
    for field in ("semantic_intent", "operational_measurement"):
        if not isinstance(s[field], str) or not s[field].strip():
            raise ValueError(f"{field} must state a nonempty intent or measurement")
    if s["scorer_output"] not in {"cosine", "logit", "sigmoid"}:
        raise ValueError("Unsupported scorer output")
    factors = _mapping(s["factors"], "factors")
    if not factors:
        raise ValueError("At least one factor is required")
    for name, factor in factors.items():
        _name(name, "factor name")
        f = _mapping(factor, f"factor {name}")
        _keys(f, {"kind", "states"}, {"generator", "slot", "description"}, f"factor {name}")
        if f["kind"] not in {"image", "caption"}:
            raise ValueError("Factor kind must be image or caption")
        states = f["states"]
        if not isinstance(states, list) or len(states) < 2 or len(set(states)) != len(states):
            raise ValueError(f"Factor {name} needs distinct states")
        for state in states:
            _name(state, "factor state")
        for field in ("generator", "slot", "description"):
            if field in f and (not isinstance(f[field], str) or not f[field].strip()):
                raise ValueError(f"Invalid factor {name} {field}")
        if f["kind"] == "image" and "generator" not in f:
            raise ValueError(f"Image factor {name} needs generator metadata")
        if f["kind"] == "caption" and "slot" not in f:
            raise ValueError(f"Caption factor {name} needs slot metadata")
    lattice = _mapping(s["score_lattice"], "score_lattice")
    _keys(lattice, {"image_states", "caption_states", "prompt"}, set(), "score_lattice")
    axes = []
    for axis, kind in (("image_states", "image"), ("caption_states", "caption")):
        states = lattice[axis]
        if not isinstance(states, list) or not states:
            raise ValueError(f"{axis} must be a nonempty unique list")
        for cell in states:
            c = _mapping(cell, f"{axis} cell")
            _keys(c, {"id", "factors"}, set(), "score cell")
            _name(c["id"], "score state id")
            values = _mapping(c["factors"], "score state factors")
            expected = {k for k, f in factors.items() if f["kind"] == kind}
            if set(values) != expected:
                raise ValueError(f"Score state {c['id']} does not assign every {kind} factor")
            for k, v in values.items():
                if v not in factors[k]["states"]:
                    raise ValueError(f"Unknown state {v!r} of factor {k}")
        axes.append([c["id"] for c in states])
        if len(set(axes[-1])) != len(axes[-1]):
            raise ValueError(f"Duplicate {axis} id")
    prompt = _mapping(lattice["prompt"], "prompt")
    _keys(prompt, {"templates", "aggregation"}, set(), "prompt")
    if not isinstance(prompt["templates"], list) or not prompt["templates"] or any(
            not isinstance(t, str) or not t.strip() for t in prompt["templates"]):
        raise ValueError("Prompt templates must be nonempty strings")
    if prompt["aggregation"] not in {"mean_normalized_embedding", "mean_score", "none"}:
        raise ValueError("Unsupported prompt aggregation")
    contrasts = _mapping(s["contrasts"], "contrasts")
    if not contrasts:
        raise ValueError("At least one signed contrast is required")
    for name, terms in contrasts.items():
        _name(name, "contrast")
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"Contrast {name} needs terms")
        seen = set()
        total = 0.0
        for term in terms:
            t = _mapping(term, "contrast term")
            _keys(t, {"image", "caption", "weight"}, set(), "contrast term")
            cell = (t["image"], t["caption"])
            if cell[0] not in axes[0] or cell[1] not in axes[1] or cell in seen:
                raise ValueError(f"Duplicate or unknown score cell in {name}")
            seen.add(cell)
            weight = _finite_number(t["weight"], "contrast weight")
            if weight == 0:
                raise ValueError("Contrast weights must be nonzero")
            total += weight
        if not math.isclose(total, 0, abs_tol=1e-12):
            raise ValueError(f"Contrast {name} must annihilate a constant score")
    th = _mapping(s["thresholds"], "thresholds")
    _keys(th, {"rule", "fractions"}, set(), "thresholds")
    if th["rule"] != "A3-v1-fixed-base":
        raise ValueError("Threshold rule must be A3-v1-fixed-base")
    fractions = _mapping(th["fractions"], "threshold fractions")
    for k, v in fractions.items():
        _name(k, "threshold name")
        _finite_number(v, f"threshold {k}", nonnegative=True)
    if not fractions:
        raise ValueError("Threshold fractions are required")
    clauses = s["clauses"]
    if not isinstance(clauses, list) or not clauses:
        raise ValueError("Clauses must be a nonempty list")
    names = []
    for c in clauses:
        c = _mapping(c, "clause")
        _keys(c, {"name", "kind"}, {"contrast", "lower", "upper", "bound", "binding"}, "clause")
        names.append(_name(c["name"], "clause name"))
        kind = c["kind"]
        if kind not in _KINDS:
            raise ValueError(f"Unsupported clause kind: {kind}")
        expected = {"lower": {"contrast", "bound"}, "magnitude": {"contrast", "bound"},
                    "band": {"contrast", "lower", "upper"},
                    "relative_leakage": {"contrast", "binding", "bound"},
                    "relative_context": {"contrast", "binding", "bound"}}[kind]
        if set(c) != expected | {"name", "kind"}:
            raise ValueError(f"Clause {c['name']} has invalid fields")
        if c["contrast"] not in contrasts:
            raise ValueError("Clause references unknown contrast")
        if kind in {"relative_leakage", "relative_context"}:
            bindings = c["binding"] if kind == "relative_context" else [c["binding"]]
            if kind == "relative_context" and (not isinstance(bindings, list) or len(bindings) != 2):
                raise ValueError("relative_context needs two binding contrasts")
            if any(b not in contrasts for b in bindings):
                raise ValueError("Unknown binding contrast")
        for key in expected & {"bound", "lower", "upper"}:
            if c[key] not in fractions:
                raise ValueError(f"Clause references undeclared threshold {c[key]}")
        if kind == "band" and fractions[c["lower"]] > fractions[c["upper"]]:
            raise ValueError("Band lower threshold exceeds upper threshold")
    if len(set(names)) != len(names):
        raise ValueError("Clause names must be unique")
    for field in ("anchors", "guards", "evidence_status"):
        if field in s and not isinstance(s[field], (str, dict)):
            raise ValueError(f"Invalid {field}")
    return s


@dataclass(frozen=True)
class CompiledRequirement:
    spec: dict
    unit: float
    base_model_id: str
    calibration_bank_id: str
    spec_hash: str
    weights: dict

    @property
    def shape(self):
        l = self.spec["score_lattice"]
        return (len(l["image_states"]), len(l["caption_states"]))

    @property
    def score_requests(self):
        """Ordered (image-state ID, caption-state ID) cells."""
        l = self.spec["score_lattice"]
        return [(i["id"], c["id"]) for i in l["image_states"] for c in l["caption_states"]]

    def _values_numpy(self, scores):
        x = np.asarray(scores, dtype=np.float64)
        if x.ndim < 2 or tuple(x.shape[-2:]) != self.shape or not np.isfinite(x).all():
            raise ValueError("Scores must be finite and match the score lattice")
        flat = x.reshape(*x.shape[:-2], -1) / self.unit
        return {k: flat @ w.reshape(-1) for k, w in self.weights.items()}

    def evaluate(self, scores):
        """Return normalized contrasts, clauses, complete pass; arrays retain batch axes."""
        values = self._values_numpy(scores)
        fractions = self.spec["thresholds"]["fractions"]
        clauses = {}
        for c in self.spec["clauses"]:
            kind, v = c["kind"], values[c["contrast"]]
            if kind == "lower":
                residual = v - fractions[c["bound"]]
                displayed = v
            elif kind == "magnitude":
                displayed = np.abs(v)
                residual = fractions[c["bound"]] - displayed
            elif kind == "band":
                displayed = v
                residual = np.minimum(v - fractions[c["lower"]], fractions[c["upper"]] - v)
            elif kind == "relative_leakage":
                binding = values[c["binding"]]
                displayed = np.abs(v)
                residual = fractions[c["bound"]] * np.maximum(binding, 0) - displayed
                # A ratio with no desired binding is not evidence of selectivity.
                residual = np.where(binding > 0, residual,
                                    -np.maximum.reduce(np.broadcast_arrays(displayed, -binding,
                                                                            np.full_like(displayed, 1e-8))))
            else:
                a, b = (values[z] for z in c["binding"])
                scale = (np.maximum(a, 0) + np.maximum(b, 0)) / 2
                displayed = np.abs(v)
                residual = fractions[c["bound"]] * scale - displayed
                residual = np.where(scale > 0, residual,
                                    -np.maximum.reduce(np.broadcast_arrays(displayed, -a, -b,
                                                                            np.full_like(displayed, 1e-8))))
            residual = np.where(np.abs(residual) <= 1e-12, 0., residual)
            clauses[c["name"]] = {"value": displayed, "residual": residual,
                                   "violation": np.maximum(-residual, 0), "passed": residual >= 0}
        passed = np.stack([c["passed"] for c in clauses.values()], axis=-1).all(axis=-1)
        return {"contrasts": values, "clauses": clauses, "passed": passed,
                "requirement_id": self.spec["id"], "requirement_hash": self.spec_hash,
                "calibration": {"unit": self.unit, "base_model_id": self.base_model_id,
                                "bank_id": self.calibration_bank_id}}

    def penalty(self, scores):
        """Differentiable squared-hinge losses per clause; no optimizer or guard."""
        import torch
        if not isinstance(scores, torch.Tensor) or tuple(scores.shape[-2:]) != self.shape or not bool(torch.isfinite(scores).all()):
            raise ValueError("Torch scores must be finite and match lattice")
        flat = scores.flatten(-2) / self.unit
        values = {k: flat @ scores.new_tensor(w.reshape(-1)) for k, w in self.weights.items()}
        fractions = self.spec["thresholds"]["fractions"]
        losses = {}
        for c in self.spec["clauses"]:
            kind, v = c["kind"], values[c["contrast"]]
            if kind == "lower":
                violation = (fractions[c["bound"]] - v).clamp_min(0)
            elif kind == "magnitude":
                violation = (v.abs() - fractions[c["bound"]]).clamp_min(0)
            elif kind == "band":
                violation = (fractions[c["lower"]] - v).clamp_min(0) + (v - fractions[c["upper"]]).clamp_min(0)
            elif kind == "relative_leakage":
                binding = values[c["binding"]]
                active = (v.abs() - fractions[c["bound"]] * binding).clamp_min(0)
                inactive = torch.maximum(torch.maximum(v.abs(), -binding),
                                         torch.full_like(v, 1e-8))
                violation = torch.where(binding > 0, active, inactive)
            else:
                a, b = (values[z] for z in c["binding"])
                scale = (a.clamp_min(0) + b.clamp_min(0)) / 2
                active = (v.abs() - fractions[c["bound"]] * scale).clamp_min(0)
                inactive = torch.maximum(torch.maximum(torch.maximum(v.abs(), -a), -b),
                                         torch.full_like(v, 1e-8))
                violation = torch.where(scale > 0, active, inactive)
            losses[c["name"]] = violation.square()
        return losses


def compile_requirement(spec, *, calibration_unit, base_model_id, calibration_bank_id):
    """Compile with an externally supplied, frozen base-model calibration unit."""
    validate_requirement(spec)
    spec=copy.deepcopy(spec)  # a caller cannot change a compiled requirement's thresholds by mutating its input
    unit = _finite_number(calibration_unit, "calibration_unit", positive=True)
    _name(base_model_id, "base model id")
    _name(calibration_bank_id, "calibration bank id")
    l = spec["score_lattice"]
    images = {cell["id"]: i for i, cell in enumerate(l["image_states"])}
    captions = {cell["id"]: i for i, cell in enumerate(l["caption_states"])}
    shape = (len(images), len(captions))
    weights = {}
    for name, terms in spec["contrasts"].items():
        w = np.zeros(shape, dtype=np.float64)
        for t in terms:
            w[images[t["image"]], captions[t["caption"]]] = t["weight"]
        weights[name] = w
    return CompiledRequirement(spec, unit, base_model_id, calibration_bank_id,
                               requirement_hash(spec), weights)
