"""Frozen-model behavioral pilot from verified feature caches; CPU only.

Selection and the analysis protocol are frozen before score formation. No
encoder, checkpoint selection, image editing or repair training occurs here.
"""
import argparse
from collections import defaultdict
import copy
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.core.specifications import load_requirement; from mirror.core.specifications import compile_requirement
from mirror.cases.color_binding.outcomes import caption_decisions; from mirror.cases.color_binding.outcomes import first_order_background; from mirror.cases.color_binding.outcomes import retrieval_query
from mirror.cases.color_binding.predict import cross_fitted; from mirror.cases.color_binding.predict import paired_auc_interval

OUT = ROOT / "clip/interbind_behavioral_pilot_20260922"
CACHE = ROOT / "clip/interbind_cache_v3_20260922"
QUAL = ROOT / "clip/interbind_source_quality_20260922"
NAT = ROOT / "clip/interbind_natural_localizer_quality_20260922"
CAL = ROOT / "data/models/calibration/calibration_frozen.json"
REVISION = ROOT / "FSE_VLM/plan/16_bounded_validation_and_pilot.md"
PAIRS = (("red", "blue"), ("green", "yellow"), ("purple", "orange"))
METHODS = ("blend90_luminance", "legacy_tint", "native_hsv")
EXCLUDE = {"ib2_pilot_routing_195998": "Visible person head/hat missing from cutout",
           "ib2_pilot_routing_278291": "Countertop fragment used as sink target"}


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def make_spec(family, pair, method, model, cal):
    spec = copy.deepcopy(load_requirement(Path(__file__).with_name("requirement_library") / f"{family}-v1.yaml"))
    color_map = dict(zip(("red", "blue"), pair))
    spec["id"] = f"{family}/pilot-{'-'.join(pair)}-{method}-v1"
    spec["scorer_output"] = "logit" if model.startswith("google_siglip") else "cosine"
    for name, factor in spec["factors"].items():
        if name == "context":
            factor["states"] = ["gray", "blue"]; factor["generator"] = "fixed_solid_context"
        else:
            factor["states"] = list(pair)
            if factor["kind"] == "image": factor["generator"] = method
    for axis in ("image_states", "caption_states"):
        for state in spec["score_lattice"][axis]:
            for factor, value in list(state["factors"].items()):
                state["factors"][factor] = ({"original": "gray", "scene_swap": "blue"}[value]
                                           if factor == "context" else color_map[value])
    fractions = spec["thresholds"]["fractions"]
    fractions.update(K=cal["kappa"], tau=cal["tau_fraction"])
    for key in ("rho", "gamma"):
        if key in fractions: fractions[key] = cal[key]
    if family == "background":
        spec["operational_measurement"] = "Signed binding in fixed gray and blue contexts; absolute and relative context gap. Legacy state IDs are ordered gray-0, gray-1, blue-0, blue-1."
        for c in spec["clauses"]:
            c["name"] = {"binding_original": "binding_gray", "binding_scene_swap": "binding_blue"}.get(c["name"], c["name"])
    return compile_requirement(spec, calibration_unit=cal["units"][model]["unit"],
                               base_model_id=cal["units"][model]["parent"], calibration_bank_id="natural_calibration_20260922")


def select_lattice(scores, row, family, pair_index, method, view):
    pair = PAIRS[pair_index]; prefix = f"{method}/{'-'.join(pair)}/"
    if family == "background":
        contexts = ("gray", "blue") if view == "audit" else (view,)
        states = [prefix + ctx + "/" + color for ctx in contexts for color in pair]
        captions = list(range(pair_index*2, pair_index*2+2))
    else:
        layout = "canvas" if view == "audit" else view
        states = [prefix + layout + "/" + "_".join(combo) for combo in itertools.product(pair, repeat=2)]
        captions = list(range(pair_index*4, pair_index*4+4))
    if len(row["state_names"]) != len(set(row["state_names"])): raise ValueError("Repeated cached state")
    indices = [row["state_names"].index(s) for s in states]
    return scores[np.ix_(indices, captions)]


def summarize_lattice(spec, scores, family):
    labels = (0, 1, 0, 1) if family == "background" else (0, 1, 2, 3)
    behavior = caption_decisions(scores[None], labels)
    evaluation = spec.evaluate(scores[None]); values = {k: float(v[0]) for k, v in evaluation["contrasts"].items()}
    if family == "background":
        binding = [values["d0"], values["d1"]]; unwanted = [abs(values["gap"])]; first = first_order_background(scores[None])
    else:
        binding = [v for k, v in values.items() if k.startswith("diag_")]
        unwanted = [abs(v) for k, v in values.items() if k.startswith("off_")]; first = {}
    clauses = {k: {kk: bool(vv[0]) if kk == "passed" else float(vv[0]) for kk, vv in v.items()}
               for k, v in evaluation["clauses"].items()}
    denom = sum(max(0., b) for b in binding)+sum(unwanted)
    return {"passed": bool(evaluation["passed"][0]), "clauses": clauses, "contrasts": values,
        "requirement_hash": spec.spec_hash, "audit_failure": int(behavior["any_failure"][0]),
        "audit_min_margin": float(behavior["minimum_margins"].min()/spec.unit),
        "audit_mean_margin": float(behavior["minimum_margins"].mean()/spec.unit),
        "audit_decision_failure_rate": float(behavior["decision_failure_rate"][0]),
        "violation_mean": float(np.mean([v["violation"] for v in clauses.values()])),
        "binding_shortfall": float(np.maximum(spec.spec["thresholds"]["fractions"]["K"]-np.asarray(binding), 0).mean()),
        "binding_mean": float(np.mean(binding)), "unwanted_strength": float(np.mean(unwanted)),
        "unwanted_share": sum(unwanted)/denom if denom else None,
        **{k: bool(v[0]) for k, v in first.items()}}


def freeze(out):
    cache_protocol = read(CACHE / "protocol.json")
    verified = read(CACHE / "verified_complete.json")
    if verified["verified_caches"] != 56: raise ValueError("Incomplete seven-subject cache")
    files = {str(REVISION): sha(REVISION), str(CAL): sha(CAL), str(CACHE / "verified_complete.json"): sha(CACHE / "verified_complete.json")}
    files.update(cache_protocol["input_hashes"])
    for item in verified["caches"]:
        path = ROOT / item["path"]
        if sha(path / "complete.json") != item["complete_sha256"]: raise ValueError("Cache completion changed")
        files[str(path / "complete.json")] = item["complete_sha256"]
        files.update({str(path / n): h for n, h in read(path / "complete.json")["files"].items()})
    for part in ("pilot", "reserve"):
        p = CACHE / "pixel_checks" / part / "checks.jsonl"; files[str(p)] = sha(p)
    for name in ("behavioral_pilot", "requirements", "outcomes", "predict"):
        p = Path(__file__).with_name(name + ".py"); files[str(p)] = sha(p)
    for family in ("background", "routing"):
        p = Path(__file__).with_name("requirement_library") / f"{family}-v1.yaml"; files[str(p)] = sha(p)
    dump(out / "protocol.json", {"version": "bounded-validation-frozen-model-pilot-v1", "inputs": files,
        "subjects": cache_protocol["subjects"], "primary_color_pair": list(PAIRS[0]),
        "primary_renderer": METHODS[0], "secondary_color_pairs": [list(p) for p in PAIRS[1:]],
        "secondary_renderers": list(METHODS[1:]), "excluded_anchors": EXCLUDE,
        "eligibility": "All states in a method/color lattice pass frozen pixel checks; two pre-score confirmed broken anchors excluded across all conditions/models",
        "pilot_only": True, "reserve": "Cached but not scored by this pilot",
        "audit_views": {"background": ["gray", "blue"], "routing": ["canvas"]},
        "outcome_views": {"background": ["green", "hue_cast", "scene_swap"], "routing": ["swapped_canvas", "in_situ"]},
        "natural": "Existing VG dataset-labeled restricted per-noun gallery; raw full images and existing unedited context crop; no judge-acceptance/color-threshold filter",
        "statistics": {"folds": 5, "logistic_C": 1., "bootstrap": 2000, "shuffle_null": 25, "seed": 20260922,
                       "numeric_baseline": ["audit_failure", "audit_min_margin", "audit_mean_margin"],
                       "categorical": ["model", "family", "object_class", "color_pair", "outcome_view"],
                       "numeric_extension": ["violation_mean", "binding_shortfall", "unwanted_strength"]},
        "cpu_only": True, "gpu_encoding": False, "repair_training": False,
        "interpretation": "Pilot prediction, not causal proof; do not change subsets/calibration/hypotheses to make a favorable number",
        "all_prior_findings_preserved": True})
    print("FROZEN behavioral pilot", sha(out / "protocol.json"), flush=True)


def score(out):
    p = read(out / "protocol.json"); verify_files(p["inputs"]); cal = read(CAL)
    metadata = {r["anchor_id"]: r for r in lines(QUAL / "pilot/accepted_rows.jsonl")}
    checks = {r["anchor_id"]: r for r in lines(CACHE / "pixel_checks/pilot/checks.jsonl")}
    natural_rows = {r["id"]: r for r in lines(NAT / "pilot/accepted_rows.jsonl")}
    audit_rows = []; outcome_rows = []; natural_results = []; eligibility = []; started = time.monotonic()
    for model in p["subjects"]:
        for family in ("background", "routing"):
            dest = CACHE / "features" / model / "pilot" / family
            images = np.load(dest / "images.npy", mmap_mode="r"); texts = np.load(dest / "texts.npy", mmap_mode="r")
            details = read(dest / "protocol.json")["details"]; indices = lines(dest / "index.jsonl")
            all_scores = []
            for row in indices:
                offset, count = row["image_offset"], row["image_count"]
                scores = np.asarray(images[offset:offset+count]) @ np.asarray(texts[row["text_indices"]]).T
                scores = scores.astype(np.float64)*details["score_scale"]+details["score_bias"]
                if not np.isfinite(scores).all(): raise ValueError("Nonfinite scores")
                all_scores.append(scores); meta = metadata[row["anchor_id"]]
                for method in METHODS:
                    for pair_index, pair in enumerate(PAIRS):
                        prefix = f"{method}/{'-'.join(pair)}"
                        accepted = row["anchor_id"] not in EXCLUDE and checks[row["anchor_id"]]["lattice_pass"][prefix]
                        eligibility.append({"model": model, "anchor_id": row["anchor_id"], "family": family,
                            "method": method, "color_pair": "-".join(pair), "accepted": accepted,
                            "reason": EXCLUDE.get(row["anchor_id"], None if accepted else "pixel_lattice_failure")})
                        if not accepted: continue
                        spec = make_spec(family, pair, method, model, cal)
                        lattice = select_lattice(scores, row, family, pair_index, method, "audit")
                        result = summarize_lattice(spec, lattice, family)
                        identity = {"model": model, "family": family, "method": method, "anchor_id": row["anchor_id"],
                            "source_id": str(meta["source_ids"][0]), "object_class": "+".join(meta["objects"]),
                            "color_pair": "-".join(pair), "unit": spec.unit,
                            "seen_class_stratum": meta.get("seen_class_stratum", "not_applicable")}
                        audit_rows.append({**identity, **result})
                        features = {k: result[k] for k in p["statistics"]["numeric_baseline"] + p["statistics"]["numeric_extension"]}
                        for view in p["outcome_views"][family]:
                            lattice = select_lattice(scores, row, family, pair_index, method, view)
                            labels = (0, 1) if family == "background" else (0, 1, 2, 3)
                            behavior = caption_decisions(lattice[None], labels)
                            outcome_rows.append({**identity, **features, "outcome_view": view,
                                "failure": int(behavior["any_failure"][0]),
                                "decision_failure_rate": float(behavior["decision_failure_rate"][0])})
            target = out / "scores" / model / family; target.mkdir(parents=True, exist_ok=True)
            with (target / "scores.npy").open("xb") as stream: np.save(stream, np.stack(all_scores), allow_pickle=False)
            jsonl(target / "index.jsonl", indices)
            dump(target / "complete.json", {"score_sha256": sha(target / "scores.npy"), "index_sha256": sha(target / "index.jsonl"),
                "input_cache_complete_sha256": sha(dest / "complete.json"), "shape": list(np.stack(all_scores).shape),
                "score_rule": "Raw cosine or SigLIP pre-sigmoid logit; rows/states/text_indices are explicit",
                "no_model_outcome_selection": True})
        for view in ("natural_foreground_context", "natural_full_image"):
            dest = CACHE / "features" / model / "pilot" / view
            images = np.load(dest / "images.npy", mmap_mode="r"); texts = np.load(dest / "texts.npy", mmap_mode="r")
            details = read(dest / "protocol.json")["details"]
            gallery = {r["row_id"]: r for r in lines(dest / "gallery_index.jsonl")}
            for query in lines(dest / "index.jsonl"):
                ids = query["gallery_ids"]; offsets = [gallery[i]["image_offset"] for i in ids]
                scores = np.asarray(images[offsets]) @ np.asarray(texts[query["text_indices"][0]])
                scores = scores.astype(np.float64)*details["score_scale"]+details["score_bias"]
                positive = [i for i in ids if natural_rows[i]["color"] == query["color"]]
                negative = [i for i in ids if natural_rows[i]["color"] != query["color"]]
                result = retrieval_query(scores, ids, positive, negative)
                natural_results.append({"model": model, "view": view, **query, **result,
                    "scores": dict(zip(ids, scores.tolist())), "positive_ids": positive, "incompatible_ids": negative})
        print("SCORED CPU", model, flush=True)
    jsonl(out / "audit_rows.jsonl", audit_rows); jsonl(out / "outcome_rows.jsonl", outcome_rows)
    jsonl(out / "natural_queries.jsonl", natural_results); jsonl(out / "eligibility.jsonl", eligibility)
    dump(out / "scores_complete.json", {"protocol_sha256": sha(out / "protocol.json"), "seconds": time.monotonic()-started,
        "audit_rows": len(audit_rows), "outcome_rows": len(outcome_rows), "natural_queries": len(natural_results),
        "files": {name: sha(out / name) for name in ("audit_rows.jsonl", "outcome_rows.jsonl", "natural_queries.jsonl", "eligibility.jsonl")},
        "gpu_used": False, "reserve_scored": False})


def shuffled_features(frame, columns, seed):
    result = frame.copy(); rng = np.random.default_rng(seed)
    keys = ["family", "source_id", "model", "color_pair"]
    unique = frame.drop_duplicates(keys).set_index(keys)
    for family in frame["family"].unique():
        sources = sorted(frame.loc[frame.family == family, "source_id"].unique())
        mapping = dict(zip(sources, rng.permutation(sources)))
        for i in frame.index[frame.family == family]:
            r = frame.loc[i]; key = (family, mapping[r.source_id], r.model, r.color_pair)
            result.loc[i, columns] = unique.loc[key, columns].to_numpy()
    return result


def analyze(out):
    p = read(out / "protocol.json"); completed = read(out / "scores_complete.json")
    verify_files({str(out / name): h for name, h in completed["files"].items()})
    frame = pd.DataFrame(lines(out / "outcome_rows.jsonl"))
    primary = frame[(frame.method == METHODS[0]) & (frame.color_pair == "red-blue")].reset_index(drop=True)
    stats = p["statistics"]; numeric = stats["numeric_baseline"]; extra = stats["numeric_extension"]; cats = stats["categorical"]
    results = {}; predictions = []; baseline = None; extended = None
    for family in ("pooled", "background", "routing"):
        f = primary if family == "pooled" else primary[primary.family == family].reset_index(drop=True)
        try:
            b = cross_fitted(f, numeric, cats); e = cross_fitted(f, numeric+extra, cats)
            ci = paired_auc_interval(f.failure, b["predictions"], e["predictions"], f.source_id, n=stats["bootstrap"])
            results[family] = {"n_rows": len(f), "n_sources": f.source_id.nunique(), "failure_rate": float(f.failure.mean()),
                "baseline_auc": b["auc"], "extended_auc": e["auc"], "baseline_brier": b["brier"], "extended_brier": e["brier"],
                "interval": ci, "fold_audit": b["fold_audit"]}
            if family == "pooled": baseline, extended = b, e
            for (_, row), bp, ep, fold in zip(f.iterrows(), b["predictions"], e["predictions"], b["folds"]):
                predictions.append({"analysis": family, **row.to_dict(), "baseline_prediction": float(bp),
                                    "extended_prediction": float(ep), "fold": int(fold)})
        except ValueError as exc:
            results[family] = {"status": "not_estimable", "reason": str(exc), "n_rows": len(f), "n_sources": f.source_id.nunique()}
        print("B4", family, results[family], flush=True)
    null = []
    if baseline is not None:
        for k in range(stats["shuffle_null"]):
            f = shuffled_features(primary, extra, stats["seed"]+k)
            e = cross_fitted(f, numeric+extra, cats)
            null.append({"seed": stats["seed"]+k, "delta_auc": e["auc"]-baseline["auc"]})
            if (k+1) % 5 == 0: print("NULL", k+1, "/", stats["shuffle_null"], flush=True)
        observed = extended["auc"]-baseline["auc"]
        results["source_shuffle_null"] = {"replicates": len(null), "observed_delta_auc": observed,
            "null_delta_auc": [r["delta_auc"] for r in null],
            "empirical_one_sided_p": (1+sum(r["delta_auc"] >= observed for r in null))/(1+len(null)),
            "scope": "Source-block shuffled predictor negative control; not a causal test"}
    summary = []
    for keys, group in pd.DataFrame(lines(out / "audit_rows.jsonl")).groupby(["model", "family", "method", "color_pair"]):
        correct = group[group.audit_failure == 0]
        summary.append(dict(zip(("model", "family", "method", "color_pair"), keys)) | {
            "n": len(group), "n_sources": group.source_id.nunique(), "conformance": float(group.passed.mean()),
            "caption_strict": float(1-group.audit_failure.mean()), "binding_mean": float(group.binding_mean.mean()),
            "unwanted_strength": float(group.unwanted_strength.mean()), "unwanted_share": float(group.unwanted_share.mean()),
            "caption_correct_n": len(correct), "hidden_violation_rate": float(1-correct.passed.mean()) if len(correct) else None})
    natural_summary = []
    for keys, group in pd.DataFrame(lines(out / "natural_queries.jsonl")).groupby(["model", "view"]):
        natural_summary.append({"model": keys[0], "view": keys[1], "n_queries": len(group),
            "r1": float(np.mean([r["1"] for r in group.hits])), "r5": float(np.mean([r["5"] for r in group.hits])),
            "scope": "Dataset-labeled restricted per-noun query-hit rates; no full-benchmark or exhaustive relevance claim"})
    jsonl(out / "cross_fitted_predictions.jsonl", predictions); jsonl(out / "prevalence_summary.jsonl", summary)
    jsonl(out / "natural_summary.jsonl", natural_summary); jsonl(out / "source_shuffle_null.jsonl", null)
    dump(out / "prediction_summary.json", results)
    dump(out / "analysis_complete.json", {"protocol_sha256": sha(out / "protocol.json"),
        "scores_complete_sha256": sha(out / "scores_complete.json"), "gpu_used": False,
        "reserve_scored": False, "new_repair_training": False,
        "files": {n: sha(out / n) for n in ("cross_fitted_predictions.jsonl", "prevalence_summary.jsonl", "natural_summary.jsonl", "source_shuffle_null.jsonl", "prediction_summary.json")}})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["freeze", "score", "analyze"])
    parser.add_argument("--out", type=Path, default=OUT); args = parser.parse_args(); log(args.out, "start")
    try:
        with threadpool_limits(limits=2):
            {"freeze": freeze, "score": score, "analyze": analyze}[args.action](args.out)
    except BaseException as exc: log(args.out, "failed", error=repr(exc)); raise
    log(args.out, "complete")


if __name__ == "__main__": main()
