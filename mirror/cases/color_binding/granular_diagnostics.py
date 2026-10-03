"""Post-specified granular pilot diagnostics from immutable saved scores, CPU only."""
import argparse
from collections import defaultdict
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.behavioral_pilot import OUT as PILOT; from mirror.cases.color_binding.behavioral_pilot import PAIRS; from mirror.cases.color_binding.behavioral_pilot import lines; from mirror.cases.color_binding.behavioral_pilot import select_lattice; from mirror.cases.color_binding.behavioral_pilot import shuffled_features
from mirror.cases.color_binding.outcomes import caption_decisions
from mirror.cases.color_binding.predict import cross_fitted; from mirror.cases.color_binding.predict import paired_auc_interval

OUT = ROOT / "clip/interbind_granular_pilot_20260922"
REVISION = ROOT / "FSE_VLM/plan/17_granular_behavioral_diagnostics.md"
IDENTITY = ("model", "family", "method", "anchor_id", "source_id", "object_class", "color_pair", "unit")
NUMERIC = ["audit_decision_failure_rate", "audit_min_margin", "audit_mean_margin",
           "state_audit_failure_rate", "state_audit_min_margin", "state_audit_mean_margin"]
EXTRA = ["violation_mean", "binding_shortfall", "unwanted_strength"]
CATEGORICAL = ["model", "object_class", "view", "color_state"]


def decision_rows(lattice, labels, unit):
    """One image decision, retaining all competitors; ties fail without tie-breaking."""
    matrix = np.asarray(lattice, dtype=float)
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("Positive fixed calibration unit required")
    result = caption_decisions(matrix[None], labels)
    rows = []
    for i, label in enumerate(labels):
        candidates = [j for j in range(matrix.shape[1]) if j != label]
        margins = [float(matrix[i, label] - matrix[i, j]) for j in candidates]
        rows.append({"state_index": i, "correct_caption": int(label),
                     "failure": int(not result["correct"][0, i]),
                     "margin_raw": float(result["minimum_margins"][0, i]),
                     "margin_normalized": float(result["minimum_margins"][0, i] / unit),
                     "wrong_caption_indices": candidates, "pair_margins_raw": margins,
                     "pairwise_win_rate": float(np.mean(np.asarray(margins) > 0))})
    return rows


def comparison_type(family, correct, wrong):
    if family == "background":
        return "object_color_flip"
    states = list(itertools.product(range(2), repeat=2))
    a, b = states[correct], states[wrong]
    changes = [i for i in (0, 1) if a[i] != b[i]]
    if changes == [0]: return "word_1_flip"
    if changes == [1]: return "word_2_flip"
    if changes == [0, 1]:
        return "color_assignment_exchange" if a[0] != a[1] else "both_words_same_color_flip"
    raise ValueError("Expected a genuinely incorrect caption")


def cluster_means(values, sources, n=2000, seed=20260922):
    """Row-weighted means, resampling entire source blocks (not individual decisions)."""
    values = np.asarray(values, float)
    if values.ndim == 1: values = values[:, None]
    unique, inverse = np.unique(sources, return_inverse=True)
    if not len(unique) or len(values) != len(inverse) or not np.isfinite(values).all():
        raise ValueError("Finite aligned rows and source identities required")
    sizes = np.bincount(inverse)
    sums = np.zeros((len(unique), values.shape[1]))
    np.add.at(sums, inverse, values)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=n)
    draws = (weights @ sums) / (weights @ sizes)[:, None]
    return {"mean": values.mean(axis=0).tolist(),
            "ci95": np.quantile(draws, [.025, .975], axis=0).T.tolist(),
            "n_sources": len(unique), "bootstrap_replicates": n,
            "scope": "Whole-source resampling; conditional on these fixed pretrained subjects; no seed inference"}


def freeze(out):
    files = {str(REVISION): sha(REVISION)}
    for name in ("protocol.json", "scores_complete.json", "analysis_complete.json", "integrity_audit.json",
                 "audit_rows.jsonl", "outcome_rows.jsonl"):
        files[str(PILOT / name)] = sha(PILOT / name)
    for name in ("granular_diagnostics", "behavioral_pilot", "outcomes", "predict", "io"):
        path = Path(__file__).with_name(name + ".py"); files[str(path)] = sha(path)
    for path in sorted((PILOT / "scores").glob("*/*/complete.json")):
        complete = read(path)
        files[str(path)] = sha(path)
        files[str(path.parent / "scores.npy")] = complete["score_sha256"]
        files[str(path.parent / "index.jsonl")] = complete["index_sha256"]
    verify_files(files)
    dump(out / "protocol.json", {
        "version": "author-requested-granular-pilot-v1", "input_hashes": files,
        "status": "Post-specified after strict-endpoint pilot; not confirmation",
        "primary": {"method": "blend90_luminance", "color_pair": "red-blue"},
        "baseline_numeric": NUMERIC, "interaction_extension": EXTRA, "categorical": CATEGORICAL,
        "statistics": {"bootstrap": 2000, "shuffle_replicates": 25, "seed": 20260922, "folds": 5, "logistic_C": 1},
        "prediction_unit": "One independent-view image decision; source-grouped folds; separate requirements",
        "subjects": read(PILOT / "protocol.json")["subjects"],
        "new_inference": False, "gpu_used": False, "reserve_scored": False, "repair_training": False,
        "interpretation": "Behavior and interaction are distinct; no outcome-dependent selection or recalibration"})
    print("FROZEN granular diagnostics", sha(out / "protocol.json"), flush=True)


def extract(out):
    p = read(out / "protocol.json"); verify_files(p["input_hashes"])
    audits = [r for r in lines(PILOT / "audit_rows.jsonl") if all(r[k] == v for k, v in p["primary"].items())]
    tables = {}
    for model in p["subjects"]:
        for family in ("background", "routing"):
            path = PILOT / "scores" / model / family
            scores = np.load(path / "scores.npy", mmap_mode="r")
            for i, r in enumerate(lines(path / "index.jsonl")):
                tables[model, family, r["anchor_id"]] = (scores[i], r)
    views = read(PILOT / "protocol.json")["outcome_views"]
    decisions, anchors, clauses = [], [], []
    for audit in audits:
        ident = {k: audit[k] for k in IDENTITY}; family = audit["family"]
        scores, index = tables[audit["model"], family, audit["anchor_id"]]
        lattice = select_lattice(scores, index, family, 0, p["primary"]["method"], "audit")
        labels = [0, 1, 0, 1] if family == "background" else [0, 1, 2, 3]
        audit_decisions = decision_rows(lattice, labels, audit["unit"])
        for view in ["audit", *views[family]]:
            if view == "audit": rows = audit_decisions
            else:
                lattice = select_lattice(scores, index, family, 0, p["primary"]["method"], view)
                rows = decision_rows(lattice, [0, 1] if family == "background" else [0, 1, 2, 3], audit["unit"])
            for r in rows:
                matching = [a for a in audit_decisions if a["correct_caption"] == r["correct_caption"]]
                state_margins = [a["margin_normalized"] for a in matching]
                state = (PAIRS[0][r["correct_caption"]] if family == "background" else
                         "_".join(list(itertools.product(PAIRS[0], repeat=2))[r["correct_caption"]]))
                decisions.append({**ident, **r, "view": view, "color_state": state,
                    "candidate_count": 2 if family == "background" else 4,
                    **{k: audit[k] for k in NUMERIC[:3] + EXTRA},
                    "state_audit_failure_rate": float(np.mean([a["failure"] for a in matching])),
                    "state_audit_min_margin": min(state_margins), "state_audit_mean_margin": float(np.mean(state_margins))})
            count = sum(r["failure"] for r in rows)
            anchors.append({**ident, "view": view, "error_count": count, "decision_count": len(rows),
                            "accuracy": 1-count/len(rows), "all_corners_correct": count == 0})
        for name, result in audit["clauses"].items():
            kind = ("binding" if name.startswith(("bind_", "binding_")) else
                    "relative_unwanted" if "relative" in name else "absolute_unwanted")
            clauses.append({**ident, "clause": name, "clause_group": kind, **result})
    # Reconcile with the frozen earlier outputs; do not silently redefine decisions.
    previous = {(r["model"], r["anchor_id"], r["outcome_view"]): r for r in lines(PILOT / "outcome_rows.jsonl")
                if all(r[k] == v for k, v in p["primary"].items())}
    for a in anchors:
        if a["view"] == "audit": continue
        r = previous[a["model"], a["anchor_id"], a["view"]]
        assert abs((1-a["accuracy"])-r["decision_failure_rate"]) < 1e-12
        assert int(not a["all_corners_correct"]) == r["failure"]
    jsonl(out / "decision_rows.jsonl", decisions); jsonl(out / "anchor_rows.jsonl", anchors)
    jsonl(out / "clause_rows.jsonl", clauses)
    dump(out / "extraction_complete.json", {"protocol_sha256": sha(out / "protocol.json"),
        "decisions": len(decisions), "anchor_model_views": len(anchors), "clauses": len(clauses),
        "matches_previous_decisions": True, "new_inference": False, "gpu_used": False,
        "files": {n: sha(out / n) for n in ("decision_rows.jsonl", "anchor_rows.jsonl", "clause_rows.jsonl")}})
    print("EXTRACTED", len(decisions), "image decisions", flush=True)


def summaries(out):
    p = read(out / "protocol.json"); verify_files(p["input_hashes"])
    done = read(out / "extraction_complete.json")
    verify_files({str(out / k): v for k, v in done["files"].items()})
    frame = pd.DataFrame(lines(out / "decision_rows.jsonl"))
    anchors = pd.DataFrame(lines(out / "anchor_rows.jsonl"))
    summaries, pairs, clause_summary = [], [], []
    # Per-model results plus a clearly marked fixed-model descriptive aggregate.
    groups = list(frame.groupby(["model", "family", "view"]))
    groups += [(("all_fixed_subjects", *k), g) for k, g in frame.groupby(["family", "view"])]
    for (model, family, view), g in groups:
        a = anchors[(anchors.family == family) & (anchors.view == view)]
        if model != "all_fixed_subjects": a = a[a.model == model]
        metrics = ["accuracy", "margin_raw", "margin_normalized", "pairwise_win_rate"]
        values = np.column_stack([1-g.failure, g.margin_raw, g.margin_normalized, g.pairwise_win_rate])
        intervals = cluster_means(values, g.source_id, p["statistics"]["bootstrap"])
        summaries.append({"model": model, "family": family, "view": view, "n_decisions": len(g),
            "n_sources": g.source_id.nunique(), "n_anchor_model_cases": len(a), "error_count": int(g.failure.sum()),
            "metrics": {k: {"mean": intervals["mean"][i], "ci95": intervals["ci95"][i]} for i, k in enumerate(metrics)},
            "margin_raw_quartiles": np.quantile(g.margin_raw, [.25, .5, .75]).tolist(),
            "margin_normalized_quartiles": np.quantile(g.margin_normalized, [.25, .5, .75]).tolist(),
            "errors_per_anchor": {str(i): int((a.error_count == i).sum()) for i in range(int(a.decision_count.max())+1)},
            "strict_complete_accuracy_secondary": float(a.all_corners_correct.mean()),
            "interval_scope": intervals["scope"]})
        for kind in sorted({comparison_type(family, int(r.correct_caption), j) for r in g.itertuples()
                            for j in r.wrong_caption_indices}):
            margins, sources = [], []
            for r in g.itertuples():
                for j, m in zip(r.wrong_caption_indices, r.pair_margins_raw):
                    if comparison_type(family, int(r.correct_caption), j) == kind:
                        margins.append(m/r.unit); sources.append(r.source_id)
            result = cluster_means(np.column_stack([np.asarray(margins)>0, margins]), sources, p["statistics"]["bootstrap"])
            pairs.append({"model": model, "family": family, "view": view, "comparison": kind,
                          "n_comparisons": len(margins), "n_sources": result["n_sources"],
                          "win_rate": result["mean"][0], "win_rate_ci95": result["ci95"][0],
                          "mean_normalized_margin": result["mean"][1], "scope": "Secondary pairwise diagnostic; not top-1 caption accuracy"})
    for keys, g in pd.DataFrame(lines(out / "clause_rows.jsonl")).groupby(["model", "family", "clause", "clause_group"]):
        result = cluster_means(np.column_stack([g.passed, g.violation]), g.source_id, p["statistics"]["bootstrap"])
        clause_summary.append(dict(zip(("model", "family", "clause", "clause_group"), keys)) |
                              {"n": len(g), "n_sources": result["n_sources"], "pass_rate": result["mean"][0],
                               "pass_rate_ci95": result["ci95"][0], "mean_violation": result["mean"][1]})
    jsonl(out / "behavior_summary.jsonl", summaries); jsonl(out / "pairwise_summary.jsonl", pairs)
    jsonl(out / "clause_summary.jsonl", clause_summary)
    dump(out / "summaries_complete.json", {"files": {n: sha(out / n) for n in
        ("behavior_summary.jsonl", "pairwise_summary.jsonl", "clause_summary.jsonl")}, "gpu_used": False})
    for r in summaries:
        if r["model"] == "all_fixed_subjects":
            print(r["family"], r["view"], "accuracy", r["metrics"]["accuracy"], "errors", r["errors_per_anchor"], flush=True)


def predict(out):
    p = read(out / "protocol.json"); verify_files(p["input_hashes"])
    done = read(out / "extraction_complete.json")
    verify_files({str(out / k): v for k, v in done["files"].items()})
    frame = pd.DataFrame(lines(out / "decision_rows.jsonl")); frame = frame[frame.view != "audit"]
    results, predictions, nulls = {}, [], []
    for family in ("background", "routing"):
        f = frame[frame.family == family].reset_index(drop=True)
        try:
            b = cross_fitted(f, NUMERIC, CATEGORICAL); e = cross_fitted(f, NUMERIC+EXTRA, CATEGORICAL)
        except ValueError as exc:
            results[family] = {"status": "not_estimable", "reason": str(exc)}; continue
        ci = paired_auc_interval(f.failure, b["predictions"], e["predictions"], f.source_id, n=p["statistics"]["bootstrap"])
        loss = cluster_means((e["predictions"]-f.failure.to_numpy())**2 - (b["predictions"]-f.failure.to_numpy())**2,
                             f.source_id, p["statistics"]["bootstrap"])
        results[family] = {"n_decisions": len(f), "n_sources": f.source_id.nunique(),
            "error_count": int(f.failure.sum()), "failure_rate": float(f.failure.mean()),
            "baseline_auc": b["auc"], "extended_auc": e["auc"], "auc_interval": ci,
            "baseline_brier": b["brier"], "extended_brier": e["brier"],
            "brier_change": {"effect": loss["mean"][0], "ci95": loss["ci95"][0], "negative_is_better": True},
            "fold_audit": b["fold_audit"]}
        for (_, r), bp, ep, fold in zip(f.iterrows(), b["predictions"], e["predictions"], b["folds"]):
            predictions.append({**r.to_dict(), "baseline_prediction": float(bp), "extended_prediction": float(ep), "fold": int(fold)})
        values = []
        for k in range(p["statistics"]["shuffle_replicates"]):
            seed = p["statistics"]["seed"]+k
            shuffled = shuffled_features(f, EXTRA, seed)
            change = cross_fitted(shuffled, NUMERIC+EXTRA, CATEGORICAL)["auc"]-b["auc"]
            values.append(change); nulls.append({"family": family, "seed": seed, "delta_auc": change})
            if (k+1) % 5 == 0: print("SOURCE NULL", family, k+1, flush=True)
        observed = e["auc"]-b["auc"]
        results[family]["source_shuffle_null"] = {"replicates": len(values), "observed_delta_auc": observed,
            "empirical_one_sided_p": (1+sum(v >= observed for v in values))/(1+len(values))}
        print("PER-DECISION PREDICTION", family, results[family], flush=True)
    jsonl(out / "cross_fitted_predictions.jsonl", predictions); jsonl(out / "source_shuffle_null.jsonl", nulls)
    dump(out / "prediction_summary.json", results)
    dump(out / "prediction_complete.json", {"protocol_sha256": sha(out / "protocol.json"),
        "post_specified": True, "gpu_used": False, "reserve_scored": False, "new_repair_training": False,
        "files": {n: sha(out / n) for n in ("cross_fitted_predictions.jsonl", "source_shuffle_null.jsonl", "prediction_summary.json")}})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["freeze", "extract", "summarize", "predict"])
    parser.add_argument("--out", type=Path, default=OUT); args = parser.parse_args(); log(args.out, "start")
    try:
        with threadpool_limits(limits=2):
            {"freeze": freeze, "extract": extract, "summarize": summaries, "predict": predict}[args.action](args.out)
    except BaseException as exc: log(args.out, "failed", error=repr(exc)); raise
    log(args.out, "complete")


if __name__ == "__main__": main()
