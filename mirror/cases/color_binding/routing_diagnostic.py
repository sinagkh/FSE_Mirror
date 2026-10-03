"""Complete, post-specified routing diagnostic. Reuses pilot scores; never trains."""
import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score; from sklearn.metrics import brier_score_loss
from threadpoolctl import threadpool_limits
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.behavioral_pilot import OUT as PILOT; from mirror.cases.color_binding.behavioral_pilot import CAL; from mirror.cases.color_binding.behavioral_pilot import lines; from mirror.cases.color_binding.behavioral_pilot import select_lattice
from mirror.cases.color_binding.routing_context_coverage import all_contexts
from mirror.core.specifications import load_requirement; from mirror.core.specifications import compile_requirement
from mirror.cases.color_binding.outcomes import caption_decisions
from mirror.cases.color_binding.granular_diagnostics import cluster_means
from mirror.cases.color_binding.predict import cross_fitted; from mirror.cases.color_binding.predict import paired_auc_interval

OUT = ROOT / "clip/interbind_routing_diagnostic_20260923"
REVISION = ROOT / "FSE_VLM/plan/18_complete_routing_diagnostic.md"
LIBRARY = ROOT / "mirror/core/templates/routing-v1.yaml"
BASE = ["audit_decision_failure_rate", "audit_min_margin", "audit_mean_margin",
        "state_audit_failure_rate", "state_audit_min_margin", "state_audit_mean_margin",
        "audit_swap_rb_margin", "audit_swap_br_margin"]
CATS = ["model", "object_class", "view", "color_state"]
REGIMES = ("nonpositive_response", "positive_response_preference_dominates", "both_exchange_decisions_correct")


def full_spec(calibration):
    old = load_requirement(LIBRARY); new = copy.deepcopy(old)
    original = compile_requirement(old, calibration_unit=1., base_model_id="symbolic", calibration_bank_id="symbolic")
    contexts = all_contexts(); names = {}; records = []
    states = [s["id"] for s in old["score_lattice"]["image_states"]]
    for c in contexts:
        matched = [k for k, w in original.weights.items() if np.array_equal(w, c["weights"])]
        name = matched[0] if matched else c["name"]
        key = (c["image_factor"], c["word_factor"], c["other_image_color_index"], c["other_word_color_index"])
        names[key] = name
        records.append({**{k: v for k, v in c.items() if k != "weights"}, "contrast": name,
                        "legacy": bool(matched), "weights": c["weights"].tolist()})
        if not matched:
            new["contrasts"][name] = [{"image": states[i], "caption": states[j], "weight": int(c["weights"][i, j])}
                                      for i, j in zip(*np.where(c["weights"] != 0))]
    for r in records:
        if r["kind"] == "unwanted":
            i, a = r["image_factor"], r["other_image_color_index"]
            r["relative_reference"] = names[i, i, a, a]
        if r["legacy"]: continue
        name = r["contrast"]
        if r["kind"] == "binding":
            new["clauses"].append({"name": "bind_"+name, "kind": "lower", "contrast": name, "bound": "K"})
        else:
            new["clauses"] += [
                {"name": "leak_"+name, "kind": "magnitude", "contrast": name, "bound": "tau"},
                {"name": "relative_"+name, "kind": "relative_leakage", "contrast": name,
                 "binding": r["relative_reference"], "bound": "rho"}]
    new["id"] = "routing/context-complete-v2"
    new["operational_measurement"] = (
        "All 16 conditional image-factor x word-factor second differences. Eight desired binding floors, "
        "eight absolute cross-effect bounds and eight relative bounds. Each relative cross effect is referenced "
        "to the same image factor's own-word binding with the other object's caption color matching its image color. "
        "Complete conformance certifies differential response under K>tau, not absolute exchange rankings.")
    new["evidence_status"] = "Post-specified diagnostic extension; frozen calibration; not a tested repair objective"
    new["thresholds"]["fractions"].update(K=calibration["kappa"], tau=calibration["tau_fraction"], rho=calibration["rho"])
    for key in ("slot_1_color", "slot_2_color"):
        new["factors"][key]["generator"] = "blend90_luminance"
    assert len(new["contrasts"]) == 16 and len(new["clauses"]) == 24
    # Old relative mappings must remain exactly those given by the stated rule.
    by_name = {r["contrast"]: r for r in records}
    for c in old["clauses"]:
        if c["kind"] == "relative_leakage":
            assert by_name[c["contrast"]]["relative_reference"] == c["binding"]
    return new, records, names


def decomposition(scores, unit=1.):
    x = np.asarray(scores, float)
    if x.shape != (4, 4) or not np.isfinite(x).all() or unit <= 0:
        raise ValueError("Finite 4x4 routing lattice and positive unit required")
    q0, q1 = x[1, 1]-x[1, 2], x[2, 1]-x[2, 2]
    e, b = (q0-q1)/2, (q0+q1)/2
    margins = np.array([q0, -q1])
    regime = REGIMES[0] if e <= 0 else REGIMES[1] if e <= abs(b) else REGIMES[2]
    np.testing.assert_allclose(margins, [e+b, e-b], atol=1e-12)
    return {"response_raw": float(e), "preference_raw": float(b),
            "response": float(e/unit), "preference": float(b/unit),
            "absolute_preference": float(abs(b)/unit), "surplus": float((e-abs(b))/unit),
            "regime": regime, "exchange_margins_raw": margins.tolist(),
            "exchange_margins": (margins/unit).tolist(),
            "exchange_errors": (margins <= 0).astype(int).tolist(),
            "exchange_accuracy": float((margins > 0).mean())}


def feature_columns(definition, contexts):
    legacy = set(load_requirement(LIBRARY)["contrasts"])
    old = ["signed_"+k for k in definition["contrasts"] if k in legacy]
    old += ["magnitude_"+r["contrast"] for r in contexts if r["legacy"] and r["kind"] == "unwanted"]
    full = ["signed_"+k for k in definition["contrasts"]]
    full += ["magnitude_"+r["contrast"] for r in contexts if r["kind"] == "unwanted"]
    return old, full


def freeze(out):
    cal = read(CAL); definition, contexts, _ = full_spec(cal)
    paths = [REVISION, CAL, LIBRARY, PILOT / "audit_rows.jsonl", PILOT / "protocol.json"]
    paths += [Path(__file__).with_name(n+".py") for n in
              ("routing_diagnostic", "routing_context_coverage", "requirements", "predict", "outcomes", "granular_diagnostics", "behavioral_pilot")]
    inputs = {str(p): sha(p) for p in paths}
    for path in sorted((PILOT / "scores").glob("*/routing/complete.json")):
        info = read(path); inputs[str(path)] = sha(path)
        inputs[str(path.parent / "scores.npy")] = info["score_sha256"]
        inputs[str(path.parent / "index.jsonl")] = info["index_sha256"]
    verify_files(inputs); out.mkdir(parents=True, exist_ok=True)
    with (out / "routing-context-complete-v2.yaml").open("x") as stream:
        yaml.safe_dump(definition, stream, sort_keys=False)
    model_specs = {}
    (out / "specs").mkdir(exist_ok=False)
    for model in read(PILOT / "protocol.json")["subjects"]:
        specific = copy.deepcopy(definition)
        specific["scorer_output"] = "logit" if model.startswith("google_siglip") else "cosine"
        path = out / "specs" / (model+".yaml")
        with path.open("x") as stream: yaml.safe_dump(specific, stream, sort_keys=False)
        model_specs[model] = {"path": str(path.resolve()), "sha256": sha(path), "scorer_output": specific["scorer_output"]}
    old, complete = feature_columns(definition, contexts)
    dump(out / "protocol.json", {"status": "post-specified diagnostic, not confirmation", "inputs": inputs,
        "spec_sha256": sha(out / "routing-context-complete-v2.yaml"), "contexts": contexts, "model_specs": model_specs,
        "primary": "red-blue, blend90, all seven fixed subjects, unchanged pilot sources",
        "audit": "direct canvas", "outcomes": ["swapped_canvas", "in_situ"],
        "baseline_features": BASE, "legacy_extension": old, "full_extension": complete, "categorical": CATS,
        "statistics": {"folds": 5, "logistic_C": 1., "bootstrap": 2000, "source_shuffles": 25, "seed": 20260922},
        "same_view_identity_not_prediction": True, "gpu_used": False, "reserve_scored": False,
        "training": False, "old_results_preserved": True})
    print("FROZEN diagnostic", sha(out / "protocol.json"), flush=True)


def protocol(out):
    p = read(out / "protocol.json"); verify_files(p["inputs"])
    if sha(out / "routing-context-complete-v2.yaml") != p["spec_sha256"]: raise ValueError("Specification changed")
    verify_files({r["path"]: r["sha256"] for r in p["model_specs"].values()})
    return p


def measure(out):
    p = protocol(out); cal = read(CAL); definition = load_requirement(out / "routing-context-complete-v2.yaml")
    _, contexts, names = full_spec(cal)
    audits = [r for r in lines(PILOT / "audit_rows.jsonl") if r["family"] == "routing" and
              r["method"] == "blend90_luminance" and r["color_pair"] == "red-blue"]
    tables = {}; specs = {}
    for model in sorted({r["model"] for r in audits}):
        path = PILOT / "scores" / model / "routing"
        scores = np.load(path / "scores.npy", mmap_mode="r")
        tables[model] = {r["anchor_id"]: (scores[i], r) for i, r in enumerate(lines(path / "index.jsonl"))}
        model_definition = load_requirement(p["model_specs"][model]["path"])
        specs[model] = compile_requirement(model_definition, calibration_unit=cal["units"][model]["unit"],
            base_model_id=cal["units"][model]["parent"], calibration_bank_id="natural_calibration_20260922")
    measurements, decisions = [], []; old_replay = 0; max_identity_error = 0.
    for a in audits:
        matrix, index = tables[a["model"]][a["anchor_id"]]; spec = specs[a["model"]]
        audit = select_lattice(matrix, index, "routing", 0, "blend90_luminance", "audit")
        endpoint = caption_decisions(audit[None], (0, 1, 2, 3))
        direct = decomposition(audit, spec.unit); ev = spec.evaluate(audit)
        for name, value in a["contrasts"].items():
            np.testing.assert_allclose(ev["contrasts"][name], value, atol=1e-12); old_replay += 1
        for name, value in a["clauses"].items():
            assert bool(ev["clauses"][name]["passed"]) == value["passed"]
            np.testing.assert_allclose(ev["clauses"][name]["violation"], value["violation"], atol=1e-12)
        features = {"signed_"+k: float(v) for k, v in ev["contrasts"].items()}
        features.update({"magnitude_"+r["contrast"]: abs(float(ev["contrasts"][r["contrast"]])) for r in contexts if r["kind"] == "unwanted"})
        identity = {k: a[k] for k in ("model", "family", "anchor_id", "source_id", "object_class", "color_pair", "unit")}
        for view in ("audit", *p["outcomes"]):
            lattice = audit if view == "audit" else select_lattice(matrix, index, "routing", 0, "blend90_luminance", view)
            d = decomposition(lattice, spec.unit); current = spec.evaluate(lattice)
            v = {k: float(z) for k, z in current["contrasts"].items()}
            reconstructed = (v[names[1, 1, 1, 1]]-v[names[1, 2, 1, 1]]-v[names[2, 1, 1, 1]]+v[names[2, 2, 1, 1]])/2
            error = abs(reconstructed-d["response"]); max_identity_error = max(max_identity_error, error)
            if error > 1e-10: raise AssertionError("Response/interaction identity failed")
            if bool(current["passed"]) and d["response"] < cal["kappa"]-cal["tau_fraction"]-1e-10:
                raise AssertionError("Full conformance differential-response bound violated")
            binding = np.array([v[r["contrast"]] for r in contexts if r["kind"] == "binding"])
            leakage = np.array([abs(v[r["contrast"]]) for r in contexts if r["kind"] == "unwanted"])
            denom = np.maximum(binding, 0).sum()+leakage.sum()
            measurements.append({**identity, "view": view, **d, "contrasts": v,
                "clauses": {k: {kk: bool(vv) if kk == "passed" else float(vv) for kk, vv in z.items()} for k, z in current["clauses"].items()},
                "binding_mean": float(binding.mean()), "binding_minimum": float(binding.min()),
                "unwanted_mean": float(leakage.mean()), "unwanted_share": float(leakage.sum()/denom) if denom else None,
                "full_conformance_secondary": bool(current["passed"])})
            if view == "audit": continue
            for j, state in enumerate((1, 2)):
                margin = float(endpoint["minimum_margins"][0, state]/spec.unit)
                decisions.append({**identity, "view": view, "color_state": "red_blue" if state == 1 else "blue_red",
                    "failure": d["exchange_errors"][j], "outcome_exchange_margin": d["exchange_margins"][j],
                    **{k: a[k] for k in BASE[:3]}, "state_audit_failure_rate": int(margin <= 0),
                    "state_audit_min_margin": margin, "state_audit_mean_margin": margin,
                    "audit_swap_rb_margin": direct["exchange_margins"][0], "audit_swap_br_margin": direct["exchange_margins"][1],
                    **features})
    jsonl(out / "measurements.jsonl", measurements); jsonl(out / "prediction_rows.jsonl", decisions)
    dump(out / "measurement_complete.json", {"old_contrasts_replayed": old_replay, "old_clauses_preserved": True,
        "maximum_response_identity_error": max_identity_error, "n_source_model_views": len(measurements),
        "n_independent_exchange_decisions": len(decisions), "new_inference": False, "gpu_used": False,
        "files": {n: sha(out / n) for n in ("measurements.jsonl", "prediction_rows.jsonl")}})
    print("MEASURED", len(measurements), "source/model/views; identity max error", max_identity_error, flush=True)


def shuffled(frame, columns, seed):
    """One whole source's full feature block, shared across models, views and states."""
    sources = sorted(frame.source_id.unique()); rng = np.random.default_rng(seed)
    mapping = dict(zip(sources, rng.permutation(sources)))
    unique = frame.drop_duplicates(["source_id", "model"])
    lookup = {(r.source_id, r.model): list(getattr(r, k) for k in columns) for r in unique.itertuples()}
    result = frame.copy()
    result.loc[:, columns] = np.array([lookup[mapping[r.source_id], r.model] for r in frame.itertuples()])
    return result


def checked_measurements(out):
    p = protocol(out); marker = read(out / "measurement_complete.json")
    verify_files({str(out / k): v for k, v in marker["files"].items()})
    return p


def analyze(out):
    p = checked_measurements(out)
    frame = pd.DataFrame(lines(out / "prediction_rows.jsonl"))
    fits = {name: cross_fitted(frame, BASE+extra, CATS) for name, extra in
            (("endpoint", []), ("legacy", p["legacy_extension"]), ("complete", p["full_extension"]))}
    summary = {"n_sources": frame.source_id.nunique(), "n_decisions": len(frame), "errors": int(frame.failure.sum()),
               "models": {k: {m: v[m] for m in ("auc", "brier", "fold_audit")} for k, v in fits.items()}, "contrasts": {}}
    for after, before in (("legacy", "endpoint"), ("complete", "endpoint"), ("complete", "legacy")):
        a, b = fits[before]["predictions"], fits[after]["predictions"]
        ci = paired_auc_interval(frame.failure, a, b, frame.source_id, n=p["statistics"]["bootstrap"])
        loss = cluster_means((b-frame.failure.to_numpy())**2-(a-frame.failure.to_numpy())**2,
                             frame.source_id, n=p["statistics"]["bootstrap"])
        summary["contrasts"][after+"_minus_"+before] = {"auc": ci, "brier": {"effect": loss["mean"][0], "ci95": loss["ci95"][0], "negative_is_better": True}}
    per_model = []
    for model, rows in frame.groupby("model"):
        ids = rows.index.to_numpy(); labels = rows.failure.to_numpy()
        for name, fit in fits.items():
            per_model.append({"model": model, "predictor": name, "n_sources": rows.source_id.nunique(), "n_decisions": len(rows),
                "auc": float(roc_auc_score(labels, fit["predictions"][ids])) if len(np.unique(labels)) == 2 else None,
                "brier": float(brier_score_loss(labels, fit["predictions"][ids]))})
    null = []
    for k in range(p["statistics"]["source_shuffles"]):
        seed = p["statistics"]["seed"]+k
        f = shuffled(frame, p["full_extension"], seed)
        value = cross_fitted(f, BASE+p["full_extension"], CATS)["auc"]-fits["endpoint"]["auc"]
        null.append({"seed": seed, "delta_auc": value})
        if (k+1) % 5 == 0: print("SHUFFLE", k+1, flush=True)
    observed = fits["complete"]["auc"]-fits["endpoint"]["auc"]
    summary["source_shuffle"] = {"replicates": len(null), "observed_delta_auc": observed,
        "empirical_one_sided_p": (1+sum(v["delta_auc"] >= observed for v in null))/(1+len(null))}
    prediction_rows = []
    for i, row in frame.iterrows():
        prediction_rows.append({**row.to_dict(), "fold": int(fits["endpoint"]["folds"][i]),
                                **{name+"_prediction": float(fit["predictions"][i]) for name, fit in fits.items()}})
    jsonl(out / "cross_fitted_predictions.jsonl", prediction_rows); jsonl(out / "per_model_prediction.jsonl", per_model)
    jsonl(out / "source_shuffle.jsonl", null); dump(out / "prediction_summary.json", summary)
    dump(out / "prediction_complete.json", {"files": {n: sha(out / n) for n in
        ("cross_fitted_predictions.jsonl", "per_model_prediction.jsonl", "source_shuffle.jsonl", "prediction_summary.json")}, "gpu_used": False})
    print("PREDICTION", summary, flush=True)


def report(out):
    p = checked_measurements(out)
    verify_files({str(out / n): h for n, h in read(out / "prediction_complete.json")["files"].items()})
    measurements = pd.DataFrame(lines(out / "measurements.jsonl")); summary_rows = []; clause_rows = []
    groups = list(measurements.groupby(["model", "view"]))
    groups += [(("all_fixed_subjects", view), g) for view, g in measurements.groupby("view")]
    for (model, view), g in groups:
        metrics = ["exchange_accuracy", "response", "absolute_preference", "surplus", "binding_mean", "unwanted_mean"]
        ci = cluster_means(g[metrics].to_numpy(float), g.source_id, n=p["statistics"]["bootstrap"])
        summary_rows.append({"model": model, "view": view, "n_sources": g.source_id.nunique(), "n_source_model_cases": len(g),
            "metrics": {k: {"mean": ci["mean"][i], "ci95": ci["ci95"][i]} for i, k in enumerate(metrics)},
            "regime_counts": {k: int((g.regime == k).sum()) for k in REGIMES},
            "full_conformance_secondary": float(g.full_conformance_secondary.mean())})
        if model == "all_fixed_subjects": continue
        for name in g.clauses.iloc[0]:
            values = [r[name] for r in g.clauses]
            clause_rows.append({"model": model, "view": view, "clause": name, "n_sources": g.source_id.nunique(),
                "pass_rate": float(np.mean([v["passed"] for v in values])), "mean_violation": float(np.mean([v["violation"] for v in values]))})
    jsonl(out / "diagnostic_summary.jsonl", summary_rows); jsonl(out / "clause_summary.jsonl", clause_rows)
    flat = []
    for row in summary_rows:
        flat.append({"model": row["model"], "view": row["view"], "n_sources": row["n_sources"],
                     "n_source_model_cases": row["n_source_model_cases"], **row["regime_counts"],
                     **{name+"_mean": v["mean"] for name, v in row["metrics"].items()}})
    with (out / "diagnostic_tables.csv").open("x") as stream: pd.DataFrame(flat).to_csv(stream, index=False)
    with (out / "clause_tables.csv").open("x") as stream: pd.DataFrame(clause_rows).to_csv(stream, index=False)
    prediction = read(out / "prediction_summary.json"); cal = read(CAL)
    text = ["# Complete routing diagnostic\n",
        "Completed on the unchanged seven-subject red/blue blend90 pilot. Post-specified diagnostic; "
        "no training, GPU inference, checkpoint/subset selection, or reserve outcomes. "
        "The original eight-interaction audit and every earlier result remain unchanged.\n",
        "## Executable measurement\n",
        "`routing-context-complete-v2.yaml` measures all 16 conditional image-factor × word-factor "
        "interactions, with 24 clauses: eight binding floors, eight absolute cross-effect bounds and "
        "eight relative bounds. Relative effects use own-word binding with the other object's "
        "caption color matching its image; all original relative pairings are preserved. "
        "This is full conditioning-context coverage for these pairs, not every possible interaction order.\n",
        "The main summaries are continuous effects and individual clause measurements. All-24 "
        "conformance is secondary, never substituted for behavioral accuracy.\n",
        "## What distinguishes response from preference\n",
        "Let q(I)=s(I,rb)−s(I,br). Then e=[q(rb)−q(br)]/2 is the differential assignment "
        "response, b=[q(rb)+q(br)]/2 is assignment-independent caption preference within this pair, "
        "and the two correct margins are e+b and e−b. Both decisions are correct iff e>|b|. "
        "The preference is not identified as a globally text-only cause.\n",
        "The exact identity 2e=D11(blue,blue)−D12(blue,blue)−D21(blue,blue)+D22(blue,blue) "
        "uses two previously omitted contrasts. Passing their binding and absolute clauses implies "
        f"e/u≥K−tau={cal['kappa']-cal['tau_fraction']:.6f}. It does not bound b. "
        "This is algebraic explanation on the same scores—not independent predictive evidence.\n",
        "## Observed regimes\n",
        "Each model has 49 sources and 98 exchange decisions per view. The aggregate describes "
        "these seven fixed subjects; they are not seven random seeds. Intervals resample whole "
        "sources across all repeated views/subjects. Counts below are source–model cases, not "
        "independent examples. Ties count as errors.\n",
        "| View | Cases | Individual exchange accuracy (%) [95% CI] | Nonpositive response | Positive response, preference dominates | Both correct |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        if r["model"] != "all_fixed_subjects": continue
        a = r["metrics"]["exchange_accuracy"]; n = r["regime_counts"]
        text.append(f"| {r['view']} | {r['n_source_model_cases']} | {100*a['mean']:.2f} [{100*a['ci95'][0]:.2f}, {100*a['ci95'][1]:.2f}] | "
                    f"{n[REGIMES[0]]} | {n[REGIMES[1]]} | {n[REGIMES[2]]} |")
    text += ["\nFull per-model regimes, response/preference magnitudes and uncertainty are in "
             "`diagnostic_summary.jsonl` and `diagnostic_tables.csv`; each of the 24 clauses is "
             "reported in `clause_summary.jsonl`. Same-view mathematical identities must not be "
             "reported as perfect prediction of unseen-context behavior.\n",
             "## Independent-view prediction\n",
             "Predict each exact exchange decision on swapped/in-place views from direct-canvas "
             "audit features. The endpoint baseline already includes both direct exchange margins "
             "as well as standard audit error/margin covariates. No independent-view scores or "
             "interactions enter the predictors. Five source-grouped folds, fixed C=1; all "
             "original/full feature dimensions are included, not selected by performance.\n",
             "| Predictor | AUC | Brier score (lower is better) |", "|---|---:|---:|"]
    for name, row in prediction["models"].items():
        text.append(f"| {name} | {row['auc']:.6f} | {row['brier']:.6f} |")
    text += ["\n| Difference | Delta AUC [95% CI] | Delta Brier [95% CI] |", "|---|---:|---:|"]
    for name, row in prediction["contrasts"].items():
        a, b = row["auc"], row["brier"]
        text.append(f"| {name} | {a['delta_auc']:+.6f} [{a['ci95'][0]:+.6f}, {a['ci95'][1]:+.6f}] | "
                    f"{b['effect']:+.6f} [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] |")
    text += [f"\nFull-feature source-shuffle negative control: {prediction['source_shuffle']['replicates']} draws, "
             f"one-sided empirical p={prediction['source_shuffle']['empirical_one_sided_p']:.3f}. "
             "Intervals are paired source-cluster bootstrap intervals conditional on fitted out-of-fold "
             "models, not refitting uncertainty or multiplicity-adjusted confirmation. Per-model "
             "AUC/Brier and every row prediction are retained.\n",
             "## Implication for repair\n",
             "Full coverage addresses a demonstrated measurement blind spot, not a demonstrated training benefit. "
             "Response deficiency and preference-dominated failure should not be collapsed into one violation. "
             "Any new repair must explicitly state which is targeted, preserve useful response and natural/endpoint "
             "evidence, and be compared with ranking and guard-only on the same independent outcomes. "
             "Do not claim full-context IS has worked: it has not been trained.\n",
             "## Reproduction and scope\n",
             "Run `python -m mirror.cases.color_binding.routing_diagnostic` with actions `freeze`, `measure`, "
             "`analyze`, `report` sequentially in a fresh output root. Commands are logged; "
             "inputs, specifications and outputs are hashed. Use `CUDA_VISIBLE_DEVICES='' "
             "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`. The source pilot remains exploratory; "
             "this diagnostic does not complete all Phase B benchmarks or Phase C repairs.\n"]
    with (out / "REPORT.md").open("x") as stream: stream.write("\n".join(text))
    dump(out / "diagnostic_complete.json", {"protocol_sha256": sha(out / "protocol.json"),
        "files": {n: sha(out / n) for n in ("REPORT.md", "diagnostic_summary.jsonl", "clause_summary.jsonl", "diagnostic_tables.csv", "clause_tables.csv")},
        "measurement_complete_sha256": sha(out / "measurement_complete.json"),
        "prediction_complete_sha256": sha(out / "prediction_complete.json"), "gpu_used": False, "training": False})
    print(out / "REPORT.md", flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "measure", "analyze", "report"))
    parser.add_argument("--out", type=Path, default=OUT); args = parser.parse_args(); log(args.out, "start")
    try:
        with threadpool_limits(limits=2):
            {"freeze": freeze, "measure": measure, "analyze": analyze, "report": report}[args.action](args.out)
    except BaseException as exc: log(args.out, "failed", error=repr(exc)); raise
    log(args.out, "complete")


if __name__ == "__main__": main()
