"""Score-independent full conditional second-order coverage diagnostic; no new loss."""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import verify_files
from mirror.cases.color_binding.behavioral_pilot import OUT as PILOT; from mirror.cases.color_binding.behavioral_pilot import lines; from mirror.cases.color_binding.behavioral_pilot import select_lattice
from mirror.core.specifications import compile_requirement; from mirror.core.specifications import load_requirement

OUT = ROOT / "clip/interbind_routing_context_coverage_20260923"


def all_contexts():
    states = list(itertools.product((0, 1), repeat=2)); records = []
    for image_factor, word_factor, other_image, other_word in itertools.product((0, 1), repeat=4):
        weight = np.zeros((4, 4))
        for a, b in itertools.product((0, 1), repeat=2):
            im, tx = [0, 0], [0, 0]
            im[image_factor] = a; im[1-image_factor] = other_image
            tx[word_factor] = b; tx[1-word_factor] = other_word
            weight[states.index(tuple(im)), states.index(tuple(tx))] = 1 if a == b else -1
        records.append({"name": f"image{image_factor+1}_word{word_factor+1}_context{other_image}{other_word}",
            "image_factor": image_factor+1, "word_factor": word_factor+1,
            "other_image_color_index": other_image, "other_word_color_index": other_word,
            "kind": "binding" if image_factor == word_factor else "unwanted", "weights": weight})
    return records


def run(out=OUT):
    log(out, "start")
    source = Path(__file__).with_name("requirement_library") / "routing-v1.yaml"
    legacy = compile_requirement(load_requirement(source), calibration_unit=1., base_model_id="symbolic", calibration_bank_id="symbolic")
    contexts = all_contexts()
    for context in contexts:
        matches = [name for name, w in legacy.weights.items() if np.array_equal(w, context["weights"])]
        if len(matches) > 1: raise AssertionError("Ambiguous old contrast mapping")
        context["legacy_contrast"] = matches[0] if matches else None
    if sum(c["legacy_contrast"] is not None for c in contexts) != 8:
        raise AssertionError("Every one of eight old contrasts must be reproduced exactly")
    inputs = {str(source): sha(source), str(Path(__file__)): sha(__file__),
              str(PILOT / "audit_rows.jsonl"): sha(PILOT / "audit_rows.jsonl")}
    for path in sorted((PILOT / "scores").glob("*/routing/complete.json")):
        complete = read(path)
        inputs[str(path)] = sha(path)
        inputs[str(path.parent / "scores.npy")] = complete["score_sha256"]
        inputs[str(path.parent / "index.jsonl")] = complete["index_sha256"]
    cal_path = ROOT / "data/models/calibration/calibration_frozen.json"
    inputs[str(cal_path)] = sha(cal_path); verify_files(inputs); calibration = read(cal_path)
    dump(out / "protocol.json", {"status": "Post-specified measurement-coverage diagnostic after author-requested audit-first review",
        "inputs": inputs, "selection": "Same red-blue/blend90 routing pilot, all seven frozen subjects, unchanged IDs",
        "contexts": [{**c, "weights": c["weights"].tolist()} for c in contexts],
        "rule": "All four image-factor x word-factor pairs, each in all four assignments of the other image and word factors",
        "thresholds": {"K": calibration["kappa"], "tau": calibration["tau_fraction"]},
        "scope": "Contrast coverage, signs, magnitude and existing absolute thresholds; not a revised repair loss or a new complete relative-conformance specification",
        "no_score_selected_subset": True, "new_inference": False, "gpu_used": False, "new_training": False, "reserve_scored": False})
    audits = [r for r in lines(PILOT / "audit_rows.jsonl") if r["family"] == "routing" and
              r["method"] == "blend90_luminance" and r["color_pair"] == "red-blue"]
    tables = {}
    for model in sorted({r["model"] for r in audits}):
        path = PILOT / "scores" / model / "routing"
        scores = np.load(path / "scores.npy", mmap_mode="r")
        tables[model] = {r["anchor_id"]: (scores[i], r) for i, r in enumerate(lines(path / "index.jsonl"))}
    records = []
    for audit in audits:
        matrix, index = tables[audit["model"]][audit["anchor_id"]]
        lattice = select_lattice(matrix, index, "routing", 0, "blend90_luminance", "audit")
        for c in contexts:
            value = float((lattice*c["weights"]).sum()/audit["unit"])
            if c["legacy_contrast"] is not None:
                np.testing.assert_allclose(value, audit["contrasts"][c["legacy_contrast"]], atol=1e-12)
            residual = value-calibration["kappa"] if c["kind"] == "binding" else calibration["tau_fraction"]-abs(value)
            records.append({**{k: audit[k] for k in ("model", "anchor_id", "source_id", "object_class")},
                **{k: v for k, v in c.items() if k != "weights"},
                "coverage": "previously_measured" if c["legacy_contrast"] else "previously_unmeasured",
                "normalized_value": value, "raw_value": value*audit["unit"],
                "absolute_threshold_pass": residual >= 0, "absolute_threshold_violation": max(-residual, 0)})
    frame = pd.DataFrame(records); summaries = []
    groups = list(frame.groupby(["model", "coverage", "kind"]))
    groups += [((model, "all_contexts", kind), g) for (model, kind), g in frame.groupby(["model", "kind"])]
    for (model, coverage, kind), g in groups:
        summaries.append({"model": model, "coverage": coverage, "kind": kind,
            "n_sources": g.source_id.nunique(), "n_conditional_measurements": len(g),
            "mean_signed": float(g.normalized_value.mean()), "mean_absolute": float(g.normalized_value.abs().mean()),
            "median_signed": float(g.normalized_value.median()), "nonpositive_fraction": float((g.normalized_value <= 0).mean()),
            "threshold_pass_fraction": float(g.absolute_threshold_pass.mean()),
            "mean_violation": float(g.absolute_threshold_violation.mean())})
    jsonl(out / "conditional_interactions.jsonl", records); jsonl(out / "summary.jsonl", summaries)
    with (out / "summary.csv").open("x") as stream: pd.DataFrame(summaries).to_csv(stream, index=False)
    dump(out / "complete.json", {"files": {n: sha(out / n) for n in ("conditional_interactions.jsonl", "summary.jsonl", "summary.csv")},
        "protocol_sha256": sha(out / "protocol.json"), "old_contrasts_exactly_reproduced": True,
        "n_conditional_interactions": len(records), "new_training": False, "gpu_used": False})
    print("COVERAGE AUDIT", len(records), "conditional interactions; all old values matched", flush=True)
    log(out, "complete")


if __name__ == "__main__":
    with threadpool_limits(limits=2): run()
