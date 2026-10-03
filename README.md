# MIRROR

Specifications, repair implementations, configurations, source manifests, and
per-example results for typography, color binding, and backdoors. Learned results
use seeds 42, 43, and 44. `supplement.pdf` contains Sections S1–S5 cited in the
paper. No model checkpoints or image datasets are bundled.

## Reproduce results from saved scores

Run from this directory. This CPU workflow reconstructs decisions, interactions,
means, and paired intervals from the supplied per-example scores. It does not
rerender datasets, execute a VLM, or retrain a repair. Keep the environment and
generated outputs outside the evidence package:

```bash
python3 -m venv /tmp/mirror-analysis-venv
/tmp/mirror-analysis-venv/bin/pip install -r requirements-analysis.txt
/tmp/mirror-analysis-venv/bin/python -m mirror verify
/tmp/mirror-analysis-venv/bin/python -m mirror test
/tmp/mirror-analysis-venv/bin/python -m mirror reproduce --out /tmp/mirror-reproduced --bootstrap 5000
/tmp/mirror-analysis-venv/bin/python -m mirror figures --out /tmp/mirror-reproduced/figures
```

Use a new output directory to retain previous reproductions. Add `--case
typography`, `--case color_binding`, or `--case backdoor` to `reproduce` for one
experiment. Outputs include result rows, independent score checks, and bootstrap
intervals; percentages and fractions are explicitly marked. `figures` renders
paper Figures 4, 5, and 6 from the bundled plot data, with consistency checks
against the score records. Bootstrap endpoints depend on the specified random
seed and draw count. Saved identifiers `IS`/`IS2` denote MIRROR; `previous_IS` is
the weaker plain typography penalty in Supplement S3, not a model selected by
the reproduction command.

## Executable method engines

`mirror/core/specifications.py` compiles metamorphic specifications from
`mirror/core/templates/` and `data/color_binding/specifications/`.
`mirror/core/repair.py` contains the low-rank adapter and cached-repair machinery.
`mirror/core/objectives.py` exposes the selected typography, color-binding, and
backdoor losses without acquiring assets or running a model. Set the interaction
coefficient to zero for matched Ranking in typography and backdoors. Color-binding
components must first use the clause/layout/block reduction in Supplement S1.

With PyTorch installed, compare standalone objectives against the selected
experiment implementations on synthetic CPU tensors:

```bash
python -m mirror objective-test
```

This checks loss values and gradients only: no optimizer updates, VLM inference,
downloads, or GPU execution. Experiment implementations are under `mirror/cases/`;
backbone loaders are in `mirror/core/encoders.py`. The selected run receipts record
normalization weights, source order, update budgets, and checkpoint identities.

## Raw-data training and evaluation prerequisites

Raw-data retraining is not a turnkey command in this release. The experiment
wrappers require feature caches and run-layout inputs that are not bundled;
changing only an output path does not make them runnable. A complete retraining
driver must rebuild image/text caches from the prepared manifests, resolve local
assets, initialize the declared model, execute the fixed recipe, and evaluate the
same source banks. The standalone objectives and compiler are usable independently.

`requirements-training.txt` records the available training package versions;
`provenance/training_environment.txt` is their snapshot. This has not been validated
as a clean installation. Check the PyTorch/NumPy array bridge and CUDA compatibility
before cache generation. Dataset-construction modules additionally need pycocotools,
requests, and nltk, whose original versions were not recorded. None is needed for
CPU score reproduction.

| Required input | Acquisition and matching |
| --- | --- |
| COCO 2017 | Obtain train2017, val2017, and their instance annotations from the [COCO download page](https://cocodataset.org/#download). Match bundled IDs, splits, boxes/masks, and source hashes; do not resample. Natural-retention captions are included in the color-binding training rows. |
| LVIS annotations | Needed to rebuild lexical-support selection, not to use the prepared typography training manifest `data/typography/class_support/train_filtered.json`. Archive URL and hash are in `lvis_v1_train.json.zip.download.json`. |
| Visual Genome calibration | Use the 70 rows, image URLs, boxes, and hashes in `data/models/calibration/selected_calibration.jsonl`; fixed values are in `calibration_frozen.json`. |
| Base VLMs | Architecture, backend, source, activation, and file hashes are in `data/models/registry.json` and `activation_addendum.json`. Obtain those exact public files and supply a local registry with matching hashes. Recorded absolute paths are provenance identities, not installation locations. |
| PAR victims, released repairs, and patterns | Fetch the pinned PAR repository and its released assets following its instructions. Match checkpoint/pattern hashes in backdoor run receipts and `provenance/model_assets.json`; a substitute victim does not reproduce the reported study. |
| SCAM, RTA100, ImageNetV2 | Download pinned archives using the commands below; file sizes and SHA-256 values are checked. |
| SugarCrepe, ARO, COCO retrieval | Obtain benchmark data under the original releases' terms. Preserve the captions, identities, and splits in selected `results/records/` files. No benchmark acquisition/unpacking driver is bundled. |

```bash
python -m mirror download --list
python -m mirror download --asset SCAM --out-dir /path/to/assets
python -m mirror upstream --list
python -m mirror upstream --name PAR --out-dir /path/to/upstream
```

Downloads are not automatically unpacked or routed into experiment wrappers.
Resolve assets outside this evidence package. The loader verifies model hashes;
an alternate local registry needs a matching activation addendum. Local provenance
paths do not imply that omitted assets are downloadable from this artifact.
Third-party dataset, model, and code licenses remain applicable.

Recipes are in Supplement S1 and `config/studies.json`: typography uses one shared
text-prefix token for 2,752 updates; color binding uses a rank/alpha-64 map, 75% tint,
balanced noun orders, paired layouts, and 1,944 updates; backdoors update the last
two visual blocks for 1,536 successful updates. Reported arms use fixed final
checkpoints. Weaker-penalty and genuine-target preservation results are retained.

## Evidence and integrity

`data/` contains prepared source manifests/protocols; `results/records/` contains
per-example scores; `results/reports/` contains summaries; `provenance/` binds
selected settings and original checkpoint/source hashes to packaged evidence.
Original hashes remain distinct from packaged hashes when paths or source
organization differ. Provenance receipts are not runtime cache-construction
instructions.

`python -m mirror verify` checks the package seal, sizes, hashes, entrypoints,
study/seed coverage, and absence of checkpoints. `sha256sum -c SHA256SUMS` also
checks the seal. Compiler tests cover contrast arithmetic, additive-offset
cancellation, non-finite score rejection, and compact-score tie-breaking. These
checks establish packaged-score consistency, not independent retraining or
fresh-inference equivalence.
