# MIRROR

Code, configurations, data manifests, and per-example results for typography,
color binding, and backdoors. Seeds: 42, 43, 44. No checkpoints or image datasets.
`supplement.pdf` is the supplementary material cited in the paper (Sections S1–S5).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt
.venv/bin/python -m mirror verify
.venv/bin/python -m mirror test
.venv/bin/python -m mirror reproduce --out reproduced --bootstrap 5000
.venv/bin/python -m mirror figures --out reproduced/figures
```

`mirror/core/`: specifications and repair. `mirror/cases/`: the three experiments.
`config/`: settings. `data/`: splits. `results/`: scores and tables.
`provenance/`: run settings and hashes. Public assets: `config/downloads.json`,
`config/upstream.json`. To reproduce one experiment, add
`--case typography`, `--case color_binding`, or `--case backdoor`.
`reproduce` takes about a minute on a CPU. `figures` renders the paper's
Figures 4, 5, and 6 (`answer_dependency`, `interaction_distributions`, `retest`).

Original metric identifiers are retained in the saved records (`IS`/`IS2`
denote MIRROR). Dataset and third-party code/model terms remain applicable.
