# Curved Transformer Spacetime Exploration

Companion material for **[A Crooked Path through Space and Semantics](article.md)**: a small, inspectable exploration of attention, metrics, and the difference between a bent trajectory and intrinsic curvature.

## Read the article

[**A Crooked Path through Space and Semantics**](article.md)

The Markdown edition preserves the current draft's section order and embedded-image positions. Equations use GitHub-compatible LaTeX, and all illustrations are stored in [`assets/`](assets/), not linked to temporary Google Drive URLs. See [publication notes](PUBLICATION_NOTES.md) for conversion details and the remaining source-image issue.

## Run the experiments

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MPLBACKEND=Agg python drosophila.py
python flatland_demo.py
```

On Windows, activate with `.venv\Scripts\activate` and set `MPLBACKEND=Agg` in your shell before running `python drosophila.py`.

- [`drosophila.py`](drosophila.py) is the original seeded experiment: two token positions, three-dimensional representations, one causal attention head, and four attention-only residual layers. It audits Q/K matrix shapes, checks causal masking, constructs a constant flat control metric, computes its Levi-Civita curvature, checks a unit sphere as a positive control, and runs the original 1,000-seed sweep. It saves its numerical ledger and plots under `results/`.
- [`flatland_demo.py`](flatland_demo.py) reproduces the later article/browser-demo fixture, including the specific shared Q/K matrix behind the displayed metric. This is a different fixed parameter set from the original seed-42 experiment; the distinction is intentional and documented in the script. It saves its full ledger under `results/flatland_demo.json`.

No training, tokenization, vocabulary lookup, normalization, feed-forward network, or language-generation objective is involved. Initial working representations are supplied directly.

## What the flat control establishes

The control deliberately ties the query and key projections to a fixed, invertible matrix `W`. Its specified **feature-space metric** is `M = W @ W.T`, while the changing **token-compatibility table** is `G = X @ M @ X.T`.

`M` is symmetric positive-definite and constant over representation space, hence intrinsically flat. The actual attention/residual updates need not follow its geodesics. This control illustrates that trajectory bending alone does not establish intrinsic curvature. It does **not** identify `M` with every possible geometry one could associate with a Transformer, or rule out other geometric constructions.

## Source paper

[The Curved Spacetime of Transformer Architectures, arXiv:2511.03060](https://arxiv.org/abs/2511.03060)

This repository accompanies an explanatory article and an exploratory experiment; it is not the paper's implementation or a reproduction of its trained-model experiments.
