# Curved Transformer Spacetime Exploration

Companion material for **[A Crooked Path through Space and Semantics](article.md)**: an inspectable exploration of attention, metrics, and the difference between a bent trajectory and intrinsic curvature.

## Read the article

[**A Crooked Path through Space and Semantics**](article.md)

The Markdown edition preserves the draft's section order and the positions of all eight embedded images. Equations use GitHub-compatible LaTeX. Illustrations live in [`assets/`](assets/) and are linked with relative paths, so the article does not depend on temporary image URLs. [The image manifest](assets/manifest.json) records dimensions and checksums.

**Before citing this as a finished technical package, see [publication notes](PUBLICATION_NOTES.md): the source draft's toy-model illustration and its later numerical fixture need reconciliation with the recovered experiment.** The conversion preserves the draft rather than silently changing either.

## Run the original experiment

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MPLBACKEND=Agg python drosophila.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:MPLBACKEND = "Agg"
python drosophila.py
```

[`drosophila.py`](drosophila.py) is the original, recovered seed-42 script. It has two token positions, three-dimensional working representations, one causal attention head, and four attention-only residual layers. Four 3-by-3 projection matrices per layer give 144 weight entries. The flat control ties the Q/K projections, reducing the number of independent choices without changing the stored matrix shapes.

There is no training, tokenization, vocabulary lookup, normalization, feed-forward network, or language-generation objective. Initial working representations are supplied directly. The script runs when executed (and also on import); it is an exploratory standalone script, not a packaged library.

The run performs the Q/K shape and symmetry audit, a singularity check on a four-token extension, the constant-metric flat control, a unit-sphere positive control, a 1,000-seed random-model sweep, and assertions for causal masking, normalized attention weights, residual updates, and absence of future-token leakage.

Outputs are written alongside the script in `results/`:

- `results.json`: all inputs, weights, intermediate tables, trajectories, and measured results.
- `trajectories.csv`: both tokens at the input and after each layer.
- `random_trajectory.png` and `flat_control_trajectory.png`: plots of the actual computed states, distinct from the article's explanatory illustrations.

With a noninteractive plotting backend, Matplotlib may print a harmless warning about `plt.show()`; the figures are saved before that call.

## What the flat control establishes

The control deliberately ties the query and key projections to a fixed, invertible matrix `W`. Its specified **feature-space metric** is `M = W @ W.T`, while the changing **token-compatibility table** is `G = X @ M @ X.T`.

`M` is symmetric positive-definite and constant over representation space, hence intrinsically flat. The attention/residual updates need not follow its geodesics. This control illustrates that trajectory bending alone does not establish intrinsic curvature. It does **not** identify `M` with every possible geometry associated with a Transformer, or rule out other geometric constructions.

The script also measures the flat-control trajectory in `z = x @ W` coordinates, where the chosen metric is Euclidean. The 1,000-seed sweep is a separate experiment on unrestricted random Q/K projections, not 1,000 flat controls and not a reproduction of the paper's trained-model statistics.

## Source paper

[The Curved Spacetime of Transformer Architectures, arXiv:2511.03060](https://arxiv.org/abs/2511.03060)

This repository accompanies an explanatory article and an exploratory experiment; it is not the paper's implementation or a reproduction of its trained-model experiments. Existing image attribution in the article is preserved. No new license is imposed on the article or third-party illustrations by this upload.
