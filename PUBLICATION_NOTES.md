# Publication and conversion notes

## Source and conversion scope

`article.md` is a Markdown conversion of the current **A Crooked Path through Space and Semantics** draft retrieved on 2026-09-14, including the concluding section **No Straight Answers**. The source document was not edited by this publication step.

Section order, prose, emphasis, quotations, numeric examples, existing attribution, and all eight embedded-image positions were retained. Formatting repairs remove duplicated rendered/LaTeX paste artifacts, restore superscripts/subscripts and matrix layouts, normalize display math to `$$` and inline math to `$`, and remove empty headings/list items introduced during editing. The source's `^/top` typo was interpreted as `^\top` consistently with the surrounding equations. Other prose and scientific claims were not rewritten.

The residual expression in the source was just `X + (AV)W_O` followed by a broken display delimiter; the conversion retains that expression rather than inventing a missing left-hand side. The article's numeric matrices are preserved, not replaced with newly generated examples.

Images are PNG files in `assets/`, referenced by relative Markdown links at their original text locations. The [manifest](assets/manifest.json) records their dimensions, file hashes, pixel hashes, and Git blob hashes. The imported image files were checked against the images extracted from the source document. These are explanatory illustrations, not a substitute for plots generated from the numerical experiment.

## Two source discrepancies requiring editorial review

### 1. Toy-model illustration

The image currently embedded in the source draft, preserved as `assets/drosophila-transformer.png`, depicts **12 tokens**, whereas the article and recovered experiment specify **2 token positions**. The image also labels its output-projection box `W_Q`, while the displayed equation and actual code use `W_O`. This generated illustration should be corrected before treating the article as technically reviewed. It has not been silently substituted or redesigned during conversion.

The experiment's actual architecture is: two token positions; representation width three; three-dimensional Q, K, and V; one causal attention head; four attention-only residual layers; no training, biases, normalization, feed-forward network, embedding lookup, or vocabulary-output layer.

### 2. Flatland numerical fixture

The original recovered `drosophila.py` uses seed 42. Its flat-control matrix is obtained from that run's first query projection. The later article displays a different matrix beginning with `1.563, 0.435, 0.188`, and an initial compatibility table beginning with `0.952, -0.438`.

Those displayed numbers do **not** reproduce from the recovered seed-42 script. The later interactive demo's full parameter fixture was not recovered for this upload. The article numbers have therefore been retained as source content, while the code is clearly identified as the original experiment. No substitute weights were reverse-engineered or presented as the original demo. A future update should either recover that demo fixture or explicitly switch the article to one documented reproducible run.

The article's rank example uses five tokens and query/key width three; the original script's separate rank check uses four tokens and width three. These are distinct examples of the same dimensional bottleneck, not an exact numeric replication of one another.

## Execution checks on the recovered script

The published script was run without changing its computations using NumPy 2.3.5, SymPy 1.14.0, Matplotlib 3.10.8, and the noninteractive `Agg` backend. All included assertions completed successfully.

Observed results for this particular run:

- Causal masking, row-normalized attention weights, residual arithmetic, and the no-future-token-leakage check passed.
- `G = X @ M @ X.T` held at all four flat-control layers.
- All 81 components of the supplied constant metric's Riemann tensor were exactly zero in the symbolic computation.
- The unit-sphere positive control had scalar curvature 2 and Gaussian curvature 1.
- Token B's flat-control turning angles measured using the metric's own ruler were approximately 166.077199, 26.063237, and 19.105500 degrees; its metric path-to-chord ratio was approximately 1.955709.
- The separate 1,000-seed unrestricted random-model sweep produced 1,000 bent paths and 4,000 asymmetric raw compatibility matrices in 4,000 layer evaluations.

These are results of the supplied toy experiment. They are not measurements of a trained language model, and do not establish that every possible Transformer geometry is flat.
