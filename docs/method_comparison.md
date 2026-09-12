# Method comparison: what to try beyond the paper's random forest

The paper's Sec. 3.3 says it tried multiple linear regression, polynomial
regression, and support vector regression before settling on a random forest
(5 trees, 70/30 split, R² = 0.98 on their 55-point set). `scripts/compare_models.py`
runs all three of those plus the random forest baseline and three suggested
alternatives, using repeated 5-fold cross-validation (100 train/test splits
per method total) rather than one split — with only 59 points, a single
split can make a method look better or worse than it really is, so every
number below is a mean ± standard deviation across those 100 splits.

## Results (5-fold CV × 20 repeats = 100 splits)

| Method | mean R² | std | Notes |
|---|---|---|---|
| Power-law fit (log-log linear regression) | 0.991 | 0.009 | `Nu = 0.087 * Ra^0.303 * Pr^0.023`. Both best *and* most stable — the tiny std means this holds up no matter which points land in the test fold. The Ra exponent (0.303) sits right next to the classical Nu ~ Ra^(1/3) scaling from RBC theory. |
| Gaussian process regression | 0.989 | 0.052 | Nearly as good on average, but 5x more variable fold-to-fold. Still the only method here that also reports a predictive uncertainty on each prediction. |
| Support vector regression | 0.898 | 0.157 | Untuned (C=100, epsilon=0.5); the wide std suggests it's sensitive to which points it sees. |
| Random forest (baseline, 5 trees) | 0.891 | 0.100 | The paper's method. Notably lower than the 0.971 a single lucky split gave earlier — cross-validation is the more honest number. |
| Gradient boosting | 0.885 | 0.097 | About the same as the random forest; doesn't show an edge on a dataset this small. |
| Polynomial regression (degree 2) | 0.758 | 0.269 | Single-split R² (0.945) was misleadingly high — under CV it's both worse and unstable, consistent with a degree-2 fit overreacting to whichever high-Ra points happen to be in the training fold. |
| Multiple linear regression | -0.096 | 1.101 | Mean R² *below zero* — on some folds it does worse than just predicting the average Nu. Confirms the paper's own finding that linear regression underfits this relationship, more starkly than the single split showed. |

**The single train/test split from before was actively misleading** for two
of these methods (random forest looked 8 points better than it should, degree-2
polynomial regression looked 19 points better) — this is exactly why doing
cross-validation before trusting a comparison matters here.

## What the bar chart hides: `results/top_methods_predicted_vs_actual.png`

The table above says the power-law fit and Gaussian process beat the random
forest, but not *where* the random forest goes wrong. This plot puts all
three side by side as predicted-vs-actual scatter plots (5-fold CV, each
point predicted by a model that never trained on it):

![Top methods: predicted vs actual](../results/top_methods_predicted_vs_actual.png)

The random forest panel shows a clear, systematic pattern: the four highest-Nu
points (Nu > 80) all fall *below* the perfect-prediction line — the model
underpredicts every one of them. This isn't random scatter, it's a known
limitation of trees: a random forest can only predict values it saw in
training (it averages leaf values), so it can't extrapolate past the highest
Nu it was trained on. With only a handful of points out at high Ra, whichever
ones land in the test fold get underpredicted almost by construction. The
power-law and Gaussian process panels don't show this bias — both are smooth
functions that extrapolate naturally, so they track the high-Nu points as
well as the low-Nu ones. This is the real reason the power-law fit and GP
outperform the random forest here, not just a marginally better R².

## Why these three suggestions specifically

**Gaussian process regression.** DNS values here carry real uncertainty (the
error bars in `docs/data_sources.pdf`'s tables) — a GP is one of the few
methods that reports how confident it is in each prediction, not just a
point estimate. That matters if this model is ever used to decide *which new
(Pr, Ra) point is worth simulating next* (a GP naturally supports this kind
of "where should we look next" reasoning; a random forest doesn't). The
cross-validated std (0.052, five times the power-law fit's) confirms the
earlier caveat was warranted: a GP with a flexible kernel can look great on
one lucky split and less so on others.

**Power-law fit.** Heat transport in convection is known to follow power
laws (the paper itself fits Nu ~ Pr^0.08 in Fig. 3), so fitting
`Nu = C · Ra^a · Pr^b` directly is the most physically natural model here —
and unlike a random forest or GP, it hands you two numbers you can quote and
compare against the literature. It fitting almost as well as the far more
flexible GP suggests the underlying relationship really is close to a single
global power law over this range, which is itself a useful finding to
mention in a writeup.

**Gradient boosting.** A natural "next" tree-ensemble to try given the paper
already uses trees. It doesn't win here, most likely because gradient
boosting typically needs more data than 59 points to outperform a random
forest — worth revisiting once the two missing papers are added and the
dataset grows.

## Caveats

- SVR, gradient boosting, and the GP kernel all used reasonable-but-untuned
  hyperparameters; a `GridSearchCV` pass (nested inside the cross-validation,
  to avoid leaking test-fold information into the tuning) could change their
  scores meaningfully — SVR's wide std (0.157) in particular suggests it has
  room to improve.
- `notebooks/analysis.ipynb` and `scripts/baseline_random_forest.py` still
  report the single-split R² (0.971) to match the paper's own reporting
  convention — keep in mind the cross-validated, more honest number for the
  same model is 0.891 ± 0.100.
- None of this changes once the van der Poel (2013) and Zhang & Zhou (2024)
  data are added — re-run `scripts/compare_models.py` after `data/nu_dataset_model.csv`
  is updated.
