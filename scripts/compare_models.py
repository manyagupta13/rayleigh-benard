"""
Compares the baseline random forest against other regression methods on the
Nu(Pr, Ra) data, using repeated k-fold cross-validation rather than a single
train/test split (the dataset is only 59 points, so one split can make a
method look better or worse than it really is).

Includes the three methods the paper says it tried before settling on random
forest (multiple linear regression, polynomial regression, support vector
regression), plus three suggestions worth trying next: gradient boosting,
Gaussian process regression, and a physics-motivated power-law fit.

Usage:
    python scripts/compare_models.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from baseline_random_forest import load_data, RANDOM_STATE

OUTPUT_PLOT = "results/model_comparison.png"
N_SPLITS = 5
N_REPEATS = 20  # 100 train/test splits total per method, for a stable mean +/- std


class PowerLawRegressor(BaseEstimator, RegressorMixin):
    """Nu = C * Ra^a * Pr^b, fit as log10(Nu) = log10(C) + a*log10(Ra) + b*log10(Pr).

    X is already log10(Pr), log10(Ra) (see baseline_random_forest.load_data),
    so this is just a linear regression in log-log space - the standard way
    physicists fit power-law scalings (the paper's own Nu ~ Pr^0.08 fit in
    Fig. 3 is the same idea). Written as a proper sklearn estimator so it can
    be cross-validated the same way as every other method here.
    """

    def fit(self, X, y):
        self.reg_ = LinearRegression()
        self.reg_.fit(X, np.log10(y))
        return self

    def predict(self, X):
        return 10 ** self.reg_.predict(X)

    def formula(self):
        b_pr, a_ra = self.reg_.coef_
        C = 10 ** self.reg_.intercept_
        return f"Nu = {C:.3g} * Ra^{a_ra:.3f} * Pr^{b_pr:.3f}"


def build_models():
    models = {}
    # --- Methods the paper says it tried, before choosing random forest ---
    models["Multiple linear regression"] = LinearRegression()
    models["Polynomial regression (deg=2)"] = make_pipeline(
        PolynomialFeatures(degree=2), LinearRegression()
    )
    models["Support vector regression"] = make_pipeline(
        StandardScaler(), SVR(kernel="rbf", C=100, epsilon=0.5)
    )
    # --- The paper's chosen method (baseline) ---
    models["Random forest (baseline, 5 trees)"] = RandomForestRegressor(
        n_estimators=5, random_state=RANDOM_STATE
    )
    # --- Suggested alternatives ---
    models["Gradient boosting"] = GradientBoostingRegressor(random_state=RANDOM_STATE)
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    models["Gaussian process regression"] = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=RANDOM_STATE),
    )
    models["Power-law fit (log-log linear regression)"] = PowerLawRegressor()
    return models


def main():
    X, y = load_data()
    models = build_models()
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        rows.append((name, scores.mean(), scores.std(), scores))

    rows.sort(key=lambda r: r[1], reverse=True)

    n_folds = N_SPLITS * N_REPEATS
    print(f"{len(X)} points, {N_SPLITS}-fold CV repeated {N_REPEATS} times "
          f"({n_folds} train/test splits per method)\n")
    print(f"{'Method':42s} {'mean R^2':>10s} {'std':>8s}")
    print("-" * 64)
    for name, mean, std, _ in rows:
        print(f"{name:42s} {mean:10.3f} {std:8.3f}")

    # Report the power-law formula fit on the full dataset (for interpretability,
    # not part of the CV score above)
    pl = PowerLawRegressor().fit(X, y)
    print(f"\nPower-law fit on the full dataset: {pl.formula()}")

    # Bar chart with error bars (mean +/- 1 std across folds)
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = ["#2a9d5c" if n.startswith("Random forest") else "#4a7fb5" for n in names]

    plt.figure(figsize=(8, 5))
    plt.barh(names, means, xerr=stds, color=colors, capsize=3)
    plt.xlabel(f"R^2 (mean +/- std over {n_folds} CV splits)")
    plt.title(f"Model comparison ({N_SPLITS}-fold CV x {N_REPEATS} repeats)")
    plt.gca().invert_yaxis()
    plt.xlim(left=min(0, min(m - s for m, s in zip(means, stds)) - 0.05))
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nSaved plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
