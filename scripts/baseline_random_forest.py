"""
Baseline model: reproduces Sec. 3.3 of Shreshthi & Pandey's paper.

Random forest regressor, Pr and Ra (log10-transformed, matching Figure 4's
axes) as inputs, Nu as output. 70/30 train-test split, 5 trees, R^2 on the
test set.

Usage:
    python scripts/baseline_random_forest.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

DATA_PATH = "data/nu_dataset_model.csv"
RANDOM_STATE = 42  # fixed so this script and compare_models.py use the same split


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = np.log10(df[["Pr", "Ra"]])
    y = df["Nu"]
    return X, y


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(n_estimators=5, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"Baseline: random forest (5 trees), {len(X)} points, "
          f"{len(X_train)} train / {len(X_test)} test")
    print(f"R^2 on test set: {r2:.3f}")

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, edgecolor="k")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="perfect prediction")
    plt.xlabel("Nu (actual)")
    plt.ylabel("Nu (predicted)")
    plt.title(f"Baseline random forest (R^2 = {r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/baseline_random_forest.png", dpi=150)
    print("Saved plot to results/baseline_random_forest.png")

    return r2


if __name__ == "__main__":
    main()
