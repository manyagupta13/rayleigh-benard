# rayleigh-benard

Machine-learning prediction of the Nusselt number (Nu) from Prandtl (Pr) and
Rayleigh (Ra) numbers in 2D Rayleigh-Bénard convection, following Sec. 3.3
("Forecasting heat transport using machine learning") of Shreshthi & Pandey's
paper.

```
rayleigh-benard/
├── references.md              # quick plain-text citation list
├── docs/
│   ├── data_sources.pdf       # full provenance: reproduces every source
│   │                          #   table, highlights what was kept, comments
│   │                          #   on what was excluded and why
│   └── data_sources.tex       # LaTeX source for the above
├── papers/                    # PDFs of the papers referenced above
├── data/
│   └── nu_dataset_model.csv   # Pr, Ra, Nu only - nothing else. Load this for training.
├── notebooks/
│   └── analysis.ipynb         # EDA + random forest regression, following the paper's method
├── scripts/
│   ├── baseline_random_forest.py  # standalone version of the paper's method
│   └── compare_models.py          # baseline vs. 6 other regression methods
├── docs/
│   └── method_comparison.md   # results + discussion of the method comparison
└── results/
    ├── eda_scatter.png
    ├── predicted_vs_actual.png
    ├── baseline_random_forest.png
    └── model_comparison.png
```

## What's in `nu_dataset_model.csv` and what isn't

All 59 rows describe the *same physical setup*: a closed 2D square box
(Γ=1), no-slip walls, isothermal top/bottom plates, adiabatic sidewalls,
solved with the same spectral-element code — confirmed because several
source papers report the exact same Nu (to the decimal) at the same Pr, Ra,
meaning they're literally the same simulation reused across papers.

Left out on purpose (see `docs/data_sources.pdf` for the row-by-row
reasoning): resolution-check re-runs (same physical point, re-run at finer
mesh - not new data), exact cross-paper duplicates (counted once), and the
"RBC-P" table from Pandey/Tiwari/Sreenivasan (2026) - a genuinely different
physical configuration (different solver, likely different domain) used for
a 1D/2D/3D/4D comparison, not this project's question.

## Status

`analysis.ipynb` and `scripts/baseline_random_forest.py` both run end-to-end
on the current 59-row dataset and get R² ≈ 0.97 on the test set — close to
the paper's 0.98 on their full 55-point set, even though 2 of the 5 source
papers (van der Poel 2013, Zhang & Zhou 2024 — see `references.md`) are still
missing. Re-run once those are added.

`scripts/compare_models.py` checks 6 other methods against that baseline
using repeated 5-fold cross-validation (100 splits per method, not just one)
— a physics-motivated power-law fit comes out both best and most stable
(R² = 0.991 ± 0.009), a Gaussian process is a close second but far less
stable (0.989 ± 0.052), and the random forest baseline itself drops to
0.891 ± 0.100 once cross-validated (a single lucky split had made it look
like 0.971). See `docs/method_comparison.md` for the full table and
reasoning behind each suggestion.
