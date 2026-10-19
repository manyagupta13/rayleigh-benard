# References

Papers used to build the Nu(Pr, Ra) dataset in `data/`. References 1–6 are
the ones cited in Sec. 3.3 ("Forecasting heat transport using machine
learning") of Shreshthi & Pandey; 7–10 were added afterwards because they
simulate the identical configuration (2D square box, Γ = 1, no-slip walls,
isothermal plates, adiabatic sidewalls). Row-by-row provenance, with every
source table reproduced, is in `docs/data_sources.pdf`.

1. Shreshthi and A. Pandey. "Turbulent Prandtl number in the near-wall region
   of thermal convection." (project paper).
   `papers/shreshthi_pandey_2026_turbulent_prandtl_number.pdf` — 13 rows

2. A. Pandey. "Thermal boundary layer structure in low-Prandtl-number
   turbulent convection." *J. Fluid Mech.*, 910:A13, 2021.
   doi:10.1017/jfm.2020.961 — arXiv:2101.03051
   `papers/pandey_2021_thermal_boundary_layer.pdf` — 5 rows

3. A. Pandey and K. R. Sreenivasan. "Transient and steady convection in two
   dimensions." *J. Fluid Mech.*, 1015:A42, 2025.
   doi:10.1017/jfm.2025.10357 — arXiv:2503.03080
   `papers/pandey_sreenivasan_2025_transient_steady_convection.pdf` — 29 rows

4. A. Pandey, H. Tiwari, and K. R. Sreenivasan. "Thermal convection in 1, 2,
   3, and 4 dimensions." *J. Fluid Mech.*, in press, 2026.
   arXiv:2607.26372
   `papers/pandey_tiwari_sreenivasan_2026_1234_dimensions.pdf` — 12 rows

5. E. P. van der Poel, R. J. A. M. Stevens, and D. Lohse. "Comparison between
   two- and three-dimensional Rayleigh-Bénard convection." *J. Fluid Mech.*,
   736:177-194, 2013. doi:10.1017/jfm.2013.488
   2D values at Ra = 1e8 from Appendix Table 2 — 16 rows

6. Y. Zhang and Q. Zhou. "Low-Prandtl-number effects on global and local
   statistics in two-dimensional Rayleigh-Bénard convection." *Phys. Fluids*,
   36(1):015107, 2024. doi:10.1063/5.0175011 — Table II, 15 rows

7. Y. Zhang, Q. Zhou, and C. Sun. "Statistics of kinetic and thermal energy
   dissipation rates in two-dimensional turbulent Rayleigh-Bénard
   convection." *J. Fluid Mech.*, 814:165-184, 2017. doi:10.1017/jfm.2017.19
   — Table 1, 18 rows

8. A. Xu, L. Shi, and H.-D. Xi. "Statistics of temperature and thermal
   energy dissipation rate in low-Prandtl number turbulent thermal
   convection." *Phys. Fluids*, 31:125101, 2019. doi:10.1063/1.5129818 —
   arXiv:1911.04042 — Table II, 8 rows

9. J.-C. He, M.-W. Fang, Z.-Y. Gao, S.-D. Huang, and Y. Bao. "Effects of
   Prandtl number in two-dimensional turbulent convection." *Chin. Phys. B*,
   30(9):094701, 2021. doi:10.1088/1674-1056/ac0781 — 62 rows extracted from
   the vector graphics of Fig. 5(a) (no table in the paper)

10. E. P. van der Poel, R. J. A. M. Stevens, K. Sugiyama, and D. Lohse. "Flow
    states in two-dimensional Rayleigh-Bénard convection as a function of
    aspect-ratio and Rayleigh number." *Phys. Fluids*, 24:085104, 2012.
    doi:10.1063/1.4744988 — arXiv:1206.3823 — 16 rows (Γ = 1 series)
    extracted from the vector graphics of Fig. 1(a)

Checked, not used:

11. J.-L. Yang et al. "The Pr-dependence of the critical roughness height in
    two-dimensional turbulent Rayleigh-Bénard convection." *J. Fluid Mech.*,
    911:A52, 2021. doi:10.1017/jfm.2020.1091 — smooth-cell values only in a
    low-resolution raster figure at Ra = 1e8 (already well covered); used
    only as a cross-check.

## Data provenance

Every row of `data/nu_dataset_sources.csv` has a `Source` column naming one
of references 1–10, plus `Grid`, `Notes` and `Flag` columns. Five rows are
flagged (`multistable`, `flow-state`, `conflict`); see Section 13 of
`docs/data_sources.pdf`.
