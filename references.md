# References

Papers cited in Sec. 3.3 ("Forecasting heat transport using machine learning")
of Shreshthi & Pandey, used to build the Nu(Pr, Ra) dataset in `data/`.

1. Shreshthi and A. Pandey. "Turbulent Prandtl number in the near-wall region
   of thermal convection." (project paper).
   `papers/shreshthi_pandey_2026_turbulent_prandtl_number.pdf`

2. A. Pandey. "Thermal boundary layer structure in low-Prandtl-number
   turbulent convection." *J. Fluid Mech.*, 910:A13, 2021.
   doi:10.1017/jfm.2020.961 — arXiv:2101.03051
   `papers/pandey_2021_thermal_boundary_layer.pdf`

3. A. Pandey and K. R. Sreenivasan. "Transient and steady convection in two
   dimensions." *J. Fluid Mech.*, 1015:A42, 2025.
   doi:10.1017/jfm.2025.10357 — arXiv:2503.03080
   `papers/pandey_sreenivasan_2025_transient_steady_convection.pdf`

4. A. Pandey, H. Tiwari, and K. R. Sreenivasan. "Thermal convection in 1, 2,
   3, and 4 dimensions." *J. Fluid Mech.*, in press, 2026.
   arXiv:2607.26372
   `papers/pandey_tiwari_sreenivasan_2026_1234_dimensions.pdf`

5. E. P. van der Poel, R. J. A. M. Stevens, and D. Lohse. "Comparison between
   two- and three-dimensional Rayleigh-Bénard convection." *J. Fluid Mech.*,
   736:177-194, 2013. doi:10.1017/jfm.2013.488
   **Not yet obtained** — reports Nu only as scatter plots (Fig. 5a, 6b), no
   table; needs institutional access to digitize.

6. Y. Zhang and Q. Zhou. "Low-Prandtl-number effects on global and local
   statistics in two-dimensional Rayleigh-Bénard convection." *Phys. Fluids*,
   36(1):015107, 2024. doi:10.1063/5.0175011
   **Not yet obtained** — fully paywalled, needs institutional access.
   https://pubs.aip.org/aip/pof/article/36/1/015107/2932406

## Data provenance

Every row in `data/nu_dataset.csv` has a `Source` column pointing to one of
references 1-4 above (5 and 6 are still missing). See that file's `Notes`
column for rows that are likely duplicates across papers, or from a
different flow configuration.
