# Multi-hazard adaptation prioritisation across Southeast Asia

Code accompanying the manuscript *Machine Learning-Based Multi-Hazard Adaptation
Across Major Land Systems in Southeast Asia* (Ms. No. INDIC-6889, submitted to
*Environmental and Sustainability Indicators*).

The repository covers hazard susceptibility modelling for landslides, wildfires
and floods across eleven Southeast Asian countries, the validation of those
models, and the collinearity diagnostics reported in Section 4.2 of the paper.

---

## Please read this before using the code

**The scripts that produced the primary results were not retained.** The
susceptibility maps and the AUC values reported in Section 3.1 of the manuscript
come from an earlier Google Earth Engine workflow whose source is no longer
available. The code here is an independent reimplementation of that workflow,
written from the documented method.

What follows from that:

- The cross-validation results in **Table 5**, and the collinearity diagnostics
  in **Tables 3 and 4** and **Figure 12**, were produced by this code and can be
  reproduced from it.
- The susceptibility maps and headline AUC values in **Section 3.1 cannot** be
  reproduced from this code. Re-running the pipeline will give values close to,
  but not identical with, those figures. The manuscript states this and reports
  the validation results as a robustness assessment rather than as a
  reproduction.
- Point counts differ slightly from those in some earlier drafts because the
  wildfire inventory is filtered on the MODIS hotspot `type` attribute (see
  below).

We would rather state this plainly than present the repository as something it
is not.

---

## Repository layout

```
src/
  01_check_points.py   Inspect the hazard point files: class attribute,
                       counts, CRS, geometry type
  02_check_dates.py    Recover the inventory date ranges from the point
                       attributes (event_date, acq_date, dfo_began/ended)
  03_pipeline.py       Main pipeline: predictor stack, pseudo-absence handling,
                       sampling, VIF and correlation, cross-validation,
                       susceptibility export
  04_figure9.py        Figure 9, country-level hazard vs environment scatterplots

  README.md            What the pipeline expects, and where each input comes from
results/
  (written by the scripts)
```

Scripts are numbered in the order they are meant to be run. Each is
self-contained and carries its configuration in a block at the top of the file.

---

## Requirements

```bash
pip install -r requirements.txt
earthengine authenticate
```

An Earth Engine account with an associated cloud project is required for
`03_pipeline.py`. The remaining scripts run locally.

The pipeline was developed and run in Google Colab. `01`, `02`, `03` and `05`
are written so that they can be pasted directly into a notebook cell or run from
the command line.

---

## How to run

**1. Check the inputs.** Confirm that the point files carry the class attribute
and the counts you expect, and recover the inventory date ranges.

```bash
python src/01_check_points.py
python src/02_check_dates.py
```

**2. Sample the predictors.** Set `DRIVE_DIR`, `EE_PROJECT`, `DATE_START` and
`DATE_END` at the top of `03_pipeline.py`, then run it. Predictor values are
pulled to memory in chunks of 250 points, so no batch export is needed and the
Earth Engine synchronous timeout is not reached. Results are cached to Drive, so
an interrupted session does not force a restart.

```bash
python src/03_pipeline.py
```

**3. Regenerate Figure 9** if the underlying zonal statistics change.

```bash
python src/04_figure9.py
```

---

## Method summary

**Analysis grid.** All predictors are resampled to a common 250 m grid.
Continuous variables use bilinear interpolation; land cover and soil texture
class use nearest neighbour so that class membership is preserved.

**Conditioning factors**, following the method described in the manuscript:

| Hazard | Conditioning factors |
|---|---|
| Landslide | precipitation, slope, elevation, soil texture, soil water content, land cover, NDVI |
| Wildfire | precipitation, slope, elevation, land cover, land surface temperature, wind speed, NDVI |
| Flood | precipitation, slope, elevation, land cover, NDVI, distance to river, distance to road |

**Wildfire hotspot filtering.** The MODIS active fire product classifies each
detection by inferred type: `0` presumed vegetation fire, `1` active volcano,
`2` other static land source, `3` offshore. The retrieval for Southeast Asia
returned 1,919 hotspots, of which 1,762 are type 0. The 65 volcanoes and 92
static land sources are excluded, because they are thermal anomalies of
non-vegetative origin that the meteorological and vegetation predictors cannot
explain. Set `POINT_FILTERS = {}` in `03_pipeline.py` to disable this.

**Pseudo-absence.** Two modes are provided. `legacy` uses the non-event point
files as supplied; `buffered` regenerates them as random points outside a
distance buffer around every recorded event. The second mode exists so that the
effect of the sampling scheme can be measured rather than assumed. Run both and
compare with `compare_absence_modes()`.

**Validation.** Three schemes are reported side by side: the original 70:30
hold-out, repeated five-fold random cross-validation, and five-fold spatial
block cross-validation in which whole one-degree blocks are assigned to folds so
that no testing observation falls within the autocorrelation range of a training
observation, following Roberts et al. (2017) and Ploton et al. (2020).

**Collinearity.** Variance inflation factors are computed with an explicit
constant term in the design matrix. Without it, `variance_inflation_factor`
uses an uncentred coefficient of determination and returns inflated values for
variables with a large mean relative to their spread; this is a common and
easily missed error. Pearson and Spearman coefficients are both reported,
because the classifiers are tree ensembles and are therefore invariant to
monotonic transformations of their predictors, which makes rank correlation the
more relevant measure of the redundancy that affects variable importance.

---

## Known limitations

- The scripts for the original model runs are not available, as set out above.
- The right-hand distributional panels of Figures 2 to 4 in the manuscript
  cannot be regenerated: the derived tabulation of area by country,
  susceptibility class and landscape type was not retained, and the code to
  produce it is not in this repository.
- `dist_road` is a constant unless `ROADS_ASSET` is set in `03_pipeline.py`. The
  flood model reported in the manuscript was run without this predictor.
- The three hazard inventories cover different periods (landslides 2007–2018,
  floods 2000–2018, wildfires concentrated in 2018–2019) while the conditioning
  factors are long-term means. The manuscript discusses the consequences.

---

## Data

Input datasets are third-party products and are not redistributed here. See
`data/README.md` for the full list and for what the pipeline expects on disk.

---


## License

MIT. See `LICENSE`.
