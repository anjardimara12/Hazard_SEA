# Data

Input datasets are third-party products and are not redistributed in this
repository. This file records what the pipeline expects, and where each input
comes from.

## Hazard point files

Six shapefiles, supplied as zip archives. `src/01_check_points.py` inspects them
and reports counts, CRS, geometry type and the class attribute.

| Hazard | Event points | Non-event points |
|---|---|---|
| Landslide | 2,483 | 1,100 |
| Wildfire | 1,919 retrieved, 1,762 after filtering on hotspot type | 1,089 |
| Flood | 150 | 146 |

The class attribute is named `Potensi`: `1` for event points, `0` for non-event
points. Each file holds a single class.

The non-event points were placed in areas where the conditioning factors
indicated very low susceptibility. This selects absences using the same
predictors later supplied to the classifiers, and is expected to inflate
apparent discrimination. `src/03_pipeline.py` provides a `buffered` mode that
regenerates absences independently of the predictors, so the effect can be
measured rather than assumed.

Distances from each non-event point to the nearest recorded event, computed by
the pipeline:

| Hazard | Minimum | 5th percentile | Median |
|---|---|---|---|
| Landslide | 0.5 km | 6.7 km | 55.2 km |
| Wildfire | 2.5 km | 17.9 km | 175.5 km |
| Flood | 9.2 km | 32.5 km | 187.9 km |

Set `DRIVE_DIR` in `01_check_points.py`, `02_check_dates.py` and
`03_pipeline.py` to the folder holding these archives.

## Inventory sources and periods

Recovered from the date attributes of the point files by
`src/02_check_dates.py`.

| Hazard | Source | Period | Note |
|---|---|---|---|
| Landslide | Global Landslide Catalog (Kirschbaum et al., 2019) | 2007–2018 | 1,439 of 2,483 records carry a usable date; the remainder were retained on location alone |
| Wildfire | MODIS Collection 6 active fire (Vadrevu et al., 2019) | 2018–2019 | 47 further records extend into 2021 |
| Flood | Global Flood Database (Tellman et al., 2021) | 2000–2018 | 913 events mapped globally; 150 fall within the study area |

## Conditioning factors

All are Earth Engine catalogue products, resolved by `03_pipeline.py` at run
time. No manual download is required.

| Variable | Collection | Native resolution |
|---|---|---|
| Elevation | `USGS/SRTMGL1_003` | 30 m |
| Slope | derived from the ALOS DSM | 30 m |
| Land cover | `COPERNICUS/Landcover/100m/Proba-V-C3/Global` | 100 m |
| Soil water content | `OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01` | 250 m |
| Soil texture class | `OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02` | 250 m |
| Precipitation | `UCSB-CHG/CHIRPS/PENTAD` | ~5.5 km |
| NDVI | `MODIS/061/MOD13Q1` | 250 m |
| Land surface temperature | `MODIS/061/MOD11A2` | 1 km |
| Wind speed | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | ~28 km |
| Distance to river | `WWF/HydroSHEDS/v1/FreeFlowingRivers` | vector |
| Distance to road | not set by default | vector |

Five-day and eight-day composites are used in place of the daily collections.
Long-term means over the same window are equivalent, and the reduction from
several thousand images to a few hundred is what keeps the sampling step within
Earth Engine's limits.

`dist_road` requires a roads asset. Without one it is a constant and is dropped
from the model, which is what happened for the flood model reported in the
manuscript. Set `ROADS_ASSET` in `03_pipeline.py` to include it.

## Study area

Eleven countries, from `USDOS/LSIB_SIMPLE/2017`. Note that Myanmar appears in
that dataset as **Burma**; filtering on "Myanmar" silently removes the country
and roughly 677,000 km² of the study area, along with every point inside it. The
computed area of the eleven countries is 4,454,132 km².

## Not included

- The derived tabulation of area by country, susceptibility class and landscape
  type behind the right-hand panels of Figures 2 to 4. This was not retained.
- The rice production surfaces of Ramankutty and Foley (2008) used to place the
  cropland suitability training points.
- The scripts for the original model runs.
