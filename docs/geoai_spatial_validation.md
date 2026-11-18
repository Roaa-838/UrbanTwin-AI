# GeoAI error analysis and buffered spatial robustness

## Research question and evidence boundary

This secondary study asks how class imbalance and short train/test distances affect the
project's apparent agreement with ESA WorldCover 2021. It does not replace the frozen GeoAI
benchmark in `results/geoai`, and it is not an independent land-cover accuracy assessment.
WorldCover is a historic pseudo-label product derived partly from remotely sensed inputs; it is
not field-survey ground truth.

The analysis uses the existing real Sentinel-2 L2A stack in EPSG:32636 and the aligned
WorldCover raster. Contains modified Copernicus Sentinel data 2021. © ESA WorldCover project
2021 / Contains modified Copernicus Sentinel data (2021) processed by the ESA WorldCover
consortium; WorldCover 2021 is provided under CC BY 4.0.

## Frozen-benchmark error analysis

The original three repeated group-disjoint holdouts remain unchanged. Metrics below are
recalculated from their stored confusion counts. The same held-out pixels can occur in more than
one repeat, so repeats are descriptive paired evaluations, not independent observations.

| Model | Features | Macro-F1 | Weighted-F1 | Balanced accuracy |
|---|---|---:|---:|---:|
| Random Forest | Bands | 0.5572 | 0.9211 | 0.6558 |
| Random Forest | Bands + NDVI/NDBI | 0.5573 | 0.9220 | 0.6489 |
| Histogram gradient boosting | Bands | 0.5516 | 0.9176 | 0.6481 |
| Histogram gradient boosting | Bands + NDVI/NDBI | 0.5398 | 0.9158 | 0.6295 |

Mean held-out support per repeat is dominated by built-up pixels (7,883 of 9,117; 86.47%).
The mean class-level F1 ranges across the four model/feature configurations show the resulting
disagreement between weighted and macro summaries:

| WorldCover pseudo-label | Mean support per repeat | Mean F1 range |
|---|---:|---:|
| Built-up | 7,883.3 | 0.9423–0.9464 |
| Water | 716.3 | 0.9231–0.9391 |
| Tree cover | 375.0 | 0.4564–0.4792 |
| Grassland | 104.0 | 0.6467–0.6764 |
| Shrubland | 31.0 | 0.2550–0.3033 |
| Bare / sparse vegetation | 7.3 | 0.0130–0.0169 |

A baseline chosen only from each training partition always predicts built-up. Across the three
unique splits it obtains accuracy 0.8517–0.8743 and weighted-F1 0.7835–0.8157, but macro-F1 only
0.1533–0.1555 and balanced accuracy exactly 1/6. This demonstrates why weighted scores alone
are not adequate evidence of minority-class agreement.

Adding NDVI/NDBI does not provide a consistent gain. Mean paired changes relative to bands only
are +0.0001 macro-F1 and −0.0069 balanced accuracy for Random Forest, and −0.0118 macro-F1 and
−0.0186 balanced accuracy for histogram gradient boosting. These indices are deterministic
ratios of bands already available to both tree models; the small mixed changes are descriptive,
not evidence of a general benefit or harm.

## Prespecified buffered protocol

The robustness experiment uses nine contiguous test tiles aligned to the original 20 × 20-pixel
group grid. The 9 block rows are partitioned 3/3/3, and the 11 block columns 4/4/3. Every valid
pixel is assigned to exactly one test tile. Training pixels must come from different original
groups and, in the buffered condition, their centres must be strictly more than 500 m from every
test-pixel centre.

The 500 m distance was fixed before any new model score was calculated. It is the largest member
of a prespecified 100, 200, 300, 400, 500 and 600 m feasibility sequence for which every fold
retains at least three training-supported classes. At 600 m one fold falls below that criterion.
The unbuffered tiled condition uses the same test targets and models, providing the direct
comparison; the original random grouped benchmark is retained only as historical context.

Class eligibility is derived exclusively from buffered training support. The intersection across
all nine folds is tree cover (10), built-up (50), and water (80). These fixed classes are used for
both distance conditions and every model/feature comparison. Other classes are reported as
geographically unsupported rather than silently filtered: among all valid pixels, shrubland has
76 pixels, grassland 137, cropland 2, bare/sparse vegetation 49, and wetland 1. Their localized
coverage cannot support a nine-fold 500 m generalization test in this AOI.

Models and numerical settings are inherited from the frozen benchmark: Random Forest and
histogram gradient boosting, each with bands-only and bands-plus-NDVI/NDBI inputs. Fold seeds are
fixed in the versioned configuration. Training subsampling, class selection, predictions,
skipped folds, low-support classes, and exact pixel-centre separation are recorded.

## Reproduction and artifact isolation

The secondary experiment has its own configuration and ignored output directory:

```powershell
.\.venv\Scripts\urbantwin.exe geoai-spatial-validate
```

The command refuses to overwrite an existing output bundle. Its manifest binds the real input
stack, EO provenance, frozen benchmark details, versioned configuration snapshot, and generated
tables with SHA-256 hashes. Run the bundle validator before interpreting or publishing figures.
Compact figures may be tracked only with a source manifest and the Copernicus/WorldCover
acknowledgements above; large input rasters remain ignored.

## Interpretation limits

- The AOI is about 2.16 km × 1.70 km. A 500 m buffer tests local extrapolation within one compact
  urban scene, not transfer to an independent district, city, sensor date, or reference survey.
- Fixed core classes make the model comparison fair, but narrow the scientific target to three
  classes. The study cannot establish rare-class generalization.
- WorldCover and Sentinel-2 are temporally aligned to 2021 but are not statistically independent
  evidence sources. Agreement may reflect shared spatial or remote-sensing structure.
- Contiguous folds and distance exclusion reduce local leakage; they cannot remove scene-level
  autocorrelation or establish field accuracy.
- No buffer was selected using held-out performance. Buffer comparisons remain descriptive, and
  folds must not be treated as nine independent regional samples.

An external, legally reusable and temporally appropriate reference survey with documented class
definitions and sampling uncertainty would be required for independent accuracy validation.
None is included in this study.
