# GeoAI spatial-validation figure manifest

These compact figures were rendered from the strictly validated local
`urbantwin.geoai-spatial-validation/v1` bundle. They contain no measured-traffic data,
survey geometry, personal information, signed asset URLs, or large raster inputs.

| Figure | Evidence shown | SHA-256 |
|---|---|---|
| `baseline_per_class_f1_support.png` | Frozen benchmark mean per-split F1 and held-out support by WorldCover pseudo-label class | `bfc51fdec93105d564d648716f9ddc4cf4079015fd5f5842e5e7c5258a99389f` |
| `spatial_fold_design.png` | Central prespecified test tile, strict 500 m exclusion region, and retained training pixels | `f00d19c8189055dd20379ee08b0f1ed47d5260af75e0240a8e0f616d27dc6d7c` |
| `spatial_robustness_comparison.png` | Three-class out-of-fold macro-F1 for paired 0 m and 500 m conditions | `82506a920f1c5a322ebde6d231a6568c78ed42ec048a7b1ea6dfaf408b833222` |
| `spatial_confusion_random_forest_indices.png` | Row-normalized three-class confusion matrices for the pre-existing RF + indices configuration | `0a1af2b2dea8982682707fa33185fce27e19557ed345abb85ffacb7a91103730` |

The complete machine-readable binding is in
[`publication_manifest.json`](publication_manifest.json). It records source-bundle provenance
SHA-256 `f61ed82ed29cc939e8b17efc670eec521b2cf80929db51595edd00de47b5e1d0`,
every source artifact hash, and the rendered figure hashes. The local bundle itself remains
Git-ignored; the figures can be regenerated without refitting models by running:

```powershell
.\.venv\Scripts\urbantwin.exe geoai-spatial-figures `
  --output-dir "docs\assets\geoai_spatial_validation_reproduction"
```

Contains modified Copernicus Sentinel data 2021. © ESA WorldCover project 2021 / Contains
modified Copernicus Sentinel data (2021) processed by the ESA WorldCover consortium. WorldCover
2021 is provided under CC BY 4.0. Scores are agreement with historic pseudo-labels, not
independent field-survey accuracy.
