# Observed-traffic error analysis and paired uncertainty

## Scope

This secondary study diagnoses the completed observed-traffic benchmark. It does not refit the
frozen models, alter the final holdout, or use holdout performance to replace the model selected
from training-only grouped cross-validation. The analysis asks where measurement-free estimates
are inaccurate and how stable paired model differences are across inferred sites.

The study remains exploratory. Every directional segment was surveyed on one date, so date and
site are confounded. The final partition is a joint future-date and unseen-inferred-site holdout,
not separate evidence of temporal and geographic generalization. It contains only ten inferred
sites, which limits uncertainty estimation and influence analysis.

Earth-observation descriptors do not enter this analysis. The available 2024 imagery covers only
the selected contextual link pair, and some acquisitions are retrospective relative to its
survey. No EO contribution to multi-site prediction accuracy is evaluated.

## Immutable evidence gate

The error analysis reads an already completed benchmark only after the existing strict artifact
validator succeeds. Validation binds the analysis to:

- the normalized real-observation source manifest and recorded source-archive hashes;
- the frozen train/holdout and grouped-CV manifests;
- the exact prediction, metric, source, road-context and EO-context artifact hashes;
- the recorded source labels and simulation blocker; and
- the configuration-hash contract declared by the benchmark schema.

Historical observed-benchmark v2 artifacts remain bound to the exact raw bytes recorded in
`config_sha256`. They are validated using an isolated root containing the configuration bytes
from their recorded revision and the matching hashed inputs. A raw-byte mismatch is an error:
validation does not fall back to parsed or canonical equivalence, and no historic manifest or
hash is rewritten. Future v3 bundles retain their separately versioned canonical-hash contract.

The analysis refuses missing, corrupt, stale, incompatible, or partly aligned inputs. It does not
substitute a synthetic dataset when the real validated bundle is unavailable. Synthetic fixtures
are used only by automated tests of the analytical code.

## Analysis unit and residual convention

The unit is one directional 20-minute interval in the unchanged frozen holdout. For observation
`i`, the residual is defined as

```text
residual_i = prediction_i - observed_volume_i
```

Negative residuals are underpredictions and positive residuals are overpredictions. For each model
and inferred site, the analysis reports the number of intervals, mean absolute error (MAE), root
mean squared error (RMSE), and mean residual (prediction bias). Inferred sites are analytical
geometry clusters rather than publisher-supplied location identifiers.

All measurement-free specifications must have exactly the same observation membership before a
paired comparison is allowed. Observation ID, inferred site and observed target must agree on both
sides of every pair. Duplicate or partially overlapping rows are rejected rather than silently
joined.

## Volume-stratified diagnostics

Observed-volume quantile boundaries are calculated once from the unique frozen-holdout
observations, not from model-specific duplicated prediction rows. The same boundaries and
memberships are then applied to every measurement-free specification. Quantile summaries include
MAE, RMSE, bias, underprediction rate and overprediction rate.

The highest-volume stratum is used to describe high-volume underprediction; the lowest-volume
stratum is used to describe low-volume overprediction. These are post-evaluation diagnostics based
on the observed target. They are not input features, tuning criteria, deployment rules, or evidence
of a causal volume effect.

## Prespecified paired comparisons

The secondary analysis makes three paired comparisons on identical holdout rows:

1. Random Forest with combined features versus histogram gradient boosting with combined
   features.
2. Random Forest with combined features versus Random Forest with spatial-only features.
3. Random Forest with combined features versus the training-only historical-mean baseline.

For each comparison, the reported difference uses `left minus right`. A negative MAE difference
therefore favors the left specification. These comparisons do not change the official primary
model: that model remains the specification selected by training-only, inferred-site-grouped CV.

## Whole-site paired bootstrap

Uncertainty is estimated by resampling whole inferred sites with replacement. Each replicate:

1. draws the same number of sites as the frozen holdout;
2. retains every interval belonging to each sampled site, including repeated copies when a site is
   sampled more than once;
3. applies the identical sampled-site sequence to both models in a comparison; and
4. recomputes the paired MAE and RMSE differences.

The random seed and replicate count are versioned in the analysis configuration. The interval is
the percentile interval from the paired replicate distribution. Individual rows are never sampled
as IID observations. The resulting interval reflects sensitivity to the ten available site
clusters; it is not a citywide confidence interval, a formal proof of superiority, or a correction
for the confounded survey-date design.

## Leave-one-site-out influence

Influence diagnostics repeat each paired comparison after excluding one complete inferred site.
The change from the full-sample paired difference shows whether a conclusion depends strongly on a
particular site cluster. Sites are represented by deterministic anonymous labels in private
outputs. Large leave-one-site-out changes are reported as fragility, not treated as a reason to
remove an observation or choose a different model.

## Training development versus final holdout

For every measurement-free model/feature specification, the analysis places the five-fold
training-CV mean and dispersion beside the frozen-holdout error. The difference is descriptive:
the partitions contain different inferred sites and different dates, and the design cannot
separate those two sources of shift. Final-holdout values never enter model fitting, preprocessing,
hyperparameter choice or the official selection record.

## Persistence is a separate information budget

One-step persistence is evaluated only where a previous observation exists at the same directional
segment exactly 20 minutes earlier. It consumes a measured count unavailable to the
measurement-free estimators. Its coverage and error diagnostics are therefore shown separately;
they are not used in paired model rankings or presented as a like-for-like baseline.

## Private outputs and publication boundary

The audited traffic records do not currently carry an explicit record-level license authorizing
redistribution of derived measured-traffic products. Consequently, row-level predictions,
site-level diagnostics, numerical aggregate tables, analytical figures and the private research
report remain in Git-ignored output directories. The tracked repository contains only analytical
code, versioned configuration, deterministic synthetic tests and this methods description.

Public availability of a download is not treated as an open license. Derived numerical outputs
should be published only after the applicable publisher terms explicitly authorize that scope.
Source attribution and the broader evidence categories remain documented in
[DATA_SOURCES.md](DATA_SOURCES.md).

## Local execution

The analysis destination must be new; existing evidence is never overwritten. For a current
canonical bundle, the project root can also be the validation root. A historical v2 bundle requires
an isolated validation root containing its exact recorded configuration bytes and matching hashed
road/EO inputs.

```powershell
.\.venv\Scripts\urbantwin.exe observed-error-analysis `
  --bundle-dir "results\observed_traffic" `
  --output-dir ".urbantwin\private\observed_error_analysis_v1" `
  --config "configs\observed_error_analysis_v1.yaml" `
  --validation-root $legacyValidationRoot `
  --source-dir "data\observed\tfc\normalized_v1"
```

The output path must remain private and Git-ignored. A successful private bundle records the
upstream provenance hash, every source artifact hash used by the validator, the
analysis-configuration hash, the code revision, and hashes for all generated tables, figures and
the private report.

## Interpretation limits

- Ten inferred holdout sites provide limited support for site-level uncertainty estimation.
- Survey date and site cannot be disentangled in this release.
- Inferred sites are reproducible spatial clusters, not confirmed physical-site identifiers.
- Target-based volume strata are retrospective diagnostics and cannot be used prospectively.
- Static road descriptors and later map geometry do not establish a causal spatial mechanism.
- No EO accuracy gain, calibrated simulation, controller improvement, observed emissions reduction,
  or Greater Cairo-wide generalization is claimed.
