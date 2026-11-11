# Greater Cairo observed-data feasibility audit

Audit date: 2026-11-08
Repository baseline: `71a163f296c87a38913e45fba1085257e4c61294`
Study status: **observed link-flow study feasible with qualifications; intersection-control calibration not feasible from the audited data alone**

## Scope and exit-gate decision

This phase inspected primary data records and the existing repository before any new traffic model was built. It did not alter the existing abstract SUMO benchmark, create an MLflow run, download an Egypt-scale road extract, or claim that a calibrated Greater Cairo traffic model exists.

The three Transport for Cairo (TfC) records were accessed anonymously and their relationship was validated:

- 2024 traffic counts: accessed and fully inspected (6,618 rows).
- 2024 vehicle occupancy: record, schema, units, and a small sample inspected; the full 37.2 MB archive was not retained or scanned in this phase.
- Traffic survey locations: all 145 directional geometries accessed through bounded WFS requests and inspected.

The records support an observed **directional road-link flow** study. They do not contain intersection turning movements, signal-controller observations, phase plans, cycle lengths, offsets, detector data, or a complete demand matrix. Consequently, this audit stops before route synthesis, signal-plan construction, SUMO calibration, or optimization.

The nearest defensible observed-link candidate is the 6th of October Bridge pair that intersects the current Tahrir AOI. It is suitable for a link-flow/remote-sensing context study, not a signal-control experiment. No signalized-intersection study site is verified. The public geometry has no physical `location_id`, so the announcement's exact 60-site grouping also remains publisher-dependent even though a documented spatial clustering reproduces 60 groups.

## Primary records and access status

The official [TfC LinkedIn organization page](https://www.linkedin.com/company/transport-for-cairo/) announces three linked records and describes 60 locations, two travel directions, 12 vehicle categories, and 20-minute intervals. Its short links resolve to these official portal records:

| Record | Official catalogue page | Machine metadata | Direct data access | Anonymous access |
|---|---|---|---|---|
| Traffic counts | [document 190](https://data.transportforcairo.com/catalogue/#/document/190) | [resource 190](https://data.transportforcairo.com/api/v2/resources/190) | [asset 252](https://data.transportforcairo.com/api/v2/assets/252/download) | Verified |
| Vehicle occupancy | [document 192](https://data.transportforcairo.com/catalogue/#/document/192) | [resource 192](https://data.transportforcairo.com/api/v2/resources/192) | [asset 254](https://data.transportforcairo.com/api/v2/assets/254/download) | Verified |
| Survey locations | [dataset 193](https://data.transportforcairo.com/catalogue/#/dataset/193) | [resource 193](https://data.transportforcairo.com/api/v2/resources/193) | [bounded WFS GeoJSON](https://data.transportforcairo.com/geoserver/ows?service=WFS&version=2.0.0&request=GetFeature&typeNames=geonode%3A0_4_traffic_survey_locations&count=145&startIndex=0&outputFormat=application%2Fjson&srsName=EPSG%3A4326) | Verified |

Official announcement short links:

- [traffic counts](https://lnkd.in/eQerP2qC) to document 190;
- [vehicle occupancy](https://lnkd.in/eQtmHsgi) to document 192;
- [survey locations](https://lnkd.in/efHPjC2T) to dataset 193.

No account, registration, authenticated cookie, or signed asset URL was required. The API advertises anonymous view and download permission. The portal metadata gives the attribution `Transport for Cairo (TfC)` and describes the observations as raw, unfactored, unannualized, and not seasonally adjusted. All three records have portal publication metadata dated 2026-10-19; the observed data span 2024.

### License status

The announcement and portal describe the records as open data, but each of the three API records has an empty `license.identifier`; the constraints field is also blank. TfC's general open-data policy is not a substitute for a record-specific legal grant. A Creative Commons license used by a different TfC project must not be transferred to these records by assumption.

**Permission update:** after this feasibility audit, the project author reported receiving direct TfC permission to continue the academic UrbanTwin-AI research and reuse the data. The original communication and personal contact details remain private. This author-reported permission supersedes the earlier conclusion that no research-reuse authorization had been obtained, but it does not convert the records into openly licensed data or establish public redistribution rights that are not present in the exact message.

Therefore:

- inspection and validation of the public downloads were possible;
- attribution to TfC is required by the record metadata;
- academic research and data reuse are authorized according to the project author's report;
- the portal license remains unspecified, and public redistribution of source files, normalized rows, derived maps, aggregate results, PDFs or GitHub artifacts requires checking the exact private permission terms;
- no open-source, Creative Commons or general public-redistribution license is inferred.

The repository contains no copied TfC record or real-data fixture. Offline tests use generated toy records.

## Actual schemas, units, and quality findings

### Traffic counts

The downloaded archive was 849,566 bytes. SHA-256 of that archive at audit time was `d90703a022b2cad08e0533a4d9538236e45034c2fe260673a08c6995a56bbd7b`; its single CSV member was 849,406 bytes with SHA-256 `6bb9c42cd1fabcafab8478059d5e63d983a6f8d073f7f9b6461ced9982f3f52a`.

Actual columns, in source order:

```text
osm_id, Day, Date, road_name, direction, hour_start, hour_end,
motorcycle, car, taxi, pickup, 2AxTruck, 3AxTruck, >3AxTruck,
other, mini_microbus, publicbus, privatebus, tuktuk, segmentId
```

The 12 vehicle fields are non-negative integer **vehicle counts per 20-minute directional link interval**. They are not rates, passenger counts, annual average daily traffic, or turning movements.

Verified profile:

| Check | Result |
|---|---:|
| Data rows | 6,618 |
| Collection range | 2024-07-21 through 2024-11-05 |
| Distinct survey dates | 30 |
| Distinct `segmentId` values | 142 |
| Distinct `osm_id` values | 128 |
| Interval duration | 20 minutes for all 6,618 rows |
| Direction codes | `0`: 3,292 rows; `1`: 3,326 rows |
| Vehicle-count cells with non-integers or negatives | 0 |
| Duplicate interval keys | 0 |
| Blank values | 239 blank `road_name`; no other blank cells |
| Total vehicles across 12 classes | 2,961,483 |

The published metadata calls the directions inbound/outbound, while the files encode only `0` and `1`. There is no machine-readable legend proving which code is inbound. The Arabic `survey_line_name` in the geometry sometimes gives a destination, but it is not present for every feature and must not be used as a universal codebook. The observation timezone is not stated.

`Day` values are correct for `Date` after whitespace trimming, but four weekday spellings contain trailing spaces. Six directional segments account for the 239 blank road names; all six can be associated with a nonblank geometry road name through `segmentId`, but a validator reports rather than silently fills those source cells.

### Vehicle occupancy

The portal metadata reports 299,450 records over the same 2024-07-21 to 2024-11-05 period. The actual sampled header is:

```text
osm_id, Day, Date, road_name, direction, hour_start, hour_end,
pax_motorcycle, pax_car, pax_taxi, pax_pickup, Pax_2AxTruck,
Pax_3AxTruck, Pax_>3AxTruck, pax_other, pax_mini_microbus,
pax_publicbus, pax_privatebus, pax_tuktuk, segmentId
```

The source is mixed-unit by design:

- motorcycle, car, taxi, pickup, truck, tuktuk, and other fields contain observed occupant counts for an individual observed vehicle;
- minibus/microbus, public-bus, and private-bus fields contain percentage occupancy strings such as `50%`, `75%`, and `100%`.

Sample rows place an observation in the relevant class field and use zeroes elsewhere. Interpreting each row as one observed vehicle is consistent with the sample and announcement, but the portal supplies no separate data dictionary. These rows are not interval-level average occupancy factors without additional aggregation. Exact full-file missingness and domain frequencies are **not verified** because the full occupancy archive was not scanned under the small-sample constraint.

### Survey-location geometry

Actual WFS schema:

```text
fid: integer
osm_id: integer
segmentid: string
direction: string
road_name: string
highway: string
oneway: string, nullable
survey_line_name: string, nullable
osm_extract_date: date
geom: MultiLineString, EPSG:4326
```

The WFS schema above uses `fid`, lower-case `segmentid`, and `MultiLineString`. The original 75,205-byte downloadable archive uses `segmentId`, omits `fid`, and contains 145 `LineString` features. Its audit-time SHA-256 was `d8b8e329be14b2c008460e7caed55c3dc820dc151119f4d0d37fa4a6070b42b1`. These are two serializations of the same 145 source records; the validator accepts both explicitly.

Verified profile:

| Check | Result |
|---|---:|
| Features | 145 directional lines (`LineString` download; `MultiLineString` WFS) |
| Geometry CRS | CRS84 / EPSG:4326 longitude-latitude |
| Valid, nonempty geometries | 145 |
| Distinct segment IDs | 145 |
| Distinct `osm_id` values | 131 |
| Direction codes | `0`: 71; `1`: 74 |
| Feature length in EPSG:32636 | 65.86-261.17 m; median 137.13 m |
| Missing `oneway` | 9 |
| Missing `survey_line_name` | 2 |
| Other missing schema fields | 0 |
| Source OSM extract date | 144 features: 2026-07-15; 1 feature: 2026-07-20 |

The extent is longitude 30.50059345-31.89547358 and latitude 29.61521685-30.50295063. The supplied lines therefore cover Greater Cairo, not just the current Tahrir AOI.

The source OSM geometry dates are in 2026, roughly two years after the measured traffic. They are valid third-party map context for locating the records, but they are not contemporaneous 2024 road observations.

### Count-to-geometry association

The documented keys are `segmentId`/`segmentid` and `osm_id`. Validation found:

- after trimming one geometry segment ID with a trailing newline, every one of the 142 counted segments has a geometry;
- three geometry segments have no traffic-count rows;
- five counted segments, affecting 156 count rows, have an `osm_id` that conflicts with the `osm_id` on the geometry row carrying the same segment ID;
- all matched segment direction codes agree.

For audit and publisher follow-up, the trimmed segment ID is the effective observed join key. Production use must not silently accept the five two-key conflicts:

| Segment ID | Count `osm_id` | Geometry `osm_id` |
|---|---:|---:|
| `2cb11255-8dd0-4b9c-ada6-3ca1d849c1b5` | 544278163 | 628911446 |
| `3e884148-6dd2-473a-a298-ea93a6f8c74f` | 628911446 | 544278163 |
| `61b492ea-0953-40d6-b6aa-5d1ee3ff4f86` | 680915920 | 972297890 |
| `83800cf8-2840-4797-8381-f22558648120` | 972297890 | 753869090 |
| `8b3e3b85-1b44-496a-ae5a-e358e0783bee` | 99894381 | 680915920 |

The geometry-only segment IDs are:

```text
1ee50603-9594-4c9d-b427-6398fd628ede
3619a904-a27f-4720-b9ae-76044858e93d
6a5b83a3-60fe-4a83-b6b5-a155c17ae4e7
```

OSM way IDs are not permanent identifiers; the TfC metadata itself instructs users to preserve `osm_extract_date` and the direction field. A later OSM snapshot must be rematched spatially and must not be joined by an unverified current way ID alone.

## Spatial audit of the stated 60 survey locations

The public geometry has no physical-site identifier. It has 145 directional road pieces because a survey line can cross separated carriageways or side roads. The publisher's exact 60-location identities therefore cannot be verified from an official key.

As a diagnostic only, this audit tested whether the supplied geometry is spatially consistent with the announcement:

1. transform all supplied lines from EPSG:4326 to the repository's metric CRS, EPSG:32636;
2. make an undirected graph connecting two supplied lines when their geometric separation is at most 250 m;
3. treat connected components as **inferred site clusters**.

This yields 60 clusters: 46 contain two features, 3 contain three, and 11 contain four. Every cluster contains both direction codes. The result stays at 60 from a 250 m through 750 m threshold and falls to 58 at 1,000 m. That is useful corroboration, but the threshold-based grouping is an inference, not a replacement for a publisher-supplied `location_id` or point geometry.

Distances below are minimum geometry-to-AOI distances in EPSG:32636, using the configured Tahrir bbox `[31.225, 30.038, 31.247, 30.053]`. Directional totals are measured vehicles summed over all 12 classes; they are descriptive source checks, not model performance.

| Rank | Inferred site / representative coordinate | AOI distance | Road type | Survey dates | 20-min rows (dir 0 / dir 1) | Vehicles (dir 0 / dir 1) | Signal-node check |
|---:|---|---:|---|---|---:|---:|---|
| 1 | 6th of October Bridge (31.229902, 30.049172) | 0 m | motorway | 2024-10-31, both | 72 / 72 | 107,028 / 69,206 | None mapped within 500 m |
| 2 | 15th May Bridge (31.227073, 30.057712) | 489 m | motorway | 2024-07-22 / 2024-07-24 | 72 / 72 | 38,370 / 52,496 | Not queried |
| 3 | Salah Salem Street north (31.264351, 30.028118) | 1,982 m | trunk | 2024-10-27, both | 53 / 50 | 49,239 / 67,221 | None mapped within 500 m |
| 4 | Salah Salem Street south (31.235022, 30.015686) | 2,453 m | trunk | 2024-07-29, both | 51 / 51 | 22,119 / 26,376 | None mapped within 500 m |
| 5 | Abbas Bridge (31.220595, 30.015880) | 2,472 m | trunk | 2024-07-23 / 2024-07-22 | 72 / 72 | 48,120 / 41,556 | Not queried |
| 6 | Rod El Farag Bridge (31.229459, 30.085772) | 3,621 m | primary | 2024-07-22, both | 72 / 72 | 63,571 / 39,956 | Not queried |
| 7 | Nasr Road (31.297184, 30.056084) | 4,781 m | trunk | 2024-07-29, both | 51 / 51 | 32,953 / 32,497 | None mapped within 500 m |

A single bounded Overpass query checked current `highway=traffic_signals` nodes within 500 m of ranks 1, 3, 4, and 7. It returned zero nodes at OSM base timestamp `2026-11-08T10:45:55Z`. That means only that no such node was mapped in those four radii in that snapshot. It does not prove physical signals are absent.

### Reproducible site decision

For a **link-flow and EO-context study**, provisionally select rank 1, the 6th of October Bridge pair, because it is the only supplied site cluster that intersects the existing AOI and it has same-day, full-day coverage in both direction codes. This rule uses location and measurement completeness only; it does not inspect any model outcome. The physical-site interpretation remains provisional until TfC confirms the grouping. Academic research reuse is author-reported as authorized; any public release still depends on the exact private permission terms covering that publication category.

For an **intersection-control study**, select no site yet. Rank 1 is a motorway bridge, not an at-grade controlled junction. The nearest non-motorway candidates lack verified signal proximity, and the TfC files lack turning movements and signal timing. A suitable intersection requires a publisher/agency site key, field-checked geometry, controller/timing records, and movement-level counts.

## What the traffic records can and cannot establish

The files establish directional link volumes by class and interval. They do not distinguish left, through, or right movements; identify approaches to a common junction; provide lane-by-lane observations; or record routes through a network. Two directional counts on a road are not a complete turning-flow matrix.

The occupancy file can support observed passenger-load summaries after its mixed units are confirmed and its full quality profile is run. It cannot by itself assign passengers to turns or reveal signal state.

Do not derive unobserved turning ratios by forcing link totals to balance and then label those values measured. Any such balancing, temporal expansion, route choice, lane assignment, or controller program is an inferred or simulated input and must be reported as such.

## Road geometry feasibility

Primary road sources:

- [Geofabrik Egypt](https://download.geofabrik.de/africa/egypt.html) provides a daily Egypt `.osm.pbf`, checksum, polygon, daily changes, and historical files. Egypt has no smaller Geofabrik subregion.
- [Geofabrik download and license documentation](https://www.geofabrik.de/data/download.html) identifies PBF as raw OSM data and applies ODbL 1.0 with OSM attribution/share-alike duties.
- [Overpass API guidance](https://wiki.openstreetmap.org/wiki/Overpass_API) supports small, selective live queries; repeated work should be cached, rate-limited, and moved to regional extracts when appropriate.
- [SUMO's official OSM importer](https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html) imports OSM XML, converts WGS84 to UTM, and warns that missing speeds, ramps, junction joins, and traffic lights may be inferred heuristically.
- [Osmium extract documentation](https://docs.osmcode.org/osmium/latest/osmium-extract.html) describes buffered, reference-complete clipping modes.

As of this audit, `egypt-latest.osm.pbf` was approximately 170 MB and contained data through 2026-11-07T21:21:36Z. The archive exposes `egypt-240101.osm.pbf` (164,912,017 bytes), but that is an early-January snapshot and is not automatically representative of each July-November 2024 survey date.

A lightweight Overpass test for the configured Tahrir AOI first returned HTTP 502 and then, after backoff, reported 1,162 highway ways at OSM base `2026-11-08T10:37:51Z`; a separate count reported 44 mapped signal nodes in the AOI. A subsequent geometry sample returned 502. These checks establish source reachability and the presence of current map content, not a downloaded, connected, study-ready network. The 44 nodes are not 44 verified controllers and contain no observed timing.

The current repository road function uses one OSMnx/Overpass call and exports geometry only. It lacks a project-owned durable cache, bounded handling of 502 responses, snapshot/query hashes, a Geofabrik fallback, turn-restriction preservation, component QA, and count-site anchoring. `results/roads.json` is absent. Therefore no genuine connected study network is currently validated.

### Recommended network acquisition after the gate is cleared

1. Obtain the publisher's physical-site identifier and a verified study polygon around the chosen observed site.
2. For a small current-map check, use a named cache under `.urbantwin/osm_cache`, an identifiable user agent, no parallel requests, bounded exponential backoff for 429/502/504, and preserve query text, endpoint, retrieval time, `osm_base`, version, and SHA-256.
3. For reproducible offline work, download a pinned Geofabrik snapshot once, verify its published checksum plus a local SHA-256, and record ODbL attribution. Do not rely only on mutable `latest`.
4. Clip with a justified buffer. Preserve complete ways and validate turn-restriction relation references. Convert the clipped PBF to OSM XML before `netconvert`.
5. Match the observed lines to candidate network edges in EPSG:32636 using a declared distance/heading rule; retain ambiguity and unmatched reports. Check the connected component that contains the measured link, rather than blindly keeping every component or the largest component.
6. Inspect lane, speed, one-way, junction, boundary, and restriction missingness. Preserve original OSM IDs and verify WGS84-to-SUMO coordinate round trips.
7. Stop before traffic-light construction unless observed controller and movement data are acquired. Any SUMO-guessed junction, timing plan, route, or turn remains inferred/simulated.

## Temporal alignment with Earth observation

The existing real EO bundle is valid for its stated purpose but is from 2021:

- three Sentinel-2 L2A scenes acquired on 2021-08-01, 2021-08-06, and 2021-08-26;
- ESA WorldCover 2021 v200;
- projected grid EPSG:32636 at 10 m over the Tahrir AOI;
- Sentinel reflectance resampled bilinearly; SCL and WorldCover categoricals resampled nearest-neighbour;
- clear SCL classes 4, 5, 6, and 7; median composite on a shared clear mask;
- source labels and hashes accepted by the repository's strict provenance checks.

Those inputs are not temporally aligned with the 2024 counts. Existing GeoAI scores are agreement with a 2021 WorldCover pseudo-label product, not surveyed 2024 land-cover accuracy.

For 2024 context, use the existing real-ingestion path and `sentinel-2-l2a` collection to query acquisitions covering the chosen site and the actual survey date. A defensible protocol is a preregistered short window around the survey date, widened only under a documented cloud-availability rule, using the same projected 10 m grid and SCL mask. Record every item ID, acquisition time, scene-cloud metadata, clear-pixel support, processing version, and final hash; do not retain signed STAC asset URLs. Treat the 2024 reflectance/composite as remotely sensed context. Continue to label WorldCover 2021 as a historic categorical pseudo-label/reference, never as 2024 ground truth.

No 2024 Sentinel scene was downloaded during this feasibility phase.

## Scientific evidence classes

| Class | Items in scope | Permitted interpretation |
|---|---|---|
| Measured transport inputs | TfC 2024 directional class counts and sampled occupancy observations | Raw roadside observations at supplied intervals and links |
| Remotely sensed observations | Existing 2021 Sentinel-2 L2A; future verified 2024 Sentinel-2 L2A | Surface-reflectance context after documented masking/compositing |
| Historic pseudo-label | ESA WorldCover 2021 | Historic categorical reference; not field truth and not 2024 truth |
| Third-party map data | TfC's 2026 OSM-matched lines; Overpass/Geofabrik OSM | Map geometry/tags of stated snapshot and variable completeness |
| Inferred parameters | 250 m diagnostic clustering, edge matches, missing lanes/speeds, turn ratios, temporal factors, controller grouping | Must be labeled as inference with sensitivity/QA; never measured |
| Entirely simulated variables | SUMO routes, demand not directly observed, phase plans, vehicle interactions, emissions, optimizer outcomes | Simulator output only; not observed Cairo performance or environmental measurement |

The existing 198-request traffic benchmark remains an abstract synthetic SUMO experiment. It was not rerun, changed, or used to choose a site. This project must not be described as a calibrated Cairo digital twin.

## Offline validation command

The repository includes a read-only validator for locally acquired count and location files. It does not download data, substitute synthetic records, write research results, or invoke SUMO.

```powershell
New-Item -ItemType Directory -Force data\observed\tfc | Out-Null

Invoke-WebRequest `
  -Uri "https://data.transportforcairo.com/documents/190/download" `
  -OutFile "data\observed\tfc\2024_traffic_counts.zip"

Invoke-WebRequest `
  -Uri "https://data.transportforcairo.com/api/v2/assets/255/download" `
  -OutFile "data\observed\tfc\2024_traffic_survey_locations.zip"

urbantwin observed-data-audit `
  --counts "data\observed\tfc\2024_traffic_counts.zip" `
  --locations "data\observed\tfc\2024_traffic_survey_locations.zip"
```

The command emits a deterministic JSON summary with input hashes, schema and cardinality checks, missingness, dates, directions, duplicate intervals, geometry validity, and join anomalies. Warnings such as the five OSM-ID conflicts remain visible; the validator does not repair source data.

The full occupancy archive can be downloaded manually from [document 192](https://data.transportforcairo.com/catalogue/#/document/192) only when the exact recorded permission covers that research use. It is intentionally outside the current validator because the full table and its mixed-unit domain have not yet passed an exact audit.

`data/observed/` and ZIP archives are ignored by Git. Record hashes, retrieval UTC timestamps, source record IDs, the portal-license status and the author-reported permission basis in a separate provenance manifest. Do not commit the raw records unless the exact recorded terms permit redistribution and the project intentionally adopts that policy.

## Required publisher or field follow-up

Before downstream modeling, request or verify:

1. the exact scope, restrictions and required attribution of TfC's direct permission, including whether public source or derived-artifact redistribution is allowed;
2. the original 60-location identifier and survey-point geometry;
3. a direction-code legend for `0` and `1`, plus the observation timezone;
4. correction or explanation of the five `osm_id` conflicts and three geometry-only segments;
5. full occupancy quality checks, including missingness and percentage domains;
6. movement-level turning counts for a selected intersection;
7. controller inventory, signal phase sequence, cycle lengths, offsets, timing schedule, and survey-time changes;
8. lane configuration, approach geometry, queue observations, and any detector/source methodology needed for calibration and validation;
9. a road snapshot appropriate to each 2024 observation date, or a documented reason to use a later map.

If these cannot be obtained, retain the work as an observed link-flow/EO context study. A separate NGSIM Peachtree Street study may be considered for a movement/signal-method case, but it must use its own provenance, geography, labels, and conclusions and must never be represented as Cairo data.

## Recommended next implementation steps

1. Record TfC's direct permission privately and obtain any remaining clarification about public redistribution and the join/site-key issues.
2. Run the offline validator on locally downloaded official files and archive its hash-only summary outside authoritative research results.
3. Freeze the link-flow site rule or, if genuine controller data becomes available, preregister a separate intersection-site rule before model evaluation.
4. Acquire 2024 Sentinel-2 context for that site/date under the existing real-data provenance protocol.
5. Build and test a cached, provenance-rich road retrieval layer with a pinned Geofabrik fallback; validate a connected network before any SUMO import.
6. Define observed-versus-inferred calibration targets. Link counts alone may constrain link flows but cannot identify all turns or signal timings.
7. Only then design a small calibration/validation experiment with held-out intervals or dates. Keep all simulated outputs labeled simulator-only and keep the current abstract benchmark separate.

## Phase conclusion

Actual Cairo traffic-count records and their directional road geometries were accessed and validated. Anonymous download works, and the project author reports direct TfC permission for academic research and data reuse. The portal still provides no open license, and public redistribution depends on the exact private permission terms. A provisional link-flow/EO site can be selected reproducibly, subject to publisher confirmation of the physical grouping. A matching signal-controlled study intersection, genuine connected simulation-ready road network, turning-flow observations, and controller timing were **not verified**. Downstream traffic modeling stops at this gate.
