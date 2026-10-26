# Transfer application data

Refreshed October 25, 2026 from [Kaggle Transfermarkt snapshot 679](https://www.kaggle.com/datasets/davidcariboo/player-scores/versions/679), published September 5, 2026. [Upstream project](https://github.com/dcaribou/transfermarkt-datasets). License: CC0.

The upstream collection is paused. Games end July 6, 2026; player appearances June 28, 2026; valuations June 12, 2026. No 2026–27 squads are covered. Season labels are starting years: 2025 denotes the 2025–26 season where the league uses that calendar.

The original processing notebook produced 59,793 player-squad rows, 3,147 eligible 19-player squads, and 3,116 training rows across 474 clubs and seasons 2012–2025. Its 5 Attack / 6 Midfield / 7 Defender / 1 Goalkeeper cohort, 40 model features, and target calculations are retained. National-team IDs do not overlap club IDs, and unmatched national-team appearances are excluded by the original club merge/grouping. All output squads belong to real source clubs.

`feature-preprocessing.json` stores the MinMax scale/offset and 24 global league categories fitted on the same 3,147-row pre-join squad frame. The captured transform exactly matches the original notebook's normalized features. Both CSVs, transformation, and fixtures were regenerated together. Inference does not clip values.

The valuation optimization matches the original lookup on real rows, cutoffs, missing values, and duplicate-key fallback. The original unseeded shuffle is pinned to 20261024 for reproducibility. Equal-minute ordering differs between the training and inference ordering for 37 of 3,116 training squads; this existing behavior is preserved and recorded in the fixtures. No predictive-quality claim is made.

The compressed CSVs are lossless:

- `squad_history.csv.gz`: 1,619,837 bytes; SHA-256 `ed989ac226cc6ddfef8ac9cecbe662c37a9782de31bc874995980f4da6ea10b6`.
- `squad_and_performance.csv.gz`: 784,478 bytes; SHA-256 `cfe1b1486e739ff40384bceeb82989bc555db0c19196d8b6e31ae9c2f990c6d3`.

Source ZIP SHA-256: `5971ec002b9d7b3e99358ed28b0518e7f384549db53075efc273cc57f7dacdf1`. Input hashes, timings, notebook hash, and output checks are in `rebuild-report.json`. Machine-readable source and coverage metadata are in `source-provenance.json` and `dataset-metadata.json`.

The previous April 2024 snapshot remains in Git history. The prior restoration did not establish byte identity with the lost Backblaze files; this refresh rebuilds from a newer authentic upstream snapshot.
