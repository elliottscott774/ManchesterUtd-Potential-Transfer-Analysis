# Transfer Recommendation System

[Open the app](https://transfer-recommendation-system.streamlit.app/)

Choose a club, season, player age range, and the team performance metrics to improve. The app tests player substitutions with the original DecisionTreeRegressor and returns five players per metric, plus an overall list, linked to Transfermarkt.

The bundled data was refreshed on October 25, 2026 from [Transfermarkt dataset version 679](https://www.kaggle.com/datasets/davidcariboo/player-scores/versions/679). It contains 474 clubs and 3,147 eligible squads across season starting years 2012–2025. The upstream dataset is paused: games end July 6, 2026; appearances June 28; valuations June 12. It does not cover 2026–27 squads.

## Run

```sh
pip install -r requirements.txt
streamlit run app.py
python -m unittest discover -s tests -v
```

Data is stored in `data/`; no Backblaze account, credentials, or API keys are required. The season selector uses the seasons actually available for the chosen club.

## Rebuild the data

Download and extract the six input CSVs listed in `scripts/rebuild_data.py` from the pinned source snapshot. Run from the repository root, using a separate output directory:

```sh
python scripts/rebuild_data.py --raw-directory path/to/raw-v679 --output-directory path/to/rebuild --source-provenance data/source-provenance.json
```

The script executes the original `process_data.ipynb` output-producing cells and records input hashes, output hashes, fitted feature preprocessing, and authentic regression fixtures. Rebuild both datasets and preprocessing together. App CSVs use lossless gzip compression; see [data provenance](data/PROVENANCE.md) and [the run report](data/rebuild-report.json) for this refresh.

## Model and limits

Each squad contains 5 attackers, 6 midfielders, 7 defenders, and 1 goalkeeper. The original 40 features, targets, substitution rules, ranking, and chronological 80% training split are retained. A tree is fitted when recommendations are requested. Inference applies the saved MinMax transformation and global league mapping from the same pipeline run.

Tests compare 18 authentic squads across six leagues and seasons 2012, 2018, and 2025 with their normalized training rows. This checks preprocessing and integration, not predictive quality. The original equal-minute ordering behavior is documented in the fixtures. Recommendations are exploratory shortlists for further scouting, not a substitute for evaluating players as people and athletes.
