# Transfer application data

Source: [Kaggle football-datasets snapshot 405](https://www.kaggle.com/datasets/davidcariboo/player-scores/versions/405), April 19, 2024.

Rebuilt on October 24, 2026 through the original processing notebook, retaining its 5 attackers / 6 midfielders / 7 defenders / 1 goalkeeper cohort and 40 model features. Seasons 2012–2023. Original model and inference code are unchanged.

These rebuilt inputs are not proven byte-identical to the lost Backblaze files. Gzip compression is lossless. The valuation lookup uses an equivalent faster join; checks against the original lookup cover actual rows, cutoffs, missing values and duplicate-key fallback. The original model scaling behavior is preserved; executing recommendations is not a validation of predictive quality.

- `squad_history.csv.gz`: 1,358,425 bytes; SHA-256 `af2743b1b35d5399ee9c10ded1b506faa6259936f95eb840499a371e3f895782`.
- `squad_and_performance.csv.gz`: 658,431 bytes; SHA-256 `4eb51ca40ad6af0b714ab8e274f34b386f3c1ec8d8e8b47571335137ec591b1a`.
