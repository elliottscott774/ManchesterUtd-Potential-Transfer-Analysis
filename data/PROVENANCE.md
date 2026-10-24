# Transfer application data

Source: [Kaggle football-datasets snapshot 405](https://www.kaggle.com/datasets/davidcariboo/player-scores/versions/405), April 19, 2024.

Rebuilt on October 24, 2026 through the original processing notebook, retaining its 5 attackers / 6 midfielders / 7 defenders / 1 goalkeeper cohort and 40 model features. Seasons 2012–2023. The original DecisionTree type, training targets and cohort are retained; inference preprocessing is corrected as described below.

These rebuilt inputs are not proven byte-identical to the lost Backblaze files. Gzip compression is lossless. The valuation lookup uses an equivalent faster join; checks against the original lookup cover actual rows, cutoffs, missing values and duplicate-key fallback. The original fitted feature transformation is now consistently applied during inference; executing recommendations is not a validation of predictive quality.

- `squad_history.csv.gz`: 1,358,425 bytes; SHA-256 `af2743b1b35d5399ee9c10ded1b506faa6259936f95eb840499a371e3f895782`.
- `squad_and_performance.csv.gz`: 658,431 bytes; SHA-256 `4eb51ca40ad6af0b714ab8e274f34b386f3c1ec8d8e8b47571335137ec591b1a`.

### Runtime feature preprocessing

`feature-preprocessing.json` stores the exact MinMax scale/offset and global league categories fitted by the original notebook on its 2,641-row pre-scaling squad frame. Inference applies that same transformation before the original DecisionTree predicts; this corrects the original training/inference scale mismatch. Values are not clipped. The model type, target variables and training cohort remain unchanged. The source code is therefore not claimed byte-identical to the original app.
