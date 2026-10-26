"""Use the feature preprocessing fitted by the original data pipeline."""
import json
from pathlib import Path

import pandas as pd


def training_preprocessing():
    path = Path(__file__).resolve().parents[1] / 'data' / 'feature-preprocessing.json'
    return json.loads(path.read_text(encoding='utf-8'))


def encode_leagues(leagues):
    return leagues.map(training_preprocessing()['league_mapping'])


def transform_features(squad):
    preprocessing = training_preprocessing()
    features = preprocessing['features']
    scale = pd.Series(preprocessing['scale'], index=features)
    offset = pd.Series(preprocessing['offset'], index=features)
    return squad.loc[:, features].astype(float).mul(scale).add(offset)
