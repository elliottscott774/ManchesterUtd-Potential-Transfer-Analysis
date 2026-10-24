"""Regression for the real training/inference feature-space mismatch.

Uses historical rebuilt squads and their corresponding normalized training rows.
Predictions execute on the actual fitted DecisionTreeRegressor; the call spy only
records the matrix delivered to it.
"""
from pathlib import Path
import json
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from utils.model import train_model, predict
from utils.recommendation import process_squad

ROOT = Path(__file__).resolve().parents[1]


class FeaturePreprocessingRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((ROOT / 'data/feature-preprocessing.json').read_text(encoding='utf-8'))
        fixture_file = ROOT / 'tests/feature-preprocessing-fixtures.json'
        cls.fixtures = json.loads(fixture_file.read_text(encoding='utf-8'))['cases']
        cls.history = pd.read_csv(ROOT / 'data/squad_history.csv.gz')
        cls.training = pd.read_csv(ROOT / 'data/squad_and_performance.csv.gz')
        cls.model = train_model(['wins'], cls.training)
        assert len({case['league'] for case in cls.fixtures}) >= 3
        assert len({case['season'] for case in cls.fixtures}) >= 2
        assert len(cls.metadata['features']) == 40

    def squad(self, case):
        squad = self.history[(self.history.club_code == case['club_code']) &
                             (self.history.season == case['season'])].copy()
        self.assertEqual(len(squad), 19)
        return squad

    def test_predict_matches_corresponding_training_feature_space(self):
        features = self.metadata['features']
        for case in self.fixtures:
            with self.subTest(club=case['club_code'], season=case['season'], league=case['league']):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)
                    processed = process_squad(self.squad(case))
                with patch.object(self.model, 'predict', wraps=self.model.predict) as actual_predict:
                    result = predict(self.model, processed)
                self.assertTrue(np.isfinite(result).all())
                actual_features = actual_predict.call_args.args[0]
                self.assertEqual(list(actual_features.columns), features)
                expected = np.array([case['expected_features'][name] for name in features], dtype=float)
                np.testing.assert_allclose(actual_features.iloc[0].to_numpy(dtype=float), expected,
                                           rtol=1e-12, atol=1e-12,
                                           err_msg='Inference must match the original normalized training row')

    def test_process_squad_uses_training_global_league_codes(self):
        mapping = self.metadata['league_mapping']
        for case in self.fixtures:
            with self.subTest(club=case['club_code'], season=case['season'], league=case['league']):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)
                    processed = process_squad(self.squad(case))
                self.assertEqual(int(processed.league_code.iloc[0]), mapping[case['league']])

    def test_fixture_outputs_match_saved_training_csv(self):
        features = self.metadata['features']
        for case in self.fixtures:
            with self.subTest(club=case['club_code'], season=case['season']):
                saved = self.training[self.training['Unnamed: 0'] == case['training_csv_index']]
                self.assertEqual(len(saved), 1)
                expected = np.array([case['expected_features'][name] for name in features], dtype=float)
                np.testing.assert_allclose(saved.iloc[0][features].to_numpy(dtype=float), expected,
                                           rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
