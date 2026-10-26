"""Data-only refreshes must also replace fitted feature transformation values."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from utils.preprocessing import training_preprocessing, transform_features

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingRefreshRegression(unittest.TestCase):
    def test_inference_uses_replaced_preprocessing_file(self):
        metadata = json.loads((ROOT / 'data/feature-preprocessing.json').read_text(encoding='utf-8'))
        cases = json.loads((ROOT / 'tests/feature-preprocessing-fixtures.json').read_text(encoding='utf-8'))['cases']
        case = max(cases, key=lambda value: value['season'])
        squad = pd.DataFrame([case['raw_features']])
        clear_cache = getattr(training_preprocessing, 'cache_clear', lambda: None)
        clear_cache()
        try:
            with tempfile.TemporaryDirectory(prefix='preprocessing-refresh-', dir=ROOT / 'tests') as temporary_directory:
                temporary_root = Path(temporary_directory)
                (temporary_root / 'data').mkdir()
                metadata_path = temporary_root / 'data/feature-preprocessing.json'
                metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
                with patch('utils.preprocessing.__file__', str(temporary_root / 'utils/preprocessing.py')):
                    before = transform_features(squad)
                    # A replacement scaler must take effect in a warm process.
                    season_index = metadata['features'].index('season')
                    metadata['offset'][season_index] += 1
                    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
                    after = transform_features(squad)
                    self.assertAlmostEqual(after['season'].iloc[0], before['season'].iloc[0] + 1,
                                           msg='Inference must reload replaced fitted preprocessing.')
        finally:
            clear_cache()


if __name__ == '__main__':
    unittest.main()
