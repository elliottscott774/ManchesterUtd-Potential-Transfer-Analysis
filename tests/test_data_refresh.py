"""Exercise a real Streamlit rerun after only tracked data files change.

The same application source first reads an authentic pre-2024 subset, then the
current published dataset. Cached old rows must not survive a data deployment.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import unittest

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DataRefreshRegression(unittest.TestCase):
    def test_data_only_refresh_updates_available_seasons_without_source_change(self):
        current_history = pd.read_csv(ROOT / 'data/squad_history.csv.gz')
        old_history = current_history[current_history.season <= 2023]
        latest_season = int(current_history.loc[current_history.club_code == 'manchester-united', 'season'].max())
        self.assertGreater(latest_season, 2023)
        current_metadata = json.loads((ROOT / 'data/dataset-metadata.json').read_text(encoding='utf-8'))
        original_source = (ROOT / 'app.py').read_text(encoding='utf-8')
        st.cache_data.clear()
        previous_temporary_directory = tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory(prefix='data-refresh-', dir=ROOT / 'tests') as temporary_directory:
                temporary_root = Path(temporary_directory)
                # AppTest creates a module-level temporary directory on import.
                # Keep that under the same workspace-owned cleanup scope.
                tempfile.tempdir = str(temporary_root)
                from streamlit.testing.v1 import AppTest
                data_directory = temporary_root / 'data'
                data_directory.mkdir()
                source = original_source.replace(
                    "DATA_DIR = Path(__file__).resolve().parent / 'data'",
                    f'DATA_DIR = Path({str(data_directory)!r})',
                )
                self.assertNotEqual(source, original_source, 'The temporary app must use isolated test data.')
                application_path = temporary_root / 'app.py'
                application_path.write_text(source, encoding='utf-8')
                source_sha256 = hashlib.sha256(application_path.read_bytes()).hexdigest()

                history_path = data_directory / 'squad_history.csv.gz'
                training_path = data_directory / 'squad_and_performance.csv.gz'
                old_history.to_csv(history_path, index=False, compression='gzip')
                shutil.copyfile(ROOT / 'data/squad_and_performance.csv.gz', training_path)
                old_metadata = json.loads(json.dumps(current_metadata))
                old_metadata['seasons']['max'] = 2023
                old_metadata['games_through'] = '2024-04-19'
                old_metadata['refreshed_on'] = '2026-10-24'
                old_metadata['files']['squad_history.csv.gz']['sha256'] = hashlib.sha256(history_path.read_bytes()).hexdigest()
                old_metadata['files']['squad_history.csv.gz']['bytes'] = history_path.stat().st_size
                (data_directory / 'dataset-metadata.json').write_text(json.dumps(old_metadata), encoding='utf-8')

                application = AppTest.from_file(str(application_path)).run(timeout=60)
                self.assertEqual(len(application.exception), 0)
                self.assertEqual(max(int(value) for value in application.select_slider[0].options), 2023)

                # Simulate a tracked-data deployment while the source and cache
                # stay in place, exactly as a warm Streamlit process can see it.
                shutil.copyfile(ROOT / 'data/squad_history.csv.gz', history_path)
                (data_directory / 'dataset-metadata.json').write_text(json.dumps(current_metadata), encoding='utf-8')
                self.assertNotEqual(old_metadata['files']['squad_history.csv.gz']['sha256'],
                                    current_metadata['files']['squad_history.csv.gz']['sha256'])
                self.assertEqual(hashlib.sha256(application_path.read_bytes()).hexdigest(), source_sha256)
                application.run(timeout=60)

                self.assertEqual(len(application.exception), 0)
                self.assertTrue(any(current_metadata['games_through'] in item.value for item in application.caption),
                                'Fresh metadata must be visible on the rerun.')
                self.assertEqual(max(int(value) for value in application.select_slider[0].options), latest_season,
                                 'A data-only deployment must invalidate cached squad rows.')
                self.assertEqual(application.select_slider[0].value, latest_season)
        finally:
            tempfile.tempdir = previous_temporary_directory
            st.cache_data.clear()


if __name__ == '__main__':
    unittest.main()
