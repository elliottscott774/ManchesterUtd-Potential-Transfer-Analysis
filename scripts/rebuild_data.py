"""Execute the original notebook pipeline against an explicitly recorded raw snapshot."""
from pathlib import Path
import argparse
import hashlib
from importlib.metadata import version
import json
import os
import platform
import shutil
import time

import numpy as np
import pandas as pd

from capture_preprocessing import capture, fixtures
from valuation_equivalent import make_fast_lookup, validate_equivalence


RAW_FILES = (
    'appearances.csv', 'clubs.csv', 'games.csv', 'players.csv',
    'player_valuations.csv', 'competitions.csv',
)
PIPELINE_CELLS = (1, 13, 17, 18, 21, 22, 24, 26, 27, 29, 32, 34, 35,
                  37, 39, 43, 44, 48, 51, 52, 53, 54, 55)
SHUFFLE_SEED = 20261024


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stage_raw_files(raw_directory, data_directory):
    raw_files = []
    for filename in RAW_FILES:
        source_path = raw_directory / filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_hash = file_hash(source_path)
        target_path = data_directory / filename
        if target_path.exists():
            if file_hash(target_path) != source_hash:
                raise ValueError(f'Staged raw data differs from source: {target_path}')
        else:
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copyfile(source_path, target_path)
        raw_files.append({'file': filename, 'bytes': source_path.stat().st_size,
                          'sha256': source_hash})
    return raw_files


def audit_raw_data(data_directory):
    games = pd.read_csv(data_directory / 'games.csv')
    appearances = pd.read_csv(data_directory / 'appearances.csv')
    valuations = pd.read_csv(data_directory / 'player_valuations.csv')
    clubs = pd.read_csv(data_directory / 'clubs.csv')
    competitions = pd.read_csv(data_directory / 'competitions.csv')
    score_columns = ['home_club_goals', 'away_club_goals']
    if games[score_columns].isna().any().any() or games[score_columns].lt(0).any().any():
        raise ValueError('Unplayed or missing game scores require review before aggregation.')
    national_competitions = set(competitions.loc[
        competitions['type'] == 'national_team_competition', 'competition_id'])
    national_games = games[games['competition_id'].isin(national_competitions)]
    national_team_ids = set(national_games['home_club_id']) | set(national_games['away_club_id'])
    club_ids = set(clubs['club_id'])
    national_club_overlap = sorted(national_team_ids & club_ids)
    if national_club_overlap:
        raise ValueError(f'National-team and club IDs overlap: {national_club_overlap}')
    return {
        'games': len(games), 'appearances': len(appearances),
        'valuations': len(valuations), 'clubs': len(clubs),
        'minimum_game_season': int(games['season'].min()),
        'maximum_game_season': int(games['season'].max()),
        'latest_game_date': str(games['date'].max()),
        'latest_appearance_date': str(appearances['date'].max()),
        'latest_valuation_date': str(valuations['date'].max()),
        'duplicate_valuation_keys': int(valuations.duplicated(['player_id', 'date']).sum()),
        'national_team_games': len(national_games),
        'national_team_appearances': int(appearances['game_id'].isin(national_games['game_id']).sum()),
        'national_team_club_id_overlap': national_club_overlap,
        'appearances_without_matching_club': int((~appearances['player_club_id'].isin(club_ids)).sum()),
        'national_team_handling': 'Original left club merge and groupby exclude unmatched national-team rows; output club IDs are checked.',
    }


def run_pipeline(arguments):
    raw_directory = arguments.raw_directory.resolve()
    output_directory = arguments.output_directory.resolve()
    notebook_path = arguments.notebook.resolve()
    provenance_path = arguments.source_provenance.resolve()
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    for field in ('dataset', 'source_url', 'version', 'zip_sha256'):
        if not provenance.get(field):
            raise ValueError(f'Source provenance must contain {field}')
    if not (provenance.get('published') or provenance.get('uploaded')):
        raise ValueError('Source provenance must contain published or uploaded timestamp')
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    notebook_hash = file_hash(notebook_path)
    work_directory = output_directory / 'work'
    data_directory = work_directory / 'data'
    data_directory.mkdir(parents=True, exist_ok=True)
    raw_files = stage_raw_files(raw_directory, data_directory)
    raw_audit = audit_raw_data(data_directory)
    (output_directory / 'raw-data-audit.json').write_text(json.dumps(raw_audit, indent=2), encoding='utf-8')
    print('Raw data audit:', json.dumps(raw_audit), flush=True)
    np.random.seed(SHUFFLE_SEED)
    namespace = {'__name__': '__rebuild__', 'np': np, 'pd': pd}
    timings = []
    original_directory = Path.cwd()
    try:
        os.chdir(work_directory)
        for cell_index in PIPELINE_CELLS:
            code = ''.join(notebook['cells'][cell_index]['source'])
            if cell_index == 1:
                unused_line = "df_game_lineups = pd.read_csv('./data/game_lineups.csv')"
                assert unused_line in code
                code = code.replace(unused_line, '# Unused transfer-history branch is not run.')
            if cell_index == 13:
                unused_line = "transfer_history = pd.read_csv('./data/player_transfer_history.csv')"
                assert unused_line in code
                code = code.replace(unused_line, '# Unused transfer-history table is not loaded.')
            if cell_index == 51:
                marker = "final_squad_df = final_squad_df.drop(columns='league')"
                assert marker in code
                capture_mapping = "_global_league_mapping = {str(value): int(index) for index, value in enumerate(final_squad_df['league'].astype('category').cat.categories)}\n"
                code = code.replace(marker, capture_mapping + marker)
            if cell_index == 52:
                captured_raw = namespace['final_squad_df'].copy(deep=True)
            print(f'Executing original notebook cell {cell_index}', flush=True)
            started_at = time.monotonic()
            exec(compile(code, f'{notebook_path.name}:cell{cell_index}', 'exec'), namespace)
            if cell_index == 21:
                original_lookup = namespace['combine_player_stats_and_valuations']
                fast_lookup = make_fast_lookup(original_lookup)
                validation = validate_equivalence(namespace['player_history'], namespace['df_player_valuations'], original_lookup, fast_lookup)
                (output_directory / 'valuation-equivalence.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
                namespace['combine_player_stats_and_valuations'] = fast_lookup
                print('Valuation equivalence:', json.dumps(validation), flush=True)
            if cell_index == 52:
                captured_scaled = namespace['final_squad_df'].copy(deep=True)
                capture(captured_raw, captured_scaled, namespace['_global_league_mapping'],
                        output_directory, notebook_hash, provenance)
                print('Captured exact original preprocessing for 40 features.', flush=True)
            elapsed_seconds = round(time.monotonic() - started_at, 3)
            timings.append({'cell': cell_index, 'seconds': elapsed_seconds})
            print(f'Finished cell {cell_index} in {elapsed_seconds}s', flush=True)
    finally:
        os.chdir(original_directory)
    history = pd.read_csv(data_directory / 'squad_history.csv')
    training = pd.read_csv(data_directory / 'squad_and_performance.csv')
    club_ids = set(pd.read_csv(data_directory / 'clubs.csv')['club_id'])
    if not set(history['club_id']).issubset(club_ids):
        raise ValueError('Output contains a squad without a source club.')
    squad_sizes = history.groupby('squad_id').size()
    if not squad_sizes.eq(19).all():
        raise ValueError('Output squad size differs from the original 19-player cohort.')
    if training.isna().any().any():
        raise ValueError('Training output contains missing values.')
    fixture_report = fixtures(captured_raw, captured_scaled, namespace['_global_league_mapping'],
                              history, training, output_directory)
    outputs = []
    for filename in ('squad_history.csv', 'squad_and_performance.csv'):
        path = data_directory / filename
        frame = pd.read_csv(path)
        outputs.append({'file': filename, 'rows': len(frame), 'columns': list(frame.columns),
                        'sha256': file_hash(path), 'bytes': path.stat().st_size})
    report = {
        'runtime': {'python': platform.python_version(), 'pandas': version('pandas'),
                    'numpy': version('numpy'), 'scikit-learn': version('scikit-learn')},
        'source_notebook_sha256': notebook_hash,
        'source_provenance': provenance,
        'raw_snapshot_version': provenance['version'],
        'raw_snapshot_published': provenance.get('published', provenance.get('uploaded')),
        'raw_files': raw_files, 'raw_audit': raw_audit,
        'shuffle_seed': SHUFFLE_SEED,
        'notes': [
            'Original output-producing notebook cells executed, with unused game-lineup and transfer-history reads omitted.',
            'Original 5 Attack / 6 Midfield / 7 Defender / 1 Goalkeeper criteria, feature transform, targets and output schemas retained.',
            'Original unseeded shuffle is seeded for reproducibility; exact-minute tie behavior is documented in fixture ordering audit.',
            'Valuation backward-asof optimization validated against original lookup; duplicate player/date keys use original fallback.',
        ],
        'timings': timings, 'outputs': outputs,
        'output_validation': {'history_squads': int(history['squad_id'].nunique()),
                              'clubs': int(history['club_id'].nunique()),
                              'minimum_season': int(history['season'].min()),
                              'maximum_season': int(history['season'].max()),
                              'all_squads_have_19_players': True,
                              'all_output_club_ids_in_raw_clubs': True,
                              'training_has_no_missing_values': True,
                              'fixture_cases': len(fixture_report['cases'])},
    }
    (output_directory / 'rebuild-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--notebook', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'process_data.ipynb')
    parser.add_argument('--source-provenance', type=Path, required=True)
    run_pipeline(parser.parse_args())


if __name__ == '__main__':
    main()
