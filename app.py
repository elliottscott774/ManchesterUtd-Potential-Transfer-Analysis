from pathlib import Path
import json
import pandas as pd
import numpy as np

import streamlit as st

from utils.recommendation import get_recommendation


# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / 'data'


# ------------------------------------------------------
#                        CACHING
# ------------------------------------------------------
@st.cache_data
def get_data(data_file_hashes):
    """Invalidate cached frames when tracked dataset contents change."""
    squad_and_performance = pd.read_csv(DATA_DIR / 'squad_and_performance.csv.gz')
    squad_history = pd.read_csv(DATA_DIR / 'squad_history.csv.gz')
    return squad_and_performance, squad_history


# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
# ------------------------------
# PART 0 : Getting Data
# ------------------------------
refresh_metadata = json.loads((DATA_DIR / 'dataset-metadata.json').read_text(encoding='utf-8'))
data_file_hashes = {filename: details['sha256'] for filename, details in refresh_metadata['files'].items()}
squad_and_performance, squad_history = get_data(data_file_hashes)


# ------------------------------
# Processing User Input
# ------------------------------

st.write(
'''
# Transfer Recommendation System
\n\n
Want to know which players your favorite football team should sign? 
\n\n
To begin, choose the following properties to receive recommendations:
\n\n
1. A team\n\n
2. One or many metrics to improve\n\n
3. A season\n\n
4. Age range of recommendations\n\n

Each metric will return a list of 5 players. The overall category are 5 players who have the best combined rank in all metrics.\n\n
Each player will have a link to their Transfermarkt webpage.
'''
)


# parse out the list of teams in top_players_per_squad
team_list = squad_history['club_code'].unique().tolist()
default_team_index = team_list.index('manchester-united') if 'manchester-united' in team_list else 0
st.caption(f"Data through {refresh_metadata['games_through']} · Refreshed {refresh_metadata['refreshed_on']}")


LL, ML, MR, RR = st.columns(4)

with LL:
    selected_team = st.selectbox(label = 'Choose a team:',
    options =(team_list),
    index = default_team_index
    )

with ML:
    selected_target = st.multiselect(label = 'What do you want to improve?',
                                     options = ['wins', 'draws', 'loss', 'goals_scored',
       'goals_conceded', 'clean_sheets', 'points', 'total_games', 'win_rate',
       'loss_rate', 'goal_difference', 'avg_goals_scored_per_game',
       'avg_goals_conceded_per_game', 'clean_sheet_rate',
       'goals_scored_per_win', 'goals_conceded_per_loss']
                                     )


with MR:
    available_seasons = sorted(squad_history.loc[squad_history['club_code'] == selected_team, 'season'].unique().astype(int).tolist())
    selected_season = st.select_slider(label = 'Choose a season: ', options = available_seasons, value = available_seasons[-1])


with RR:
    selected_age_range = st.slider(label ="Age range of Recommendations:",
    value=(16, 35),
    max_value = 40,
    min_value = 16)


# ------------------------------
# Get Recommendations
# ------------------------------

if selected_target:
    try:
        top_players = get_recommendation(squad_and_performance, squad_history, selected_team, selected_target, selected_season, selected_age_range)
        for category, player_ids in top_players.items():
            # Filter the DataFrame to get rows matching the player IDs
            filtered_players = squad_history[squad_history['player_id'].isin(player_ids)].drop_duplicates(subset=['player_id'])
        
            st.write(f"{category.capitalize()}:")
            for _, row in filtered_players.iterrows():
                player_name = row['player_name'].replace(" ", "-")
                player_id = row['player_id']  # Assuming this is the correct Transfermarkt ID
                transfermarkt_url = f"https://www.transfermarkt.com/{player_name}/profil/spieler/{player_id}"
                link_label = f"View {row['player_name']}'s profile on Transfermarkt"
            
                # Display the link with a label
                try:
                    st.page_link(transfermarkt_url, label=link_label, icon="🌎")
                except AttributeError:
                    # If st.page_link is not available, use markdown to create a hyperlink
                    st.markdown(f"[{link_label}]({transfermarkt_url})", unsafe_allow_html=True)
    except Exception as e:
        st.write("Sorry, that team does not meet the model criteria")
        st.write(str(e))  # It's helpful to print the exception for debugging purposes.

        

