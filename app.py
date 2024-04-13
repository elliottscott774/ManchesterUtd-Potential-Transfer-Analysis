import os
import pickle
import pandas as pd
import sklearn
import numpy as np

import streamlit as st
import altair as alt
from dotenv import load_dotenv

from utils.b2 import B2
from utils.recommendation import get_recommendation

# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------


# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
load_dotenv()

# load Backblaze connection
# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])


# ------------------------------------------------------
#                        CACHING
# ------------------------------------------------------
@st.cache_data
def get_data():
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    squad_and_performance = b2.get_df('squad_and_performance.csv')
    squad_history = b2.get_df('squad_history.csv')
    return squad_and_performance, squad_history


# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
# ------------------------------
# PART 0 : Overview
# ------------------------------
squad_and_performance, squad_history = get_data()


st.write(
'''
# Transfer Recommendation System
\n\n
Football fans love discussing which players their favorite clubs should sign.
 The purpose of this system is to suggest players who are likely to enhance the team's performance.
\n\n
'''
)
# ------------------------------
# PART 1 : Processing User Input
# ------------------------------

# parse out the list of teams in top_players_per_squad
team_list = squad_history['club_code'].unique().tolist()


LL, ML, MR, RR = st.columns(4)

with LL:
    selected_team = st.selectbox(label = 'Choose a team:',
    options =(team_list),
    index = 179
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
    selected_season = st.slider(label = 'Choose a season: ', min_value = 2012, max_value = 2023, value = 2023, step = 1)


with RR:
    selected_age_range = st.slider(label ="Age range of Recommendations:",
    value=(16, 35),
    max_value = 40,
    min_value = 16)


top_players = get_recommendation(squad_and_performance, squad_history, selected_team, selected_target, selected_season, selected_age_range)

# Convert player_ids in top_players to player_names using squad_history, ensuring each player is only counted once
for category, player_ids in top_players.items():
    unique_player_names = squad_history.drop_duplicates(subset=['player_id']).loc[squad_history['player_id'].isin(player_ids)]['player_name'].tolist()
    top_players[category] = unique_player_names

# Making the output more reader-friendly
st.write("Top Players in Each Category:")
for category, player_names in top_players.items():
    st.write(f"{category.capitalize()}: {', '.join(map(str, player_names))}")

# ------------------------------
# PART 2 : Plot
# ------------------------------