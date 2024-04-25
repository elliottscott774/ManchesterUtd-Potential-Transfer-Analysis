import os
import pandas as pd
import numpy as np

import streamlit as st
from dotenv import load_dotenv

from utils.b2 import B2
from utils.recommendation import get_recommendation


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
# PART 0 : Getting Data
# ------------------------------
squad_and_performance, squad_history = get_data()


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


# ------------------------------
# Get Recommendations
# ------------------------------

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

        

