import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st
import altair as alt
from dotenv import load_dotenv

from utils.b2 import B2

# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'players.csv'

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
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df_players = b2.get_df(REMOTE_DATA)

    # average sentiment scores for the whole dataset
    benchmarks = df_players[['market_value_in_eur']] \
                    .agg(['mean', 'median'])
    
    return df_players, benchmarks


# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
# ------------------------------
# PART 0 : Overview
# ------------------------------
st.write(
'''
# Exercise 1
Articulate a quantitative question of the data you plan to use in your final project for this class. 
Build a visualization which addresses this question. For larger datasets, you can feel free to use a subset of the data.

### QUESTION:\n
**What are the current values of Manchester United players?**

# Exercise 2
Build and deploy a very simple Streamlit web app with the following:

Your visualization from Exercise 1
A view of a subset of your data using st.dataframe.
Try to store (and access) your data using Backblaze.
''')

df_players, benchmarks = get_data()

# ------------------------------
# PART 1 : Filter Data
# ------------------------------

st.write(
'''
Who are the current Manchester United Players?:
''')

manchester_united_players = df_players[df_players['current_club_name'] == 'Manchester United Football Club']
current_manchester_united_players = manchester_united_players[manchester_united_players['last_season'] == 2023]
current_manchester_united_players = pd.DataFrame(current_manchester_united_players)
current_manchester_united_players = current_manchester_united_players.sort_values(by="market_value_in_eur", ascending = False)

st.dataframe(current_manchester_united_players)



# ------------------------------
# PART 2 : Plot
# ------------------------------

st.write(
'''
## Visualize
What are the market values of these players?
'''
)

st.write(alt.Chart(current_manchester_united_players).mark_bar().encode(
    x = alt.X('name', sort = None),
    y = 'market_value_in_eur',
))