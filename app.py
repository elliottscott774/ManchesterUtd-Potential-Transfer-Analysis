import os
import pickle
import pandas as pd
import sklearn
import numpy as np

import streamlit as st
import altair as alt
from dotenv import load_dotenv

from utils.b2 import B2

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
    transfer_history = b2.get_df('player_transfer_history.csv')
    player_history = b2.get_df('player_history.csv')
    squad_and_performance = b2.get_df('squad_and_performance.csv')
    squad_history = b2.get_df('squad_history.csv')
    return transfer_history, player_history, squad_and_performance, squad_history


# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
# ------------------------------
# PART 0 : Overview
# ------------------------------
transfer_history, player_history, squad_and_performance, squad_history = get_data()


st.write(
'''
# Transfer Recommendation System
\n\n
Fans of football debate endlessly on which player(s) their favorite football club should bring to their team. The goal of this system is to 
recommend potential players based on how they are likely to improve the club.
\n\n
Through careful data processing I was able to create a player history data set, which tracks the performance of each player over the years, and a transfer history data set, 
which tracks the transfers of each player over the years.
\n\n
## Player History
''')

st.dataframe(player_history)
st.write('\n\n# Transfer History ')
st.dataframe(transfer_history)

st.write(
'''
\n\n 
A model is trained to predict the success of a squad. 
Every squad from 2013-2022 is compiled and their performances 
are aggregated to give a performance score. The model is then 
trained to predict the performance score of the 
squad. This trained model can then be used to 
estimate the success of any squad of players. 
\n\n
A transfer recommender system is built using this model. The 
user can input the name of a team and the system will return 
the top 10 players with the highest score from the model. 
Multiple models can be trained based on desired performance
metrics. For example a model could be trained to recommend 
players that are most likely to improve the team's overall performance, 
or a model could be trained to recommend players that are 
most likely to improve the team's attacking performance. 
Multiple models could be built and polled to give a more 
nuanced recommendation.
\n\n
## Issues
I am behind in the model building process. A lot of effort 
was been spent processing the data to be in a format that 
can be used to train the model. I have a way of constructing
all team squads and constructing player transfer history.
\n\n
I haven't figured out a good way to build a model based on 
squads however. A squad of players can vary in the 
number of players and the model needs to be able to handle 
that. I could simplify the data by using only the top 5 players
in each position based on appearances. This would make every
squad the same size, but would be less realistic.
\n\n
I also need to figure out a way of scoring the performance of a
team based on multiple statistics. How do I know which team is best?

## Next Steps
''')
st.image('Transfer_Recommender.png')

st.write(
'''
The above images are my plans for how I will train the squad
success prediction model and construct the recommendation system.
\n\n
For each season I will aggregate the players for each team.
\n\n
Every team will have performance data related to them that I will use to 
calculate the success of squad. The performance data is data such as 
games won, games drawn, game lost, goals for, goals against, number of assists,
clean sheets, and final league position. The squad's success value becomes
the target value of the prediction model of which the input of is a squad of players
\n\n
The recommendation system will be built using the trained model. The user of the app
selects a team they want player transfer recommendations for. The system constructs this
squad of players and iteratively adds a player to the squad. This new squad becomes the 
input into the model. The model will output a score and at the end the top 10 players with
 the highest score will be returned to the user.
   




'''
)

# ------------------------------
# PART 1 : Processing User Input
# ------------------------------

# parse out the list of teams in top_players_per_squad
team_list = squad_history['club_code'].unique().tolist()


# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Choose a team:',
    (team_list)
)






# ------------------------------
# PART 2 : Plot
# ------------------------------