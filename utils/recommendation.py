import pandas as pd
import numpy as np
from utils.model import train_model, predict
import streamlit as st

def get_recommendation(squad_and_performance, squad_history, selected_team, targets, season, age_range):
    
    selected_squad = squad_history[(squad_history['club_code'] == selected_team) & (squad_history['season'] == season)]
    available_players = squad_history[(squad_history['club_code'] != selected_team) & (squad_history['season'] == season) & (squad_history['age'] >= age_range[0]) & (squad_history['age'] <= age_range[1])]
    player_predictions = {}  # Initialize an empty dictionary to store predictions
    
    latest_iteration = st.empty()
    progress_text = "Finding the best players. Please wait."
    bar = st.progress(0, text = progress_text)

    # Calculate the total number of iterations
    total_iterations = len(targets) * len(available_players)
    current_iteration = 0  # Initialize a counter to track the current iteration

    for target in targets:
        print('Modeling: ', target)
        model = train_model([target], squad_and_performance)  # Train model for the current target
        for player in available_players.itertuples():  # Iterate through available players
            new_squad = replace_player_with_lowest_minutes(selected_squad, player.player_id, squad_history, season)
            processed_squad = process_squad(new_squad)
            prediction = predict(model, processed_squad)
            # Initialize a sub-dictionary for the player if it doesn't exist
            if player.player_id not in player_predictions:
                player_predictions[player.player_id] = {}
            # Store the prediction for the current target
            player_predictions[player.player_id][target] = prediction
            
            # Update the current iteration and calculate progress
            current_iteration += 1
            progress = int((current_iteration / total_iterations) * 100)
            bar.progress(progress, text = progress_text)

    top_players = find_top_players_in_categories(player_predictions, targets)       
    return top_players 

def replace_player_with_lowest_minutes(selected_squad, new_player, squad_history, season):
            
            """
            Replaces a player in the selected squad with a new player from the squad history,
            targeting the player with the lowest minutes played in the same position for a specific season.
            The new player's club_id and league are made the same as the rest of the squad.

            Parameters:
            - selected_squad (DataFrame): The current selected squad.
            - new_player (str): The player ID of the new player to be inserted.
            - squad_history (DataFrame): The historical data of squads including minutes played and positions.
            - season (int): The season for which the replacement should be considered.

            Returns:
            - DataFrame: The updated selected squad with the new player replacing the one with the lowest minutes played in the same position.
            """
            # Find the position of the new player for the specified season
            new_player_data = squad_history[(squad_history['player_id'] == new_player) & (squad_history['season'] == season)]
            
            # If the new player has multiple entries (played for more than one club), select the entry with the most minutes played
            if len(new_player_data) > 1:
                new_player_data = new_player_data.loc[new_player_data['minutes_played'].idxmax()]

            # If only one entry exists, use it directly
            else:
                new_player_data = new_player_data.iloc[0]

            new_player_position = new_player_data['position']

            # Filter selected_squad for players in the same position
            same_position_players = selected_squad[selected_squad['position'] == new_player_position]

            # Find the player with the lowest minutes played in the same position
            if not same_position_players.empty:
                lowest_minutes_player = same_position_players.loc[same_position_players['minutes_played'].idxmin()]

                # Replace the player with the lowest minutes played with the new player
                selected_squad = selected_squad.drop(lowest_minutes_player.name)
                
                # Ensure the new player row has the same columns as the selected_squad
                new_player_row = new_player_data[selected_squad.columns]

                # Update the new player's club_id and league to match the rest of the squad
                new_player_row['club_id'] = selected_squad['club_id'].iloc[0]
                new_player_row['league'] = selected_squad['league'].iloc[0]

                # Append the new player to the selected squad
                selected_squad = pd.concat([selected_squad, pd.DataFrame([new_player_row])], ignore_index=True)

            return selected_squad


def process_squad(squad):
        # Define a custom order for positions
        position_order = {'Attack': 1, 'Midfield': 2, 'Defender': 3, 'Goalkeeper': 4}

        # Map the custom order to a new column for sorting
        squad['position_order'] = squad['position'].map(position_order)

        # Sort the DataFrame by squad_id, position_order, and minutes_played
        top_players_per_squad_sorted = squad.sort_values(by=['position_order', 'minutes_played'], ascending=[True, False])

        # Drop unnecessary columns before pivoting
        columns_to_drop = ['squad_id','player_name', 'club_code', 'goals', 'games', 'assists', 'minutes_played', 'goals_for', 'goals_against', 'clean_sheet', 'season_end_valuation', 'sub_position']
        top_players_per_squad_sorted.drop(columns=columns_to_drop, inplace=True)
        
        # Generate a unique identifier for each player within their squad for pivoting
        top_players_per_squad_sorted['player_rank'] = top_players_per_squad_sorted.groupby('club_id').cumcount() + 1

        # Pivot 'player_id' and 'age' columns
        player_ids_pivot = top_players_per_squad_sorted.pivot(index='club_id', columns='player_rank', values='player_id').add_prefix('player_id_')
        ages_pivot = top_players_per_squad_sorted.pivot(index='club_id', columns='player_rank', values='age').add_prefix('age_')

        # Merge the pivoted DataFrames back together
        squad_wide_format = player_ids_pivot.join(ages_pivot).reset_index()

        # Extract unique squad identifiers (season, league, club_id, squad_id) and drop duplicates
        squad_identifiers = top_players_per_squad_sorted[['season', 'league', 'club_id']].drop_duplicates()

        # Merge the squad identifiers with the wide format DataFrame
        final_squad_df = pd.merge(squad_identifiers, squad_wide_format, on='club_id', how='inner')

        final_squad_df['league_code'] = final_squad_df['league'].astype('category').cat.codes
        #drop league column
        final_squad_df = final_squad_df.drop(columns=['league', 'club_id'])

        return(final_squad_df)


def find_top_players_in_categories(player_predictions, model_targets):
        """
        Finds the top 5 players in each category specified in target
        and the top 5 players overall based on the weighted sum of their ranks across all specified categories.

        Parameters:
        - player_predictions (dict): A dictionary containing player IDs as keys and dictionaries of model targets and
            their predicted values as values.
        - target (str or list): A single category (model target) or a list of categories to find top players for.

        Returns:
        - dict: A dictionary with keys for each category in target and 'overall', each containing the top 5 player IDs for that category.
        """
        # Ensure target is a list
        if not isinstance(model_targets, list):
            model_targets = [model_targets]

        for player_id, predictions in player_predictions.items():
            for target, prediction in predictions.items():
                # Ensure the prediction is a scalar numeric value
                if isinstance(prediction, np.ndarray) and prediction.size == 1:
                    player_predictions[player_id][target] = prediction.item()  # Convert single-element array to scalar
                elif isinstance(prediction, list) and len(prediction) == 1:
                    player_predictions[player_id][target] = prediction[0]  # Convert single-element list to scalar
               
        # Now, when you create the DataFrame, the data should be in a suitable format for numeric operations
        predictions_df = pd.DataFrame.from_dict(player_predictions, orient='index')

        # Ensure all data in predictions_df are of a numeric type
        predictions_df = predictions_df.apply(pd.to_numeric, errors='coerce')

        # Proceed with the rest of the function as before

        # Define categories where a lower rank is better
        lower_is_better = ['wins', 'goals_scored', 'clean_sheets', 'points', 'win_rate', 'goal_difference', 'avg_goals_scored_per_game', 'clean_sheet_rate', 'goals_scored_per_win']
        # Define categories where a higher rank is better (i.e., fewer occurrences are better)
        higher_is_better = ['loss', 'goals_conceded', 'draws', 'total_games', 'loss_rate', 'avg_goals_conceded_per_game', 'goals_conceded_per_loss']

        # Rank the players within each category based on model_targets. Lower rank is better for specified categories.
        rankings = predictions_df[model_targets].rank(method='min', ascending=True)
        # Adjust ranking for categories where higher rank is better
        for category in higher_is_better:
            if category in model_targets:
                rankings[category] = rankings[category].rank(method='min', ascending=False)

        
        # Initialize a dictionary to store top 5 players in each category including overall
        top_players = {}

        # Find the top 5 players in each category specified in model_targets
        for category in model_targets:
            top_players[category] = rankings[category].nlargest(5).index.tolist()

        # Calculate overall ranking based on the weighted sum of ranks across all specified categories
        rankings['overall'] = rankings[model_targets].sum(axis=1)
        top_players['overall'] = rankings['overall'].nlargest(5).index.tolist()

        return top_players