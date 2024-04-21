from sklearn.tree import DecisionTreeRegressor


def train_model(target, squad_and_performance):
    squad_and_performance = squad_and_performance.sort_values(by='season')
    df_train = squad_and_performance.iloc[:int(0.8*len(squad_and_performance))]
    df_test = squad_and_performance.iloc[int(0.8*len(squad_and_performance)):]
    
    # only numeric features
    features = ['season', 'player_id_1',
       'player_id_2', 'player_id_3', 'player_id_4', 'player_id_5',
       'player_id_6', 'player_id_7', 'player_id_8', 'player_id_9',
       'player_id_10', 'player_id_11', 'player_id_12', 'player_id_13',
       'player_id_14', 'player_id_15', 'player_id_16', 'player_id_17',
       'player_id_18', 'player_id_19', 'age_1', 'age_2', 'age_3', 'age_4',
       'age_5', 'age_6', 'age_7', 'age_8', 'age_9', 'age_10', 'age_11',
       'age_12', 'age_13', 'age_14', 'age_15', 'age_16', 'age_17', 'age_18',
       'age_19', 'league_code']

    X = df_train[features]
    y = df_train[target]

    dt_0 = DecisionTreeRegressor()
    dt_0.fit(X, y)
    return dt_0


def predict(model, squad):
    y_pred = model.predict(squad)
    return y_pred