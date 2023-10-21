import numpy as np
import pandas as pd
import joblib
from expected_vaep_model.features.formula import value
from expected_vaep_model.modelling_data_contract import ModellingDataContract

### Converting chain data to SPADL format

def play_left_to_right(chains):
    
    # Want everyone to be playing from left to right perspective
    chains['new_x'] = chains['x'].copy()
    chains['new_y'] = chains['y'].copy()

    chains['new_x'] = np.where((chains['Team'] == chains['Team_Chain']), chains['x'], -1*chains['new_x'])
    chains['new_y'] = np.where((chains['Team'] == chains['Team_Chain']), chains['y'], -1*chains['new_y'])

    return chains

def create_duration(chains):
    max_quarter_durations = chains.groupby(['Match_ID', "Quarter"])['Quarter_Duration'].max().reset_index()
    max_quarter_durations = max_quarter_durations.rename(columns = {'Quarter_Duration':'Quarter_Duration_Max'})
    max_quarter_durations = max_quarter_durations.pivot(index = 'Match_ID', columns='Quarter', values='Quarter_Duration_Max')
    chains = chains.merge(max_quarter_durations, how='left', on = ['Match_ID'])
    chains['Duration'] = np.where(chains['Quarter'] == 1, chains['Quarter_Duration'],
                                np.where(chains['Quarter'] == 2, chains[1] + chains['Quarter_Duration'],
                                        np.where(chains['Quarter'] == 3, chains[1] + chains[2] + chains['Quarter_Duration'],
                                                    np.where(chains['Quarter'] == 4, chains[1] + chains[2] + chains[3] + chains['Quarter_Duration'],
                                                            0))))
    
    return chains

def get_action_types(chains):
    
    schema_chains = chains.copy()

    schema_chains[~((schema_chains['Description'] == "Contested Mark") & (schema_chains['Team'] != schema_chains['Team_Chain']))]
    schema_chains[~((schema_chains['Description'] == "Uncontested Mark") & (schema_chains['Team'] != schema_chains['Team_Chain']))]
    schema_chains[~((schema_chains['Description'] == "Contested Knock On") & (schema_chains['Team'] != schema_chains['Team_Chain']))]
    schema_chains[~((schema_chains['Description'] == "Gather from Opposition") & (schema_chains['Team'] != schema_chains['Team_Chain']))]
    schema_chains[~((schema_chains['Description'] == "Loose Ball Get") & (schema_chains['Team'] != schema_chains['Team_Chain']))]
    schema_chains[~((schema_chains['Description'] == "Hard Ball Get") & (schema_chains['Team'] != schema_chains['Team_Chain']))]
    schema_chains[~((schema_chains['Description'] == "Pack Mark (O)") & (schema_chains['Team'] != schema_chains['Team_Chain']))]

    schema_chains['action_type'] = schema_chains['Description'].copy()
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Handball Received", "Carry", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Kickin play on", "Carry", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Bounce", "Carry", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Gather From Hitout", "Gather", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Gather from Opposition", "Gather", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Mark On Lead", "Uncontested Mark", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Hard Ball Get Crumb", "Hard Ball Get", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Loose Ball Get Crumb", "Loose Ball Get", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Ruck Hard Ball Get", "Hard Ball Get", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Free For: In Possession", "Free For", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Free For: Off The Ball", "Free For", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Free For: Before the Bounce", "Free For", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Kickin short", "Kick", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Pack Mark (P)", "Contested Mark", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Pack Mark (O)", "Contested Mark", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "No Pressure Error", "Error", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Debit", "Error", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Contested Knock On", "Knock On", schema_chains['action_type'])
    schema_chains['action_type'] = np.where(schema_chains['Description'] == "Ground Kick", "Kick", schema_chains['action_type'])

    schema_chains['action_type'] = np.where((schema_chains['Description'] == "Kick") & (schema_chains['Shot_At_Goal'] == True), "Shot", schema_chains['action_type'])

    return schema_chains['action_type']

def get_outcome_types(chains):
    
    schema_chains = chains.copy()
    schema_chains['NextTeam'] = schema_chains.groupby('Match_ID')['Team'].shift(-1).fillna(0)
    schema_chains['outcome_type'] = "effective"
    
    schema_chains['outcome_type'] = np.where(schema_chains['action_type'].isin(["Kick", "Handball", "Shot"]), schema_chains['Disposal'], schema_chains['outcome_type'])
    schema_chains['outcome_type'] = np.where((schema_chains['action_type'].isin(["Free For", "Knock On"])) & (schema_chains['Team'] != schema_chains['NextTeam']), "ineffective", schema_chains['outcome_type'])
    schema_chains['outcome_type'] = np.where(schema_chains['action_type'] == "Error", "ineffective", schema_chains['outcome_type'])
            
    return schema_chains['outcome_type']

def convert_chains_to_schema(chains):
    
    schema_chains = chains.copy()
    
    schema_chains = play_left_to_right(schema_chains)
    
    schema_chains = create_duration(schema_chains)

    schema_chains['match_id'] = schema_chains['Match_ID']
    schema_chains['chain_number'] = schema_chains['Chain_Number']
    schema_chains['order'] = schema_chains['Order']
    schema_chains['match_id'] = schema_chains['Match_ID']
    schema_chains['quarter'] = schema_chains['Quarter']
    schema_chains['quarter_seconds'] = schema_chains['Quarter_Duration']
    schema_chains['overall_seconds'] = schema_chains['Duration']
    schema_chains['team'] = schema_chains['Team']
    schema_chains['player'] = schema_chains['Player']

    schema_chains['start_x'] = schema_chains['new_x']
    schema_chains['start_y'] = schema_chains['new_y']
    schema_chains['end_x'] = schema_chains.groupby('Match_ID')['new_x'].shift(-1).fillna(0)
    schema_chains['end_y'] = schema_chains.groupby('Match_ID')['new_y'].shift(-1).fillna(0)

    schema_chains['action_type'] = get_action_types(schema_chains)
    schema_chains['outcome_type'] = get_outcome_types(schema_chains)

    schema_chains = schema_chains.dropna(subset=['Player'])
    schema_chains = schema_chains[schema_chains['action_type'].isin(ModellingDataContract.action_types)]

    schema_chains = schema_chains[['match_id', 'chain_number', 'order', 'quarter', 'quarter_seconds', 'overall_seconds', 'team', 'player', 'start_x', 'start_y', 'end_x', 'end_y', 'action_type', 'outcome_type', 'xScore']]
    
    return schema_chains

### Creating Features

def gamestates(actions, num_prev_actions: int = 3):
    """ Convert a dataframe of actions to gamestates.
    
    Each gamestate is represented as the num_prev_actions previous actions.

    Parameters
    ----------
    actions : AFLActions
        A DataFrame with the actions of a game.
    num_prev_actions : int, default = 3
        The number of previous actions included in the gamestate.

    Returns
    -------
    GameStates
        The num_prev_actions previous actions for each action.
    """
    
    if num_prev_actions < 1:
        raise ValueError('The gamestate should include at least one preceding action.')
    
    states = [actions]
    for i in range(1, num_prev_actions):
        prev_actions = actions.copy().shift(i, fill_value=0)
        prev_actions.iloc[:i] = pd.concat([actions[:1]] * i, ignore_index=True)
        states.append(prev_actions)
        
    return states

def action_type(actions):
    
    return actions[['action_type']]

def action_type_onehot(actions):
    
    X = {}
    for action_type in ModellingDataContract.action_types:
        col = 'type_' + action_type
        X[col] = actions['action_type'] == action_type
    return pd.DataFrame(X, index=actions.index)

def outcome(actions):
    
    return actions[['outcome_type']]

def outcome_onehot(actions):
    
    X = {}
    for outcome_type in ModellingDataContract.outcome_types:
        col = 'outcome_' + outcome_type
        X[col] = actions['outcome_type'] == outcome_type
    return pd.DataFrame(X, index=actions.index)

def action_outcome_onehot(actions):
    
    action_type = action_type_onehot(actions)
    outcome_type = outcome_onehot(actions)
    X = {}
    for type_col in list(action_type):
        for outcome_col in list(outcome_type):
            X[type_col + '_' + outcome_col] = action_type[type_col] & outcome_type[outcome_col]
    return pd.DataFrame(X, index=actions.index)

def time(actions):
        
    return actions[['quarter', 'quarter_seconds', 'overall_seconds']].copy()

def start_location(actions):
    
    return actions[['start_x', 'start_y']]

def end_location(actions):
    
    return actions[['end_x', 'end_y']]

def movement(actions):
    
    move = pd.DataFrame(index=actions.index)
    move['dx'] = actions['end_x'] - actions['start_x']
    move['dy'] = actions['end_y'] - actions['start_y']
    move['movement'] = np.sqrt(move['dx']**2 + move['dy']**2)
    return move

def team(gamestates):
    """ Check whether the possession changed during the game state. 
    
    For each action, True if the team that performed the action is the same team that performed the last action.
    Otherwise False
    """
    
    a0 = gamestates[0]
    team_df = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        team_df['team_'+(str(i+1))] = a['team'] == a0['team']
    return team_df

def time_delta(gamestates):
    """ Get the number of seconds between the last and previous actions. 
    
    """
  
    a0 = gamestates[0]
    dt = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dt['time_delta'+(str(i+1))] = a['overall_seconds'] - a0['overall_seconds']
    return dt
    
def space_delta(gamestates):
    """ Get the distance covered between the last and previous actions. 
    
    """
  
    a0 = gamestates[0]
    space_delta = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dx = a['end_x'] - a0['start_x']
        space_delta['dx_a0' + (str(i+1))] = dx
        dy = a['end_y'] - a0['start_y']
        space_delta['dy_a0' + (str(i+1))] = dy
        space_delta['move_a0' + str(i+1)] = np.sqrt(dx**2 + dy**2)
        
    return space_delta    

def goal_score(gamestates):
    """ Get the number of goals scored by each team after the action. """
    
    actions = gamestates[0]
    teamA = actions['team'].values[0]
    goals = actions['action_type'].str.contains('Shot') & (actions['outcome_type'] == "effective")
    
    teamisA = actions['team'] == teamA
    teamisB = ~teamisA
    goals_teamA = (goals & teamisA)
    goals_teamB = (goals & teamisB)
    
    goal_score_teamA = goals_teamA.cumsum() - goals_teamA
    goal_score_teamB = goals_teamB.cumsum() - goals_teamB
    
    score_df = pd.DataFrame(index=actions.index)
    score_df['goalscore_team'] = (goal_score_teamA * goals_teamA) + (goal_score_teamB * goals_teamB)
    score_df['goalscore_opponent'] = (goal_score_teamA * goals_teamB) + (goal_score_teamA * goals_teamB)
    score_df['goalscore_diff'] = score_df['goalscore_team'] - score_df['goalscore_opponent']

    return score_df

def create_match_gamestate_features(actions, match_id, num_prev_actions=3):
    
    match_actions = actions[actions['match_id'] == match_id]

    states = gamestates(match_actions, num_prev_actions)
    
    states_features = []
    for actions in range(len(states)):
        state = pd.concat([
            # action_type(states[actions]),
            action_type_onehot(states[actions]),
            # outcome(states[actions]),
            outcome_onehot(states[actions]),
            action_outcome_onehot(states[actions]),
            time(states[actions]),
            start_location(states[actions]),
            end_location(states[actions]),
            movement(states[actions])
        ], axis=1)
        state.columns = [x+'_a'+str(actions) for x in list(state.columns)]
        states_features.append(state)
        
    features = pd.concat([
        team(states),
        time_delta(states),
        space_delta(states),
        goal_score(states)
        ], axis=1)
    
    states_features.append(features)
    
    gamestate_features = pd.concat(states_features, axis=1) 
    
    return gamestate_features

def create_gamestate_features(chains):
    
    match_id_list = list(chains['match_id'].unique())
    match_gamestate_feature_list = []
    for match in match_id_list:
        match_gamestate_features = create_match_gamestate_features(chains, match_id=match, num_prev_actions=3)
        match_gamestate_feature_list.append(match_gamestate_features)
        
    gamestate_features = pd.concat(match_gamestate_feature_list, axis=0)
    
    return gamestate_features

### Creating Response - Scoring and Conceding Labels

def expected_scores(actions, num_actions: int = 10) -> pd.DataFrame:
    """Determines whether the team possessing the ball had a shot within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game
    num_actions : int, default=10
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'exp_scores' and a row for each action set to the expected score
        value of the shot if a shot was taken within the next x actions.
        Otherwise 0.
    """
    
    shots = (actions['action_type'] == "Shot")
    y = pd.concat([shots, actions[['team', 'xScore']]], axis=1)
    y.columns = ['shots', 'team', 'xScore']
    for i in range(num_actions):
        for c in ['team', 'xScore']:
            shifted = y[c].shift(-i).fillna(0)
            y['%s+%d' % (c, i)] = shifted
        y['xScore+'+str(i)] = y['xScore+'+str(i)] * (y['team+'+str(i)] == y['team'])
    xscore_cols = [x for x in y if 'xScore+' in x]
    res = y[xscore_cols].max(axis=1)
    
    return pd.DataFrame(res, columns=['exp_scores'])

def expected_concedes(actions, num_actions: int = 10) -> pd.DataFrame:
    """Determines whether the team possessing the ball conceded a goal within the next x actions.
    

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game
    num_actions : int, default=10
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes' and a row for each action set to True if a goal was
        conceded by the team possessing the ball within the next x actions.
        Otherwise False.
    """
    
    shots = (actions['action_type'] == "Shot")
    y = pd.concat([shots, actions[['team', 'xScore']]], axis=1)
    y.columns = ['shots', 'team', 'xScore']
    for i in range(num_actions):
        for c in ['team', 'xScore']:
            shifted = y[c].shift(-i).fillna(0)
            y['%s+%d' % (c, i)] = shifted
        y['xScore+'+str(i)] = y['xScore+'+str(i)] * (y['team+'+str(i)] != y['team'])
    xscore_cols = [x for x in y if 'xScore+' in x]
    res = y[xscore_cols].max(axis=1)
    
    return pd.DataFrame(res, columns=['exp_concedes'])

def create_match_gamestate_labels(actions, match_id, num_future_actions=10):
    
    match_actions = actions[actions['match_id'] == match_id]

    gamestate_labels = pd.concat([
        expected_scores(match_actions),
        expected_concedes(match_actions)
    ], axis=1)
    
    return gamestate_labels

def create_gamestate_labels(chains):
    
    match_id_list = list(chains['match_id'].unique())
    match_gamestate_label_list = []
    for match in match_id_list:
        match_gamestate_labels = create_match_gamestate_labels(chains, match_id=match, num_future_actions=10)
        match_gamestate_label_list.append(match_gamestate_labels)
        
    gamestate_labels = pd.concat(match_gamestate_label_list, axis=0)
    
    return gamestate_labels

final_state_map = {
    'goal':'goal',
    'behind':'behind',
    'turnover':'miss',
    'rushed':'miss',
    'outOfBounds':'miss',
    'ballUpCall':'miss',
    'endQuarter':'miss',
    'rushedOpp':'miss',
}

def expected_score_response_processing(chain_data):
    
    chain_data['Final_State'] = chain_data['Final_State'].replace(final_state_map)

    chain_data['Goal'] = np.where((chain_data['Shot_At_Goal'] == True) & (chain_data['Final_State'] == "goal"), 1, 0)
    chain_data['Behind'] = np.where((chain_data['Shot_At_Goal'] == True) & (chain_data['Final_State'] == "behind"), 1, 0)
    chain_data['Miss'] = np.where((chain_data['Shot_At_Goal'] == True) & (chain_data['Final_State'] == "miss"), 1, 0)

    chain_data['Score'] = np.where(chain_data['Goal']==1, 6,
                                np.where(chain_data['Behind']==1, 1, 
                                            0))
    
    return chain_data

def expected_score_feature_engineering(chain_data):
        
    chain_data['Event_Type0'] = chain_data['Description']
    chain_data['Event_Type1'] = chain_data['Description'].shift(1)
    chain_data['Event_Type2'] = chain_data['Description'].shift(2)
    chain_data['Event_Type3'] = chain_data['Description'].shift(3)

    chain_data['x0'] = chain_data['x']
    chain_data['x1'] = chain_data['x'].shift(1)
    chain_data['x2'] = chain_data['x'].shift(2)
    chain_data['x3'] = chain_data['x'].shift(3)

    chain_data['y0'] = chain_data['y']
    chain_data['y1'] = chain_data['y'].shift(1)
    chain_data['y2'] = chain_data['y'].shift(2)
    chain_data['y3'] = chain_data['y'].shift(3)

    chain_data['Quarter_Duration0'] = chain_data['Quarter_Duration']
    chain_data['Quarter_Duration1'] = chain_data['Quarter_Duration'].shift(1)
    chain_data['Quarter_Duration2'] = chain_data['Quarter_Duration'].shift(2)
    chain_data['Quarter_Duration3'] = chain_data['Quarter_Duration'].shift(3)
    
    chain_data['Time_Since_Last_Action'] = chain_data['Quarter_Duration0'] - chain_data['Quarter_Duration1']
    chain_data['Distance_Since_Last_Action'] = ((chain_data['x1'] - chain_data['x0'])**2 + (chain_data['y1'] - chain_data['y0'])**2)**0.5
    
    chain_data['Chain_Duration'] = chain_data['Quarter_Duration'] - chain_data['Quarter_Duration_Chain_Start']
    
    chain_data['Distance_to_Right_Goal_x'] = chain_data['Venue_Length']/2 - chain_data['x0']
    chain_data['Distance_to_Middle_y'] = abs(chain_data['y0'])

    chain_data['Distance_to_Middle_Goal'] = (chain_data['Distance_to_Right_Goal_x']**2 + chain_data['Distance_to_Middle_y']**2)**0.5
    chain_data['Angle_to_Middle_Goal'] = np.arctan2(chain_data['Distance_to_Middle_y'], chain_data['Distance_to_Right_Goal_x'])
    chain_data['Angle_to_Middle_Goal_degrees'] = np.degrees(chain_data['Angle_to_Middle_Goal'])

    chain_data['Visible_Goal_Angle'] = (6.4*chain_data['Distance_to_Right_Goal_x']) / (chain_data['Distance_to_Right_Goal_x']**2 + chain_data['Distance_to_Middle_y']**2-(6.4/2)**2)
    chain_data['Visible_Goal_Angle_degrees'] = np.degrees(chain_data['Visible_Goal_Angle'])

    chain_data['Visible_Behind_Angle'] = ((6.4*3)*chain_data['Distance_to_Right_Goal_x']) / (chain_data['Distance_to_Right_Goal_x']**2 + chain_data['Distance_to_Middle_y']**2-((3*6.4)/2)**2)
    chain_data['Visible_Behind_Angle_degrees'] = np.degrees(chain_data['Visible_Behind_Angle'])
    
    chain_data['Squared_Distance_to_Middle_Goal'] = chain_data['Distance_to_Right_Goal_x']**2
    chain_data['Log_Distance_to_Middle_Goal'] = np.log(chain_data['Distance_to_Right_Goal_x'])
    
    return chain_data

def split_shots(chain_data):
    
    chain_data['Event_Type1'] = chain_data['Description'].shift(1)
    df_shots = chain_data[chain_data['Shot_At_Goal'] == True]
    df_shots['Set_Shot'] = df_shots['Event_Type1'].apply(lambda x: ("Mark" in x) or ("Free" in x))
    df_set_shots = df_shots[df_shots['Set_Shot']]
    df_open_shots = df_shots[~df_shots['Set_Shot']]
    
    return df_set_shots, df_open_shots

def get_expected_scores(chain_data, expected_scores_path_dict):
    
    # Preprocess
    chain_data = expected_score_response_processing(chain_data)
    df_set_shots, df_open_shots = split_shots(chain_data)

    goal_set_preproc = joblib.load(expected_scores_path_dict['set']['goal']['preprocessor'])
    behind_set_preproc = joblib.load(expected_scores_path_dict['set']['behind']['preprocessor'])
    miss_set_preproc = joblib.load(expected_scores_path_dict['set']['miss']['preprocessor'])
    goal_open_preproc = joblib.load(expected_scores_path_dict['open']['goal']['preprocessor'])
    behind_open_preproc = joblib.load(expected_scores_path_dict['open']['behind']['preprocessor'])
    miss_open_preproc = joblib.load(expected_scores_path_dict['open']['miss']['preprocessor'])

    set_goal_features = goal_set_preproc.transform(chain_data)
    set_behind_features = behind_set_preproc.transform(chain_data)
    set_miss_features = miss_set_preproc.transform(chain_data)
    open_goal_features = goal_open_preproc.transform(chain_data)
    open_behind_features = behind_open_preproc.transform(chain_data)
    open_miss_features = miss_open_preproc.transform(chain_data)

    # Load models
    expected_goal_set_model = joblib.load(expected_scores_path_dict['set']['goal']['model'])
    expected_behind_set_model = joblib.load(expected_scores_path_dict['set']['behind']['model'])
    expected_miss_set_model = joblib.load(expected_scores_path_dict['set']['miss']['model'])

    expected_goal_open_model = joblib.load(expected_scores_path_dict['open']['goal']['model'])
    expected_behind_open_model = joblib.load(expected_scores_path_dict['open']['behind']['model'])
    expected_miss_open_model = joblib.load(expected_scores_path_dict['open']['miss']['model'])

    # Score models
    df_set_shots['xGoals'] = expected_goal_set_model.predict_proba(set_goal_features, calibrate=True)
    df_set_shots['xBehinds'] = expected_behind_set_model.predict_proba(set_behind_features, calibrate=True)
    df_set_shots['xMiss'] = expected_miss_set_model.predict_proba(set_miss_features, calibrate=True)

    df_open_shots['xGoals'] = expected_goal_open_model.predict_proba(open_goal_features, calibrate=True)
    df_open_shots['xBehinds'] = expected_behind_open_model.predict_proba(open_behind_features, calibrate=True)
    df_open_shots['xMiss'] = expected_miss_open_model.predict_proba(open_miss_features, calibrate=True)

    # Expected Score
    df_shots = pd.concat([df_set_shots, df_open_shots], axis=0)
    df_shots = df_shots.sort_values(by = ['Match_ID', "Chain_Number", "Order"])

    df_shots['xGoals_normalised'] = df_shots['xGoals'] / (df_shots['xGoals'] + df_shots['xBehinds'] + df_shots['xMiss'])
    df_shots['xBehinds_normalised'] = df_shots['xBehinds'] / (df_shots['xGoals'] + df_shots['xBehinds'] + df_shots['xMiss'])
    df_shots['xMiss_normalised'] = df_shots['xMiss'] / (df_shots['xGoals'] + df_shots['xBehinds'] + df_shots['xMiss'])

    df_shots['xScore'] = df_shots['xGoals_normalised']*6 + df_shots['xBehinds_normalised']

    # Merge xScore to Chain
    chain_data = chain_data.merge(df_shots[['Match_ID', "Chain_Number", "Order", 'xGoals', 'xBehinds', 'xMiss', 'xGoals_normalised', 'xBehinds_normalised', 'xMiss_normalised', 'xScore']], how = "left", on = ['Match_ID', "Chain_Number", "Order"])

    return chain_data

def get_stratified_train_test_val_columns(data, response):
    
    from sklearn.model_selection import train_test_split
    
    X, y = data.drop(columns=[response]), data[response]
    X_modelling, X_test, y_modelling, y_test = train_test_split(X, y, test_size = 0.2, random_state=2407)
    X_train, X_val, y_train, y_val = train_test_split(X_modelling, y_modelling, test_size = 0.2, random_state=2407)
    X_train[response+'TrainingSet'] = True
    X_test[response+'TestSet'] = True
    X_val[response+'ValidationSet'] = True
    
    if [response+'TrainingSet', response+'TestSet', response+'ValidationSet'] not in list(data):
        data = pd.merge(data, X_train[response+'TrainingSet'], how="left", left_index=True, right_index=True) 
        data = pd.merge(data, X_test[response+'TestSet'], how="left", left_index=True, right_index=True) 
        data = pd.merge(data, X_val[response+'ValidationSet'], how="left", left_index=True, right_index=True)
        data[[response+'TrainingSet', response+'TestSet', response+'ValidationSet']] = data[[response+'TrainingSet', response+'TestSet', response+'ValidationSet']].fillna(False) 
        
    return data

def calculate_exp_vaep_values(schema_chains):
    
    match_list = list(schema_chains['match_id'].unique())
    match_exp_vaep_list = []
    for match in match_list:
        match_chains = schema_chains[schema_chains['match_id'] == match]
        v = value(match_chains, match_chains['exp_scores'], match_chains['exp_concedes'])
        match_exp_vaep_list.append(v)
        
    exp_vaep_values = pd.concat(match_exp_vaep_list, axis=0)
    
    return pd.concat([schema_chains, exp_vaep_values], axis=1)