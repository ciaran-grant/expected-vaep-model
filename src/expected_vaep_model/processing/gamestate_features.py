import itertools
import numpy as np
import pandas as pd
from expected_vaep_model.processing.types import Types

### Creating Gamestate Features

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
    for action_type in Types().action_types:
        col = f'type_{action_type}'
        X[col] = actions['action_type'] == action_type
    return pd.DataFrame(X, index=actions.index)

def outcome(actions):
    
    return actions[['outcome_type']]

def outcome_onehot(actions):
    
    X = {}
    for outcome_type in Types().outcome_types:
        col = f'outcome_{outcome_type}'
        X[col] = actions['outcome_type'] == outcome_type
    return pd.DataFrame(X, index=actions.index)

def action_outcome_onehot(actions):
    
    action_type = action_type_onehot(actions)
    outcome_type = outcome_onehot(actions)
    X = {
        type_col + '_' + outcome_col: action_type[type_col]
        & outcome_type[outcome_col]
        for type_col, outcome_col in itertools.product(
            list(action_type), list(outcome_type)
        )
    }
    return pd.DataFrame(X, index=actions.index)

def time(actions):
        
    return actions[['period', 'period_seconds', 'overall_seconds']]

def start_location(actions):
    
    return actions[['left_right_start_x', 'left_right_start_y']]

def end_location(actions):
    
    return actions[['left_right_end_x', 'left_right_end_y']]

def movement(actions):
    
    move = pd.DataFrame(index=actions.index)
    move['dx'] = actions['left_right_end_x'] - actions['left_right_start_x']
    move['dy'] = actions['left_right_end_y'] - actions['left_right_start_y']
    move['movement'] = np.sqrt(move['dx']**2 + move['dy']**2)
    return move

def contested(actions):
    
    return actions[['contested']]

def mark(actions):
    
    return actions[['mark']]

def team(gamestates):
    """ Check whether the possession changed during the game state. 
    
    For each action, True if the team that performed the action is the same team that performed the last action.
    Otherwise False
    """
    
    a0 = gamestates[0]
    team_df = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        team_df[f'team_{str(i + 1)}'] = a['team'] == a0['team']
    return team_df

def time_delta(gamestates):
    """ Get the number of seconds between the last and previous actions. 
    
    """
  
    a0 = gamestates[0]
    dt = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dt[f'time_delta{str(i + 1)}'] = a['overall_seconds'] - a0['overall_seconds']
    return dt
    
def space_delta(gamestates):
    """ Get the distance covered between the last and previous actions. 
    
    """
  
    a0 = gamestates[0]
    space_delta = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dx = a['end_x'] - a0['left_right_start_x']
        space_delta[f'dx_a0{str(i + 1)}'] = dx
        dy = a['end_y'] - a0['left_right_start_y']
        space_delta[f'dy_a0{str(i + 1)}'] = dy
        space_delta[f'move_a0{str(i + 1)}'] = np.sqrt(dx**2 + dy**2)

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
            action_type_onehot(states[actions]),
            outcome_onehot(states[actions]),
            action_outcome_onehot(states[actions]),
            time(states[actions]),
            start_location(states[actions]),
            end_location(states[actions]),
            movement(states[actions]),
            contested(states[actions]),
            mark(states[actions])
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

    return pd.concat(states_features, axis=1)

def create_gamestate_features(chains):
    
    match_id_list = list(chains['match_id'].unique())
    match_gamestate_feature_list = []
    for match in match_id_list:
        match_gamestate_features = create_match_gamestate_features(chains, match_id=match, num_prev_actions=3)
        match_gamestate_feature_list.append(match_gamestate_features)

    return pd.concat(match_gamestate_feature_list, axis=0)
