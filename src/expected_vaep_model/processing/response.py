import pandas as pd
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
    y = pd.concat([shots, actions[['team', 'xscore']]], axis=1)
    y.columns = ['shots', 'team', 'xscore']

    for i in range(num_actions):
        for c in ['team', 'xscore']:
            shifted = y[c].shift(-i).fillna(0)
            y['%s+%d' % (c, i)] = shifted
        y[f'xscore+{str(i)}'] = y[f'xscore+{str(i)}'] * (
            y[f'team+{str(i)}'] == y['team']
        )
        
    xscore_cols = [x for x in y if 'xscore+' in x]
    res = y[xscore_cols].max(axis=1)

    return pd.DataFrame(res, columns=['exp_scores_label'])

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
    y = pd.concat([shots, actions[['team', 'xscore']]], axis=1)
    y.columns = ['shots', 'team', 'xscore']
    for i in range(num_actions):
        for c in ['team', 'xscore']:
            shifted = y[c].shift(-i).fillna(0)
            y['%s+%d' % (c, i)] = shifted
        y[f'xscore+{str(i)}'] = y[f'xscore+{str(i)}'] * (
            y[f'team+{str(i)}'] != y['team']
        )
    xscore_cols = [x for x in y if 'xscore+' in x]
    res = y[xscore_cols].max(axis=1)

    return pd.DataFrame(res, columns=['exp_concedes_label'])

def create_match_gamestate_labels(actions, match_id, num_future_actions=10):
    
    match_actions = actions[actions['match_id'] == match_id]

    return pd.concat(
        [expected_scores(match_actions, num_actions=num_future_actions), 
         expected_concedes(match_actions, num_actions=num_future_actions)],
        axis=1,
    )

def create_gamestate_labels(chains):
    
    match_id_list = list(chains['match_id'].unique())
    match_gamestate_label_list = []
    for match in match_id_list:
        match_gamestate_labels = create_match_gamestate_labels(chains, match_id=match, num_future_actions=10)
        match_gamestate_label_list.append(match_gamestate_labels)

    return pd.concat(match_gamestate_label_list, axis=0)

