import pandas as pd
def merge_xscores_to_chains(chains, xscore):

    keys = ['Match_ID', 'Chain_Number', 'Order', 'CD_Player_ID', 'Player', 'Team', 'Player_ID']
    shot_cols = [
        'predicted_result',
        'behind_probas',
        'goal_probas',
        'miss_probas',
        'xscore',
        'result',
        'score',
        'miss',
        'behind',
        'goal']
            
    return pd.merge(chains, xscore[keys + shot_cols], how = 'left', on = keys)