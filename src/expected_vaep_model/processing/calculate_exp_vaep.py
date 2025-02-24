import pandas as pd
from expected_vaep_model.features.formula import value

def calculate_exp_vaep_values(schema_chains):
    
    match_list = list(schema_chains['match_id'].unique())
    match_exp_vaep_list = []
    for match in match_list:
        match_chains = schema_chains[schema_chains['match_id'] == match]
        v = value(match_chains, match_chains['exp_scores'], match_chains['exp_concedes'])
        match_exp_vaep_list.append(v)
        
    exp_vaep_values = pd.concat(match_exp_vaep_list, axis=0)
    
    return pd.concat([schema_chains, exp_vaep_values], axis=1)