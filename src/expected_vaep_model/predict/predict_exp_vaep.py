import pandas as pd
import numpy as np
import joblib
from expected_vaep_model.features.preprocessing import convert_chains_to_schema
def load_preprocessor():
    
    return joblib.load("models/preprocessors/exp_vaep_preprocessor_v6.joblib")

def load_scores_model():
    
    return joblib.load("models/models/exp_vaep_scores_v6.joblib")

def load_concedes_model():
    
    return joblib.load("models/models/exp_vaep_concedes_v6.joblib")

def merge_chains_shots(chains, shots):
    
    shot_ids = ['Match_ID', 'Chain_Number', 'Order', 'CD_Player_ID', 'Player', 'Team', 'Player_ID']
    shots = shots[shot_ids + [
        'Goal',
        'Behind',
        'Miss',
        'Score',
        'Event_Type1',
        'Set_Shot',
        'xGoals',
        'xBehinds',
        'xMiss',
        'xGoals_normalised',
        'xBehinds_normalised',
        'xMiss_normalised',
        'xScore']]

    # Preprocess
    chains['Quarter'] = chains['Period_Number']
    chains['Quarter_Duration'] = chains['Period_Duration']
    chains['Quarter_Duration_Chain_Start'] = chains['Period_Duration_Chain_Start']
    chains['Shot_At_Goal'] = np.where(chains['Shot_At_Goal'] == "TRUE", True, False)

    chains = chains.merge(shots, how = "left", on = shot_ids)
    
    return chains

def create_features(chains, shots, scores_model, concedes_model, preprocessor):

    chains = merge_chains_shots(chains, shots)
    schema_chains = convert_chains_to_schema(chains)

    # Processing
    chain_features = preprocessor.transform(chains)
    schema_chains['exp_scores'] = np.clip(scores_model.predict(chain_features), 0, 6)
    schema_chains['exp_concedes'] = np.clip(concedes_model.predict(chain_features), 0, 6)

    return schema_chains     
