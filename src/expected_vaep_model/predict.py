import pandas as pd
import numpy as np
import joblib

from AFLPy.AFLData_Client import load_data
from expected_vaep_model.features.preprocessing import convert_chains_to_schema, calculate_exp_vaep_values, create_gamestate_labels

def predict_exp_vaep(ID = None):

    # Load data
    chain_data = load_data(Dataset_Name='AFL_API_Match_Chains', ID = "AFL")
    shots = load_data(Dataset_Name="CG_Expected_Score", ID = "AFL")
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
    chain_data['Quarter'] = chain_data['Period_Number']
    chain_data['Quarter_Duration'] = chain_data['Period_Duration']
    chain_data['Quarter_Duration_Chain_Start'] = chain_data['Period_Duration_Chain_Start']
    chain_data['Shot_At_Goal'] = np.where(chain_data['Shot_At_Goal'] == "TRUE", True, False)

    chain_data = chain_data.merge(shots, how = "left", on = shot_ids)

    # Processing
    preproc = joblib.load("models/preprocessors/exp_vaep_preprocessor_v6.joblib")
    chain_features = preproc.transform(chain_data)

    schema_chains = convert_chains_to_schema(chain_data)
    exp_vaep_labels = create_gamestate_labels(schema_chains)
    schema_chains = pd.concat([schema_chains, chain_features, exp_vaep_labels], axis=1)
    schema_chains = schema_chains.rename(columns={'exp_scores':'exp_scores_label',
                                                  'exp_concedes':'exp_concedes_label'})

    # Load model
    exp_score_model = joblib.load("models/models/exp_vaep_scores_v6.joblib")
    schema_chains['exp_scores'] = np.clip(exp_score_model.predict(chain_features), 0, 6)
    exp_concede_model = joblib.load("models/models/exp_vaep_scores_v6.joblib")
    schema_chains['exp_concedes'] = np.clip(exp_concede_model.predict(chain_features), 0, 6)

    return calculate_exp_vaep_values(schema_chains)        