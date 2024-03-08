import pandas as pd
import numpy as np
import joblib

from AFLPy.AFLData_Client import load_data
from expected_vaep_model.features.preprocessing import convert_chains_to_schema, calculate_exp_vaep_values, create_gamestate_labels

def predict_exp_vaep(ID = None):

    # Load data
    chain_data = load_data(Dataset_Name='AFL_API_Match_Chains', ID = ID)
    shots = load_data(Dataset_Name="CG_Expected_Score", ID = ID)

    # Preprocess
    chain_data['Quarter_Duration'] = chain_data['Period_Duration']
    chain_data['Quarter_Duration_Chain_Start'] = chain_data['Period_Duration_Chain_Start']
    chain_data['Shot_At_Goal'] = np.where(chain_data['Shot_At_Goal'] == "TRUE", True, False)
    chain_data = chain_data.merge(shots, how = "left", on = ['Match_ID', 'Chain_Number', 'Order'])

    # Processing
    preproc = joblib.load("models/preprocessors/exp_vaep_preprocessor_v5.joblib")
    chain_features = preproc.transform(chain_data)
    
    schema_chains = convert_chains_to_schema(chain_data)
    exp_vaep_labels = create_gamestate_labels(schema_chains)
    schema_chains = pd.concat([schema_chains, chain_features, exp_vaep_labels], axis=1)
    schema_chains = schema_chains.rename(columns={'exp_scores':'exp_scores_label',
                                                  'exp_concedes':'exp_concedes_label'})
    
    # Load model
    exp_score_model = joblib.load("models/models/exp_vaep_scores_v5.joblib")
    schema_chains['exp_scores'] = np.clip(exp_score_model.predict(chain_features), 0, 6)
    exp_concede_model = joblib.load("models/models/exp_vaep_scores_v5.joblib")
    schema_chains['exp_concedes'] = np.clip(exp_concede_model.predict(chain_features), 0, 6)

    # Scoring
    scored_chains = calculate_exp_vaep_values(schema_chains)
    
    # Merge back to chains
    chain_data = chain_data.merge(scored_chains.drop(columns = ['xScore']), how = "left", left_on=['Match_ID', 'Chain_Number', 'Order'], right_on=['match_id', 'chain_number', 'order'])

    return chain_data        