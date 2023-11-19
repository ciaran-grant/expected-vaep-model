import pandas as pd
import numpy as np
import joblib

from expected_score_model.config import exp_behind_open_model_file_path, exp_behind_open_preprocessor_file_path
from expected_score_model.config import exp_behind_set_model_file_path, exp_behind_set_preprocessor_file_path
from expected_score_model.config import exp_goal_open_model_file_path, exp_goal_open_preprocessor_file_path
from expected_score_model.config import exp_goal_set_model_file_path, exp_goal_set_preprocessor_file_path
from expected_score_model.config import exp_miss_open_model_file_path, exp_miss_open_preprocessor_file_path
from expected_score_model.config import exp_miss_set_model_file_path, exp_miss_set_preprocessor_file_path

from expected_vaep_model.features.preprocessing import get_expected_scores, convert_chains_to_schema, calculate_exp_vaep_values, create_gamestate_labels
from expected_vaep_model.config import chain_file_path, exp_vaep_preprocessor_file_path, exp_vaep_score_model_file_path, exp_vaep_concede_model_file_path, exp_vaep_chain_output_path

def predict_model(chain_file_path, expected_scores_path_dict, exp_vaep_chain_output_path):

    # Load data
    chains = pd.read_csv(chain_file_path)
    print("Chain data loaded.")

    # Processing
    preproc = joblib.load(exp_vaep_preprocessor_file_path)
    chain_features = preproc.transform(chains)
    
    chains = get_expected_scores(chains, expected_scores_path_dict)
    schema_chains = convert_chains_to_schema(chains)
    exp_vaep_labels = create_gamestate_labels(schema_chains)
    schema_chains = pd.concat([schema_chains, chain_features, exp_vaep_labels], axis=1)
    schema_chains = schema_chains.rename(columns={'exp_scores':'exp_scores_label',
                                                  'exp_concedes':'exp_concedes_label'})
    
    print("Preprocessing.. Complete.")

    # Load model
    exp_score_model = joblib.load(exp_vaep_score_model_file_path)
    schema_chains['exp_scores'] = np.clip(exp_score_model.predict(chain_features), 0, 6)
    exp_concede_model = joblib.load(exp_vaep_concede_model_file_path)
    schema_chains['exp_concedes'] = np.clip(exp_concede_model.predict(chain_features), 0, 6)

    # Scoring
    scored_chains = calculate_exp_vaep_values(schema_chains)
    print("Scoring.. complete.")
    
    # Merge back to chains
#     chains = chains.merge(scored_chains.drop(columns = ['xScore']), how = "left", left_on=['Match_ID', 'Chain_Number', 'Order'], right_on=['match_id', 'chain_number', 'order'])

    # Export data
    scored_chains.to_csv(exp_vaep_chain_output_path, index=False)
    print("Exporting.. complete.")

if __name__ == "__main__":
        
    expected_scores_path_dict = {
        'set':{'goal':{'preprocessor':exp_goal_set_preprocessor_file_path,
                        'model':exp_goal_set_model_file_path},
                'behind':{'preprocessor':exp_behind_set_preprocessor_file_path,
                        'model':exp_behind_set_model_file_path},
                'miss':{'preprocessor':exp_miss_set_preprocessor_file_path,
                        'model':exp_miss_set_model_file_path}},
        'open':{'goal':{'preprocessor':exp_goal_open_preprocessor_file_path,
                        'model':exp_goal_open_model_file_path},
                'behind':{'preprocessor':exp_behind_open_preprocessor_file_path,
                        'model':exp_behind_open_model_file_path},
                'miss':{'preprocessor':exp_miss_open_preprocessor_file_path,
                        'model':exp_miss_open_model_file_path}}
        }
    
    predict_model(chain_file_path, expected_scores_path_dict, exp_vaep_chain_output_path)
    
    
    
    
    
    