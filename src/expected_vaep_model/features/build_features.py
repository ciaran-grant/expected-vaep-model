import pandas as pd
import joblib

import expected_score_model.config as exp_score_config
import expected_vaep_model.config as config
from expected_vaep_model.features.data_preprocessor import ExpVAEPPreprocessor
from expected_vaep_model.features.preprocessing import get_expected_scores, convert_chains_to_schema, create_gamestate_labels, get_stratified_train_test_val_columns

import warnings
warnings.filterwarnings('ignore')

def match_chains_to_modelling_data(chain_file_path, model_output_path, preprocessor_output_path):
    
    chains = pd.read_csv(chain_file_path)
    print('Chain data loaded.')
    
    expected_scores_path_dict = {
        'set':{
            'goal':{
                'preprocessor':exp_score_config.exp_goal_set_preprocessor_file_path,
                'model':exp_score_config.exp_goal_set_model_file_path},
            'behind':{
                'preprocessor':exp_score_config.exp_behind_set_preprocessor_file_path,
                'model':exp_score_config.exp_behind_set_model_file_path},
            'miss':{
                'preprocessor':exp_score_config.exp_miss_set_preprocessor_file_path,
                'model':exp_score_config.exp_miss_set_model_file_path}},
        'open':{
            'goal':{
                'preprocessor':exp_score_config.exp_goal_open_preprocessor_file_path,
                'model':exp_score_config.exp_goal_open_model_file_path},
            'behind':{
                'preprocessor':exp_score_config.exp_behind_open_preprocessor_file_path,
                'model':exp_score_config.exp_behind_open_model_file_path},
            'miss':{
                'preprocessor':exp_score_config.exp_miss_open_preprocessor_file_path,
                'model':exp_score_config.exp_miss_open_model_file_path}}
    }
    
    print('Getting xScores for chains.')
    score_chains = get_expected_scores(chains, expected_scores_path_dict)
    schema_chains = convert_chains_to_schema(score_chains)
    
    print("Creating Expected VAEP Preprocessor.")
    preproc = ExpVAEPPreprocessor(expected_scores_path_dict)
    preproc.fit(chains)
    
    print("Creating modelling features.")
    exp_vaep_features = preproc.transform(chains)
    print("Creating labels.")
    exp_vaep_labels = create_gamestate_labels(schema_chains)
    exp_vaep_modelling_data = pd.concat([schema_chains, exp_vaep_features, exp_vaep_labels], axis=1)
    
    print("Generating train, test and validation sets.")
    exp_vaep_modelling_data = get_stratified_train_test_val_columns(exp_vaep_modelling_data, response="exp_scores")
    exp_vaep_modelling_data = get_stratified_train_test_val_columns(exp_vaep_modelling_data, response="exp_concedes")
    
    print("Exporting modelling data to: {}".format(model_output_path))
    exp_vaep_modelling_data.to_csv(model_output_path, index=False)
    print("Exporting preprocessor to: {}".format(preprocessor_output_path))
    joblib.dump(preproc, preprocessor_output_path)
    
if __name__ == "__main__":
    
    match_chains_to_modelling_data(chain_file_path=config.chain_file_path,
                                   model_output_path=config.exp_vaep_modelling_file_path,
                                   preprocessor_output_path=config.exp_vaep_preprocessor_file_path)
    
