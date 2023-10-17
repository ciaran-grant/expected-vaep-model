import pandas as pd
import joblib
import logging
import expected_vaep_model.config as config
from expected_vaep_model.features.data_preprocessor import ExpVAEPPreprocessor
from expected_vaep_model.features.preprocessing import get_expected_scores, convert_chains_to_schema, create_gamestate_labels, get_stratified_train_test_val_columns

import sys
sys.path.append(config.exp_score_dir)

import warnings
warnings.filterwarnings('ignore')

def match_chains_to_modelling_data():
    logging.basicConfig(level=logging.INFO)
    
    logging.info('Loading chain data.')
    chains = pd.read_csv(config.chain_file_path)
    logging.info('Chain data loaded.')
    
    expected_scores_path_dict = {
        'set':{
            'goal':{
                'preprocessor':config.exp_goal_set_preprocessor_file_path,
                'model':config.exp_goal_set_model_file_path},
            'behind':{
                'preprocessor':config.exp_behind_set_preprocessor_file_path,
                'model':config.exp_behind_set_model_file_path},
            'miss':{
                'preprocessor':config.exp_miss_set_preprocessor_file_path,
                'model':config.exp_miss_set_model_file_path}},
        'open':{
            'goal':{
                'preprocessor':config.exp_goal_open_preprocessor_file_path,
                'model':config.exp_goal_open_model_file_path},
            'behind':{
                'preprocessor':config.exp_behind_open_preprocessor_file_path,
                'model':config.exp_behind_open_model_file_path},
            'miss':{
                'preprocessor':config.exp_miss_open_preprocessor_file_path,
                'model':config.exp_miss_open_model_file_path}}
    }
    
    logging.info('Getting xScores for chains.')
    score_chains = get_expected_scores(chains, expected_scores_path_dict)
    schema_chains = convert_chains_to_schema(score_chains)
    
    logging.info("Creating Expected VAEP Preprocessor.")
    preproc = ExpVAEPPreprocessor(expected_scores_path_dict)
    preproc.fit(chains)
    
    logging.info("Creating modelling features.")
    exp_vaep_features = preproc.transform(chains)
    logging.info("Creating labels.")
    exp_vaep_labels = create_gamestate_labels(schema_chains)
    exp_vaep_modelling_data = pd.concat([schema_chains, exp_vaep_features, exp_vaep_labels], axis=1)
    
    logging.info("Generating train, test and validation sets.")
    exp_vaep_modelling_data = get_stratified_train_test_val_columns(exp_vaep_modelling_data, response="exp_scores")
    exp_vaep_modelling_data = get_stratified_train_test_val_columns(exp_vaep_modelling_data, response="exp_concedes")
    
    logging.info("Exporting modelling data to: {}".format(config.exp_vaep_modelling_file_path))
    exp_vaep_modelling_data.to_csv(config.exp_vaep_modelling_file_path, index=False)
    logging.info("Exporting preprocessor to: {}".format(config.exp_vaep_preprocessor_file_path))
    joblib.dump(preproc, config.exp_vaep_preprocessor_file_path)
    
if __name__ == "__main__":
    match_chains_to_modelling_data()
    
