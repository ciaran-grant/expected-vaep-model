import pandas as pd
import numpy as np
import joblib
from expected_vaep_model.processing.merge_xscore_to_chains import merge_xscores_to_chains

def load_preprocessor(file_name = "exp_vaep_preprocessor.joblib"):
    
    return joblib.load(f"model_outputs/preprocessors/{file_name}")

def load_scores_model(scores_file = "exp_vaep_scores.joblib"):
    
    return joblib.load(f"model_outputs/models/{scores_file}")

def load_concedes_model(concedes_file = "exp_vaep_concedes.joblib"):
    
    return joblib.load(f"model_outputs/models/{concedes_file}")

def predict_scores_concedes(chains, xscore, scores_model, concedes_model, preprocessor):

    xchains = merge_xscores_to_chains(chains, xscore)

    # Processing
    schema_chains, gamestate_features, gamestate_labels = preprocessor.transform(xchains)
    gamestate_features[['mark_a1', 'mark_a2']] = gamestate_features[['mark_a1', 'mark_a2']].astype(bool)
    schema_chains['exp_scores'] = np.clip(scores_model.predict(gamestate_features), 0, 6)
    schema_chains['exp_concedes'] = np.clip(concedes_model.predict(gamestate_features), 0, 6)
    
    schema_chains = pd.concat([schema_chains, gamestate_features, gamestate_labels], axis=1)

    return schema_chains