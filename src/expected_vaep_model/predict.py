import pandas as pd
import numpy as np
import joblib
from expected_vaep_model.processing.merge_xscore_to_chains import merge_xscores_to_chains

def load_preprocessor():
    
    return joblib.load("model_outputs/preprocessors/exp_vaep_preprocessor.joblib")

def load_scores_model():
    
    return joblib.load("model_outputs/models/exp_vaep_scores.joblib")

def load_concedes_model():
    
    return joblib.load("model_outputs/models/exp_vaep_concedes.joblib")

def predict_scores_concedes(chains, xscore, scores_model, concedes_model, preprocessor):

    xchains = merge_xscores_to_chains(chains, xscore)

    # Processing
    schema_chains, gamestate_features, gamestate_labels = preprocessor.transform(xchains)
    schema_chains['exp_scores'] = np.clip(scores_model.predict(gamestate_features), 0, 6)
    schema_chains['exp_concedes'] = np.clip(concedes_model.predict(gamestate_features), 0, 6)
    
    schema_chains = pd.concat([schema_chains, gamestate_features, gamestate_labels], axis=1)

    return schema_chains