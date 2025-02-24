from expected_vaep_model.processing.convert_to_schema import convert_chains_to_schema
from expected_vaep_model.features.gamestate_features import create_gamestate_features
from expected_vaep_model.features.response import create_gamestate_labels

class ExpVAEPPreprocessor:
    
    def __init__(self):
        pass
    
    def fit(self, y=None):
        pass
    
    def transform(self, X):
        X_schema = convert_chains_to_schema(X)
        features = create_gamestate_features(X_schema)
        labels = create_gamestate_labels(X_schema)
        return X_schema, features, labels