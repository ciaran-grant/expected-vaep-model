from preprocessing import get_expected_scores, convert_chains_to_schema, create_gamestate_features
from modelling_data_contract import ModellingDataContract

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

# from exp_vaep.domain.contracts.mappings import Mappings

class ExpVAEPPreprocessor(BaseEstimator, TransformerMixin):
    """ Preprocessing class and functions for training total game score model.
    """
    
    def __init__(self, expected_scores_path_dict):
        """ Specify mappings and rolling average columns to create.

        Args:
            Mappings (Mappings): Mappings object specifying mapping and transformations.
            rolling_dict (Dict): Dictionary specifying columns and types of rolling average columns.
        """
        self.ModellingDataContract = ModellingDataContract
        self.expected_scores_path_dict = expected_scores_path_dict
       
        
    def fit(self, X):
        """ Fits preprocessor to training data.
            Learns expected columns and mean imputations. 

        Args:
            X (Dataframe): Training dataframe to fit preprocessor to.

        Returns:
            self: Preprocessor learns expected colunms and means to impute.
        """
        
        # Keep only modelling columns
        self.modelling_cols = ModellingDataContract.feature_list_scores
                        
        return self
    
    def transform(self, X):
        """ Applies transformations and preprocessing steps to dataframe.

        Args:
            X (Dataframe): Training or unseen data to transform.

        Returns:
            Dataframe: Transformed data with modelling columns and no missing values.
        """
        
        # Get xScores
        X = get_expected_scores(X, self.expected_scores_path_dict)
        
        # Get Expected VAEP
        X_schema = convert_chains_to_schema(X)
        
        X_features = create_gamestate_features(X_schema)
        X_features = X_features[self.modelling_cols]
                
        return X_features
    