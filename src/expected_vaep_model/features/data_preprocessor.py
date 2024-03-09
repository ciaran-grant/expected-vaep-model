from expected_vaep_model.features.preprocessing import get_expected_scores, convert_chains_to_schema, create_gamestate_features
from expected_vaep_model.modelling_data_contract import ModellingDataContract

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

# from exp_vaep.domain.contracts.mappings import Mappings

class ExpVAEPPreprocessor(BaseEstimator, TransformerMixin):
    """ Preprocessing class and functions for training total game score model.
    """
    
    def __init__(self):
        """
        """
        self.ModellingDataContract = ModellingDataContract
       
        
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
        
        # Get Expected VAEP
        X_schema = convert_chains_to_schema(X)

        return create_gamestate_features(X_schema)
    