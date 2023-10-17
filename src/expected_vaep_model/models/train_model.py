import pandas as pd
from expected_vaep_model.modelling_data_contract import ModellingDataContract
from expected_vaep_model.models.hyperparameter_tuning import XGBHyperparameterTuner
from expected_vaep_model.models.supermodel import SuperXGBRegressor
from expected_vaep_model.models.optuna_xgb_param_grid import OptunaXGBParamGrid

def train_model(input_file_path, output_file_path, target, model_version):

    model_name = 'exp_vaep_'+target
    model_file_name = model_name + '_v' + str(model_version)

    if target == "scores":
        RESPONSE = ModellingDataContract.RESPONSE_SCORES
        FEATURES = ModellingDataContract.feature_list_scores
        MONOTONE_CONSTRAINTS = ModellingDataContract.monotone_constraints_scores
    else:
        RESPONSE = ModellingDataContract.RESPONSE_CONCEDES
        FEATURES = ModellingDataContract.feature_list_concedes
        MONOTONE_CONSTRAINTS = ModellingDataContract.monotone_constraints_concedes

    print('Loading modelling data.')        
    df_modelling = pd.read_csv(input_file_path)
    
    training_data = df_modelling[(df_modelling[RESPONSE+"TrainingSet"]) | (df_modelling[RESPONSE+"ValidationSet"])]
    test_data = df_modelling[df_modelling[RESPONSE+"TestSet"]]

    X_train, y_train = training_data.drop(columns=[RESPONSE]), training_data[RESPONSE]
    X_test, y_test = test_data.drop(columns=[RESPONSE]), test_data[RESPONSE]
    
    X_train_preproc = X_train[FEATURES]
    X_test_preproc = X_test[FEATURES]

    print('Starting hyperparameter tuning.') 
    xgb_tuner = XGBHyperparameterTuner(X_train_preproc, y_train, monotonicity_constraints=MONOTONE_CONSTRAINTS)
    xgb_tuner.tune_hyperparameters()
    print('Hyperparameter tuning complete.') 

    params = xgb_tuner.get_best_params()
    params['objective'] = OptunaXGBParamGrid.error
    params['num_rounds'] = OptunaXGBParamGrid.num_rounds
    params['early_stopping_rounds'] = OptunaXGBParamGrid.early_stopping_rounds
    params['verbosity'] = OptunaXGBParamGrid.verbosity
    params['monotone_constraints'] = MONOTONE_CONSTRAINTS
    
    print('Fitting model.') 
    super_xgb = SuperXGBRegressor(X_train = X_train_preproc, 
                               y_train = y_train, 
                               X_test = X_test_preproc, 
                               y_test = y_test,
                               params = params)
    super_xgb.fit()
    
    print('Exporting model to: {}'.format(output_file_path)) 
    super_xgb.export_model(output_file_path + "/" + model_file_name + ".joblib")
    
