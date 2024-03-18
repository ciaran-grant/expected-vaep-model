from expected_vaep_model.predict.predict_exp_vaep import load_preprocessor, load_scores_model, load_concedes_model, create_features
from expected_vaep_model.features.preprocessing import calculate_exp_vaep_values

from AFLPy.AFLData_Client import load_data, upload_data

from flask import Flask, request

app = Flask(__name__)

@app.route("/model/expectedvaep/predict", methods=["GET", "POST"])
def predict(ID = None):
    
    chains = load_data(Dataset_Name="AFL_API_Match_Chains", ID = request.json['ID'])
    shots = load_data(Dataset_Name='CG_Expected_Scores', ID = request.json['ID'])
    
    preprocessor = load_preprocessor()
    scores_model = load_scores_model()
    concedes_model = load_concedes_model()
    
    schema_chains = create_features(chains, shots, scores_model, concedes_model, preprocessor)

    data = calculate_exp_vaep_values(schema_chains) 
    
    # upload_data(Dataset_Name="CG_Expected_VAEP", Dataset=data, overwrite=True, update_if_identical=True)
    
    return data.to_json(orient='records')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=False)