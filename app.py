from expected_vaep_model.predict import load_preprocessor, load_scores_model, load_concedes_model, predict_scores_concedes
from expected_vaep_model.features.preprocessing import calculate_exp_vaep_values

from AFLPy.AFLData_Client import load_data, upload_data

from flask import Flask, request

app = Flask(__name__)

@app.route("/model/expectedvaep/predict", methods=["GET", "POST"])
def predict(ID = None):
    
    chains = load_data(Dataset_Name="AFL_API_Match_Chains", ID = request.json['ID'])
    xscore = load_data(Dataset_Name='CG_Expected_Score', ID = request.json['ID'])
    
    preprocessor = load_preprocessor()
    scores_model = load_scores_model()
    concedes_model = load_concedes_model()
    
    schema_chains = predict_scores_concedes(chains, xscore, scores_model, concedes_model, preprocessor)
    schema_chains = schema_chains.drop_duplicates(subset = ['Chain_Number', 'Order'])

    xvaep_chains = calculate_exp_vaep_values(schema_chains)
    
    upload_data(Dataset_Name="CG_Expected_VAEP", Dataset=xvaep_chains, overwrite=True, update_if_identical=True)
    
    return xvaep_chains.to_json(orient='records')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=False)