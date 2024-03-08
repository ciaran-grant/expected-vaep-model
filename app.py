from expected_vaep_model.predict import predict_exp_vaep
from AFLPy.AFLData_Client import upload_data

from flask import Flask, request

app = Flask(__name__)

@app.route("/model/expectedscore/predict", methods=["GET", "POST"])
def predict(ID = None):
    data = request.json
    
    data = predict_exp_vaep(data, ID = request.json['ID'])
    
    upload_data(Dataset_Name="CG_Expected_VAEP", Dataset=data, overwrite=True, update_if_identical=True)
    
    return data.to_json(orient='records')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=False)