from expected_vaep_model.processing.calculate_exp_vaep import calculate_exp_vaep_values
from expected_vaep_model.predict.predict import predict_scores_concedes
from expected_vaep_model.predict.predict import load_preprocessor, load_scores_model, load_concedes_model
from expected_vaep_model.visualisation.plot_team_rolling_averages import create_team_rolling, plot_team_rolling_ax, plot_all_team_rolling_figure
from AFLPy.AFLData_Client import load_data, upload_data
from expected_vaep_model.fonts.fonts import load_fonts

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot

import matplotlib.pyplot as plt
from flask import Flask, request, send_file
import io
import os

app = Flask(__name__)

@app.route("/model/expectedvaep/predict", methods=["GET", "POST"])
def predict(ID = None):
    
    chains = load_data(Dataset_Name="AFL_API_Match_Chains", ID = request.json['ID'])
    xscore = load_data(Dataset_Name='CG_Expected_Score', ID = request.json['ID'])
    
    preprocessor = load_preprocessor("exp_vaep_preprocessor.joblib")
    scores_model = load_scores_model("exp_vaep_scores.joblib")
    concedes_model = load_concedes_model("exp_vaep_concedes.joblib")
    
    schema_chains = predict_scores_concedes(chains, xscore, scores_model, concedes_model, preprocessor)
    schema_chains = schema_chains.drop_duplicates(subset = ['chain_number', 'order'])

    xvaep_chains = calculate_exp_vaep_values(schema_chains)
    
    upload_data(Dataset_Name="CG_Expected_VAEP", Dataset=xvaep_chains, overwrite=True, update_if_identical=True)
    
    return xvaep_chains.to_json(orient='records')

@app.route("/model/expectedvaep/plot_team_rolling_xscore", methods=["GET", "POST"])
def plot_team_rolling_xvaep():
    # Get data from request body
    data = request.get_json()  # Ensure the request has JSON
    if not data or 'team' not in data:
        return {"error": "Missing 'team' in request body"}, 400

    team = data['team']
    window = data.get('window', 10)  # Default to 10 if not provided
    metric = data.get('metric', 'exp_vaep_value')
    annotate = data.get('annotate', True)
    years = data.get('years', None)
    
    dpi = data.get('dpi', 300)
    figsize = data.get('figsize', (10, 6))
    style = data.get('style', 'rolling_dark')

    # Load chains with expected VAEP data
    xchains = load_data(Dataset_Name = "CG_Expected_VAEP", ID = [team])
    xchains['Year'] = xchains['Match_ID'].apply(lambda x: int(x.split("_")[1]))
    xchains['Round'] = xchains['Match_ID'].apply(lambda x: x.split("_")[2])

    team_rolling = create_team_rolling(xchains, team, window, metric=metric)

    # Generate plot
    style_path = os.path.join(os.path.dirname(__file__), 'src', 'expected_vaep_model', 'visualisation', 'styles', f'{style}.mplstyle')
    plt.style.use(style_path)

    font_path = os.path.join(os.path.dirname(__file__), 'src', 'expected_vaep_model', 'fonts')
    load_fonts(font_path)

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax = plot_team_rolling_ax(ax=ax, team=team, team_rolling=team_rolling, annotate=annotate, years=years)

    # Convert plot to image
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)  # Free memory

    return send_file(img, mimetype='image/png')

@app.route("/model/expectedvaep/plot_all_team_rolling_xscore", methods=["POST"])
def plot_all_team_rolling_xvaep():
    data = request.get_json()  # Ensure the request has JSON

    window = data.get('window', 10)  # Default to 10 if not provided
    metric = data.get('metric', 'exp_vaep_value')
    annotate = data.get('annotate', True)
    add_title = data.get('add_title', True)
    years = data.get('years', None)
    
    dpi = data.get('dpi', 300)
    style = data.get('style', 'rolling_dark')

    # Load shots with xscore
    ID = years if years is not None else "AFL"
    xchains = load_data(Dataset_Name = "CG_Expected_VAEP", ID = ID)
    xchains['Year'] = xchains['Match_ID'].apply(lambda x: int(x.split("_")[1]))
    xchains['Round'] = xchains['Match_ID'].apply(lambda x: x.split("_")[2])

    style_path = os.path.join(os.path.dirname(__file__), 'src', 'expected_vaep_model', 'visualisation', 'styles', f'{style}.mplstyle')
    plt.style.use(style_path)
    
    font_path = os.path.join(os.path.dirname(__file__), 'src', 'expected_vaep_model', 'fonts')
    load_fonts(font_path)

    fig, ax = plot_all_team_rolling_figure(
        xchains, 
        window=window, 
        metric = metric, 
        annotate = annotate,
        add_title=add_title,
        years=years)
    fig.set_dpi(dpi)

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=False)