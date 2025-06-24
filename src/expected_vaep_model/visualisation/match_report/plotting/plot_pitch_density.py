
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from mplfooty.pitch import Pitch
from expected_vaep_model.visualisation.match_report.plotting.utils import match_id_to_home_away, min_max_normalize
from expected_vaep_model.visualisation.afl_colours import team_colours

def plot_pitch_density_ax(ax, xchains, X, Y):
    
    match_id = list(xchains['match_id'].unique())[0]
    home_team, away_team = match_id_to_home_away(match_id)

    home_x = xchains[xchains['team'] == home_team]['left_right_start_x']
    home_y = xchains[xchains['team'] == home_team]['left_right_start_y']
    away_x = xchains[xchains['team'] == away_team]['left_right_start_x']
    away_y = xchains[xchains['team'] == away_team]['left_right_start_y']
    
    grid_coords = np.vstack([X.ravel(), Y.ravel()])

    home_kde = gaussian_kde(np.vstack([home_x, home_y]))
    away_kde = gaussian_kde(np.vstack([away_x, away_y]))

    home_z = home_kde(grid_coords).reshape(X.shape)
    away_z = away_kde(grid_coords).reshape(X.shape)

    z = min_max_normalize(home_z - away_z)
    df = pd.DataFrame({
        'x': X.ravel(),
        'y': Y.ravel(),
        'home_z': home_z.ravel(),
        'away_z': away_z.ravel(),
        'z': z.ravel(),
    })   
    
    # Create custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [team_colours[away_team]['primary'], team_colours[home_team]['primary']], N=256)

    home_team_colour = team_colours[home_team]['primary'] if team_colours[home_team]['primary'] != "white" else team_colours[home_team]['secondary']
    away_team_colour = team_colours[away_team]['primary'] if team_colours[away_team]['primary'] != "white" else team_colours[away_team]['secondary']
    custom_cmap = LinearSegmentedColormap.from_list(
        "extreme_diverge",
        [away_team_colour, "white", home_team_colour]
    )
    
    # Plot with seaborn
    pitch = Pitch(pitch_width=135, pitch_length=165, line_zorder=2, line_colour = 'black', pad_bottom=1, pad_top = 1)
    pitch.draw(ax=ax, tight_layout=False)
    pitch.kdeplot(x=df['x'], y=df['y'], weights=df['z'], cmap=custom_cmap, ax=ax, fill=True, levels=100)

    return ax