import numpy as np
import pandas as pd
from highlight_text import ax_text
from expected_vaep_model.visualisation.match_report.plotting.utils import match_id_to_home_away
from expected_vaep_model.visualisation.afl_colours import team_colours

def calculate_team_average_start_distance(xchains, pitch_length=165):
    start_x = xchains.groupby('chain_number')['left_right_start_x'].first()
    start_y = xchains.groupby('chain_number')['left_right_start_y'].first()
    distances = np.sqrt((start_x - pitch_length / 2) ** 2 + start_y ** 2)
    return distances.mean()

def calculate_team_average_progression(xchains, pitch_length = 165):
    team_chains_start_loc = xchains.groupby('chain_number').first()[['left_right_start_x', 'left_right_start_y']].reset_index()
    team_chains_end_loc = xchains.groupby('chain_number').last()[['left_right_start_x', 'left_right_start_y']].reset_index()
    team_chains_start_loc['distance'] = np.sqrt((team_chains_start_loc['left_right_start_x'] - pitch_length/2)**2 + (team_chains_start_loc['left_right_start_y'])**2)
    team_chains_end_loc['distance'] = np.sqrt((team_chains_end_loc['left_right_start_x'] - pitch_length/2)**2 + (team_chains_end_loc['left_right_start_y'])**2)
    return 1 - (team_chains_end_loc['distance'] / team_chains_start_loc['distance']).mean()

def calculate_team_circulation(xchains):
    
    team_disposals = xchains[(xchains['Disposal'] == 'effective') & (xchains['action_type'] != 'shot')]
    total_disposal_distance = abs(team_disposals['start_distance_to_right_goal'] - team_disposals['end_distance_to_right_goal']).sum()

    team_progressive_disposals = team_disposals[team_disposals['start_distance_to_right_goal'] > team_disposals['end_distance_to_right_goal']]
    total_positive_disposal_distance = (team_progressive_disposals['start_distance_to_right_goal'] - team_progressive_disposals['end_distance_to_right_goal']).sum()

    return 1 - (total_positive_disposal_distance / total_disposal_distance)

def calculate_team_chain_efficiency(xchains):
    return xchains['Shot_At_Goal'].sum() / xchains['chain_number'].nunique()

def calculate_team_expected_score(xchains):
    return xchains[xchains['Shot_At_Goal'] == True]['xscore'].sum()

def calculate_team_expected_action_value(xchains):
    return xchains['xvaep'].sum()

def calculate_team_nonshot_expected_action_value(xchains):
    return xchains['nonshot_xvaep'].sum()

def calculate_team_expected_efficiency(xchains):
    return (xchains['xscore'].sum() / xchains['xvaep'].sum()).round(2)

def create_match_stats_star_metrics(xchains, match_id):
    
    match_xchains = xchains[xchains['match_id'] == match_id]
    home_team, away_team = match_id_to_home_away(match_id)
    if home_team is None or away_team is None:
        raise ValueError("Home and away teams could not be determined from the match ID.")
    if home_team not in match_xchains['Team_Chain'].unique() or away_team not in match_xchains['Team_Chain'].unique():
        raise ValueError(f"Teams {home_team} or {away_team} not found in the match data.")

    home_xchains = xchains[xchains['Team_Chain'] == home_team]
    away_xchains = xchains[xchains['Team_Chain'] == away_team]

    return {
        'home': {
            'Start Distance': calculate_team_average_start_distance(home_xchains),
            'Progression': calculate_team_average_progression(home_xchains),
            'Circulation': calculate_team_circulation(home_xchains),
            'Expected Score': calculate_team_expected_score(home_xchains),
            'Action Value': calculate_team_expected_action_value(home_xchains),
            'Expected Efficiency': calculate_team_expected_efficiency(home_xchains),
            'Non-shot EAV': calculate_team_nonshot_expected_action_value(home_xchains),
            'Chain Efficiency': calculate_team_chain_efficiency(home_xchains),
        },
        'away': {
            'Start Distance': calculate_team_average_start_distance(away_xchains),
            'Progression': calculate_team_average_progression(away_xchains),
            'Circulation': calculate_team_circulation(away_xchains),
            'Expected Score': calculate_team_expected_score(away_xchains),
            'Action Value': calculate_team_expected_action_value(away_xchains),
            'Expected Efficiency': calculate_team_expected_efficiency(away_xchains),
            'Non-shot EAV': calculate_team_nonshot_expected_action_value(away_xchains),
            'Chain Efficiency': calculate_team_chain_efficiency(away_xchains),
        },
    }
    
def calculate_league_average_metrics(xchains_all):
        
    team_start_distances = xchains_all.groupby(['match_id', 'Team_Chain']).apply(lambda x: calculate_team_average_start_distance(x))
    team_average_possession = xchains_all.groupby(['match_id', 'Team_Chain']).apply(lambda x: calculate_team_average_progression(x))
    team_circulation = xchains_all.groupby(['match_id', 'Team_Chain']).apply(lambda x: calculate_team_circulation(x))
    team_xscore = xchains_all.groupby(['match_id', 'Team_Chain']).apply(lambda x: calculate_team_expected_score(x))
    team_xvaep = xchains_all.groupby(['match_id', 'Team_Chain']).apply(lambda x: calculate_team_expected_action_value(x))
    team_nonshot_vaep = xchains_all.groupby(['match_id', 'Team_Chain']).apply(lambda x: calculate_team_nonshot_expected_action_value(x))
    team_xefficiency = xchains_all.groupby(['match_id', 'Team_Chain']).apply(lambda x: calculate_team_expected_efficiency(x))
    team_chain_efficiency = xchains_all.groupby(['match_id', 'Team_Chain']).apply(lambda x: calculate_team_chain_efficiency(x))
    
    overall_average_start_distance = team_start_distances.mean().round(1)
    overall_average_progression = team_average_possession.mean().round(2)
    overall_team_circulation = team_circulation.mean().round(2)
    overall_team_xscore = team_xscore.mean().round(1)
    overall_team_xvaep = team_xvaep.mean().round(1)
    overall_team_nonshot_vaep = team_nonshot_vaep.mean().round(1)
    overall_team_xefficiency = team_xefficiency.mean().round(2)
    overall_team_chain_efficiency = team_chain_efficiency.mean().round(2)
    
    _, start_distance_bins = pd.qcut(team_start_distances, 5, retbins=True)
    _, progression_bins = pd.qcut(team_average_possession, 5, retbins=True)
    _, circulation_bins = pd.qcut(team_circulation, 5, retbins=True)
    _, xscore_bins = pd.qcut(team_xscore, 5, retbins=True)
    _, xvaep_bins = pd.qcut(team_xvaep, 5, retbins=True)
    _, nonshot_vaep_bins = pd.qcut(team_nonshot_vaep, 5, retbins=True)
    _, xefficiency_bins = pd.qcut(team_xefficiency, 5, retbins=True)
    _, chain_efficiency_bins = pd.qcut(team_chain_efficiency, 5, retbins=True)


    return {
        'Start Distance': {
            'Average':overall_average_start_distance,
            'bins': start_distance_bins
        },
        'Progression': {
            'Average':overall_average_progression,
            'bins': progression_bins
        },
        'Circulation': {
            'Average':overall_team_circulation,
            'bins': circulation_bins
        },
        'Expected Score': {
            'Average':overall_team_xscore,
            'bins': xscore_bins
        },
        'Action Value': {
            'Average':overall_team_xvaep,
            'bins': xvaep_bins
        },
        'Expected Efficiency': {
            'Average':overall_team_xefficiency,
            'bins': xefficiency_bins
        },
        'Non-shot EAV': {
            'Average':overall_team_nonshot_vaep,
            'bins': nonshot_vaep_bins
        },
        'Chain Efficiency': {
            'Average':overall_team_chain_efficiency,
            'bins': chain_efficiency_bins
        },
    }
    
league_average_metrics = {
    'Start Distance': {
        'Average': 105.9,
        'bins': np.array([ 90.04766068, 101.77816504, 104.54213829, 107.23610987, 110.30480081, 123.71120263])},
    'Progression': {
        'Average': 0.11,
        'bins': np.array([-0.09151512,  0.06923864,  0.10207973,  0.12971038,  0.15846863, 0.31159663])},
    'Circulation': {
        'Average': 0.21,
        'bins': np.array([0.08953372, 0.17567791, 0.1967178 , 0.214262  , 0.23774259, 0.33309658])},
    'Expected Score': {
        'Average': 77.5,
        'bins': np.array([ 11.62111497,  60.84275346,  71.36173702,  81.05715714, 94.13117964, 179.49794726])},
    'Action Value': {
        'Average': 160.4,
        'bins': np.array([ 26.51407546, 131.980559  , 151.11447236, 167.41812069, 188.62267019, 304.06982916])},
    'Expected Efficiency': {
        'Average': 0.48,
        'bins': np.array([0.25, 0.43, 0.47, 0.5 , 0.54, 0.72])},
    'Non-shot EAV': {
        'Average': 82.9,
        'bins': np.array([ 13.71359503,  67.21754445,  77.3139684 ,  87.22911264, 99.0730223 , 149.12092681])},
    'Chain Efficiency': {
        'Average': 0.21,
        'bins': np.array([0.07, 0.16666667, 0.19491525, 0.22117086, 0.25217391, 0.43478261])}}

def calculate_star_rating(value, bins):
    """
    Calculate star rating based on the value and bin ranges.

    Parameters:
    - value (float): The value to evaluate.
    - bins (list or array): Array of 6 bin edges.

    Returns:
    - int: Star rating (1 to 5).
    """
    return next((i for i in range(1, 6) if bins[i - 1] <= value < bins[i]), 5)


def convert_match_metrics_to_stars(match_metrics, league_average_metrics):    
    ratings = {}
    for team in ['home', 'away']:
        ratings[team] = {}
        for stat, value in match_metrics[team].items():
            ratings[team][stat] = calculate_star_rating(value, league_average_metrics[stat]['bins'])
    return ratings

import matplotlib.pyplot as plt
def draw_match_stats_star_rating_plot_ax(ax, match_metrics, league_average_metrics, match_id):
    
    home_team, away_team = match_id_to_home_away(match_id)

    ratings = convert_match_metrics_to_stars(match_metrics, league_average_metrics)
    home_stars = [ratings['home'][metric] for metric in match_metrics['home']]
    away_stars = [ratings['away'][metric] for metric in match_metrics['away']]
    home_colour = team_colours[home_team]['primary'] if team_colours[home_team]['primary'] != "white" else team_colours[home_team]['secondary']
    away_colour = team_colours[away_team]['primary'] if team_colours[away_team]['primary'] != "white" else team_colours[away_team]['secondary']
    team_colors = [home_colour, away_colour]
    
    circle_size = 40
    circle_spacing = 0.05

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')  # Turn off actual plot axes

    for i, metric in enumerate(match_metrics['home']):

        y_pos = 0.9 - 0.1 * i
        
        # Metric label
        ax.text(0.5, 0.9-0.1*i, metric, ha='center', va='center', fontsize=32, fontweight='bold', color='black', fontname='Karla')
        
        # Home team stars
        ax_text(0.5-0.17, y_pos, 
                "<         >",
                highlight_textprops=[{
                    "bbox":{
                        "facecolor": team_colors[0] if match_metrics['home'][metric] > match_metrics['away'][metric] else "white", 
                        "edgecolor": team_colors[0], 
                        "pad":1,
                        "boxstyle": "round,pad=0.35"},
                    "color": "white" if match_metrics['home'][metric] > match_metrics['away'][metric] else team_colors[0],
                    "fontweight": "bold",
                    "font" : "Karla",
                    "ha":'center', "va":'center'
                }],
                ha='center', va='center', fontsize=32, fontweight='bold', color=team_colors[0], fontname='Karla', ax=ax)
        
        ax_text(0.5-0.17, y_pos, 
                    f"<{match_metrics['home'][metric]:.2f}>",
                    highlight_textprops=[{
                        "color": "white" if match_metrics['home'][metric] > match_metrics['away'][metric] else team_colors[0],
                        "fontweight": "bold",
                        "font" : "Karla",
                        "ha":'center', "va":'center'
                    }],
                ha='center', va='center', fontsize=32, fontweight='bold', color=team_colors[0], fontname='Karla', ax=ax)

        # Away team stars
        ax_text(0.5+0.17, y_pos, 
                "<         >",
                highlight_textprops=[{
                    "bbox":{
                        "facecolor": team_colors[1] if match_metrics['away'][metric] > match_metrics['home'][metric] else "white", 
                        "edgecolor": team_colors[1], 
                        "pad":1,
                        "boxstyle": "round,pad=0.35"},
                    "color": "white" if match_metrics['away'][metric] > match_metrics['home'][metric] else team_colors[1],
                    "fontweight": "bold",
                    "font" : "Karla",
                    "ha":'center', "va":'center'
                }],
                ha='center', va='center', fontsize=32, fontweight='bold', color=team_colors[1], fontname='Karla')
        
        ax_text(0.5+0.17, y_pos, 
                    f"<{match_metrics['away'][metric]:.2f}>", 
                    highlight_textprops=[{
                        "color": "white" if match_metrics['away'][metric] > match_metrics['home'][metric] else team_colors[1],
                        "fontweight": "bold",
                        "font" : "Karla",
                        "ha":'center', "va":'center'
                    }],
                    ha='center', va='center', fontsize=32, fontweight='bold', color=team_colors[1], fontname='Karla')

        # Home team (left)
        for d in range(5):       
            filled = (4-d) < home_stars[i]
            ax.plot(0.05 + d * circle_spacing, y_pos, 'o', markersize=circle_size,
                    color=team_colors[0] if filled else "white", 
                    markeredgecolor = team_colors[0],
                    markeredgewidth=2,
                    transform=ax.transAxes)

        # Away team (right)
        for d in range(5):
            filled = (4-d) < away_stars[i]
            ax.plot(0.95 - d * circle_spacing, y_pos, 'o', markersize=circle_size,
                    color=team_colors[1] if filled else "white", 
                    markeredgecolor = team_colors[1],
                    markeredgewidth=2,
                    transform=ax.transAxes)
                
    return ax

def plot_match_stats_star_rating_plot_ax(ax, xchains, match_id, league_average_metrics):

    match_metrics = create_match_stats_star_metrics(xchains, match_id)

    ax = draw_match_stats_star_rating_plot_ax(ax, match_metrics, league_average_metrics, match_id)
    
    return ax