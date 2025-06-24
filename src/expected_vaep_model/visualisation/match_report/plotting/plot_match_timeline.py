from expected_vaep_model.visualisation.match_report.plotting.utils import match_id_to_home_away
from expected_vaep_model.visualisation.afl_colours import team_colours
import numpy as np

def create_match_timeline_data(xchains):
    
    
    match_id = list(xchains['match_id'].unique())[0]
    home_team, away_team = match_id_to_home_away(match_id)
    
    period_seconds_map = xchains.groupby('period')['period_seconds'].max().cumsum().shift(1).fillna(0).to_dict()
    xchains['seconds'] = xchains['period'].map(period_seconds_map) + xchains['period_seconds']
    xchains['minutes'] = xchains['seconds'] // 60

    match_timeline_data = xchains.groupby(['minutes', 'team']).agg(
        xvaep = ('xvaep', 'sum'),
        nonshot_xvaep = ('nonshot_xvaep', 'sum')
    ).pivot_table(index='minutes', columns='team', values='xvaep', aggfunc='sum').fillna(0)
    
    match_timeline_data[f"{home_team}_cs"] = match_timeline_data[home_team].cumsum()
    match_timeline_data[f"{away_team}_cs"] = match_timeline_data[away_team].cumsum()
    
    match_timeline_data['diff'] = match_timeline_data[f"{home_team}"] - match_timeline_data[f"{away_team}"]
    match_timeline_data['diff_cs'] = match_timeline_data[f"{home_team}_cs"] - match_timeline_data[f"{away_team}_cs"]
    match_timeline_data['rolling_diff'] = match_timeline_data['diff'].rolling(5, min_periods=0).mean()
    
    score_timeline_data = xchains.groupby(['minutes', 'team']).agg(
        score = ('score', 'sum'),
    ).pivot_table(index='minutes', columns='team', values='score', aggfunc='sum').fillna(0)
    score_timeline_data = score_timeline_data.rename(columns={home_team: f"{home_team}_score", away_team: f"{away_team}_score"})
    
    return match_timeline_data.merge(score_timeline_data, left_index=True, right_index=True)

def plot_match_timeline_ax(ax, xchains):
    
    match_id = list(xchains['match_id'].unique())[0]
    match_timeline_data = create_match_timeline_data(xchains)
    
    home_team_data = np.clip(match_timeline_data['diff'], 0, None)
    away_team_data = np.clip(match_timeline_data['diff'], None, 0)
    
    home_team, away_team = match_id_to_home_away(match_id)
    
    home_team_colour = team_colours[home_team]['primary'] if team_colours[home_team]['primary'] != "white" else team_colours[home_team]['secondary']
    away_team_colour = team_colours[away_team]['primary'] if team_colours[away_team]['primary'] != "white" else team_colours[away_team]['secondary']
    
    ax.bar(match_timeline_data.index, home_team_data, color = home_team_colour, alpha=0.5, label=home_team)
    ax.bar(match_timeline_data.index, away_team_data, color = away_team_colour, alpha=0.5, label=home_team)
    
    ax.set_ylim(-11, 11)
    ax.axis('off')
    return ax
