
from matplotlib.gridspec import GridSpecFromSubplotSpec
from mplfooty.pitch import VerticalPitch
from highlight_text import ax_text
from expected_vaep_model.visualisation.match_report.plotting.utils import match_id_to_home_away
from expected_vaep_model.visualisation.afl_colours import team_colours

def plot_expected_score_map_ax(ax, xchains, team):
    
    pitch = VerticalPitch(pitch_length=165, pitch_width=135, line_width=1, line_zorder=1, line_colour = 'grey', half = True, pad_bottom =0)
    pitch.draw(ax=ax)
    colour = team_colours[team]['primary'] if team_colours[team]['primary'] != "white" else team_colours[team]['secondary']

    goals = xchains[(xchains['team'] == team) & (xchains['Shot_At_Goal'] == True) & (xchains['Final_State'] == 'goal')]
    pitch.scatter(goals['left_right_start_x'], goals['left_right_start_y'], ax=ax,
                color = colour, s = goals['xscore']*50, label = team)
    for i, row in goals.iterrows():
        pitch.plot([row['x'], pitch.pitch_length/2], [row['y'], 0], color=colour, alpha=1, linewidth=1, ax=ax)

    behinds = xchains[(xchains['team'] == team) & (xchains['Shot_At_Goal'] == True) & (xchains['Final_State'] == 'behind')]
    pitch.scatter(behinds['left_right_start_x'], behinds['left_right_start_y'], ax=ax,
                color = 'white', s = behinds['xscore']*50, label = team, edgecolor = colour, linewidth = 2)
    
    misses = xchains[(xchains['team'] == team) & (xchains['Shot_At_Goal'] == True) & (xchains['Final_State'] != 'goal') & (xchains['Final_State'] != 'behind')]
    pitch.scatter(misses['left_right_start_x'], misses['left_right_start_y'], ax=ax,
                color = 'white', s = misses['xscore']*50, label = team, edgecolor = colour, linewidth = 2, alpha = 0.5)
    
    total = (goals['xscore'].sum() + behinds['xscore'].sum() + misses['xscore'].sum()).round(1)
    opponent_goals = xchains[(xchains['team'] != team) & (xchains['Shot_At_Goal'] == True) & (xchains['Final_State'] == 'goal')]
    opponent_behinds = xchains[(xchains['team'] != team) & (xchains['Shot_At_Goal'] == True) & (xchains['Final_State'] == 'behind')]
    opponent_misses = xchains[(xchains['team'] != team) & (xchains['Shot_At_Goal'] == True) & (xchains['Final_State'] != 'goal') & (xchains['Final_State'] != 'behind')]
    opponent_total = (opponent_goals['xscore'].sum() + opponent_behinds['xscore'].sum() + opponent_misses['xscore'].sum()).round(1)
    ax_text(ax=ax, 
            x=0, 
            y=-20, s=f"<{total} xs>",
            highlight_textprops=[{
                "bbox":{
                    "facecolor": colour if total > opponent_total else "white", 
                    "edgecolor": colour, 
                    "pad":1,
                    "boxstyle": "round,pad=0.3"},
                "color": "white" if total > opponent_total else colour,
                "fontweight": "bold",
                "font" : "Karla",
                "ha":'left', "va":'center'
            }], 
            fontsize=50, 
            fontweight='bold', color='black', fontname='Karla', ha='center', va='center')
    
    return ax

def create_expected_shot_maps_both_teams_ax(ax, xchains):
    fig = ax.figure
    parent_spec = ax.get_subplotspec()  # Get GridSpec slice from the Dashboard layout
    ax.remove()  # Remove the placeholder Axes

    # Create a nested GridSpec with 1 row and 3 columns
    inner_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=parent_spec, wspace=0, width_ratios=[1, 1])

    # Create axes for possession bar, pitch density, and field tilt
    ax_home = fig.add_subplot(inner_gs[0, 0])
    ax_away = fig.add_subplot(inner_gs[0, 1])
    
    match_id = xchains['match_id'].iloc[0]
    home_team, away_team = match_id_to_home_away(match_id)
    
    ax_home = plot_expected_score_map_ax(ax=ax_home, xchains=xchains, team=home_team)
    ax_away = plot_expected_score_map_ax(ax=ax_away, xchains=xchains, team=away_team)

    return [ax_home, ax_away]