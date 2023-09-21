from visualization.afl_colours import team_colours
import numpy as np

action_map = {
    'Hard Ball Get': "Hard Get",
    'Handball': "Handball",
    'Loose Ball Get': "Loose Get",
    'Carry': "Carry",
    'Kick': "Kick",
    'Gather':"Gather",
    'Shot':"Shot",
    'Spoil':'Spoil',
    'Knock On':'Knock On',
    'Uncontested Mark':'U. Mark',
    'Contested Mark':'C. Mark',
    'Free For':'Free',
    'Error':"Error",
    'Tackle':'Tackle'
}
team_map = {
    'Brisbane Lions': "BRI",
    'Sydney':"SYD",
    'Western Bulldogs':'WBD',
    'Collingwood':"COL",
    'Hawthorn':"HAW",
    'Essendon':"ESS",
    'Greater Western Sydney':'GWS',
    'St Kilda':'STK',
    'Fremantle':'FRE',
    'Melbourne':'MEL',
    'Port Adelaide':'POR',
    'North Melbourne':'NOR',
    'Richmond':'RIC',
    'Carlton':'CAR',
    'West Coast':"WST",
    'Gold Coast':'GCT',
    'Geelong':"GEE",
    'Adelaide':"ADE"
}

def get_chain(chain_data, match_id, chain_number):
    
    return chain_data[(chain_data['Match_ID'] == match_id) & (chain_data['Chain_Number'] == chain_number)]

def get_venue_dimensions(chain, match_id):
    
    # Get match chain information
    match = chain[chain['Match_ID'] == match_id]

    return list(set(match['Venue_Length']))[0], list(set(match['Venue_Width']))[0]

def get_team(chain):
    
    return list(set(chain['Team_Chain']))[0]

def get_team_colour(team):
    
    return team_colours[team]['positive']

def plot_chain_vaep(chain, pitch, ax):
    
    chain_copy = chain.copy()
    
    # Create new end_x, end_y back for sake of visualising if Chain_Team not same as Team
    chain_copy['x_1'] = chain_copy['x'].shift(-1).fillna(0)
    chain_copy['y_1'] = chain_copy['y'].shift(-1).fillna(0)
    
    # If shot, get end location
    if ((chain_copy['Shot_At_Goal'] > 0).any()) | (len(chain_copy[chain_copy['Description'] == "Shot"])):
        if len(chain_copy[chain_copy['Description'] == "Goal"]) > 0:
            shot_end_x = pitch.dim.right
            shot_end_y = 0
        elif len(chain_copy[chain_copy['Description'] == "Behind"]) > 0:
            behind_detail = chain_copy[chain_copy['Description'] == "Behind"]['Behind_Detail']
            if (behind_detail == "missLeft").any():
                shot_end_x = pitch.dim.right
                shot_end_y = pitch.dim.goal_width
            else:
                shot_end_x = pitch.dim.right
                shot_end_y = -1*(pitch.dim.goal_width)
    # if turnover at the end, set end x/y to same location (to avoid flipping the pitch)
    if list(set(chain_copy['Final_State'] == "turnover"))[0]:
        chain_copy.iloc[-1, chain_copy.columns.get_loc('x_1')] = chain_copy.iloc[-1, chain_copy.columns.get_loc('x')]
        chain_copy.iloc[-1, chain_copy.columns.get_loc('y_1')] = chain_copy.iloc[-1, chain_copy.columns.get_loc('y')]

    # Reset order to only include SPADL action_types
    chain_spadl = chain_copy[~chain_copy['action_type'].isnull()].reset_index(drop=True)
    chain_spadl['order'] = list(range(1, chain_spadl.shape[0]+1))

    # Round floats
    chain_spadl = round(chain_spadl, 3)
    
    # Get colours
    teamA = get_team(chain)
    teamB = teamA

    if len(set(chain_spadl['Team'])) > 1:
        teamB = list(set(chain_spadl['Team']) - set([teamA]))[0]
    # Plot action locations
    for team in [teamA, teamB]: 
        carries = chain_spadl[(chain_spadl['action_type'] == "Carry") & (chain_spadl['Team'] == team)]
        pitch.scatter(carries['x'], carries['y'], ax=ax, marker=".", color = get_team_colour(team), zorder=2)

        handballs = chain_spadl[(chain_spadl['action_type'] == "Handball") & (chain_spadl['Team'] == team)]
        pitch.scatter(handballs['x'], handballs['y'], ax=ax, marker=".", color = get_team_colour(team), zorder=2)
        
        knock_on = chain_spadl[(chain_spadl['action_type'] == "Knock On") & (chain_spadl['Team'] == team)]
        pitch.scatter(knock_on['x'], knock_on['y'], ax=ax, marker=".", color = get_team_colour(team), zorder=2)

        kicks = chain_spadl[(chain_spadl['action_type'] == "Kick") & (chain_spadl['Team'] == team)]
        pitch.scatter(kicks['x'], kicks['y'], ax=ax, marker="o", color = get_team_colour(team),zorder=2) # make line an arrow

        free = chain_spadl[(chain_spadl['action_type'] == "Free For") & (chain_spadl['Team'] == team)]
        pitch.scatter(free['x'], free['y'], ax=ax, marker="o", color = get_team_colour(team),zorder=2)

        shot = chain_spadl[(chain_spadl['action_type'] == "Shot") & (chain_spadl['Team'] == team)]
        pitch.scatter(shot['x'], shot['y'], ax=ax, marker="o", edgecolors="w", color = get_team_colour(team),zorder=2)

        uncontested_mark = chain_spadl[(chain_spadl['action_type'] == "Uncontested Mark") & (chain_spadl['Team'] == team)]
        pitch.scatter(uncontested_mark['x'], uncontested_mark['y'], ax=ax, marker="s", color = get_team_colour(team),zorder=2)

        contested_mark = chain_spadl[(chain_spadl['action_type'] == "Contested Mark") & (chain_spadl['Team'] == team)]
        pitch.scatter(contested_mark['x'], contested_mark['y'], ax=ax, marker="s", color = get_team_colour(team),zorder=2)

        loose_ball_get = chain_spadl[(chain_spadl['action_type'] == "Loose Ball Get") & (chain_spadl['Team'] == team)]
        pitch.scatter(loose_ball_get['x'], loose_ball_get['y'], ax=ax, marker="^", color = get_team_colour(team),zorder=2)

        hard_ball_get = chain_spadl[(chain_spadl['action_type'] == "Hard Ball Get") & (chain_spadl['Team'] == team)]
        pitch.scatter(hard_ball_get['x'], hard_ball_get['y'], ax=ax, marker="^", color = get_team_colour(team),zorder=2)

        gather = chain_spadl[(chain_spadl['action_type'] == "Gather") & (chain_spadl['Team'] == team)]
        pitch.scatter(gather['x'], gather['y'], ax=ax, marker="^", color = get_team_colour(team),zorder=2)

        spoil = chain_spadl[(chain_spadl['action_type'] == "Spoil") & (chain_spadl['Team'] == team)]
        pitch.scatter(spoil['x'], spoil['y'], ax=ax, marker="X", color = get_team_colour(team),zorder=2)

        error = chain_spadl[(chain_spadl['action_type'] == "Error") & (chain_spadl['Team'] == team)]
        pitch.scatter(error['x'], error['y'], ax=ax, marker="x", color = get_team_colour(team),zorder=2)

        tackle = chain_spadl[(chain_spadl['action_type'] == "Tackle") & (chain_spadl['Team'] == team)]
        pitch.scatter(tackle['x'], tackle['y'], ax=ax, marker="x", color = get_team_colour(team),zorder=2)

    for i, row in chain_spadl.iterrows():
        if (row['action_type'] == "Carry"):
            pitch.annotate(text="", xytext = (row['x'], row['y']), xy=(row['x_1'], row['y_1']), 
                        ha='center', va='center', ax=ax, fontsize=4, zorder=1,
                        arrowprops=dict(color="white", arrowstyle = "-", linestyle = "--"))
            pitch.annotate(text = row['order'], xy = (row['x'], row['y']+3), ax=ax, fontsize = 5, fontweight="bold")
        elif (row['action_type'] == "Kick"):
            pitch.annotate(text="", xytext = (row['x'], row['y']), xy=(row['x_1'], row['y_1']), 
                        ha='center', va='center', ax=ax, fontsize=4, zorder=1,
                        arrowprops=dict(color="white", arrowstyle = "->"))
            pitch.annotate(text = row['order'], xy = (row['x'], row['y']+3), ax=ax, fontsize = 5, fontweight="bold")
        elif (row['action_type'] == "Shot"):
            pitch.annotate(text="", xytext = (row['x'], row['y']), xy=(shot_end_x, shot_end_y), 
                        ha='center', va='center', ax=ax, fontsize=4, zorder=1,
                        arrowprops=dict(color="white", arrowstyle = "->"))
            pitch.annotate(text = row['order'], xy = (row['x'], row['y']+3), ax=ax, fontsize = 5, fontweight="bold")
        else:
            pitch.annotate(text="", xytext = (row['x'], row['y']), xy=(row['x_1'], row['y_1']), 
                        ha='center', va='center', ax=ax, fontsize=4, zorder=1,
                        arrowprops=dict(color="white", arrowstyle = "-"))
            pitch.annotate(text = row['order'], xy = (row['x'], row['y']+3), ax=ax, fontsize = 5, fontweight="bold")

    return ax

def add_chain_table_vaep(chain, fig, left, bottom, width, height):
    
    # Reset order to only include SPADL action_types
    chain_spadl = chain[~chain['action_type'].isnull()].reset_index(drop=True)
    chain_spadl['order'] = list(range(1, chain_spadl.shape[0]+1))
    chain_spadl = round(chain_spadl, 2)
    chain_spadl['Table_Team'] = chain_spadl['Team'].replace(team_map)
    chain_spadl['action_type'] = chain_spadl['action_type'].replace(action_map)

    # Get colours
    team = get_team(chain)
    team_colour = get_team_colour(team)

    ncols = 9
    nrows = chain_spadl['order'].max()
    fontsize = 5
    
    ax_table = fig.add_axes((left, bottom, width, height))

    ax_table.set_xlim(0, ncols+1)
    ax_table.set_ylim(0, nrows+1)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax_table.spines[spine].set_visible(False)
    ax_table.tick_params(left=False)
    ax_table.tick_params(labelleft=False)
    ax_table.tick_params(bottom=False)
    ax_table.tick_params(labelbottom=False)
    ax_table.grid(False)

    positions = [0.5, 1.5, 3, 4.75, 6.25, 7.5, 8.5, 9.5]
    columns = ['order', 'marker', 'action_type', 'Player', 'Table_Team', 'exp_offensive_value', 'exp_defensive_value', 'exp_vaep_value']

    for y in range(nrows):
        for j, column in enumerate(columns):
            if column == "marker":
                if chain_spadl['action_type'][y] in ['Spoil', 'Error', 'Tackle']:
                    ax_table.scatter(positions[j], nrows-y, marker="x", color = get_team_colour(chain_spadl['Team'][y]),zorder=2)
                if chain_spadl['action_type'][y] in ['Handball', 'Carry', 'Knock On']:
                    ax_table.scatter(positions[j], nrows-y, marker=".", color = get_team_colour(chain_spadl['Team'][y]),zorder=2)
                if chain_spadl['action_type'][y] in ['Kick', 'Free']:
                    ax_table.scatter(positions[j], nrows-y, marker="o", color = get_team_colour(chain_spadl['Team'][y]),zorder=2)
                if chain_spadl['action_type'][y] in ['Shot']:
                    ax_table.scatter(positions[j], nrows-y, marker="o", edgecolors="w", color = get_team_colour(chain_spadl['Team'][y]),zorder=2)
                if chain_spadl['action_type'][y] in ['U. Mark', 'C. Mark']:
                    ax_table.scatter(positions[j], nrows-y, marker="s", color = get_team_colour(chain_spadl['Team'][y]),zorder=2)                                  
                if chain_spadl['action_type'][y] in ['Loose Get', 'Hard Get', 'Gather']:
                    ax_table.scatter(positions[j], nrows-y, marker="^", color = get_team_colour(chain_spadl['Team'][y]),zorder=2)                   
            else:
                ax_table.annotate(xy=(positions[j], nrows-y), text=chain_spadl[column][y], ha='center', color="w", fontsize=fontsize)

    column_names = ['Order', 'Marker', 'Action', 'Player', 'Team', 'O.V', 'D.V', 'E-VAEP']
    for index, c in enumerate(column_names):
        ax_table.annotate(xy=(positions[index], nrows+1), text=column_names[index], weight="bold", ha="center", color="w", fontsize=fontsize)

    ax_table.plot([ax_table.get_xlim()[0], ax_table.get_xlim()[1]], [nrows+0.6, nrows+0.6], lw=1, color='w', marker='', zorder=4)
    ax_table.plot([ax_table.get_xlim()[0], ax_table.get_xlim()[1]], [0.6, 0.6], lw=1, color='w', marker='', zorder=4)
    for x in range(1, nrows):
        ax_table.plot([ax_table.get_xlim()[0], ax_table.get_xlim()[1]], [x+0.6, x+0.6], lw=0.5, color='grey', ls=':', zorder=3 , marker='')

    return ax_table

