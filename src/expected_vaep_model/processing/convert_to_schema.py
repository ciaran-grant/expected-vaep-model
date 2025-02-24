import numpy as np
import pandas as pd
from expected_vaep_model.processing.types import Types

types = Types()
action_types = types.action_types
outcome_types = types.outcome_types

### Converting chain data to SPADL format
def create_indicator(chains, column_name, condition, shift_value=0, true_value=True, false_value=np.nan):
    chains[column_name] = np.where(chains['Description'].shift(shift_value) == condition, true_value, false_value)
    return chains

def create_centre_bounce(chains):
    return create_indicator(chains, 'From_Centre_Bounce', 'Centre Bounce', shift_value=1, true_value=True, false_value=False)

def create_to_out_of_bounds(chains):
    return create_indicator(chains, 'To_Out_of_Bounds', 'Out of Bounds', shift_value=-1)

def create_kick_inside50(chains):
    return create_indicator(chains, 'Kick_Inside50', 'Kick Into F50', shift_value=-1)

def create_ball_up_call(chains):
    chains = create_indicator(chains, 'To_Ball_Up', 'Ball Up Call', shift_value=-1)
    return create_indicator(chains, 'From_Ball_Up', 'Ball Up Call', shift_value=1)

def create_rushed_behind(chains):
    return create_indicator(chains, 'Rushed_Behind', 'Rushed', shift_value=-1)

def create_contest_target(chains):
    chains['Contest_Target'] = np.where(chains['Description'].shift(-1) == "Contest Target", chains['Player'].shift(-1), np.nan)
    return chains

def create_goal(chains):
    return create_indicator(chains, 'Goal', 'Goal', shift_value=-1)

def create_behind(chains):
    chains = create_indicator(chains, 'Behind', 'Behind', shift_value=-1)
    chains['Behind_Detail'] = chains['Behind_Detail'].shift(-1)
    return chains

def create_out_on_full(chains):
    chains = create_indicator(chains, 'To_Out_On_Full', 'Out On Full After Kick', shift_value=-1)
    return create_indicator(chains, 'From_Out_On_Full', 'OOF Kick In', shift_value=1)

def create_error(chains):
    return create_indicator(chains, 'Error', 'No Pressure Error', true_value=True, false_value=np.nan)

def create_kick_in(chains):
    return create_indicator(chains, 'From_Kick_In', 'Kickin play on', shift_value=1)

def create_mark(chains):
    conditions = ["Uncontested Mark", "Contested Mark", "Mark On Lead"]
    chains['Mark'] = chains['Description'].isin(conditions)
    return chains

def create_contested(chains):
    uncontested_conditions = ["Uncontested Mark", "Loose Ball Get", "Loose Ball Get Crumb", "Gather", "Kickin playon", "Mark On Lead", "Gather From Opposition", "Gather from Opposition", "No Pressure Error", "Knock On"]
    contested_conditions = ["Contested Mark", "Hard Ball Get", "Hard Ball Get Crumb", "Ruck Hard Ball Get", "Spoil", "Gather From Hitout", "Free For", "Contested Knock On", "Ground Kick", "Free For: In Possession", "Free For: Off The Ball"]
    chains['Contested'] = np.nan
    chains['Contested'] = np.where(chains['Description'].isin(uncontested_conditions), False, chains['Contested'])
    chains['Contested'] = np.where(chains['Description'].isin(contested_conditions), True, chains['Contested'])
    return chains

def create_free(chains):
    conditions = ["Free For", "Free For: In Possession", "Free For: Off The Ball"]
    chains['Free'] = chains['Description'].isin(conditions)
    return chains


def create_action_type(chains):
    action_map = {
        "Handball Received": "Receive",
        "Uncontested Mark": "Mark",
        "Contested Mark": "Mark",
        "Mark On Lead": "Mark",
        "Loose Ball Get": "Loose Ball Get",
        "Loose Ball Get Crumb": "Loose Ball Get",
        "Hard Ball Get": "Hard Ball Get",
        "Hard Ball Get Crumb": "Hard Ball Get",
        "Ruck Hard Ball Get": "Hard Ball Get",
        "Gather": "Gather",
        "Gather From Hitout": "Gather",
        "Gather from Opposition": "Gather",
        "Free For": "Free",
        "Free For: In Possession": "Free",
        "Free Advantage": "Free",
        "Free For: Off The Ball": "Free",
        "Ground Kick": "Kick",
        "Kickin short": "Kick",
        "Contested Knock On": "Knock On",
        "Knock On": "Knock On"
    }
    chains['action_type'] = chains['Description'].map(action_map).fillna(chains['Description'])
    chains['action_type'] = np.where(chains['Shot_At_Goal'] == 'TRUE', 'Shot', chains['action_type'])
    return chains

def create_chain_variables(chains):
    functions = [
        create_action_type, create_contested, create_mark, create_free, create_centre_bounce,
        create_kick_inside50, create_ball_up_call, create_rushed_behind, create_contest_target,
        create_goal, create_behind, create_out_on_full, create_error, create_kick_in
    ]
    
    for func in functions:
        chains = func(chains)
    
    return chains

def remove_descriptions(chains):
    
    columns_to_remove = [
        'Centre Bounce', 'Out of Bounds', 'Kick Into F50', 'Kick Inside 50 Result', 
        'Ball Up Call', 'Shot At Goal', 'Rushed', 'Contest Target', 'Goal', 'Behind', 
        'Bounce', 'Mark Fumbled', 'Mark Dropped', 'Out On Full After Kick', 'OOF Kick In', 
        'Kickin play on'
    ]

    chains = chains[~chains['Description'].isna()]
    chains = chains[~chains['Description'].isin(columns_to_remove)]
    return chains

def remove_missing_players(chains):
    return chains[~chains['Player'].isna()]

def filter_action_type(chains):
    return chains[chains['action_type'].isin(action_types)]

def postprocess_end_xy(chains):
    
    # When start x, y swaps to -1*x, -1*y without changing teams, just change end x, y to the start x, y
    same_team_condition = (chains['x'] == -1 * chains['end_x']) & (chains['y'] == -1 * chains['end_y']) & (chains['Team_Chain'] == chains['Team'])
    chains.loc[same_team_condition, ['end_x', 'end_y']] = chains.loc[same_team_condition, ['x', 'y']].values
    
    # When they swap round because the opponent gets possession, those rows are duplicated, so can remove
    different_team_condition = (chains['x'] == -1 * chains['end_x']) & (chains['y'] == -1 * chains['end_y']) & (chains['Team_Chain'] != chains['Team'])
    chains = chains[~different_team_condition]
    
    return chains

def create_end_xy(chains):
    
    # Create end x, y columns to fill in later
    chains['end_x'] = np.nan
    chains['end_y'] = np.nan
    
    # Add x, y location of out of bounds to end of previous action
    chains['end_x'] = np.where(chains['Description'].shift(-1) == "Out of Bounds", chains['x'].shift(-1), chains['end_x'])
    chains['end_y'] = np.where(chains['Description'].shift(-1) == "Out of Bounds", chains['y'].shift(-1), chains['end_y'])
    
    # Move Kick Inside 50 Result x, y coordinates to end x, y of kick
    chains['end_x'] = np.where(chains['Description'].shift(-2) == "Kick Inside 50 Result", chains['x'].shift(-2), chains['end_x'])
    chains['end_y'] = np.where(chains['Description'].shift(-2) == "Kick Inside 50 Result", chains['y'].shift(-2), chains['end_y'])
    
    # Add x, y location of out on full to previous action
    chains['end_x'] = np.where(chains['Description'].shift(-1) == "Out On Full After Kick", chains['x'].shift(-1), chains['end_x'])
    chains['end_y'] = np.where(chains['Description'].shift(-1) == "Out On Full After Kick", chains['y'].shift(-1), chains['end_y'])
    
    chains = remove_descriptions(chains)
    chains = remove_missing_players(chains)
    chains = filter_action_type(chains)
    
    # Add remaining x, y locations of next actions to previous action
    chains['end_x'] = np.where(chains['end_x'].isna(), chains['x'].shift(-1), chains['end_x'])
    chains['end_y'] = np.where(chains['end_y'].isna(), chains['y'].shift(-1), chains['end_y'])

    # Removing duplicates or possession turnovers
    chains = postprocess_end_xy(chains)

    return chains

def create_pitch_xy(chains):
    def adjust_coordinates(direction, team, x, y):
        return np.where((direction == "right") & (team == chains['Away_Team']), -1 * x,
                        np.where((direction == "left") & (team == chains['Home_Team']), -1 * x, x)), \
               np.where((direction == "right") & (team == chains['Away_Team']), -1 * y,
                        np.where((direction == "left") & (team == chains['Home_Team']), -1 * y, y))

    chains['pitch_start_x'], chains['pitch_start_y'] = adjust_coordinates(chains['Home_Team_Direction_Q1'], chains['Team_Chain'], chains['x'], chains['y'])
    chains['pitch_end_x'], chains['pitch_end_y'] = adjust_coordinates(chains['Home_Team_Direction_Q1'], chains['Team_Chain'], chains['end_x'], chains['end_y'])

    return chains

def play_left_to_right(chains):
    chains['start_x'] = chains['x']
    chains['start_y'] = chains['y']
    
    for coord in ['start_x', 'start_y', 'end_x', 'end_y']:
        chains[f'left_right_{coord}'] = chains[coord].copy()
        chains[f'left_right_{coord}'] = np.where((chains['Team'] == chains['Team_Chain']) | (chains['Team'].isna()), chains[f'left_right_{coord}'], -1 * chains[f'left_right_{coord}'])
    return chains

def create_end_distance_metrics(chains):
    for coord in ['start', 'end']:
        chains[f'{coord}_distance_to_right_goal'] = np.sqrt(
            np.square(chains[f'left_right_{coord}_x'] - chains['Venue_Length'] / 2) + 
            np.square(chains[f'left_right_{coord}_y'])
        )
    return chains

def create_inside50(chains):
    chains['Inside50'] = np.where((chains['start_distance_to_right_goal'] > 50) & (chains['end_distance_to_right_goal'] < 50), True, np.nan)
    return chains

def create_duration(chains):
    max_quarter_durations = chains.groupby(['Match_ID', "Period_Number"])['Period_Duration'].max().reset_index()
    max_quarter_durations = max_quarter_durations.rename(columns = {'Period_Duration':'Period_Duration_Max'})
    max_quarter_durations = max_quarter_durations.pivot(index = 'Match_ID', columns='Period_Number', values='Period_Duration_Max')
    chains = chains.merge(max_quarter_durations, how='left', on = ['Match_ID'])
    chains['Duration'] = np.where(chains['Period_Number'] == 1.0, chains['Period_Duration'],
                                np.where(chains['Period_Number'] == 2.0, chains[1.0] + chains['Period_Duration'],
                                        np.where(chains['Period_Number'] == 3.0, chains[1.0] + chains[2.0] + chains['Period_Duration'],
                                                    np.where(chains['Period_Number'] == 4.0, chains[1.0] + chains[2.0] + chains[3.0] + chains['Period_Duration'],
                                                            0))))
    
    return chains

def get_outcome_types(chains):
    chains['NextTeam'] = chains.groupby('Match_ID')['Team'].shift(-1).fillna(0)
    chains['outcome_type'] = "effective"
    
    chains.loc[chains['action_type'].isin(["Kick", "Handball", "Shot"]), 'outcome_type'] = chains['Disposal']
    chains.loc[(chains['action_type'].isin(["Free For", "Knock On"])) & (chains['Team'] != chains['NextTeam']), 'outcome_type'] = "ineffective"
    chains.loc[chains['action_type'] == "Error", 'outcome_type'] = "ineffective"
    
    return chains['outcome_type']

def convert_chains_to_schema(chains):
    
    schema_chains = chains.copy()
    
    schema_chains = create_chain_variables(schema_chains)
    schema_chains = create_end_xy(schema_chains)
    schema_chains = create_pitch_xy(schema_chains)
    schema_chains = play_left_to_right(schema_chains)
    schema_chains = create_end_distance_metrics(schema_chains)
    schema_chains = create_inside50(schema_chains)
    schema_chains = create_duration(schema_chains)
    
    schema_chains['outcome_type'] = get_outcome_types(schema_chains)

    schema_chains = schema_chains.dropna(subset=['Player'])
    schema_chains = filter_action_type(schema_chains)
    
    columns_mapping = {
        'Match_ID': 'match_id',
        'Chain_Number': 'chain_number',
        'Order': 'order',
        'Period_Number': 'period',
        'Period_Duration': 'period_seconds',
        'Duration': 'overall_seconds',
        'Team': 'team',
        'Player': 'player',
        'Contested': 'contested',
        'Mark': 'mark'
    }

    schema_chains = schema_chains.rename(columns=columns_mapping)

    return schema_chains