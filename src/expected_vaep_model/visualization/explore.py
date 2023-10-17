import numpy as np

def get_player(vaep_data, player):
    
    return vaep_data[vaep_data['player'] == player]

def get_team(vaep_data, team):
    
    return vaep_data[vaep_data['team'] == team]

def get_player_vaep(chains):
    
    players_vaep = chains.groupby(['player', 'team']).agg(
        exp_vaep_value = ("exp_vaep_value", "sum"),
        exp_offensive_value = ("exp_offensive_value", "sum"),
        exp_defensive_value = ("exp_defensive_value", "sum"),
        num_actions = ("action_type", "count")
        ).reset_index()
    players_vaep = players_vaep[['player', 'team', 'exp_vaep_value', 'exp_offensive_value', 'exp_defensive_value', 'num_actions']]
    
    return players_vaep

def get_player_minutes(player_stats):
    
    player_minutes = player_stats.groupby("Player").agg(
        minutes_percent = ("Percent_Played", "sum"),
        games = ("Player", "count")
    ).reset_index()
    player_minutes = player_minutes.rename(columns = {"Player":"player"})
    
    return player_minutes

def convert_vaep_to_rating(player_vaep_mins, minutes_percent_threshold = 1000):
    
    player_ratings = player_vaep_mins[player_vaep_mins['minutes_percent'] > minutes_percent_threshold]
    player_ratings['vaep_rating'] = player_ratings['exp_vaep_value'] * 100 / player_ratings['minutes_percent']
    player_ratings['offensive_rating'] = player_ratings['exp_offensive_value'] * 100 / player_ratings['minutes_percent']
    player_ratings['defensive_rating'] = player_ratings['exp_defensive_value'] * 100 / player_ratings['minutes_percent']
    
    return player_ratings    

def get_match(chains, match_id):
    return chains[chains['match_id'] == match_id]

def get_match_rankings(match_vaep):
    
    match_vaep['vaep_ranking'] = match_vaep['exp_vaep_value'].rank(ascending = False)
    match_vaep['offensive_ranking'] = match_vaep['exp_offensive_value'].rank(ascending = False)
    match_vaep['defensive_ranking'] = match_vaep['exp_defensive_value'].rank(ascending = False)

    match_vaep['normalised_exp_vaep_value'] = (match_vaep['exp_vaep_value'] - match_vaep['exp_vaep_value'].min()) / (match_vaep['exp_vaep_value'].max() - match_vaep['exp_vaep_value'].min())
    match_vaep['normalised_exp_offensive_value'] = (match_vaep['exp_offensive_value'] - match_vaep['exp_offensive_value'].min()) / (match_vaep['exp_offensive_value'].max() - match_vaep['exp_offensive_value'].min())
    match_vaep['normalised_exp_defensive_value'] = (match_vaep['exp_defensive_value'] - match_vaep['exp_defensive_value'].min()) / (match_vaep['exp_defensive_value'].max() - match_vaep['exp_defensive_value'].min())

    match_vaep['z_exp_vaep_value'] = (match_vaep['exp_vaep_value'] - match_vaep['exp_vaep_value'].mean()) / np.std(match_vaep['exp_vaep_value'])
    match_vaep['z_exp_offensive_value'] = (match_vaep['exp_offensive_value'] - match_vaep['exp_offensive_value'].mean()) / np.std(match_vaep['exp_offensive_value'])
    match_vaep['z_exp_defensive_value'] = (match_vaep['exp_defensive_value'] - match_vaep['exp_defensive_value'].mean()) / np.std(match_vaep['exp_defensive_value'])
    
    return match_vaep

def get_vaep_action_summary(vaep_data):
    
    action_vaep = vaep_data.groupby('action_type').agg(
        exp_vaep_value = ("exp_vaep_value", "sum"),
        exp_offensive_value = ("exp_offensive_value", "sum"),
        exp_defensive_value = ("exp_defensive_value", "sum"),
        num_actions = ("action_type", "count")
        ).reset_index()
    action_vaep['action_%'] = action_vaep['num_actions'] / action_vaep['num_actions'].sum()

    action_vaep['exp_vaep_value_per_action'] = action_vaep['exp_vaep_value'] / action_vaep['num_actions']
    action_vaep['off_value_per_action'] = action_vaep['exp_offensive_value'] / action_vaep['num_actions']
    action_vaep['def_value_per_action'] = action_vaep['exp_defensive_value'] / action_vaep['num_actions']

    action_vaep['vaep_%'] = action_vaep['exp_vaep_value'] / action_vaep['exp_vaep_value'].sum()
    action_vaep['offensive_%'] = action_vaep['exp_offensive_value'] / action_vaep['exp_offensive_value'].sum()
    action_vaep['defensive_%'] = action_vaep['exp_defensive_value'] / action_vaep['exp_defensive_value'].sum()
        
    return action_vaep