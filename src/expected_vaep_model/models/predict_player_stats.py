import pandas as pd
import numpy as np

from expected_vaep_model.config import exp_vaep_chain_output_path, player_stats_file_path, exp_vaep_player_stats_output_path

def predict_player_stats(scored_chains, player_stats_file_path, scored_player_stats_output_path):
    
    chains = pd.read_csv(scored_chains)
    player_stats = pd.read_csv(player_stats_file_path)

    def get_player_totals_by_match(chains):
        return chains.groupby(['Match_ID', 'Player', 'Team'])[['xScore', 'exp_vaep_value', 'exp_offensive_value', 'exp_defensive_value']].sum().reset_index().sort_values(by=['Match_ID', "Player", "Team"])

    player_value = get_player_totals_by_match(chains)
    player_stats = player_stats.merge(player_value, how = "left", on = ['Match_ID', 'Player', 'Team'])

    def get_receiver_totals_by_match(chains):
        return chains.groupby(['Match_ID', 'Receiver'])[['exp_vaep_value', 'exp_offensive_value', 'exp_defensive_value']].sum().reset_index().sort_values(by=['Match_ID', "Receiver"])

    chains['Receiver'] = chains['Player'].shift(-1)
    receiver_value = get_receiver_totals_by_match(chains)
    receiver_value = receiver_value.rename(columns = {
        "Receiver":"Player",
        "exp_vaep_value":"exp_vaep_value_received"})

    player_stats = player_stats.merge(receiver_value[['Match_ID', 'Player', 'exp_vaep_value_received']], how = "left", on = ['Match_ID', 'Player'])

    player_stats.to_csv(scored_player_stats_output_path, index=False)
    
if __name__ == "__main__":
    predict_player_stats(exp_vaep_chain_output_path, player_stats_file_path, exp_vaep_player_stats_output_path)