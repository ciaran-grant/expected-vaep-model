import numpy as np

from expected_vaep_model.visualisation.afl_colours import team_colourmaps

from expected_vaep_model.visualisation.match_report.dashboard import Dashboard
from expected_vaep_model.visualisation.match_report.plotting.plot_expected_action_value_heatmap import plot_expected_action_value_heatmap_ax
from expected_vaep_model.visualisation.match_report.plotting.plot_match_table import plot_match_table_ax
from expected_vaep_model.visualisation.match_report.plotting.plot_match_summary import plot_match_summary_ax
from expected_vaep_model.visualisation.match_report.plotting.plot_match_timeline import plot_match_timeline_ax
from expected_vaep_model.visualisation.match_report.plotting.plot_pitch_density import plot_pitch_density_ax
from expected_vaep_model.visualisation.match_report.plotting.plot_star_rating import plot_match_stats_star_rating_plot_ax, league_average_metrics
from expected_vaep_model.visualisation.match_report.plotting.plot_expected_score import create_expected_shot_maps_both_teams_ax
from expected_vaep_model.visualisation.match_report.plotting.plot_expected_action_value_heatmap import plot_expected_action_value_heatmap_ax

from expected_vaep_model.visualisation.match_report.plotting.load_data import (
    load_xchains,
    load_match_summary,
    load_player_stats,
    load_positions
)

from expected_vaep_model.visualisation.match_report.plotting.utils import match_id_to_home_away, load_fonts

font_path = r"/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-vaep-model/src/expected_vaep_model/fonts"
logo_file_path = r"/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-vaep-model/src/expected_vaep_model/visualisation/logos"
match_report_output_path = r"/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-vaep-model/figures/match_reports"
def match_report(match_id):
    
    load_fonts(font_path)
    
    xchains = load_xchains(match_id)
    match_summary = load_match_summary(match_id)
    player_stats = load_player_stats(match_id)
    positions = load_positions(match_id)
    
    print(f"Loaded data for match ID: {match_id}")
    
    home_team, away_team = match_id_to_home_away(match_id)
    
    xmin, xmax, ymin, ymax = -100, 100, -100, 100
    res = 100
    X, Y = np.meshgrid(np.linspace(xmin, xmax, num=res), np.linspace(ymin, ymax, num=res))
    
    dashboard = Dashboard(5, 3, figsize=(50, 40), height_ratios=[1, 1, 2, 1.5, 2], wspace=0, hspace=0)
    home_logo_loc = {'x': 0.25, 'y': 0.35, 'width': 0.15}
    away_logo_loc = {'x': 0.75, 'y': 0.35, 'width': 0.15}

    print("Creating dashboard plots...")
    
    # Home team plots
    dashboard.add_plot((0, 0, 3, 1), plot_expected_action_value_heatmap_ax, xchains=xchains, team=home_team, team_colourmaps=team_colourmaps)
    print("Added home team expected action value heatmap plot.")
    dashboard.add_plot((3, 0, 2, 1), plot_match_table_ax, positions=positions, player_stats=player_stats, xchains=xchains, team=home_team)
    print("Added home team match table plot.")
    # Middle Plots
    dashboard.add_plot((0, 1, 1, 1), plot_match_summary_ax, match_summary=match_summary, logo_file_path=logo_file_path, home_logo_loc=home_logo_loc, away_logo_loc=away_logo_loc)
    print("Added match summary plot.")
    dashboard.add_plot((1, 1, 1, 1), plot_match_timeline_ax, xchains=xchains)
    print("Added match timeline plot.")
    dashboard.add_plot((2, 1, 1, 1), plot_pitch_density_ax, xchains=xchains, X=X, Y=Y)
    print("Added pitch density plot.")
    dashboard.add_plot((3, 1, 1, 1), plot_match_stats_star_rating_plot_ax, xchains=xchains, match_id=match_id, league_average_metrics=league_average_metrics)
    print("Added star rating plot.")
    dashboard.add_plot((4, 1, 1, 1), create_expected_shot_maps_both_teams_ax, xchains=xchains)
    print("Added expected shot maps plot.")
    # Away team plots
    dashboard.add_plot((0, 2, 3, 1), plot_expected_action_value_heatmap_ax, xchains=xchains, team=away_team, team_colourmaps=team_colourmaps)
    print("Added away team expected action value heatmap plot.")   
    dashboard.add_plot((3, 2, 2, 1), plot_match_table_ax, positions=positions, player_stats=player_stats, xchains=xchains, team=away_team)
    print("Added away team match table plot.")
    
    # Save the dashboard to a file
    dashboard.save(f"{match_report_output_path}/match_report_{match_id}.png")
    print(f"Dashboard saved as match_report_{match_id}.png")
    # Optionally, return the dashboard object for further manipulation or inspection
    return dashboard

if __name__ == "__main__":
    match_id = "AFL_2025_15_Geelong_Brisbane"
    dashboard = match_report(match_id)
    print(f"Match report for {match_id} generated successfully.")
    # You can change the match_id to test with different matches.
    # Ensure that the match_id corresponds to a valid match in your dataset.
    # For example, you can use "AFL_2023_08_Collingwood_Melbourne" or any other valid match ID.
    # Make sure the data files for the specified match are available in the expected directories.
    # This will generate a match report and save it as a PNG file.
    # The dashboard will contain various plots including expected action value heatmaps, match tables, timelines,
    # pitch density plots, star ratings, and expected shot maps for both teams.
    # The generated report will be saved in the current working directory with the filename format "match_report_{match_id}.png".
    # You can adjust the match_id variable to generate reports for different matches as needed.
    
    
    