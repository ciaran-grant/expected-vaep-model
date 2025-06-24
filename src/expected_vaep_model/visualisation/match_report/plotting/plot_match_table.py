
from plottable import Table, ColumnDefinition
from plottable.formatters import decimal_to_percent
from plottable.cmap import normed_cmap
from matplotlib.colors import LinearSegmentedColormap
from expected_vaep_model.visualisation.afl_colours import team_colourmaps

cmap = LinearSegmentedColormap.from_list(
    name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256
)
position_map = {
    'Back Pocket Left': 'BPL',
    'Back Pocket Right': 'BPR',
    'Centre': 'C',
    'Centre Half Back': 'CHB',
    'Centre Half Forward': 'CHF',
    'EMERG': 'EM',
    'Forward Pocket Left': 'FPL',
    'Forward Pocket Right': 'FPR',
    'Full Back': 'FB',
    'Full Forward': 'FF',
    'Half Back Flank Left': 'HBL',
    'Half Back Flank Right': 'HBR',
    'Half Forward Flank Left': 'HFL',
    'Half Forward Flank Right': 'HFR',
    'Interchange': 'INT',
    'Ruck': 'RU',
    'Ruck Rover': 'RR',
    'Rover': 'R',
    'Wing Left': 'WL',
    'Wing Right': 'WR',
    'Substitute': 'SUB'
}

def create_match_table_data(positions, player_stats, xchains, team):
    team_positions = positions[positions['Team'] == team]
    team_positions = team_positions[['Position', 'Player']].drop_duplicates().sort_values('Position')
    team_positions['Position'] = team_positions['Position'].map(position_map)
    team_positions.set_index('Position', inplace=True)
    team_positions = team_positions[['Player']]
    team_positions = team_positions.loc[[
        'BPL','FB', 'BPR', 'HBL', 'CHB', 'HBR', 'WL', 'C', 'WR', 'HFL', 'CHF', 'HFR', 
        'FPL', 'FF', 'FPR', 'RU', 'RR', 'R', 'INT', 'SUB'
    ], ['Player']
    ]
    team_positions = team_positions.reset_index()
    
    team_positions_stats = team_positions.merge(player_stats[['Player', 'Percent_Played', 'Disposals', 'Goals', 'Behinds']], left_on='Player', right_on='Player', how='left')
    team_positions_stats = team_positions_stats.rename(columns={'Percent_Played': '% Played'})
    team_positions_stats['% Played'] = team_positions_stats['% Played'] * 0.01
    
    team_xchains = xchains[xchains['Team_Chain'] == team]
    team_player_xstats = team_xchains.groupby(['player']).agg(
        xscore=('xscore', 'sum'),
        xvaep=('xvaep', 'sum'),
        nonshot_xvaep=('nonshot_xvaep', 'sum')
    ).reset_index().round(1)
    team_player_xstats = team_player_xstats.rename(columns={'player':'Player'})
    team_position_xstats = team_positions_stats.merge(team_player_xstats, left_on='Player', right_on='Player', how='left')

    return team_position_xstats[['Position', 'Player', '% Played', 'xscore', 'nonshot_xvaep', 'xvaep']].set_index('Position')

def create_match_table_ax(ax, team_positions, team):
    
    col_defs = (
        [
            ColumnDefinition(
                name = 'Position',
                title = "",
                textprops={'ha': 'center', 'va': 'center'},
                width=0.5,
            ),
            ColumnDefinition(
                name = 'Player',
                textprops={'ha': 'left', 'va': 'center', "fontsize": 24, "font": "Roboto"},
                width=1.25,
            ),
            ColumnDefinition(
                "% Played",
                textprops={'ha': 'center', 'va': 'center'},
                width=0.5,
                formatter=decimal_to_percent,
                cmap=cmap,
            ),
            ColumnDefinition(
                name="xscore",
                group = "Expected Action Value",
                width=0.75,
                textprops={
                    "fontsize": 24,
                    "ha": "center",
                },
                cmap=normed_cmap(team_positions["xscore"], cmap=team_colourmaps[team], num_stds=2.5)
                ),
            ColumnDefinition(
                name="nonshot_xvaep",
                group = "Expected Action Value",
                width=0.75,
                textprops={
                    "fontsize": 24,
                    "ha": "center",
                },
                cmap=normed_cmap(team_positions["nonshot_xvaep"], cmap=team_colourmaps[team], num_stds=2.5)
                ),
            ColumnDefinition(
                name="xvaep",
                group = "Expected Action Value",
                width=0.75,
                textprops={
                    "fontsize":24,
                    "ha": "center",
                },
                cmap=normed_cmap(team_positions["xvaep"], cmap=team_colourmaps[team], num_stds=2.5)
                ),
        ]
    )
    
    tab = Table(
        team_positions,
        column_definitions=col_defs,
        textprops={"fontsize": 24, "font": "Roboto"},
        ax=ax,
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
        row_divider_kw={"linewidth": 1, "linestyle": "-"},
        # odd_row_color="ghostwhite",
        # even_row_color="white",
        )
    
    tab.columns["xscore"].set_fontcolor("w")
    tab.columns["xvaep"].set_fontcolor("w")
    tab.columns["nonshot_xvaep"].set_fontcolor("w")
    
    return tab
        
def plot_match_table_ax(ax, positions, player_stats, xchains, team):
    team_positions = create_match_table_data(positions, player_stats, xchains, team)
    ax = create_match_table_ax(ax=ax, team_positions=team_positions, team=team)    
    return ax