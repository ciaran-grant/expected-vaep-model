import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from highlight_text import fig_text, ax_text
import matplotlib.gridspec as gridspec
import itertools

from expected_vaep_model.visualisation.afl_colours import team_colours

def create_team_rolling(chains, team, window=10, metric='xvaep'):
    
    team_chains = chains[chains['team'] == team]
    team_chains_for = team_chains.groupby(['year', 'round'])[[metric]].sum().rename(columns={metric:'for'})

    opp_chains = chains[((chains['Home_Team'] == team) | (chains['Away_Team'] == team)) & (chains['team'] != team)]
    team_chains_against = opp_chains.groupby(['year', 'round'])[[metric]].sum().rename(columns={metric:'against'})

    team_rolling = team_chains_for.merge(team_chains_against, left_index=True, right_index = True)

    team_rolling[['for_rolling', 'against_rolling']] = team_rolling[['for', 'against']].rolling(window=window, min_periods=0).mean()
    team_rolling['diff_rolling'] = team_rolling['for_rolling'] - team_rolling['against_rolling']
    
    return team_rolling

def plot_team_rolling_ax(ax, team, team_rolling, annotate=True, years = None):
    
    if years is None:
        data = team_rolling
    else:
        data = team_rolling.loc[(team_rolling.index.get_level_values('Year').isin(years))]
    
    x = [f"{str(x[0])}{str(x[1])}" for x in list(data.index)]
    y_for = data['for_rolling'].values
    y_against = data['against_rolling'].values

    colour_for = team_colours[team]['primary']
    colour_against = team_colours[team]['secondary']

    line_for = ax.plot(x, y_for, label = 'for', color = colour_for, lw=1.5)
    line_against = ax.plot(x, y_against, label = 'against', color= colour_against, lw=1.5)

    ax.fill_between(x, y_against, y_for, where = y_for > y_against, interpolate=True, alpha=0.85, zorder=3, color=line_for[0].get_color())
    ax.fill_between(x, y_against, y_for, where = y_against >= y_for, interpolate=True, alpha=0.85, zorder=3, color=line_against[0].get_color())

    ax.set_ylim(0, 125)

    years = list(data.reset_index()['year'].unique())
    first_rounds = data.reset_index().groupby('year').first().reset_index()['round'].tolist()
    xticks = [str(y) + str(r) for y, r in zip(years, first_rounds)] + [
        f'{str(max(years) + 1)}00'
    ]
    ax.set_xticks(xticks)
    xticklabels = years + [max(years)+1]
    ax.set_xticklabels(xticklabels)

    if annotate:
        ax = annotate_team_rolling_ax(ax, team, data, colour_for, colour_against)

    return ax

def annotate_team_rolling_ax(ax, team, team_rolling, colour_for, colour_against, font = 'Karla'):
    
    for_number = team_rolling['for_rolling'].iloc[-1]
    against_number = team_rolling['against_rolling'].iloc[-1]
    text_colour_for = "black" if colour_for == "white" else "white"
    text_colour_against = "black" if colour_against == "white" else "white"
    ax_text(
        x=0, y=140,
        s=f'<{team}>\n<xvaep for: {for_number:.1f}>  <xvaep against: {against_number:.1f}>',
        highlight_textprops=[
            {'weight':'bold', 'font':'Karla'},
            {'size':'10', 'bbox':{'edgecolor':colour_for, 'facecolor':colour_for, 'pad':1}, 'color':text_colour_for},
            {'size':'10', 'bbox':{'edgecolor':colour_against, 'facecolor':colour_against, 'pad':1}, 'color':text_colour_against},
        ],
        font="Karla",
        ha="left",
        size=14,
    )
    
def plot_all_team_rolling_figure(chains, window, metric = 'xvaep', annotate = True, add_title=True, years = None):
    
    fig = plt.figure(figsize=(18, 24), dpi=300)

    nrows=6
    ncols=3
    gspec = gridspec.GridSpec(
        ncols=ncols, nrows=nrows, figure=fig,
        hspace=0.3
    )
    
    team_list = list(team_colours.keys())
    for plot_counter, (row, col) in enumerate(itertools.product(range(nrows), range(ncols))):
        team = team_list[plot_counter]
        ax = plt.subplot(gspec[row, col])
        team_rolling = create_team_rolling(chains, team, window, metric)
        ax = plot_team_rolling_ax(ax, team, team_rolling, annotate = annotate, years = years)
        
    if add_title:
        fig_text(
            x=0.13, y=0.91,
            s = "AFL - xvaep for & against - 10-game rolling average.",
            size = 22,
            font = "Karla"
        )
    
    return fig, ax