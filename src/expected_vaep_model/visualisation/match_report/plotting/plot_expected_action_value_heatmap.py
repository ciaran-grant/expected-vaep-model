from mplfooty.pitch import VerticalPitch

def plot_expected_action_value_heatmap_ax(ax, xchains, team, team_colourmaps):
    
    pitch = VerticalPitch(pitch_length=165, pitch_width=135, line_width=1, line_zorder=4, line_colour='black', pad_left=0, pad_right=0, line_alpha=0.5)
    pitch.draw(ax=ax)

    x = xchains[xchains['team'] == team]['left_right_start_x']
    y = xchains[xchains['team'] == team]['left_right_start_y']
    xvaep = xchains[xchains['team'] == team]['xvaep']

    stats = pitch.bin_statistic(x, y, bins=(7, 5), statistic='count', values=xvaep, normalize=False)
    pitch.heatmap(stats, edgecolors="black", cmap=team_colourmaps[team], ax=ax)
    
    return ax