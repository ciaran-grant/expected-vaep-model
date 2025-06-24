from PIL import Image
import pandas as pd
from highlight_text import ax_text

from expected_vaep_model.visualisation.match_report.plotting.utils import inset_image, match_id_to_home_away

def plot_score_ax(ax, match_summary):
    # Extract info
    score = match_summary['Q4_Score'].values[0]
    home_score = score.split('-')[0]
    home_final_score = home_score.split('.')[-1]
    away_score = score.split('-')[1]
    away_final_score = away_score.split('.')[-1]
    
    # --- Plot score ---
    score_text = f"{home_final_score} - {away_final_score}"
    
    ax_text( 
        s=score_text, 
        x=0.5, y=0.6,  # higher to leave room for logos
        ax=ax, 
        fontsize=60,
        font="Karla",
        fontproperties={'weight': 'bold'},
        ha='center', va='center'
    )
    
def plot_date_venue_ax(ax, match_summary):
    # Extract info
    date = match_summary['Date'].values[0]
    date = pd.to_datetime(date).strftime('%A, %d %B %Y')
    venue = match_summary['Venue'].values[0]
    
    # --- Plot date and venue ---
    date_venue_text = f"<Australian Football League>\n<{date}>\n<{venue}>"
    
    ax_text( 
        s=date_venue_text, 
        x=0.5, y=0.35,
        ax=ax, 
        color = "grey",
        fontsize=20,
        font="Karla",
        textalign = "center",
        fontproperties={'weight': 'bold'},
        ha='center', va='center'
    )
    
def inset_team_logo(ax, logo_file_path, team_name, x, y, width=None, height=None, vertical=False):
    """
    Inset a team logo into the given axis.

    Parameters:
    - ax: The axis to inset the logo into.
    - logo_file_path: Path to the logo file.
    - team_name: Name of the team.
    - x, y: Coordinates for the inset position.
    - width, height: Size of the inset logo.
    """
    logo_url = f"{logo_file_path}/{team_name}.png"
    logo = Image.open(logo_url)
    ax = inset_image(x, y, logo, width=width, height=height, vertical=vertical, ax=ax)
    
    return ax

def plot_match_summary_ax(ax, match_summary, logo_file_path, home_logo_loc, away_logo_loc):
    
    match_id = list(match_summary['Match_ID'].unique())[0]
    home_team, away_team = match_id_to_home_away(match_id)
    
    plot_score_ax(ax=ax, match_summary=match_summary)
    plot_date_venue_ax(ax=ax, match_summary=match_summary)
    _ = inset_team_logo(ax=ax, logo_file_path=logo_file_path, team_name=home_team, x=home_logo_loc['x'], y=home_logo_loc['y'], width=home_logo_loc['width'])
    _ = inset_team_logo(ax=ax, logo_file_path=logo_file_path, team_name=away_team, x=away_logo_loc['x'], y=away_logo_loc['y'], width=away_logo_loc['width'])
    ax.axis('off')
        
    return ax