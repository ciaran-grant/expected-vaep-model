import re
from PIL import Image
import os
import matplotlib.font_manager as fm

def add_space_before_capitals(text):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)

def match_id_to_home_away(match_id):
    """
    Convert match ID to home and away teams.

    Parameters:
    - match_id (str): Match ID in the format "AFL_YYYY_MM_Team1_Team2".

    Returns:
    - tuple: Home team and away team names.
    """
    teams = match_id.split("_")[3:]
    home_team = add_space_before_capitals(teams[0])
    away_team = add_space_before_capitals(teams[1])
    return home_team, away_team

def validate_ax(ax):
    """ Error message when ax is missing."""
    if ax is None:
        msg = "Missing 1 required argument: ax. A Matplotlib axis is required for plotting."
        raise TypeError(msg)
    
def get_aspect(ax):
    """ Get the aspect ratio of an axes.
    From Stackoverflow post by askewchan:
    https://stackoverflow.com/questions/41597177/get-aspect-ratio-of-axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes, default None
    Returns
    -------
    float
    """
    left_bottom, right_top = ax.get_position() * ax.figure.get_size_inches()
    width, height = right_top - left_bottom
    return height / width * ax.get_data_ratio()
    
def inset_image(x, y, image, width=None, height=None, vertical=False, ax=None, **kwargs):
    """ Adds an image as an inset_axes.

    Parameters
    ----------
    x, y: float
    image: array-like or PIL image
        The image data.
    width, height: float, default None
        The width, height of the inset_axes for plotting the image.
        By default, in the data coordinates.
    vertical : bool, default False
        If the orientation is vertical (True), then the code switches the x and y coordinates.
    ax : matplotlib.axes.Axes, default None
        The axis to plot on.

    **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.imshow.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from PIL import Image
    >>> from urllib.request import urlopen
    >>> from mplsoccer import inset_image
    >>> fig, ax = plt.subplots()
    >>> image_url = 'https://upload.wikimedia.org/wikipedia/commons/b/b8/Messi_vs_Nigeria_2018.jpg'
    >>> image = urlopen(image_url)
    >>> image = Image.open(image)
    >>> ax_image = inset_image(0.5, 0.5, image, width=0.2, ax=ax)
    """
    validate_ax(ax)

    if isinstance(image, Image.Image):
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]
    image_aspect = image_height / image_width

    ax_aspect = ax.get_aspect()
    if ax_aspect == 'auto':
        ax_aspect = get_aspect(ax)

    if vertical:
        x, y = y, x

    if height is not None and width is not None:
        raise TypeError('Invalid argument: you must only give one of height or width not both')
    if height is None and width is None:
        raise TypeError('Invalid argument: you must supply one of height or width')

    if width is None:
        width = height / image_aspect * ax_aspect
    elif height is None:
        height = width * image_aspect / ax_aspect

    bbox = (x - width / 2, y - height / 2, width, height)
    ax_inset = ax.inset_axes(bbox, transform=ax.transData, xlim=(0, image_width),
                             ylim=(image_height, 0), **kwargs)
    ax_inset.imshow(image, **kwargs)
    ax_inset.axis('off')
    return ax_inset

def min_max_normalize(x):
    """Min-max normalization to scale values between 0 and 1."""
    return (x - x.min()) / (x.max() - x.min())

def load_fonts(font_path):
    
    for x in os.listdir(font_path):
        if x.split(".")[-1] == "ttf":
            fm.fontManager.addfont(f"{font_path}/{x}")
            try:
                fm.FontProperties(weight=x.split("-")[-1].split(".")[0].lower(), fname=x.split("-")[0])
            except Exception:
                continue