import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class Dashboard:
    def __init__(self, rows, cols, figsize=(12, 8), width_ratios=None, height_ratios=None, wspace=0.1, hspace=0.1):
        """
        Initialize the dashboard with a grid layout.

        Parameters:
        - rows (int): Number of rows in the grid.
        - cols (int): Number of columns in the grid.
        - figsize (tuple): Size of the figure.
        """
        self.fig = plt.figure(figsize=figsize, dpi=300, constrained_layout=True)
        self.gs = GridSpec(
            rows, cols, 
            figure=self.fig, 
            width_ratios=width_ratios, height_ratios=height_ratios,
            wspace=wspace, hspace=hspace)
        self.axes = {}

    def add_plot(self, position, plot_func, title=None, **kwargs):
        """
        Add a plot to the dashboard.

        Parameters:
        - position (tuple): Grid position (row, col, rowspan, colspan).
        - plot_func (callable): Function to create the plot. It should accept an `Axes` object as its first argument.
        - title (str): Title of the plot (optional).
        - kwargs: Additional arguments to pass to the plot function.
        """
        row, col, rowspan, colspan = position
        ax = self.fig.add_subplot(self.gs[row:row+rowspan, col:col+colspan])
        plot_func(ax, **kwargs)
        if title:
            ax.set_title(title)
        self.axes[title if title else f"Plot_{len(self.axes)+1}"] = ax

    def show(self):
        """Display the dashboard."""
        plt.show()
        
    def save(self, filename, dpi=300):
        """
        Save the dashboard to a file.

        Parameters:
        - filename (str): Name of the file to save the dashboard.
        - dpi (int): Dots per inch for the saved figure.
        """
        self.fig.savefig(filename, dpi=dpi)
        print(f"Dashboard saved as {filename}")