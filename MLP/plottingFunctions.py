import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd

def plot_subdivided_plane_with_data(data, titles=None, x_labels=None, figsize=(10, 5), color="black", dashes = False):
    """
    Plot multiple line graphs on a subdivided plane from a DataFrame using Seaborn.

    Parameters:
        data (DataFrame): The DataFrame containing the data to be plotted.
        titles (list or None): List of titles for each subplot. If None, no titles will be shown.
        x_labels (list or None): List of labels for the x-axis of each subplot. If None, no x-axis labels will be shown.
        figsize (tuple): Figure size (width, height) for the plot.

    Returns:
        None
    """

    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    num_plots = len(data.columns)

    if titles is None:
        titles = [f"Plot {i+1}" for i in range(num_plots)]
    elif len(titles) != num_plots:
        raise ValueError("Number of titles should match the number of columns in the data.")

    if x_labels is None:
        x_labels = [f"X-axis {i+1}" for i in range(num_plots)]
    elif len(x_labels) != num_plots:
        raise ValueError("Number of x_labels should match the number of columns in the data.")

    rows = int(np.sqrt(num_plots))
    cols = int(np.ceil(num_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            ax.set_title(titles[i])
            sns.lineplot(data=data.iloc[:, i], ax=ax, color=color, dashes=dashes)
            ax.set_xlabel(x_labels[i])
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Restore warnings to their default behavior
    warnings.resetwarnings()

def plot_multiple_graphs(data:pd.DataFrame, title=None, x_label=None, y_label=None, figsize=(10, 6), legend=True, palette="BuGn_r", dashes = False):
    """
    Plot multiple line graphs from a DataFrame using Seaborn.

    Parameters:
        data (DataFrame): The DataFrame containing the data to be plotted.
        title (str or None): Title for the plot. If None, no title will be shown.
        x_label (str or None): Label for the x-axis. If None, no x-axis label will be shown.
        y_label (str or None): Label for the y-axis. If None, no y-axis label will be shown.
        figsize (tuple): Figure size (width, height) for the plot.
        legend (bool): Whether to show the legend. Set to True to show legends, False to hide.

    Returns:
        None
    """

    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette(palette, n_colors=len(data.columns))

    # Plot each column in the data as a separate line
    sns.lineplot(data=data, palette=colors, ax=ax, legend=legend, dashes=dashes)

    if title:
        ax.set_title(title)
    
    if x_label:
        ax.set_xlabel(x_label)
    
    if y_label:
        ax.set_ylabel(y_label)

    if legend:
        ax.legend()

    plt.show()

    # Restore warnings to their default behavior
    warnings.resetwarnings()


def plot_heatmap(data_matrix, x_labels=[], y_labels=[], title=None, figsize=(8, 6), cmap="viridis", annot=False):
    """
    Plot a heatmap from a data matrix using Seaborn.

    Parameters:
        data_matrix (2D array-like): The data matrix for the heatmap.
        x_labels (list or None): Labels for the x-axis (columns). If None, no labels will be shown.
        y_labels (list or None): Labels for the y-axis (rows). If None, no labels will be shown.
        title (str or None): Title for the heatmap. If None, no title will be shown.
        figsize (tuple): Figure size (width, height) for the plot.
        cmap (str): Colormap to use for the heatmap (e.g., "viridis", "coolwarm", "Blues").
        annot (bool): Whether to annotate the cells with the data values.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    sns.heatmap(data_matrix, cmap=cmap, annot=annot, fmt=".2f", xticklabels=x_labels, yticklabels=y_labels)
    
    if title:
        plt.title(title)
    
    plt.xlabel('')  # Remove x-axis label
    plt.ylabel('')  # Remove y-axis label
    plt.show()

def plot_3d(Expected, Predicted=None, label=[None, None], figsize=(10, 6), x_label='X', y_label='Y', z_label='Z', title=None,
            color=['grey', 'red'], marker=['.','.']):
    """
    Create a 3D scatter plot from expected and predicted data points using Matplotlib.

    Parameters:
        Expected (numpy.ndarray): Array containing expected data points with three columns (x, y, z).
        Predicted (numpy.ndarray or None): Array containing predicted data points with three columns (x, y, z).
            If None, only the expected data will be plotted.
        labels (list): List of labels for the data points (e.g., ['Expected', 'Predicted']).
        figsize (tuple): Figure size (width, height) for the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        z_label (str): Label for the z-axis.
        expected_color (str): Color for the expected data points.
        predicted_color (str): Color for the predicted data points.
        expected_marker (str): Marker style for the expected data points.
        predicted_marker (str): Marker style for the predicted data points.

    Returns:
        None
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z coordinates from "Expected"
    x_expected, y_expected, z_expected = Expected[:, 0], Expected[:, 1], Expected[:, 2]

    # Plot the "Expected" points
    ax.scatter(x_expected, y_expected, z_expected, c=color[0], marker=marker[0], label=label[0])

    if Predicted is not None:
        # Extract x, y, and z coordinates from "Predicted"
        x_predicted, y_predicted, z_predicted = Predicted[:, 0], Predicted[:, 1], Predicted[:, 2]

        # Plot the "Predicted" points
        ax.scatter(x_predicted, y_predicted, z_predicted, c=color[1], marker=marker[1], label=label[1])

    # Set labels, title and a legend
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

    # Restore warnings to their default behavior
    warnings.resetwarnings()