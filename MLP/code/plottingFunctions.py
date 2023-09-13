import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_subdivided_plane_with_data(data, titles, x_labels):
    num_plots = len(data.columns)

    if len(titles) != num_plots:
        raise ValueError("Number of titles should match the number of columns in the data.")

    rows = int(np.sqrt(num_plots))
    cols = int(np.ceil(num_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            ax.set_title(titles[i])
            sns.lineplot(data=data.iloc[:, i], ax=ax)
            ax.set_xlabel(x_labels[i])  # Customize the plot type here
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()



def plot_multiple_graphs(data, title, x_label):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each column in the data as a separate line
    for i, column in enumerate(data.columns):
        sns.lineplot(data=data[column], label=column, ax=ax)

    # Set the title, legend, and x-axis label
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel(x_label)

    # Show the plot
    plt.show()