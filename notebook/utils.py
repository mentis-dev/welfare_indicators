import os
import matplotlib.pyplot as plt

# Ensure the plots directory exists
plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)


def save_plot(fig_plt, filename_base, extension=".png"):
    """
    Saves a given Matplotlib figure to the 'plots' folder.
    Ensures unique filenames by adding a counter if a file already exists.

    Parameters:
    - fig: Matplotlib figure object to save
    - filename_base: Name for the saved file (without extension)
    - extension: File extension (default = .png)
    """
    counter = 0
    filename = f"{filename_base}{extension}"
    filepath = os.path.join(plot_folder, filename)

    # Ensure unique filenames
    while os.path.exists(filepath):
        counter += 1
        filename = f"{filename_base}_{counter}{extension}"
        filepath = os.path.join(plot_folder, filename)

    fig_plt.savefig(filepath)
