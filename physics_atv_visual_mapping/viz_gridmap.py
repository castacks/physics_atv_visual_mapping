import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# Variables
gridmap_path = "/wheelsafe_ws/gridmap_data.npy"

# Load gridmap
gridmap_npy = np.load(gridmap_path)  # [8, 64, 64]

gridmap_npy = gridmap_npy.transpose(1, 2, 0)

# Visualize
# TODO: figure out how to support multiple viz types
gridmap_rgb = gridmap_npy[..., :3]
vmin = gridmap_rgb.reshape(-1, 3).min(axis=0).reshape(1, 1, 3)
vmax = gridmap_rgb.reshape(-1, 3).max(axis=0).reshape(1, 1, 3)
print(gridmap_rgb.dtype)
print(vmin.dtype)
gridmap_cs = (gridmap_rgb - vmin) / (vmax - vmin)
gridmap_cs = (gridmap_cs * 255.0).astype(np.int32)

# Make GUI to get gridmap_values at selected index
# Create a figure and axis
fig, ax = plt.subplots()

# Display the gridmap RGB image
img = ax.imshow(gridmap_cs)
# Initialize a variable to store the rectangle to show index selected
click_rect = None
# Define a click event handler
def onclick(event):
    global click_rect  # Access the global rectangle object
    # Get the x and y pixel coordinates
    ix, iy = int(event.xdata), int(event.ydata)

    # Check if the click is inside the image
    if ix >= 0 and iy >= 0 and ix < gridmap_rgb.shape[1] and iy < gridmap_rgb.shape[0]:
        # Get the corresponding gridmap values at the clicked coordinates
        gridmap_values = gridmap_npy[
            iy, ix, :
        ]  # Extract the values from gridmap_npy at the clicked point

        # Update the title with the gridmap values
        title = f"GridMap at ({ix}, {iy}): {gridmap_values}"
        ax.set_title(title)
        print(title)

        # Remove the previous rectangle if it exists
        if click_rect:
            click_rect.remove()

        # Draw a red rectangle around the clicked point
        click_rect = patches.Rectangle(
            (ix - 0.5, iy - 0.5), 1, 1, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(click_rect)

        # Redraw the figure to update the rectangle and title
        fig.canvas.draw()


# Connect the click event handler to the figure
cid = fig.canvas.mpl_connect("button_press_event", onclick)

# Show the plot
plt.show()
