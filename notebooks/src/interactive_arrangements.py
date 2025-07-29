import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pandas as pd

# create polygon
poly = Polygon([
    [-66, 56], [-30, 40], [23, 5], [57, -15],
    [27, -60], [-12, -36], [-67, 14], [-80, 36]
])


# Define the figure and axes
fig, ax = plt.subplots()

# Set up the coordinate plane
ax.set_xlim([-80, 60])
ax.set_ylim([-60, 60])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)

# Plot the polygon
xp, yp = poly.exterior.xy
plt.plot(xp, yp)

# Function to handle mouse clicks
x_list = []
y_list = []


def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        ax.plot(x, y, marker='o', markersize=5, color='blue')
        plt.draw()

        # save to list
        x_list.append(round(x, 1))
        y_list.append(round(y, 1))


# Connect the click event to the function
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Display the plot
plt.show()

points = pd.DataFrame({
    'location': [*range(1, len(x_list)+1)],
    'x': x_list,
    'y': y_list
})

print(points)

# determine folder location
if (input("Is this nn_optimal? y/n: ") == 'y'):
    nn_folder = "nn_optimal"
else:
    nn_folder = "nn_suboptimal"

if (input("Is this arrangement 10 points? y/n: ") == 'y'):
    point_num = "10"
else:
    point_num = "8"

num_file = input("What number arrangement is this? ")

file_name = "designed_"+point_num+"p_"+nn_folder+num_file+".csv"

points.to_csv("arrangements/"+nn_folder+"/"+point_num+"_points/"+file_name)
