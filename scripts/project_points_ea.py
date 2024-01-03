import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Number of bins in the elevation and azimuth dimension (before resizing).
# NOTE: These should be set such that the "grid" representing the bin divisions in the
# elevation-azimuth view fits the grid-like distribution of points projecte
ELEVATION_SIZE = 28
AZIMUTH_SIZE = 44

def project_points_to_ea_view(point_cloud_file_path, image_resolution):
        # Read the point cloud data file and get its points.
        point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
        points = np.asarray(point_cloud.points)

        # Extract the azimuth, elevation, and range values (x, y, z) from the point cloud.
        azimuth_values = points[:, 0]
        elevation_values = points[:, 1]
        range_values = points[:, 2]

        # Read the point cloud data file and extract the power and Doppler values.
        power_values = []
        with open(point_cloud_file_path, 'r') as f:
            lines = f.readlines()
            flag = False
            # Get all data lines.
            for line in lines:
                if line.startswith("DATA"):
                    flag = True
                    continue
                if flag:
                    # Get the power values for the points of this point cloud.
                    values = line.split()
                    power = float(values[3])  # Assuming the power value of the point is at in field of index 3.
                    power_values.append(power)

        # Convert the power values to NumPy arrays.
        power_values = np.array(power_values)

        # Calculate the inverse of the range values.
        inverse_range = 10.0 / range_values

        # Create a scatter plot with points colored based on power, and with sizes inversely correlated to their range.
        fig = plt.figure()
        scatter = plt.scatter(azimuth_values, elevation_values, c=power_values, cmap='viridis', s=inverse_range)

        # Add labels and remove ticks.
        plt.ylabel('Elevation')
        plt.xlabel('Azimuth')
        plt.xticks([])
        plt.yticks([])

        # Set the limits to match the image resolution.
        plt.xlim(0, image_resolution[0])
        plt.ylim(image_resolution[1], 0)

        # Calculate the half bin width
        half_bin_width = (image_resolution[0] / AZIMUTH_SIZE) / 2

        # Add vertical lines to divide the plot into AZIMUTH_SIZE bins.
        vertical_bin_edges = np.linspace(-half_bin_width, image_resolution[0] + half_bin_width, AZIMUTH_SIZE+1)
        plt.vlines(vertical_bin_edges, 0, image_resolution[1], colors='red', linewidth=0.25)

        # Calculate the half bin height.
        half_bin_height = (image_resolution[1] / ELEVATION_SIZE) / 2

        # Add horizontal lines to divide the plot into ELEVATION_SIZE bins.
        horizontal_bin_edges = np.linspace(-half_bin_height, image_resolution[1] + half_bin_height, ELEVATION_SIZE+1)
        plt.hlines(horizontal_bin_edges, 0, image_resolution[0], colors='red', linewidth=0.25)

        return fig